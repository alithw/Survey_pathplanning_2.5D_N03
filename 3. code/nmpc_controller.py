import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from scipy.interpolate import interp1d

# Import A* Planner từ file bạn đã tạo trước đó
from a_star_cost import AStarPlanner

# ==========================================
# CẤU HÌNH THÔNG SỐ NMPC
# ==========================================
N = 20          # Prediction Horizon (Nhìn trước 20 bước ~ 2 giây)
dt = 0.1        # Thời gian lấy mẫu (10Hz)

# Giới hạn vật lý của Robot (Constraints)
V_MAX = 2.0     # Vận tốc dài tối đa (m/s)
V_MIN = 0.0     # Không đi lùi
W_MAX = 1.0     # Vận tốc góc tối đa (rad/s)
W_MIN = -1.0

# Trọng số ma trận Q (Tracking) và R (Control Effort)
Q_x = 20.0
Q_y = 20.0
Q_theta = 2.0
R_v = 1.0
R_w = 0.5

# ==========================================
# HÀM LOAD MAP VÀ CHẠY A* ĐỂ LẤY WAYPOINTS
# ==========================================
def get_astar_trajectory():
    base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
    filename = "000_025.pt"
    dataset_path = os.path.join(base_dir, filename)
    
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, f"env_{filename}")
        if not os.path.exists(dataset_path):
            print(f"[LỖI] Không tìm thấy file map: {dataset_path}")
            return None, None

    # Load data
    data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    tensors = data['tensors']
    def clean_tensor(tensors, name):
        t = tensors[name].squeeze()
        if len(t.shape) > 2: t = t[0]
        return t.numpy()

    heights = clean_tensor(tensors, 'heights')
    slopes = clean_tensor(tensors, 'slopes')
    t_classes = clean_tensor(tensors, 't_classes')
    GRID_SIZE = heights.shape[0]

    # Khởi tạo Planner giống hệt config của bạn
    planner = AStarPlanner(
        heights=heights, slopes=slopes, t_classes=t_classes, grid_size=GRID_SIZE,
        ssf=1.2, smooth_radius=2, max_step_cost=350.0, max_total_cost=10000.0,
        heuristic_weight=3.0, min_step_cost=2.0
    )

    START_MANUAL = (33, 8)
    GOAL_MANUAL = (40, 58)
    print(f"Đang chạy A* từ {START_MANUAL} đến {GOAL_MANUAL}...")
    
    path, _ = planner.plan(START_MANUAL, GOAL_MANUAL)
    if not path:
        print("A* không tìm được đường!")
        return None, None
        
    return np.array(path), heights

# ==========================================
# HÀM CHÍNH: NMPC ĐIỀU KHIỂN ROBOT THEO A*
# ==========================================
def run_nmpc_simulation():
    # 1. LẤY QUỸ ĐẠO TỪ A*
    astar_path, heights_map = get_astar_trajectory()
    if astar_path is None: return
    
    # A* trả về (row, col). Trục y của ảnh là row, trục x là col.
    x_astar = astar_path[:, 1] 
    y_astar = astar_path[:, 0]

    print(f"Nội suy {len(astar_path)} điểm lưới của A* thành quỹ đạo thời gian thực...")
    
    # Tính toán khoảng cách tích lũy dọc theo đường A*
    dx = np.diff(x_astar)
    dy = np.diff(y_astar)
    dist = np.sqrt(dx**2 + dy**2)
    cum_dist = np.insert(np.cumsum(dist), 0, 0)
    
    # --- BỘ LỌC TOÁN HỌC ---
    # Loại bỏ các điểm có khoảng cách = 0 (trùng nhau) để tránh lỗi chia cho 0 trong interp1d
    _, unique_idx = np.unique(cum_dist, return_index=True)
    unique_idx.sort() # Giữ nguyên thứ tự quỹ đạo
    
    # Nội suy với vận tốc mục tiêu (Target Velocity) = 1.0 m/s
    V_TARGET = 1.0
    t_astar = cum_dist[unique_idx] / V_TARGET
    
    x_astar_clean = x_astar[unique_idx]
    y_astar_clean = y_astar[unique_idx]
    
    total_time = t_astar[-1]
    
    if total_time <= 0:
        print("[LỖI] Quỹ đạo A* quá ngắn hoặc không hợp lệ!")
        return
    
    t_ref = np.arange(0, total_time, dt)
    fx = interp1d(t_astar, x_astar_clean, kind='linear', fill_value="extrapolate")
    fy = interp1d(t_astar, y_astar_clean, kind='linear', fill_value="extrapolate")
    
    x_ref_global = fx(t_ref)
    y_ref_global = fy(t_ref)
    
    # Tính góc hướng theta tham chiếu
    dx_dt = np.gradient(x_ref_global)
    dy_dt = np.gradient(y_ref_global)
    theta_ref_global = np.arctan2(dy_dt, dx_dt)
    
    # Padding dữ liệu ở cuối để NMPC luôn nhìn thấy đủ N bước tương lai khi tới đích
    x_ref_global = np.pad(x_ref_global, (0, N+1), 'edge')
    y_ref_global = np.pad(y_ref_global, (0, N+1), 'edge')
    theta_ref_global = np.pad(theta_ref_global, (0, N+1), 'edge')

    # =======================================================
    # 2. THIẾT LẬP CASADI OPTI STACK
    # =======================================================
    print("Khởi tạo bộ giải NMPC bằng CasADi Opti Stack...")
    opti = ca.Opti()

    X = opti.variable(3, N+1) 
    U = opti.variable(2, N)   

    X0 = opti.parameter(3)          
    X_ref = opti.parameter(3, N+1)  
    slip_ratio = opti.parameter()   

    # Phương trình Động học có trượt (Slip Kinematics)
    for k in range(N):
        x_next = X[0, k] + dt * U[0, k] * (1 - slip_ratio) * ca.cos(X[2, k])
        y_next = X[1, k] + dt * U[0, k] * (1 - slip_ratio) * ca.sin(X[2, k])
        theta_next = X[2, k] + dt * U[1, k]

        opti.subject_to(X[0, k+1] == x_next)
        opti.subject_to(X[1, k+1] == y_next)
        opti.subject_to(X[2, k+1] == theta_next)

    opti.subject_to(X[:, 0] == X0)
    opti.subject_to(opti.bounded(V_MIN, U[0, :], V_MAX))
    opti.subject_to(opti.bounded(W_MIN, U[1, :], W_MAX))

    # Hàm mục tiêu J
    cost = 0
    Q = ca.diag([Q_x, Q_y, Q_theta])
    R = ca.diag([R_v, R_w])

    for k in range(N):
        state_error = X[:, k] - X_ref[:, k]
        cost += ca.mtimes([state_error.T, Q, state_error])  
        cost += ca.mtimes([U[:, k].T, R, U[:, k]])          

    term_error = X[:, N] - X_ref[:, N]
    cost += ca.mtimes([term_error.T, Q*5, term_error])

    opti.minimize(cost)

    p_opts = {"expand": True, "print_time": False}
    s_opts = {"max_iter": 100, "print_level": 0, "sb": "yes"}
    opti.solver("ipopt", p_opts, s_opts)

    # =======================================================
    # 3. CHẠY VÒNG LẶP ĐIỀU KHIỂN BÁM QUỸ ĐẠO A*
    # =======================================================
    print("Bắt đầu NMPC điều khiển Robot bám theo A*...")
    
    # Trạng thái ban đầu: Đặt robot đúng điểm Start của A*
    current_state = np.array([x_ref_global[0], y_ref_global[0], theta_ref_global[0]]) 
    
    history_x, history_y, history_theta = [], [], []
    history_v, history_w = [], []
    
    # Giả định Robot đi trên cỏ/đất bị trượt nhẹ (10%)
    current_slip = 0.10 

    num_steps = len(t_ref)
    start_time_sim = time.time()

    for step in range(num_steps):
        x_ref_horizon = x_ref_global[step : step + N + 1]
        y_ref_horizon = y_ref_global[step : step + N + 1]
        theta_ref_horizon = theta_ref_global[step : step + N + 1]
        ref_matrix = np.vstack((x_ref_horizon, y_ref_horizon, theta_ref_horizon))

        opti.set_value(X0, current_state)
        opti.set_value(X_ref, ref_matrix)
        opti.set_value(slip_ratio, current_slip)

        opti.set_initial(X, ref_matrix)
        opti.set_initial(U, np.ones((2, N)) * 0.1)

        try:
            sol = opti.solve()
            u_optimal = sol.value(U)[:, 0] 
        except:
            print(f"Solver kẹt tại bước {step}. Dùng lệnh điều khiển trước đó.")
            u_optimal = opti.debug.value(U)[:, 0]

        v_cmd = u_optimal[0]
        w_cmd = u_optimal[1]

        # Cập nhật trạng thái thực tế
        current_state[0] += dt * v_cmd * (1 - current_slip) * np.cos(current_state[2])
        current_state[1] += dt * v_cmd * (1 - current_slip) * np.sin(current_state[2])
        current_state[2] += dt * w_cmd

        history_x.append(current_state[0])
        history_y.append(current_state[1])
        history_theta.append(current_state[2])
        history_v.append(v_cmd)
        history_w.append(w_cmd)
        
        if step % 20 == 0:
            print(f"NMPC Step {step}/{num_steps} | Lỗi Tracking: {np.linalg.norm(current_state[:2] - ref_matrix[:2, 0]):.3f}m | Lệnh [v:{v_cmd:.2f}, w:{w_cmd:.2f}]")

    print(f"Mô phỏng hoàn tất! Thời gian chạy NMPC: {time.time() - start_time_sim:.2f} giây.")

    # =======================================================
    # 4. TRỰC QUAN HÓA KẾT QUẢ ĐỂ ĐƯA VÀO REPORT
    # =======================================================
    print("Đang tạo biểu đồ...")
    plt.figure(figsize=(16, 8))

    # Biểu đồ 1: Tracking trên nền Terrain
    ax1 = plt.subplot(1, 2, 1)
    if heights_map is not None:
        img = ax1.imshow(heights_map, cmap='terrain', origin='upper')
        plt.colorbar(img, ax=ax1, label='Độ cao địa hình (z)', shrink=0.8)
    
    # A* Path (Dashed)
    plt.plot(x_astar, y_astar, 'r--', linewidth=2, label="Global Plan (A*)")
    # NMPC Path (Solid Blue)
    plt.plot(history_x, history_y, 'b-', linewidth=2.5, label=f"Actual Path (NMPC - Slip {current_slip*100}%)")
    
    plt.scatter(x_astar[0], y_astar[0], c='yellow', edgecolors='black', s=100, label="Start", zorder=5)
    plt.scatter(x_astar[-1], y_astar[-1], c='blue', edgecolors='white', s=100, label="Goal", zorder=5)
    
    plt.title("NMPC Path Tracking over A* Reference", fontsize=14)
    plt.xlabel("X (Cột lưới)")
    plt.ylabel("Y (Hàng lưới)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Biểu đồ 2: Tín hiệu điều khiển thời gian thực
    ax2 = plt.subplot(1, 2, 2)
    time_axis = np.arange(0, total_time, dt)[:num_steps]
    
    ax2.step(time_axis, history_v, 'b-', label='Linear Velocity v (m/s)', linewidth=2)
    ax2.step(time_axis, history_w, 'g-', label='Angular Velocity w (rad/s)', linewidth=2)
    ax2.axhline(V_MAX, color='b', linestyle=':', alpha=0.5, label='Max v')
    ax2.axhline(W_MAX, color='g', linestyle=':', alpha=0.5, label='Max w')
    ax2.axhline(W_MIN, color='g', linestyle=':', alpha=0.5)
    
    plt.title("NMPC Control Effort over Time", fontsize=14)
    plt.xlabel("Thời gian mô phỏng (s)")
    plt.ylabel("Vận tốc lệnh (Command)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # --- TỰ ĐỘNG LƯU ẢNH TRƯỚC KHI HIỂN THỊ ---
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/nmpc_tracking_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ sắc nét vào: {save_path}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Cửa sổ đồ họa bị chặn, nhưng ảnh đã được lưu an toàn. Chi tiết lỗi GUI: {e}")

if __name__ == "__main__":
    run_nmpc_simulation()