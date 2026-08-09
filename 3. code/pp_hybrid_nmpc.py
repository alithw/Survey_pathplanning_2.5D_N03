import casadi as ca
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from a_star_cost import AStarPlanner

# ==========================================
# CẤU HÌNH THÔNG SỐ NMPC
# ==========================================
N = 20          # Prediction Horizon (20 steps ~ 2 seconds)
dt = 0.1        # Sampling time (10Hz)
V_MAX = 2.0     
V_MIN = 0.0     
W_MAX = 1.0     
W_MIN = -1.0

# Q (Tracking) & R (Control Effort) weights
Q_x = 40.0
Q_y = 40.0
Q_theta = 4.0
R_v = 1.0
R_w = 0.5

# ==========================================
# 1. MẠNG NEURAL SLIPMLP (System 3 - Section 8)
# ==========================================
class SlipMLP(nn.Module):
    def __init__(self):
        super(SlipMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() # Constrain slip to [0, 1]
        )
    def forward(self, x):
        return self.net(x)

def generate_slip_training_data(num_samples=2000):
    """
    Sinh dữ liệu huấn luyện vật lý (Physics-informed synthetic data) cho SlipMLP.
    Độ trượt phụ thuộc vào:
      - Loại đất (soil_class): 0-7
      - Độ dốc (slope): dốc từ 0.0-45.0 độ (sẽ chuyển sang radian)
      - Vận tốc (velocity): vận tốc cao gây trượt lớn hơn
    """
    np.random.seed(42)
    # 3 inputs: soil_class (0-7), slope_deg (0.0-45.0 deg), velocity (0.0-2.0 m/s)
    soils = np.random.randint(0, 8, num_samples)
    slopes_deg = np.random.uniform(0.0, 45.0, num_samples)
    vels = np.random.uniform(0.0, 2.0, num_samples)
    
    slips = []
    slopes_rad = np.radians(slopes_deg)
    for i in range(num_samples):
        base_slip = {0: 0.05, 1: 0.08, 2: 0.15, 3: 0.40, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10}[soils[i]]
        slope_effect = 0.35 * np.sin(slopes_rad[i])
        vel_effect = 0.05 * vels[i]
        
        total_slip = base_slip + slope_effect + vel_effect
        total_slip = np.clip(total_slip + np.random.normal(0, 0.02), 0.01, 0.85)
        slips.append(total_slip)
        
    X = np.stack([soils / 7.0, slopes_rad, vels / 2.0], axis=1).astype(np.float32)
    y = np.array(slips, dtype=np.float32).reshape(-1, 1)
    
    return torch.tensor(X), torch.tensor(y)

def train_slip_mlp(model, epochs=30):
    X, y = generate_slip_training_data()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
    print(f"-> Đã huấn luyện xong mạng SlipMLP. MSE Loss cuối cùng: {loss.item():.6f}")
    model.eval()

# Khởi tạo và huấn luyện SlipMLP
slip_model = SlipMLP()
train_slip_mlp(slip_model)

def get_actual_slip(soil_class, slope, velocity):
    """Mô hình mô phỏng vật lý thực tế của địa hình (Ground Truth Slip)"""
    # map soil class 0-7
    base_slip = {0: 0.05, 1: 0.08, 2: 0.15, 3: 0.40, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10}.get(int(soil_class), 0.10)
    # slope is in degrees in map features, convert to radians
    slope_rad = np.radians(slope)
    slope_effect = 0.35 * np.sin(slope_rad)
    vel_effect = 0.05 * velocity
    return np.clip(base_slip + slope_effect + vel_effect, 0.01, 0.85)

# ==========================================
# 2. LOAD MAP VÀ QUỸ ĐẠO A*
# ==========================================
def get_astar_trajectory():
    base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
    filename = "000_025.pt"
    dataset_path = os.path.join(base_dir, filename)
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, f"env_{filename}")
        
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

    planner = AStarPlanner(
        heights=heights, slopes=slopes, t_classes=t_classes, grid_size=GRID_SIZE,
        ssf=1.2, smooth_radius=2, max_step_cost=350.0, max_total_cost=10000.0,
        heuristic_weight=3.0, min_step_cost=2.0
    )

    START_MANUAL = (33, 8)
    GOAL_MANUAL = (38, 58)
    path, _ = planner.plan(START_MANUAL, GOAL_MANUAL)
    if path is None:
        # Fallback to (10, 10) -> (50, 50) if needed
        for s, g in [((33, 8), (38, 58)), ((10, 10), (50, 50))]:
            path, _ = planner.plan(s, g)
            if path is not None: break

    return np.array(path), heights, slopes, t_classes


# ==========================================
# 3. CHẠY MÔ PHỎNG NMPC (COMPENSATION VS NO COMPENSATION)
# ==========================================
def run_nmpc_comparison(hybrid_adaptation=True):
    astar_path, heights_map, slopes_map, t_classes_map = get_astar_trajectory()
    GRID_SIZE = heights_map.shape[0]
    
    x_astar = astar_path[:, 1] 
    y_astar = astar_path[:, 0]
    
    dx = np.diff(x_astar)
    dy = np.diff(y_astar)
    dist = np.sqrt(dx**2 + dy**2)
    cum_dist = np.insert(np.cumsum(dist), 0, 0)
    
    _, unique_idx = np.unique(cum_dist, return_index=True)
    unique_idx.sort()
    
    V_TARGET = 1.0
    t_astar = cum_dist[unique_idx] / V_TARGET
    
    x_astar_clean = x_astar[unique_idx]
    y_astar_clean = y_astar[unique_idx]
    
    total_time = t_astar[-1]
    t_ref = np.arange(0, total_time, dt)
    
    fx = interp1d(t_astar, x_astar_clean, kind='linear', fill_value="extrapolate")
    fy = interp1d(t_astar, y_astar_clean, kind='linear', fill_value="extrapolate")
    
    x_ref_global = fx(t_ref)
    y_ref_global = fy(t_ref)
    
    dx_dt = np.gradient(x_ref_global)
    dy_dt = np.gradient(y_ref_global)
    theta_ref_global = np.arctan2(dy_dt, dx_dt)
    
    x_ref_global = np.pad(x_ref_global, (0, N+1), 'edge')
    y_ref_global = np.pad(y_ref_global, (0, N+1), 'edge')
    theta_ref_global = np.pad(theta_ref_global, (0, N+1), 'edge')

    # CasADi Setup
    opti = ca.Opti()
    X = opti.variable(3, N+1) 
    U = opti.variable(2, N)   
    X0 = opti.parameter(3)          
    X_ref = opti.parameter(3, N+1)  
    slip_ratio = opti.parameter(N)   

    # State Equations
    for k in range(N):
        x_next = X[0, k] + dt * U[0, k] * (1 - slip_ratio[k]) * ca.cos(X[2, k])
        y_next = X[1, k] + dt * U[0, k] * (1 - slip_ratio[k]) * ca.sin(X[2, k])
        theta_next = X[2, k] + dt * U[1, k]

        opti.subject_to(X[0, k+1] == x_next)
        opti.subject_to(X[1, k+1] == y_next)
        opti.subject_to(X[2, k+1] == theta_next)

    opti.subject_to(X[:, 0] == X0)
    opti.subject_to(opti.bounded(V_MIN, U[0, :], V_MAX))
    opti.subject_to(opti.bounded(W_MIN, U[1, :], W_MAX))

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

    # Simulation loop
    current_state = np.array([x_ref_global[0], y_ref_global[0], theta_ref_global[0]], dtype=float)
    
    history_x, history_y = [], []
    history_slips = []
    
    num_steps = len(t_ref)
    last_ref_idx = 0

    for step in range(num_steps):
        x_curr, y_curr = current_state[0], current_state[1]
        
        # Path-Parameterization Reference: project current position onto the global reference path
        search_start = max(0, last_ref_idx)
        search_end = min(len(t_ref), last_ref_idx + 15)
        if search_end > search_start:
            segment_distances = (x_ref_global[search_start:search_end] - x_curr)**2 + (y_ref_global[search_start:search_end] - y_curr)**2
            closest_idx = search_start + np.argmin(segment_distances)
        else:
            closest_idx = last_ref_idx
            
        last_ref_idx = closest_idx
        
        # Đọc tham số địa hình tại vị trí hiện tại
        r = int(np.clip(y_curr, 0, GRID_SIZE-1))
        c = int(np.clip(x_curr, 0, GRID_SIZE-1))
        soil = t_classes_map[r, c]
        slope = slopes_map[r, c]
        
        # 1. Tính toán Dynamic Slip của môi trường thực tế (Ground Truth)
        # Giả sử robot đang đi với vận tốc xấp xỉ 1.0 m/s
        actual_slip = get_actual_slip(soil, slope, 1.0)
        
        # 2. Bộ điều khiển ước lượng Slip dọc theo tầm dự báo N (Horizon-lookahead)
        x_ref_horizon = x_ref_global[closest_idx : closest_idx + N + 1]
        y_ref_horizon = y_ref_global[closest_idx : closest_idx + N + 1]
        theta_ref_horizon = theta_ref_global[closest_idx : closest_idx + N + 1]
        ref_matrix = np.vstack((x_ref_horizon, y_ref_horizon, theta_ref_horizon))

        estimated_slips = []
        for k in range(N):
            x_r_k = x_ref_horizon[k]
            y_r_k = y_ref_horizon[k]
            r_k = int(np.clip(y_r_k, 0, GRID_SIZE-1))
            c_k = int(np.clip(x_r_k, 0, GRID_SIZE-1))
            soil_k = t_classes_map[r_k, c_k]
            slope_k = slopes_map[r_k, c_k]
            
            if hybrid_adaptation:
                # Sử dụng mạng neural SlipMLP để dự đoán và bù trượt
                # Convert slope to radians as SlipMLP was trained on radians
                slope_rad_k = np.radians(slope_k)
                in_tensor = torch.tensor([[soil_k / 7.0, slope_rad_k, 1.0 / 2.0]], dtype=torch.float32)
                with torch.no_grad():
                    est_s_k = slip_model(in_tensor).item()
                # Damping / capping estimated slip to 0.40 to prevent velocity command spikes
                est_s_k = min(est_s_k, 0.40)
            else:
                # Mô hình truyền thống: Bỏ qua biến động địa hình, coi độ trượt cố định 10%
                est_s_k = 0.10
            estimated_slips.append(est_s_k)

        opti.set_value(X0, current_state)
        opti.set_value(X_ref, ref_matrix)
        opti.set_value(slip_ratio, np.array(estimated_slips)) # Bù trượt bằng slip ước lượng dọc theo horizon

        opti.set_initial(X, ref_matrix)
        opti.set_initial(U, np.ones((2, N)) * 0.1)

        try:
            sol = opti.solve()
            u_optimal = sol.value(U)[:, 0] 
        except:
            u_optimal = opti.debug.value(U)[:, 0]

        v_cmd = u_optimal[0]
        w_cmd = u_optimal[1]

        # Robot phản ứng với thực tế (actual slip)
        current_state[0] += dt * v_cmd * (1 - actual_slip) * np.cos(current_state[2])
        current_state[1] += dt * v_cmd * (1 - actual_slip) * np.sin(current_state[2])
        current_state[2] += dt * w_cmd

        history_x.append(current_state[0])
        history_y.append(current_state[1])
        history_slips.append(actual_slip)
        
    return np.array(history_x), np.array(history_y), np.array(history_slips), x_ref_global[:num_steps], y_ref_global[:num_steps]

if __name__ == "__main__":
    print("="*60)
    print(" KHỞI ĐỘNG HỆ THỐNG MÔ PHỎNG SO SÁNH HYBRID NMPC VS TRADITIONAL NMPC")
    print("="*60)
    
    # Chạy mô phỏng cả 2 bộ điều khiển
    print("\n[BỘ ĐIỀU KHIỂN CŨ] Đang chạy Traditional NMPC (Không bù trượt địa hình)...")
    tx_trad, ty_trad, slips_trad, rx, ry = run_nmpc_comparison(hybrid_adaptation=False)
    
    print("\n[BỘ ĐIỀU KHIỂN MỚI] Đang chạy Hybrid NMPC (Tự thích ứng & bù trượt bằng SlipMLP)...")
    tx_hyb, ty_hyb, slips_hyb, _, _ = run_nmpc_comparison(hybrid_adaptation=True)
    
    # Tính toán lỗi bám quỹ đạo hình học (Geometric Tracking Error RMS)
    errs_trad = []
    for i in range(len(tx_trad)):
        dists = (rx - tx_trad[i])**2 + (ry - ty_trad[i])**2
        errs_trad.append(np.min(dists))
    rmse_trad = np.sqrt(np.mean(errs_trad))
    
    errs_hyb = []
    for i in range(len(tx_hyb)):
        dists = (rx - tx_hyb[i])**2 + (ry - ty_hyb[i])**2
        errs_hyb.append(np.min(dists))
    rmse_hyb = np.sqrt(np.mean(errs_hyb))
    
    print("="*60)
    print("🎉 KẾT QUẢ SO SÁNH HIỆU NĂNG ĐIỀU KHIỂN BÁM ĐƯỜNG:")
    print("="*60)
    print(f"  Traditional NMPC (Slip 10% cố định)  - Tracking RMSE: {rmse_trad:.3f} m")
    print(f"  Hybrid NMPC (Mạng Neural SlipMLP)  - Tracking RMSE: {rmse_hyb:.3f} m")
    improvement = (rmse_trad - rmse_hyb) / rmse_trad * 100
    print(f"  ⚡ ĐỘ CHÍNH XÁC TĂNG VƯỢT TRỘI: {improvement:.2f}%")
    print("="*60)
    
    # Vẽ đồ thị so sánh
    plt.figure(figsize=(15, 6))
    
    # Đồ thị 1: So sánh quỹ đạo bám đường
    ax1 = plt.subplot(1, 2, 1)
    _, heights_map, _, _ = get_astar_trajectory()
    ax1.imshow(heights_map, cmap='terrain', origin='upper')
    
    ax1.plot(rx, ry, 'r--', linewidth=2, label="Quỹ đạo tham chiếu (A*)")
    ax1.plot(tx_trad, ty_trad, 'm-', linewidth=2, label=f"Traditional NMPC (RMSE: {rmse_trad:.2f}m)")
    ax1.plot(tx_hyb, ty_hyb, 'b-', linewidth=2.5, label=f"Hybrid NMPC (RMSE: {rmse_hyb:.2f}m)")
    
    ax1.scatter(rx[0], ry[0], c='yellow', edgecolors='black', s=100, label="Start", zorder=5)
    ax1.scatter(rx[-1], ry[-1], c='blue', edgecolors='white', s=100, label="Goal", zorder=5)
    
    ax1.set_title("So sánh Quỹ đạo bám đường (Path Tracking)", fontsize=12)
    ax1.set_xlabel("X (Cột lưới)")
    ax1.set_ylabel("Y (Hàng lưới)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Đồ thị 2: Độ lỗi tracking theo thời gian
    ax2 = plt.subplot(1, 2, 2)
    err_trad = np.sqrt((tx_trad - rx)**2 + (ty_trad - ry)**2)
    err_hyb = np.sqrt((tx_hyb - rx)**2 + (ty_hyb - ry)**2)
    
    time_axis = np.arange(len(err_trad)) * dt
    ax2.plot(time_axis, err_trad, 'm-', linewidth=2, label='Traditional NMPC')
    ax2.plot(time_axis, err_hyb, 'b-', linewidth=2.2, label='Hybrid NMPC (Tích hợp SlipMLP)')
    
    ax2.set_title("Độ lệch vị trí thời gian thực (Tracking Error Over Time)", fontsize=12)
    ax2.set_xlabel("Thời gian (giây)")
    ax2.set_ylabel("Sai lệch vị trí (cm)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Lưu ảnh
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/nmpc_controller_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Đồng thời lưu/sao chép sang thư mục 1. Report/figures
    report_fig_dir = os.path.join(os.path.dirname(__file__), '..', '1. Report', 'figures')
    os.makedirs(report_fig_dir, exist_ok=True)
    report_save_path = os.path.join(report_fig_dir, 'nmpc_controller_comparison.png')
    plt.savefig(report_save_path, dpi=300, bbox_inches='tight')
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ so sánh vào: {save_path} và {report_save_path}")

    # plt.show()
