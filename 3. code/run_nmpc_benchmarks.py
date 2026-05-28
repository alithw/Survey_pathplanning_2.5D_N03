import casadi as ca
import numpy as np
import os
import glob
import time
import json
import pandas as pd
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
# 1. MẠNG NEURAL SLIPMLP (System 3)
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
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def generate_slip_training_data(num_samples=2000):
    np.random.seed(42)
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
    print(f"-> Đã huấn luyện xong SlipMLP. MSE Loss cuối cùng: {loss.item():.6f}")
    model.eval()

# Khởi tạo và huấn luyện SlipMLP
slip_model = SlipMLP()
train_slip_mlp(slip_model)

def get_actual_slip(soil_class, slope, velocity):
    base_slip = {0: 0.05, 1: 0.08, 2: 0.15, 3: 0.40, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10}.get(int(soil_class), 0.10)
    slope_rad = np.radians(slope)
    slope_effect = 0.35 * np.sin(slope_rad)
    vel_effect = 0.05 * velocity
    return np.clip(base_slip + slope_effect + vel_effect, 0.01, 0.85)

# ==========================================
# 2. LOAD MAP VÀ QUỸ ĐẠO THAM CHIẾU
# ==========================================
def get_trajectory_for_map(npz_path):
    data = np.load(npz_path)
    heights = data['heights']
    slopes = data['slopes']
    t_classes = data['t_classes']
    start = data['start']
    goal = data['goal']
    
    GRID_SIZE = heights.shape[0]
    
    # Thử lập quỹ đạo bằng A* động trước để đảm bảo tính tối ưu nhất
    try:
        planner = AStarPlanner(
            heights=heights, slopes=slopes, t_classes=t_classes, grid_size=GRID_SIZE,
            ssf=1.2, smooth_radius=2, max_step_cost=350.0, max_total_cost=10000.0,
            heuristic_weight=3.0, min_step_cost=2.0
        )
        path, _ = planner.plan(tuple(start), tuple(goal))
        if path is not None and len(path) > 5:
            return np.array(path), heights, slopes, t_classes
    except Exception as e:
        print(f"      [Cảnh báo] AStarPlanner thất bại trên {os.path.basename(npz_path)}: {e}. Sử dụng path lưu sẵn.")
        
    # Dự phòng: Sử dụng path lập sẵn lưu trong npz
    if 'path' in data and len(data['path']) > 2:
        return np.array(data['path']), heights, slopes, t_classes
    
    raise ValueError(f"Không thể tạo hoặc tải quỹ đạo cho {npz_path}")

# ==========================================
# 3. MÔ PHỎNG NMPC THEO DÕI QUỸ ĐẠO
# ==========================================
def simulate_nmpc(astar_path, heights_map, slopes_map, t_classes_map, hybrid_adaptation=True):
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
    
    if len(t_ref) < 2:
        return 999.0, 999.0, 0.0, 0.0 # Trả về lỗi nếu quỹ đạo quá ngắn
        
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
    history_v, history_w = [], []
    
    num_steps = len(t_ref)
    last_ref_idx = 0

    for step in range(num_steps):
        x_curr, y_curr = current_state[0], current_state[1]
        
        search_start = max(0, last_ref_idx)
        search_end = min(len(t_ref), last_ref_idx + 15)
        if search_end > search_start:
            segment_distances = (x_ref_global[search_start:search_end] - x_curr)**2 + (y_ref_global[search_start:search_end] - y_curr)**2
            closest_idx = search_start + np.argmin(segment_distances)
        else:
            closest_idx = last_ref_idx
            
        last_ref_idx = closest_idx
        
        r = int(np.clip(y_curr, 0, GRID_SIZE-1))
        c = int(np.clip(x_curr, 0, GRID_SIZE-1))
        soil = t_classes_map[r, c]
        slope = slopes_map[r, c]
        
        actual_slip = get_actual_slip(soil, slope, 1.0)
        
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
                slope_rad_k = np.radians(slope_k)
                in_tensor = torch.tensor([[soil_k / 7.0, slope_rad_k, 1.0 / 2.0]], dtype=torch.float32)
                with torch.no_grad():
                    est_s_k = slip_model(in_tensor).item()
                est_s_k = min(est_s_k, 0.40)
            else:
                est_s_k = 0.10
            estimated_slips.append(est_s_k)

        opti.set_value(X0, current_state)
        opti.set_value(X_ref, ref_matrix)
        opti.set_value(slip_ratio, np.array(estimated_slips))

        opti.set_initial(X, ref_matrix)
        opti.set_initial(U, np.ones((2, N)) * 0.1)

        try:
            sol = opti.solve()
            u_optimal = sol.value(U)[:, 0] 
        except:
            u_optimal = opti.debug.value(U)[:, 0]

        v_cmd = u_optimal[0]
        w_cmd = u_optimal[1]

        current_state[0] += dt * v_cmd * (1 - actual_slip) * np.cos(current_state[2])
        current_state[1] += dt * v_cmd * (1 - actual_slip) * np.sin(current_state[2])
        current_state[2] += dt * w_cmd

        history_x.append(current_state[0])
        history_y.append(current_state[1])
        history_v.append(v_cmd)
        history_w.append(w_cmd)
        
    history_x = np.array(history_x)
    history_y = np.array(history_y)
    
    # Tính các chỉ số lỗi hình học
    errs = []
    for i in range(len(history_x)):
        dists = (x_ref_global[:num_steps] - history_x[i])**2 + (y_ref_global[:num_steps] - history_y[i])**2
        errs.append(np.min(dists))
    rmse = np.sqrt(np.mean(errs))
    max_err = np.sqrt(np.max(errs))
    
    # Năng lượng điều khiển
    energy = float(np.sum(np.array(history_v)**2 + 0.5 * np.array(history_w)**2) * dt)
    
    # Vị trí đích bám
    final_dist = np.sqrt((history_x[-1] - x_ref_global[num_steps-1])**2 + (history_y[-1] - y_ref_global[num_steps-1])**2)
    success = 1.0 if (final_dist < 2.0 and max_err < 2.5) else 0.0
    
    return rmse, max_err, energy, success

# ==========================================
# 4. CHẠY THỰC NGHIỆM TRÊN 30 BẢN ĐỒ
# ==========================================
def run_nmpc_benchmarks():
    print("="*70)
    print(" KHỞI ĐỘNG HỆ THỐNG ĐÁNH GIÁ NMPC DIỆN RỘNG TRÊN TẬP DỮ LIỆU BENCHNAV")
    print("="*70)
    
    gt_dir = "gt_dataset"
    log_dir = "../4. logs"
    os.makedirs(log_dir, exist_ok=True)
    
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npz")))
    if not gt_files:
        print("[LỖI] Không tìm thấy dữ liệu *.npz nào trong gt_dataset!")
        return
        
    # Lấy 30 bản đồ đại diện trải dài đều trên 509 bản đồ
    step_size = len(gt_files) // 30
    selected_files = [gt_files[i * step_size] for i in range(30)]
    
    print(f"-> Chọn được 30 bản đồ từ tổng số {len(gt_files)} bản đồ để đánh giá NMPC.")
    
    results = []
    
    for idx, filepath in enumerate(selected_files):
        filename = os.path.basename(filepath)
        print(f"[{idx+1}/30] Đang xử lý bản đồ: {filename}...")
        
        try:
            astar_path, heights_map, slopes_map, t_classes_map = get_trajectory_for_map(filepath)
        except Exception as e:
            print(f"   [Bỏ qua] Lỗi nạp bản đồ hoặc lập quỹ đạo: {e}")
            continue
            
        # Traditional NMPC
        t_start = time.perf_counter()
        rmse_trad, max_err_trad, energy_trad, succ_trad = simulate_nmpc(
            astar_path, heights_map, slopes_map, t_classes_map, hybrid_adaptation=False
        )
        time_trad = time.perf_counter() - t_start
        
        # Hybrid NMPC
        t_start = time.perf_counter()
        rmse_hyb, max_err_hyb, energy_hyb, succ_hyb = simulate_nmpc(
            astar_path, heights_map, slopes_map, t_classes_map, hybrid_adaptation=True
        )
        time_hyb = time.perf_counter() - t_start
        
        if rmse_trad == 999.0 or rmse_hyb == 999.0:
            print("   [Bỏ qua] Mô phỏng gặp lỗi quỹ đạo cực ngắn.")
            continue
            
        improvement = (rmse_trad - rmse_hyb) / rmse_trad * 100
        
        entry = {
            'map': filename,
            'trad_rmse_m': rmse_trad,
            'trad_max_err_m': max_err_trad,
            'trad_energy': energy_trad,
            'trad_success': succ_trad,
            'trad_time_s': time_trad,
            
            'hyb_rmse_m': rmse_hyb,
            'hyb_max_err_m': max_err_hyb,
            'hyb_energy': energy_hyb,
            'hyb_success': succ_hyb,
            'hyb_time_s': time_hyb,
            'improvement_pct': improvement
        }
        results.append(entry)
        
        print(f"   Traditional NMPC - RMSE: {rmse_trad:.3f} m, Max Err: {max_err_trad:.3f} m, Success: {int(succ_trad)}")
        print(f"   Hybrid NMPC      - RMSE: {rmse_hyb:.3f} m, Max Err: {max_err_hyb:.3f} m, Success: {int(succ_hyb)}")
        print(f"   ⚡ Cải thiện độ chính xác bám đường: {improvement:.2f}%")
        print("-" * 60)
        
    # Xuất log kết quả
    df = pd.DataFrame(results)
    csv_path = os.path.join(log_dir, "nmpc_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"🎉 Đã lưu kết quả chi tiết ra CSV: {csv_path}")
    
    # Tính toán thống kê tổng hợp
    summary = {
        'total_maps_evaluated': len(df),
        'trad_avg_rmse_m': float(df['trad_rmse_m'].mean()),
        'trad_std_rmse_m': float(df['trad_rmse_m'].std()),
        'trad_avg_max_err_m': float(df['trad_max_err_m'].mean()),
        'trad_avg_energy': float(df['trad_energy'].mean()),
        'trad_success_rate': float(df['trad_success'].mean() * 100),
        'trad_avg_solve_time_s': float(df['trad_time_s'].mean()),
        
        'hyb_avg_rmse_m': float(df['hyb_rmse_m'].mean()),
        'hyb_std_rmse_m': float(df['hyb_rmse_m'].std()),
        'hyb_avg_max_err_m': float(df['hyb_max_err_m'].mean()),
        'hyb_avg_energy': float(df['hyb_energy'].mean()),
        'hyb_success_rate': float(df['hyb_success'].mean() * 100),
        'hyb_avg_solve_time_s': float(df['hyb_time_s'].mean()),
        
        'avg_rmse_improvement_pct': float((df['trad_rmse_m'].mean() - df['hyb_rmse_m'].mean()) / df['trad_rmse_m'].mean() * 100)
    }
    
    summary_path = os.path.join(log_dir, "nmpc_benchmark_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"🎉 Đã lưu thống kê tổng hợp ra JSON: {summary_path}")
    
    # In bảng tổng hợp
    print("\n" + "="*50)
    print(" BẢNG THỐNG KÊ TỔNG HỢP HIỆU NĂNG NMPC (30 BẢN ĐỒ)")
    print("="*50)
    print(f" Chỉ số                       | Traditional NMPC | Hybrid NMPC")
    print("-" * 50)
    print(f" RMSE trung bình (m)          | {summary['trad_avg_rmse_m']:.3f} ± {summary['trad_std_rmse_m']:.3f} | {summary['hyb_avg_rmse_m']:.3f} ± {summary['hyb_std_rmse_m']:.3f}")
    print(f" Lệch biên lớn nhất TB (m)    | {summary['trad_avg_max_err_m']:.3f}            | {summary['hyb_avg_max_err_m']:.3f}")
    print(f" Tỷ lệ bám đích thành công (%) | {summary['trad_success_rate']:.1f}%            | {summary['hyb_success_rate']:.1f}%")
    print(f" Năng lượng điều khiển TB     | {summary['trad_avg_energy']:.3f}            | {summary['hyb_avg_energy']:.3f}")
    print(f" Thời gian mô phỏng TB (s)    | {summary['trad_avg_solve_time_s']:.2f} s           | {summary['hyb_avg_solve_time_s']:.2f} s")
    print("-" * 50)
    print(f" ⚡ CẢI TIẾN TRUNG BÌNH RMSE  : {summary['avg_rmse_improvement_pct']:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_nmpc_benchmarks()
