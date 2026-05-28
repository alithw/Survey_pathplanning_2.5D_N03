"""
run_benchmarks_omb.py
=====================
Chạy toàn bộ thực nghiệm đánh giá trên tập dữ liệu OMB (Orthogonal Maze Benchmark)
của Nguyễn Kiều Linh, gồm hai phần:

  Phần 1: So sánh Weighted A* vs MCTD-A* (Global Path Planning)
  Phần 2: So sánh Traditional NMPC vs Hybrid NMPC (Path Tracking)

Dữ liệu OMB là lưới nhị phân 2D phẳng (không có heightmap / terrain):
  - heights  = 0.0  (mặt phẳng)
  - slopes   = 0.0  (không dốc)
  - t_classes= 0    (Grass → c_bekker = 1.0, slip = 0.05 + 0.05·v)
  - grid[r][c] == 1 → obstacle → chi phí vô cùng

Kết quả được lưu vào:
  ../4. logs/omb_planning_results.csv
  ../4. logs/omb_planning_summary.json
  ../4. logs/omb_nmpc_results.csv
  ../4. logs/omb_nmpc_summary.json
  ../4. logs/omb_benchmark_summary.json   ← tổng hợp cuối cùng
"""

import os
import sys
import json
import glob
import time
import heapq
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import casadi as ca
from scipy.interpolate import interp1d

# Thêm thư mục hiện tại vào sys.path để import từ cùng thư mục
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_system1 import System1CNN

# ─────────────────────────────────────────────────────────────
# CẤU HÌNH CHUNG
# ─────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset_omb/mazes"
LOG_DIR      = "../4. logs"
MODEL_PATH   = "system1_model.pth"
NUM_PLANNING = 50    # Số map dùng cho Phần 1 (lấy đều trên toàn dataset)
NUM_NMPC     = 15    # Số map dùng cho Phần 2  (tốn nhiều thời gian hơn)
RANDOM_SEED  = 42

# Tham số robot (2D phẳng)
SSF            = 1.2
SMOOTH_RADIUS  = 2
MAX_STEP_COST  = 500.0
MAX_TOTAL_COST = 2000000.0   # maze paths can be very long
HEURISTIC_W    = 3.0
MIN_STEP_COST  = 1.0

# Tham số NMPC
N_HOR  = 20
DT     = 0.1
V_MAX, V_MIN   = 2.0, 0.0
W_MAX, W_MIN   = 1.0, -1.0
Q_x, Q_y, Q_t  = 40.0, 40.0, 4.0
R_v, R_w        = 1.0, 0.5

os.makedirs(LOG_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL CNN (System 1)
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = System1CNN().to(device)
if os.path.exists(MODEL_PATH):
    cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    cnn_model.eval()
    print(f"-> Đã nạp System 1 CNN từ: {MODEL_PATH}")
else:
    print(f"[CẢNH BÁO] Không tìm thấy {MODEL_PATH} – dự đoán hành lang bị vô hiệu hóa.")


# ─────────────────────────────────────────────────────────────
# LOAD MAZE
# ─────────────────────────────────────────────────────────────
def snap_to_free(grid, pos, grid_size, max_radius=20):
    """
    Nếu pos (row, col) nằm trên ô obstacle, tìm ô free gần nhất
    bằng BFS từ tâm pos với bán kính tăng dần.
    """
    r, c = int(pos[0]), int(pos[1])
    if 0 <= r < grid_size and 0 <= c < grid_size and grid[r, c] == 0:
        return (r, c)   # Đã free
    # BFS spiral để tìm ô free gần nhất
    from collections import deque
    q = deque([(r, c)])
    visited = {(r, c)}
    while q:
        cr, cc = q.popleft()
        if 0 <= cr < grid_size and 0 <= cc < grid_size and grid[cr, cc] == 0:
            return (cr, cc)
        if abs(cr - r) > max_radius and abs(cc - c) > max_radius:
            break
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            nxt = (cr+dr, cc+dc)
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
    return (r, c)   # Fallback: giữ nguyên


def load_maze(filepath):
    """
    Đọc file JSON của OMB và trả về:
      grid_np  : np.ndarray (N×N) nhị phân (0=free, 1=obstacle)
      start    : (row, col)  – đã snap về ô free gần nhất
      goal     : (row, col)  – đã snap về ô free gần nhất
      grid_size: int N
    """
    with open(filepath, "r") as f:
        d = json.load(f)
    grid_size = d["size"]
    grid_np   = np.array(d["grid"], dtype=np.uint8)   # (N, N)
    start_raw = tuple(d["start"])   # [row, col]
    goal_raw  = tuple(d["goal"])    # [row, col]
    # Snap start/goal sang ô free nếu đang nằm trên obstacle
    start     = snap_to_free(grid_np, start_raw, grid_size)
    goal      = snap_to_free(grid_np, goal_raw,  grid_size)
    return grid_np, start, goal, grid_size


from collections import deque as _deque


def is_maze_connected(grid_np, start, goal, grid_size):
    """Kiem tra start va goal co cung thanh phan lien thong khong (BFS 4 chieu)."""
    if grid_np[start[0], start[1]] == 1 or grid_np[goal[0], goal[1]] == 1:
        return False
    if start == goal:
        return False
    visited = {start}
    q = _deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < grid_size and 0 <= nc < grid_size
                    and (nr, nc) not in visited and grid_np[nr, nc] == 0):
                visited.add((nr, nc))
                q.append((nr, nc))
    return False


def build_valid_filelist(all_files, verbose=True):
    """Quet dataset, giu lai cac maze ma start-goal cung thanh phan lien thong."""
    valid = []
    n_total = len(all_files)
    if verbose:
        print(f"  Kiem tra ket noi start-goal tren {n_total} maze ...", flush=True)
    for i, fp in enumerate(all_files):
        try:
            grid, start, goal, gsz = load_maze(fp)
            if is_maze_connected(grid, start, goal, gsz):
                valid.append(fp)
        except Exception:
            pass
        if verbose and (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{n_total} - hop le: {len(valid)}", flush=True)
    if verbose:
        pct = 100 * len(valid) / max(n_total, 1)
        print(f"  -> Tong maze hop le: {len(valid)}/{n_total} ({pct:.1f}%)")
    return valid


# ─────────────────────────────────────────────────────────────
# PLANNER CHO OMB  (flat 2D, obstacle-aware)
# ─────────────────────────────────────────────────────────────
class OMBAStarPlanner:
    """Weighted A* trên lưới nhị phân OMB (mặt phẳng, không terrain)."""

    def __init__(self, grid, grid_size,
                 heuristic_weight=3.0, min_step_cost=1.0,
                 max_total_cost=200000.0,
                 corridor_mask=None, corridor_threshold=0.2):
        self.grid              = grid         # (N, N) uint8
        self.grid_size         = grid_size
        self.heuristic_weight  = heuristic_weight
        self.min_step_cost     = min_step_cost
        self.max_total_cost    = max_total_cost
        self.corridor_mask     = corridor_mask
        self.corridor_threshold= corridor_threshold

    def _step_cost(self, nx, ny):
        """Chi phí 1 ô: vô cùng nếu là obstacle, 1.0 nếu free."""
        if self.grid[nx, ny] == 1:
            return float("inf")
        return 1.0

    def plan(self, start, goal):
        """Weighted A*, trả về (path, cost, nodes_explored)."""
        start_h = np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2) * self.min_step_cost
        pq      = [(self.heuristic_weight * start_h, 0.0, start)]
        dist    = {start: 0.0}
        parent  = {start: None}
        nodes   = 0

        while pq:
            f, g, (x, y) = heapq.heappop(pq)
            nodes += 1
            if g > dist.get((x, y), float("inf")):
                continue
            if (x, y) == goal:
                path = []
                cur  = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return path[::-1], g, nodes
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
                nx_, ny_ = x+dx, y+dy
                if 0 <= nx_ < self.grid_size and 0 <= ny_ < self.grid_size:
                    sc = self._step_cost(nx_, ny_) * np.sqrt(dx*dx + dy*dy)
                    if sc == float("inf"):
                        continue
                    ng = g + sc
                    if ng > self.max_total_cost:
                        continue
                    if (nx_, ny_) not in dist or ng < dist[(nx_, ny_)]:
                        dist[(nx_, ny_)] = ng
                        h  = np.sqrt((nx_-goal[0])**2+(ny_-goal[1])**2) * self.min_step_cost
                        heapq.heappush(pq, (ng + self.heuristic_weight*h, ng, (nx_, ny_)))
                        parent[(nx_, ny_)] = (x, y)
        return None, float("inf"), nodes

    def plan_with_corridor(self, start, goal):
        """
        MCTD-A*: tìm đường trong hành lang CNN; fallback A* toàn cục nếu thất bại.
        Trả về (path, cost, nodes_explored, fallback_flag).
        """
        if self.corridor_mask is None:
            path, cost, nodes = self.plan(start, goal)
            return path, cost, nodes, True

        start_h = np.sqrt((start[0]-goal[0])**2+(start[1]-goal[1])**2)*self.min_step_cost
        pq      = [(self.heuristic_weight*start_h, 0.0, start)]
        dist    = {start: 0.0}
        parent  = {start: None}
        nodes   = 0

        while pq:
            f, g, (x, y) = heapq.heappop(pq)
            nodes += 1
            if g > dist.get((x, y), float("inf")):
                continue
            if (x, y) == goal:
                path = []
                cur  = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return path[::-1], g, nodes, False
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
                nx_, ny_ = x+dx, y+dy
                if 0 <= nx_ < self.grid_size and 0 <= ny_ < self.grid_size:
                    # Ràng buộc hành lang (ngoại trừ start và goal)
                    if (nx_, ny_) != goal and (nx_, ny_) != start:
                        if self.corridor_mask[nx_, ny_] < self.corridor_threshold:
                            continue
                    sc = self._step_cost(nx_, ny_) * np.sqrt(dx*dx + dy*dy)
                    if sc == float("inf"):
                        continue
                    ng = g + sc
                    if ng > self.max_total_cost:
                        continue
                    if (nx_, ny_) not in dist or ng < dist[(nx_, ny_)]:
                        dist[(nx_, ny_)] = ng
                        h  = np.sqrt((nx_-goal[0])**2+(ny_-goal[1])**2)*self.min_step_cost
                        heapq.heappush(pq, (ng+self.heuristic_weight*h, ng, (nx_, ny_)))
                        parent[(nx_, ny_)] = (x, y)

        # Fallback toàn cục
        print("   [FALLBACK] Không tìm được đường trong hành lang → chuyển sang A* toàn cục.")
        path, cost, nodes2 = self.plan(start, goal)
        return path, cost, nodes+nodes2, True


# ─────────────────────────────────────────────────────────────
# DỰ ĐOÁN HÀNH LANG (System 1 CNN – scaled to model input)
# ─────────────────────────────────────────────────────────────
def predict_corridor_omb(grid, start, goal, model, model_size=64):
    """
    Mô hình System1CNN được huấn luyện trên ảnh 64×64.
    Ta scale ảnh đầu vào xuống model_size, chạy CNN, rồi upsample kết quả.
    """
    N = grid.shape[0]
    import torch.nn.functional as F

    # 5 kênh đầu vào:
    #   ch0 = occupancy (0=free, 1=obstacle)  → đặt vai trò "heights"
    #   ch1 = zeros                           → slopes
    #   ch2 = zeros                           → t_classes / 8
    #   ch3 = start mask
    #   ch4 = goal mask
    occ   = grid.astype(np.float32)          # obstacle map
    zeros = np.zeros((N, N), dtype=np.float32)

    start_mask = zeros.copy(); start_mask[int(start[0]), int(start[1])] = 1.0
    goal_mask  = zeros.copy(); goal_mask[int(goal[0]),  int(goal[1])]  = 1.0

    x5 = np.stack([occ, zeros, zeros, start_mask, goal_mask], axis=0)   # (5, N, N)
    x5_t = torch.tensor(x5).unsqueeze(0)                                 # (1, 5, N, N)

    # Downsample to model input size
    x_small = F.interpolate(x5_t, size=(model_size, model_size),
                            mode="bilinear", align_corners=False).to(device)

    with torch.no_grad():
        pred_small = model(x_small)                                       # (1, 1, model_size, model_size)

    # Upsample back to original grid size
    pred_full = F.interpolate(pred_small, size=(N, N),
                              mode="bilinear", align_corners=False)
    corridor  = pred_full.squeeze().cpu().numpy()                         # (N, N)
    return corridor


# ─────────────────────────────────────────────────────────────
# TIỆN ÍCH
# ─────────────────────────────────────────────────────────────
def path_length(path):
    """Độ dài hình học (pixel)."""
    arr = np.array(path)
    return float(np.sum(np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))))


def smoothness(path):
    """Tổng bình phương biến thiên góc."""
    arr = np.array(path)
    if len(arr) < 3:
        return 0.0
    dx = np.diff(arr[:, 1])
    dy = np.diff(arr[:, 0])
    ang = np.arctan2(dy, dx)
    da  = np.diff(ang)
    da  = (da + np.pi) % (2*np.pi) - np.pi
    return float(np.sum(da**2))


# ─────────────────────────────────────────────────────────────
# PHẦN 1: GLOBAL PATH PLANNING BENCHMARK
# ─────────────────────────────────────────────────────────────
def run_planning_benchmark(maze_files):
    print("\n" + "="*65)
    print("  PHẦN 1: ĐÁNH GIÁ QUY HOẠCH ĐƯỜNG ĐI TOÀN CỤC (OMB DATASET)")
    print("="*65)

    results = []
    n = len(maze_files)
    for idx, fpath in enumerate(maze_files):
        fname = os.path.basename(fpath)
        print(f"  [{idx+1:02d}/{n}] {fname}", end=" ... ")

        try:
            grid, start, goal, grid_size = load_maze(fpath)
        except Exception as e:
            print(f"LỖI ĐỌC FILE: {e}")
            continue

        planner = OMBAStarPlanner(grid, grid_size,
                                  heuristic_weight=HEURISTIC_W,
                                  min_step_cost=MIN_STEP_COST,
                                  max_total_cost=MAX_TOTAL_COST)

        # ---------- Weighted A* ----------
        t0 = time.perf_counter()
        path_pure, cost_pure, nodes_pure = planner.plan(start, goal)
        t_pure = (time.perf_counter() - t0) * 1000

        if path_pure is None:
            print("A* không tìm được đường – bỏ qua.")
            continue

        # ---------- System 1: CNN corridor prediction ----------
        t0 = time.perf_counter()
        corridor = predict_corridor_omb(grid, start, goal, cnn_model)
        t_sys1   = (time.perf_counter() - t0) * 1000

        # ---------- MCTD-A* (System 2 trong hành lang) ----------
        planner.corridor_mask      = corridor
        planner.corridor_threshold = 0.2
        t0 = time.perf_counter()
        path_mctd, cost_mctd, nodes_mctd, fallback = planner.plan_with_corridor(start, goal)
        t_sys2 = (time.perf_counter() - t0) * 1000
        t_mctd = t_sys1 + t_sys2

        if path_mctd is None:
            print("MCTD-A* không tìm được đường – bỏ qua.")
            continue

        entry = {
            "map":              fname,
            "grid_size":        grid_size,
            "a_star_time_ms":   round(t_pure, 4),
            "a_star_cost":      round(cost_pure, 4),
            "a_star_length_px": round(path_length(path_pure), 4),
            "a_star_smooth":    round(smoothness(path_pure), 6),
            "a_star_nodes":     nodes_pure,
            "a_star_waypoints": len(path_pure),

            "mctd_time_ms":     round(t_mctd, 4),
            "mctd_sys1_ms":     round(t_sys1, 4),
            "mctd_sys2_ms":     round(t_sys2, 4),
            "mctd_cost":        round(cost_mctd, 4),
            "mctd_length_px":   round(path_length(path_mctd), 4),
            "mctd_smooth":      round(smoothness(path_mctd), 6),
            "mctd_nodes":       nodes_mctd,
            "mctd_waypoints":   len(path_mctd),
            "mctd_fallback":    fallback,
        }
        results.append(entry)
        print(f"A*={t_pure:.1f}ms MCTD={t_mctd:.1f}ms nodes↓{100*(1-nodes_mctd/(nodes_pure+1)):.0f}%")

    df = pd.DataFrame(results)
    csv_path = os.path.join(LOG_DIR, "omb_planning_results.csv")
    df.to_csv(csv_path, index=False)

    # Thống kê tổng hợp
    def safe_mean(col): return float(df[col].mean()) if len(df) > 0 else 0.0
    def safe_std(col):  return float(df[col].std())  if len(df) > 1 else 0.0

    planning_summary = {
        "dataset":          "OMB (Orthogonal Maze Benchmark)",
        "maps_evaluated":   len(df),
        "a_star": {
            "avg_time_ms":   safe_mean("a_star_time_ms"),
            "std_time_ms":   safe_std("a_star_time_ms"),
            "avg_cost":      safe_mean("a_star_cost"),
            "avg_length_px": safe_mean("a_star_length_px"),
            "avg_smooth":    safe_mean("a_star_smooth"),
            "avg_nodes":     safe_mean("a_star_nodes"),
        },
        "mctd_a_star": {
            "avg_time_ms":   safe_mean("mctd_time_ms"),
            "std_time_ms":   safe_std("mctd_time_ms"),
            "avg_sys1_ms":   safe_mean("mctd_sys1_ms"),
            "avg_sys2_ms":   safe_mean("mctd_sys2_ms"),
            "avg_cost":      safe_mean("mctd_cost"),
            "avg_length_px": safe_mean("mctd_length_px"),
            "avg_smooth":    safe_mean("mctd_smooth"),
            "avg_nodes":     safe_mean("mctd_nodes"),
            "fallback_rate": float(df["mctd_fallback"].mean()*100) if len(df)>0 else 0.0,
        },
    }
    if len(df) > 0 and planning_summary["a_star"]["avg_time_ms"] > 0:
        t_a = planning_summary["a_star"]["avg_time_ms"]
        t_m = planning_summary["mctd_a_star"]["avg_time_ms"]
        planning_summary["time_improvement_pct"]   = round((t_a - t_m) / t_a * 100, 2)

        n_a = planning_summary["a_star"]["avg_nodes"]
        n_m = planning_summary["mctd_a_star"]["avg_nodes"]
        planning_summary["nodes_reduction_pct"]    = round((n_a - n_m) / (n_a+1e-9) * 100, 2)

        l_a = planning_summary["a_star"]["avg_length_px"]
        l_m = planning_summary["mctd_a_star"]["avg_length_px"]
        planning_summary["length_change_pct"]      = round((l_m - l_a) / (l_a+1e-9) * 100, 2)

        s_a = planning_summary["a_star"]["avg_smooth"]
        s_m = planning_summary["mctd_a_star"]["avg_smooth"]
        planning_summary["smoothness_improvement_pct"] = round((s_a - s_m) / (s_a+1e-9) * 100, 2)

    json_path = os.path.join(LOG_DIR, "omb_planning_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(planning_summary, f, ensure_ascii=False, indent=2)

    print(f"\n-> Đã lưu kết quả chi tiết: {csv_path}")
    print(f"-> Đã lưu tóm tắt thống kê: {json_path}")
    return planning_summary, df


# ─────────────────────────────────────────────────────────────
# PHẦN 2: NMPC PATH TRACKING BENCHMARK
# ─────────────────────────────────────────────────────────────

# ---- SlipMLP (giữ nguyên từ run_nmpc_benchmarks.py) --------
class SlipMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),  nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def train_slip_mlp(model, epochs=30):
    np.random.seed(42)
    soils      = np.random.randint(0, 8, 2000)
    slopes_deg = np.random.uniform(0.0, 45.0, 2000)
    vels       = np.random.uniform(0.0, 2.0,  2000)
    slips = []
    slopes_rad = np.radians(slopes_deg)
    base_map   = {0:0.05,1:0.08,2:0.15,3:0.40,4:0.10,5:0.10,6:0.10,7:0.10}
    for i in range(2000):
        s = base_map[soils[i]] + 0.35*np.sin(slopes_rad[i]) + 0.05*vels[i]
        s = np.clip(s + np.random.normal(0, 0.02), 0.01, 0.85)
        slips.append(s)
    X = torch.tensor(
        np.stack([soils/7.0, slopes_rad, vels/2.0], axis=1).astype(np.float32))
    y = torch.tensor(np.array(slips, dtype=np.float32).reshape(-1, 1))
    opt, crit = optim.Adam(model.parameters(), lr=0.01), nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
    model.eval()
    print(f"-> SlipMLP huấn luyện xong (MSE={loss.item():.6f})")

slip_model = SlipMLP()
train_slip_mlp(slip_model)


def simulate_nmpc_flat(path_grid, grid_size, hybrid=True):
    """
    Mô phỏng NMPC trên lưới phẳng OMB.
    path_grid: danh sách (row, col) từ A*/MCTD-A*
    Trả về (rmse, max_err, energy, success)
    """
    path_arr = np.array(path_grid)
    # Đổi (row, col) → (x=col, y=row) để đồng nhất với không gian 2D thông thường
    x_ref_raw = path_arr[:, 1].astype(float)
    y_ref_raw = path_arr[:, 0].astype(float)

    dx   = np.diff(x_ref_raw)
    dy   = np.diff(y_ref_raw)
    dist = np.sqrt(dx**2 + dy**2)
    cum_dist = np.insert(np.cumsum(dist), 0, 0)

    _, uid = np.unique(cum_dist, return_index=True)
    uid.sort()
    V_TARGET = 1.0
    t_arr    = cum_dist[uid] / V_TARGET
    xc       = x_ref_raw[uid]
    yc       = y_ref_raw[uid]

    total_t  = t_arr[-1]
    t_ref    = np.arange(0, total_t, DT)
    if len(t_ref) < 2:
        return 999.0, 999.0, 0.0, 0.0

    fx = interp1d(t_arr, xc, kind="linear", fill_value="extrapolate")
    fy = interp1d(t_arr, yc, kind="linear", fill_value="extrapolate")
    xr = fx(t_ref)
    yr = fy(t_ref)

    dxdt     = np.gradient(xr)
    dydt     = np.gradient(yr)
    theta_r  = np.arctan2(dydt, dxdt)

    xr = np.pad(xr, (0, N_HOR+1), "edge")
    yr = np.pad(yr, (0, N_HOR+1), "edge")
    theta_r = np.pad(theta_r, (0, N_HOR+1), "edge")

    # CasADi
    opti = ca.Opti()
    X_v  = opti.variable(3, N_HOR+1)
    U_v  = opti.variable(2, N_HOR)
    X0   = opti.parameter(3)
    Xref = opti.parameter(3, N_HOR+1)
    slip = opti.parameter(N_HOR)

    # Flat model: slip_grass ≈ 0.05 + 0.05·v (v=1 → slip≈0.10)
    for k in range(N_HOR):
        xn = X_v[0,k] + DT*U_v[0,k]*(1-slip[k])*ca.cos(X_v[2,k])
        yn = X_v[1,k] + DT*U_v[0,k]*(1-slip[k])*ca.sin(X_v[2,k])
        tn = X_v[2,k] + DT*U_v[1,k]
        opti.subject_to(X_v[0,k+1]==xn)
        opti.subject_to(X_v[1,k+1]==yn)
        opti.subject_to(X_v[2,k+1]==tn)

    opti.subject_to(X_v[:,0] == X0)
    opti.subject_to(opti.bounded(V_MIN, U_v[0,:], V_MAX))
    opti.subject_to(opti.bounded(W_MIN, U_v[1,:], W_MAX))

    Q  = ca.diag([Q_x, Q_y, Q_t])
    R  = ca.diag([R_v, R_w])
    cost = 0
    for k in range(N_HOR):
        e = X_v[:,k] - Xref[:,k]
        cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([U_v[:,k].T, R, U_v[:,k]])
    e_term = X_v[:,N_HOR] - Xref[:,N_HOR]
    cost  += ca.mtimes([e_term.T, Q*5, e_term])
    opti.minimize(cost)

    p_opts = {"expand": True, "print_time": False}
    s_opts = {"max_iter": 100, "print_level": 0, "sb": "yes"}
    opti.solver("ipopt", p_opts, s_opts)

    state = np.array([xr[0], yr[0], theta_r[0]], dtype=float)
    hist_x, hist_y, hist_v, hist_w = [], [], [], []
    num_steps = len(t_ref)
    last_idx  = 0

    for step in range(num_steps):
        xc_s, yc_s = state[0], state[1]
        se  = max(0, last_idx)
        ee  = min(len(t_ref), last_idx + 15)
        if ee > se:
            seg_d   = (xr[se:ee]-xc_s)**2 + (yr[se:ee]-yc_s)**2
            c_idx   = se + int(np.argmin(seg_d))
        else:
            c_idx   = last_idx
        last_idx = c_idx

        # Slip prediction: Grass, slope=0 → Traditional=0.10; Hybrid=SlipMLP
        est_slips = []
        for k in range(N_HOR):
            if hybrid:
                inp = torch.tensor([[0.0, 0.0, 1.0/2.0]], dtype=torch.float32)
                with torch.no_grad():
                    s_k = float(slip_model(inp).item())
                s_k = min(s_k, 0.40)
            else:
                s_k = 0.10   # fixed traditional
            est_slips.append(s_k)

        xr_h = xr[c_idx:c_idx+N_HOR+1]
        yr_h = yr[c_idx:c_idx+N_HOR+1]
        tr_h = theta_r[c_idx:c_idx+N_HOR+1]
        ref_mat = np.vstack((xr_h, yr_h, tr_h))

        opti.set_value(X0,   state)
        opti.set_value(Xref, ref_mat)
        opti.set_value(slip,  np.array(est_slips))
        opti.set_initial(X_v, ref_mat)
        opti.set_initial(U_v, np.ones((2, N_HOR))*0.1)

        try:
            sol = opti.solve()
            u_opt = sol.value(U_v)[:, 0]
        except:
            u_opt = opti.debug.value(U_v)[:, 0]

        v_cmd, w_cmd = u_opt[0], u_opt[1]
        # Thực tế: slip của grass = 0.05 + 0.05·v
        actual_slip = 0.05 + 0.05 * abs(v_cmd)
        state[0] += DT * v_cmd * (1 - actual_slip) * np.cos(state[2])
        state[1] += DT * v_cmd * (1 - actual_slip) * np.sin(state[2])
        state[2] += DT * w_cmd

        hist_x.append(state[0])
        hist_y.append(state[1])
        hist_v.append(v_cmd)
        hist_w.append(w_cmd)

    hx = np.array(hist_x)
    hy = np.array(hist_y)

    # RMSE theo khoảng cách gần nhất đến đường tham chiếu
    errs = []
    for i in range(len(hx)):
        d2  = (xr[:num_steps]-hx[i])**2 + (yr[:num_steps]-hy[i])**2
        errs.append(float(np.min(d2)))
    rmse    = float(np.sqrt(np.mean(errs)))
    max_err = float(np.sqrt(np.max(errs)))
    energy  = float(np.sum(np.array(hist_v)**2 + 0.5*np.array(hist_w)**2) * DT)

    final_dist = float(np.sqrt((hx[-1]-xr[num_steps-1])**2 + (hy[-1]-yr[num_steps-1])**2))
    success    = 1.0 if (final_dist < 3.0 and max_err < 5.0) else 0.0
    return rmse, max_err, energy, success


def run_nmpc_benchmark(maze_files):
    print("\n" + "="*65)
    print("  PHẦN 2: ĐÁNH GIÁ NMPC BÁM QUỸ ĐẠO (OMB DATASET – PHẲNG)")
    print("="*65)

    results = []
    n = len(maze_files)
    for idx, fpath in enumerate(maze_files):
        fname = os.path.basename(fpath)
        print(f"  [{idx+1:02d}/{n}] {fname}")

        try:
            grid, start, goal, grid_size = load_maze(fpath)
        except Exception as e:
            print(f"    LỖI: {e}"); continue

        # Tạo quỹ đạo tham chiếu bằng A*
        planner = OMBAStarPlanner(grid, grid_size,
                                  heuristic_weight=HEURISTIC_W,
                                  min_step_cost=MIN_STEP_COST,
                                  max_total_cost=MAX_TOTAL_COST)
        path, _, _ = planner.plan(start, goal)
        if path is None or len(path) < 5:
            print("    [Bỏ qua] Không tạo được quỹ đạo tham chiếu đủ dài."); continue

        # Traditional NMPC
        print("    -> Traditional NMPC ...", end=" ")
        t0 = time.perf_counter()
        try:
            rmse_t, me_t, en_t, sc_t = simulate_nmpc_flat(path, grid_size, hybrid=False)
        except Exception as e:
            print(f"LỖI: {e}"); continue
        time_t = time.perf_counter() - t0
        print(f"RMSE={rmse_t:.3f}m  MaxErr={me_t:.3f}m  Succ={int(sc_t)}")

        # Hybrid NMPC
        print("    -> Hybrid NMPC     ...", end=" ")
        t0 = time.perf_counter()
        try:
            rmse_h, me_h, en_h, sc_h = simulate_nmpc_flat(path, grid_size, hybrid=True)
        except Exception as e:
            print(f"LỖI: {e}"); continue
        time_h = time.perf_counter() - t0
        print(f"RMSE={rmse_h:.3f}m  MaxErr={me_h:.3f}m  Succ={int(sc_h)}")

        if rmse_t >= 999.0 or rmse_h >= 999.0:
            print("    [Bỏ qua] Lỗi quỹ đạo quá ngắn."); continue

        imp = (rmse_t - rmse_h) / (rmse_t + 1e-9) * 100
        results.append({
            "map":              fname,
            "grid_size":        grid_size,
            "path_length_px":   round(path_length(path), 2),
            "trad_rmse_m":      round(rmse_t, 4),
            "trad_max_err_m":   round(me_t,   4),
            "trad_energy":      round(en_t,   4),
            "trad_success":     sc_t,
            "trad_time_s":      round(time_t, 3),
            "hyb_rmse_m":       round(rmse_h, 4),
            "hyb_max_err_m":    round(me_h,   4),
            "hyb_energy":       round(en_h,   4),
            "hyb_success":      sc_h,
            "hyb_time_s":       round(time_h, 3),
            "rmse_improvement_pct": round(imp, 2),
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(LOG_DIR, "omb_nmpc_results.csv")
    df.to_csv(csv_path, index=False)

    def sm(c): return float(df[c].mean()) if len(df) > 0 else 0.0
    def ss(c): return float(df[c].std())  if len(df) > 1 else 0.0

    nmpc_summary = {
        "dataset":         "OMB (Orthogonal Maze Benchmark)",
        "maps_evaluated":  len(df),
        "traditional_nmpc": {
            "avg_rmse_m":      sm("trad_rmse_m"),
            "std_rmse_m":      ss("trad_rmse_m"),
            "avg_max_err_m":   sm("trad_max_err_m"),
            "avg_energy":      sm("trad_energy"),
            "success_rate_pct":sm("trad_success") * 100,
            "avg_time_s":      sm("trad_time_s"),
        },
        "hybrid_nmpc": {
            "avg_rmse_m":      sm("hyb_rmse_m"),
            "std_rmse_m":      ss("hyb_rmse_m"),
            "avg_max_err_m":   sm("hyb_max_err_m"),
            "avg_energy":      sm("hyb_energy"),
            "success_rate_pct":sm("hyb_success") * 100,
            "avg_time_s":      sm("hyb_time_s"),
        },
    }
    if len(df) > 0:
        t_r = nmpc_summary["traditional_nmpc"]["avg_rmse_m"]
        h_r = nmpc_summary["hybrid_nmpc"]["avg_rmse_m"]
        nmpc_summary["avg_rmse_improvement_pct"] = round((t_r-h_r)/(t_r+1e-9)*100, 2)

    json_path = os.path.join(LOG_DIR, "omb_nmpc_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nmpc_summary, f, ensure_ascii=False, indent=2)

    print(f"\n-> Đã lưu kết quả NMPC chi tiết: {csv_path}")
    print(f"-> Đã lưu tóm tắt NMPC: {json_path}")
    return nmpc_summary, df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  THỰC NGHIỆM DIỆN RỘNG TRÊN TẬP DỮ LIỆU OMB")
    print("  (Orthogonal Maze Benchmark – Nguyễn Kiều Linh, 2026)")
    print("=" * 65)

    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
    if not all_files:
        print(f"[LỖI] Không tìm thấy file JSON nào trong: {DATASET_DIR}")
        return
    print(f"-> Tổng số maze trong dataset: {len(all_files)}")

    # Loc chi giu maze co start-goal ket noi (connected) – tranh loi A* vo han
    valid_files = build_valid_filelist(all_files, verbose=True)
    if len(valid_files) < 5:
        print(f"[LOI] Chi co {len(valid_files)} maze hop le - qua it de danh gia!")
        return

    # Chon deu NUM_PLANNING map cho Phan 1 tu danh sach hop le
    step_p  = max(1, len(valid_files) // NUM_PLANNING)
    plan_fs = [valid_files[i * step_p] for i in range(min(NUM_PLANNING, len(valid_files)))]

    # Chon deu NUM_NMPC map cho Phan 2 tu danh sach hop le
    step_n  = max(1, len(valid_files) // NUM_NMPC)
    nmpc_fs = [valid_files[i * step_n] for i in range(min(NUM_NMPC, len(valid_files)))]

    print(f"-> Phần 1: {len(plan_fs)} bản đồ đại diện (Quy hoạch đường đi)")
    print(f"-> Phần 2: {len(nmpc_fs)} bản đồ đại diện (Bám quỹ đạo NMPC)")

    t_total_start = time.time()

    # ---- PHẦN 1 ----
    p_summary, p_df = run_planning_benchmark(plan_fs)

    # ---- PHẦN 2 ----
    n_summary, n_df = run_nmpc_benchmark(nmpc_fs)

    elapsed = time.time() - t_total_start

    # ---- TỔNG HỢP CUỐI CÙNG ----
    final = {
        "benchmark_dataset":    "OMB – Orthogonal Maze Benchmark",
        "total_runtime_s":      round(elapsed, 2),
        "planning_benchmark":   p_summary,
        "nmpc_benchmark":       n_summary,
    }
    fin_path = os.path.join(LOG_DIR, "omb_benchmark_summary.json")
    with open(fin_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # ---- IN KẾT QUẢ TỔNG HỢP ----
    print("\n" + "="*65)
    print("  KẾT QUẢ TỔNG HỢP THỰC NGHIỆM TRÊN TẬP DỮ LIỆU OMB")
    print("="*65)

    pa = p_summary.get("a_star", {})
    pm = p_summary.get("mctd_a_star", {})
    print("\n[PHẦN 1] QUY HOẠCH ĐƯỜNG ĐI TOÀN CỤC")
    print(f"  Số bản đồ đánh giá: {p_summary.get('maps_evaluated', 0)}")
    print(f"  {'Chỉ số':<30} {'Weighted A*':>15} {'MCTD-A*':>15}")
    print(f"  {'-'*62}")
    print(f"  {'Thời gian TB (ms)':<30} {pa.get('avg_time_ms',0):>15.2f} {pm.get('avg_time_ms',0):>15.2f}")
    print(f"  {'Độ dài TB (pixel)':<30} {pa.get('avg_length_px',0):>15.2f} {pm.get('avg_length_px',0):>15.2f}")
    print(f"  {'Độ mịn TB (↓ tốt hơn)':<30} {pa.get('avg_smooth',0):>15.4f} {pm.get('avg_smooth',0):>15.4f}")
    print(f"  {'Nodes duyệt TB':<30} {pa.get('avg_nodes',0):>15.0f} {pm.get('avg_nodes',0):>15.0f}")
    print(f"  {'Tỷ lệ Fallback (%)':<30} {'—':>15} {pm.get('fallback_rate',0):>14.1f}%")
    print(f"  ⚡ Giảm nodes tìm kiếm: {p_summary.get('nodes_reduction_pct',0):.1f}%")
    print(f"  ⚡ Thay đổi thời gian:  {p_summary.get('time_improvement_pct',0):.1f}%")

    ta = n_summary.get("traditional_nmpc", {})
    th = n_summary.get("hybrid_nmpc", {})
    print("\n[PHẦN 2] BÁM QUỸ ĐẠO NMPC")
    print(f"  Số bản đồ đánh giá: {n_summary.get('maps_evaluated', 0)}")
    print(f"  {'Chỉ số':<35} {'Traditional':>15} {'Hybrid':>15}")
    print(f"  {'-'*67}")
    print(f"  {'RMSE TB (m)':<35} {ta.get('avg_rmse_m',0):>15.4f} {th.get('avg_rmse_m',0):>15.4f}")
    print(f"  {'Lệch biên lớn nhất TB (m)':<35} {ta.get('avg_max_err_m',0):>15.4f} {th.get('avg_max_err_m',0):>15.4f}")
    print(f"  {'Tỷ lệ thành công (%)':<35} {ta.get('success_rate_pct',0):>14.1f}% {th.get('success_rate_pct',0):>14.1f}%")
    print(f"  {'Năng lượng TB':<35} {ta.get('avg_energy',0):>15.3f} {th.get('avg_energy',0):>15.3f}")
    print(f"  {'Thời gian mô phỏng TB (s)':<35} {ta.get('avg_time_s',0):>15.2f} {th.get('avg_time_s',0):>15.2f}")
    print(f"  ⚡ Cải thiện RMSE: {n_summary.get('avg_rmse_improvement_pct',0):.2f}%")

    print(f"\n  Tổng thời gian chạy: {elapsed:.1f}s")
    print(f"  Kết quả tổng hợp: {fin_path}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
