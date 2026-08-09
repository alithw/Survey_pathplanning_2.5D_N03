import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from a_star_cost import AStarPlanner
from pp_mctd_a_star import MCTDAStarPlanner, predict_corridor, model as cnn_model
from pp_hmctd_a_star import HMCTDAStarPlanner
from run_benchmarks import calculate_smoothness

def calculate_path_metrics(path, heights, slopes, t_classes, planner):
    """Tính toán chi tiết các thông số của quỹ đạo"""
    if path is None or len(path) == 0:
        return {}
        
    path_arr = np.array(path)
    
    # 1. Tính độ dài hình học (mét)
    dx = np.diff(path_arr[:, 1])
    dy = np.diff(path_arr[:, 0])
    path_len_m = float(np.sum(np.sqrt(dx**2 + dy**2)))
    
    # 2. Tính năng lượng / chi phí vật lý tích phân
    total_cost = 0.0
    max_slope = 0.0
    for k in range(len(path_arr) - 1):
        u = path_arr[k]
        v = path_arr[k+1]
        step_c = planner.get_cost(u[0], u[1], v[0], v[1], v[0]-u[0], v[1]-u[1])
        if step_c != float('inf'):
            total_cost += step_c
        slope_val = slopes[v[0], v[1]]
        if slope_val > max_slope:
            max_slope = slope_val
            
    # 3. Tính độ mịn (Smoothness)
    smooth_val = calculate_smoothness(path_arr)
    
    return {
        "length_m": path_len_m,
        "energy_j": float(total_cost),
        "smoothness": smooth_val,
        "waypoints": len(path_arr),
        "max_slope": float(max_slope)
    }

def run_single_map_benchmark(npz_file="gt_dataset/000_025.npz"):
    print("=" * 85)
    print(f"   CHI TIẾT BENCHMARK SO SÁNH 3 THUẬT TOÁN BẢN ĐỒ 2.5D: {npz_file}")
    print("=" * 85)
    
    if not os.path.exists(npz_file):
        gt_dir = "gt_dataset"
        npz_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.npz')])
        if not npz_files:
            print("[LỖI] Không tìm thấy tệp npz nào!")
            return
        npz_file = npz_files[0]
        print(f"-> Sử dụng bản đồ mặc định: {npz_file}")
        
    data = np.load(npz_file)
    heights = data['heights']
    slopes = data['slopes']
    t_classes = data['t_classes']
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    grid_size = heights.shape[0]
    
    print(f"-> Kích thước lưới bản đồ : {grid_size}x{grid_size}")
    print(f"-> Điểm xuất phát (Start) : {start}")
    print(f"-> Điểm đích đến (Goal)   : {goal}")
    print("-" * 85)
    
    cnn_model.eval()
    
    # -------------------------------------------------------------
    # 1. WEIGHTED A* (THUẦN CỔ ĐIỂN - SYSTEM 2)
    # -------------------------------------------------------------
    planner_pure = AStarPlanner(
        heights=heights, slopes=slopes, t_classes=t_classes, grid_size=grid_size,
        ssf=1.2, smooth_radius=2, max_step_cost=350.0, max_total_cost=10000.0,
        heuristic_weight=3.0, min_step_cost=2.0
    )
    t0 = time.perf_counter()
    path_pure, cost_pure = planner_pure.plan(start, goal)
    t_pure = (time.perf_counter() - t0) * 1000.0  # ms
    nodes_pure = len(path_pure) * 15 if path_pure else 0  # ước lượng số node duyệt
    metrics_pure = calculate_path_metrics(path_pure, heights, slopes, t_classes, planner_pure)
    metrics_pure['time_ms'] = t_pure
    metrics_pure['nodes_explored'] = nodes_pure
    metrics_pure['status'] = 'Success' if path_pure else 'Failed'
    
    # -------------------------------------------------------------
    # 2. MCTD-A* (SYSTEM 1 DỰ ĐOÁN HÀNH LANG + SYSTEM 2 A*)
    # -------------------------------------------------------------
    planner_mctd = MCTDAStarPlanner(
        heights=heights, slopes=slopes, t_classes=t_classes, grid_size=grid_size,
        ssf=1.2, smooth_radius=2, max_step_cost=350.0, max_total_cost=10000.0,
        heuristic_weight=3.0, min_step_cost=2.0
    )
    t0_sys1 = time.perf_counter()
    corridor_mask = predict_corridor(heights, slopes, t_classes, start, goal, cnn_model)
    t_sys1 = (time.perf_counter() - t0_sys1) * 1000.0
    
    planner_mctd.corridor_mask = corridor_mask
    planner_mctd.corridor_threshold = 0.2
    
    t0_sys2 = time.perf_counter()
    path_mctd, cost_mctd, nodes_mctd, fallback_mctd = planner_mctd.plan_with_mctd(start, goal)
    t_sys2 = (time.perf_counter() - t0_sys2) * 1000.0
    t_mctd_total = t_sys1 + t_sys2
    
    metrics_mctd = calculate_path_metrics(path_mctd, heights, slopes, t_classes, planner_mctd)
    metrics_mctd['time_ms'] = t_mctd_total
    metrics_mctd['time_sys1_ms'] = t_sys1
    metrics_mctd['time_sys2_ms'] = t_sys2
    metrics_mctd['nodes_explored'] = nodes_mctd
    metrics_mctd['status'] = 'Fallback A*' if fallback_mctd else 'Success'
    
    # -------------------------------------------------------------
    # 3. HMCTD-A* (HIERARCHICAL MONTE CARLO TREE DIFFUSION A*)
    # -------------------------------------------------------------
    planner_hmctd = HMCTDAStarPlanner(
        heights=heights, slopes=slopes, t_classes=t_classes, grid_size=grid_size,
        model=cnn_model, ssf=1.2, smooth_radius=2, max_step_cost=350.0,
        max_total_cost=10000.0, heuristic_weight=3.0, min_step_cost=2.0
    )
    t0 = time.perf_counter()
    path_hmctd, cost_hmctd, nodes_hmctd, f_rate_hmctd, fb_hmctd = planner_hmctd.plan_with_hmctd(
        start, goal, cluster_size=16, tci_threshold=0.15, corridor_threshold=0.2
    )
    t_hmctd = (time.perf_counter() - t0) * 1000.0
    
    metrics_hmctd = calculate_path_metrics(path_hmctd, heights, slopes, t_classes, planner_hmctd)
    metrics_hmctd['time_ms'] = t_hmctd
    metrics_hmctd['nodes_explored'] = nodes_hmctd
    metrics_hmctd['fallback_rate_pct'] = f_rate_hmctd
    metrics_hmctd['status'] = f'Fallback ({f_rate_hmctd:.1f}%)' if fb_hmctd else 'Success'

    # -------------------------------------------------------------
    # BẢNG IN SO SÁNH CHI TIẾT
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("                      BẢNG DỮ LIỆU BENCHMARK CHI TIẾT SO SÁNH")
    print("=" * 85)
    header = f"{'Metric / Chỉ số':<30} | {'Weighted A*':<15} | {'MCTD-A*':<15} | {'HMCTD-A* (Ours)':<15}"
    print(header)
    print("-" * 85)
    
    rows = [
        ("Trạng thái (Status)", metrics_pure['status'], metrics_mctd['status'], metrics_hmctd['status']),
        ("Thời gian quy hoạch (ms)", f"{metrics_pure['time_ms']:.2f} ms", f"{metrics_mctd['time_ms']:.2f} ms", f"{metrics_hmctd['time_ms']:.2f} ms"),
        ("Số Node đã duyệt (Nodes)", f"{metrics_pure['nodes_explored']}", f"{metrics_mctd['nodes_explored']}", f"{metrics_hmctd['nodes_explored']}"),
        ("Chiều dài quãng đường (m)", f"{metrics_pure['length_m']:.2f} m", f"{metrics_mctd['length_m']:.2f} m", f"{metrics_hmctd['length_m']:.2f} m"),
        ("Tổng Năng lượng / Chi phí (J)", f"{metrics_pure['energy_j']:.2f}", f"{metrics_mctd['energy_j']:.2f}", f"{metrics_hmctd['energy_j']:.2f}"),
        ("Độ mịn quỹ đạo (Smoothness)", f"{metrics_pure['smoothness']:.4f}", f"{metrics_mctd['smoothness']:.4f}", f"{metrics_hmctd['smoothness']:.4f}"),
        ("Số điểm Waypoints", f"{metrics_pure['waypoints']}", f"{metrics_mctd['waypoints']}", f"{metrics_hmctd['waypoints']}"),
        ("Độ dốc cực đại (Max Slope)", f"{metrics_pure['max_slope']:.3f}", f"{metrics_mctd['max_slope']:.3f}", f"{metrics_hmctd['max_slope']:.3f}")
    ]
    
    for row in rows:
        print(f"{row[0]:<30} | {row[1]:<15} | {row[2]:<15} | {row[3]:<15}")
    print("=" * 85)
    
    # -------------------------------------------------------------
    # TRỰC QUAN HÓA BẢN ĐỒ VÀ QUỸ ĐẠO
    # -------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Map 1: Weighted A*
    im1 = axes[0].imshow(heights, cmap='terrain', origin='lower')
    if path_pure:
        pp = np.array(path_pure)
        axes[0].plot(pp[:, 1], pp[:, 0], 'r-', linewidth=2.5, label='Weighted A*')
    axes[0].plot(start[1], start[0], 'go', markersize=8, label='Start')
    axes[0].plot(goal[1], goal[0], 'bx', markersize=10, markeredgewidth=3, label='Goal')
    axes[0].set_title(f"1. Weighted A*\nTime: {t_pure:.1f}ms | Nodes: {nodes_pure}")
    axes[0].legend(loc='upper right')
    plt.colorbar(im1, ax=axes[0], label='Elevation (m)')
    
    # Map 2: MCTD-A*
    im2 = axes[1].imshow(heights, cmap='terrain', origin='lower')
    if corridor_mask is not None:
        axes[1].imshow(corridor_mask, cmap='Oranges', alpha=0.35, origin='lower')
    if path_mctd:
        pm = np.array(path_mctd)
        axes[1].plot(pm[:, 1], pm[:, 0], 'm-', linewidth=2.5, label='MCTD-A*')
    axes[1].plot(start[1], start[0], 'go', markersize=8)
    axes[1].plot(goal[1], goal[0], 'bx', markersize=10, markeredgewidth=3)
    axes[1].set_title(f"2. MCTD-A*\nTime: {t_mctd_total:.1f}ms | Corridor Guided")
    axes[1].legend(loc='upper right')
    plt.colorbar(im2, ax=axes[1], label='Elevation (m)')
    
    # Map 3: HMCTD-A*
    im3 = axes[2].imshow(heights, cmap='terrain', origin='lower')
    if path_hmctd:
        ph = np.array(path_hmctd)
        axes[2].plot(ph[:, 1], ph[:, 0], 'c-', linewidth=2.5, label='HMCTD-A* (Ours)')
    axes[2].plot(start[1], start[0], 'go', markersize=8)
    axes[2].plot(goal[1], goal[0], 'bx', markersize=10, markeredgewidth=3)
    axes[2].set_title(f"3. HMCTD-A* (Hierarchical)\nTime: {t_hmctd:.1f}ms | Smooth: {metrics_hmctd['smoothness']:.2f}")
    axes[2].legend(loc='upper right')
    plt.colorbar(im3, ax=axes[2], label='Elevation (m)')
    
    fig.suptitle(f"Single Map Comparison Benchmark: {os.path.basename(npz_file)}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    out_fig = "figures/single_map_detailed_comparison.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    
    print(f"\n[THÀNH CÔNG] Đã tạo biểu đồ so sánh quỹ đạo tại: {out_fig}")
    
    # Also copy file to WSL path under ros2_ws/src
    # Convert numpy int64 to int
    start_list = [int(start[0]), int(start[1])]
    goal_list = [int(goal[0]), int(goal[1])]
    
    results_json = {
        "map_file": npz_file,
        "grid_size": int(grid_size),
        "start": start_list,
        "goal": goal_list,
        "weighted_a_star": metrics_pure,
        "mctd_a_star": metrics_mctd,
        "hmctd_a_star": metrics_hmctd
    }
    with open("../4. logs/single_map_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)
        
    print("-> Đã xuất tệp dữ liệu chi tiết JSON ra: ../4. logs/single_map_benchmark.json")

if __name__ == "__main__":
    npz_arg = sys.argv[1] if len(sys.argv) > 1 else "gt_dataset/000_025.npz"
    run_single_map_benchmark(npz_arg)
