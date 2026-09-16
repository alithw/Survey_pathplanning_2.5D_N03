# ==============================================================================
# BENCHMARK SO SÁNH THUẬT TOÁN QUY HOẠCH TOÀN CỤC TRÊN BẢN ĐỒ 2.5D
# ==============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import torch
import json

from a_star_cost import AStarPlanner
from fast_hmctd_planner import FastHMCTDPlanner

def load_map_000_025():
    base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
    filename = "000_025.pt"
    dataset_path = os.path.join(base_dir, filename)
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, f"env_{filename}")
        
    data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    tensors = data['tensors']
    def clean(name):
        t = tensors[name].squeeze()
        if len(t.shape) > 2: t = t[0]
        return t.numpy()
    return clean('heights'), clean('slopes'), clean('t_classes')

def evaluate_metrics(path, heights):
    if path is None or len(path) < 2:
        return float('inf'), float('inf')
    total_energy = 0.0
    smoothness = 0.0
    headings = []
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        d2d = np.hypot(dx, dy)
        dz = float(heights[p2[0], p2[1]] - heights[p1[0], p1[1]])
        sp = dz / d2d if d2d > 0 else 0
        cm = max(155.4*(sp**5) - 30.4*(sp**4) - 43.3*(sp**3) + 46.3*(sp**2) + 19.5*sp + 3.6, 0.5)
        total_energy += cm * d2d
        headings.append(np.arctan2(dy, dx))

    for i in range(len(headings) - 1):
        dth = abs(headings[i+1] - headings[i])
        dth = min(dth, 2 * np.pi - dth)
        smoothness += dth**2

    return total_energy, smoothness

def run_planner_benchmark():
    print('====================================================================')
    print('   BẮT ĐẦU BENCHMARK THUẬT TOÁN QUY HOẠCH TOÀN CỤC TRÊN MAP 000_025 ')
    print('====================================================================')
    
    heights, slopes, soils = load_map_000_025()
    grid_size = heights.shape[0]
    start = (33, 8)
    goal = (38, 58)

    # 1. Baseline: Weighted A*
    print('-> [1/3] Đang chạy Weighted A* chuẩn...')
    t0 = time.perf_counter()
    w_planner = AStarPlanner(
        heights=heights, slopes=slopes, t_classes=soils, grid_size=grid_size,
        ssf=1.2, smooth_radius=2, max_step_cost=350.0, max_total_cost=10000.0,
        heuristic_weight=3.0, min_step_cost=2.0
    )
    w_path, _ = w_planner.plan(start, goal)
    t_weighted = (time.perf_counter() - t0) * 1000.0
    w_path = np.array(w_path)
    w_energy, w_smooth = evaluate_metrics(w_path, heights)
    print(f'   + Thời gian: {t_weighted:.2f} ms | Năng lượng: {w_energy:.1f} J | Độ mịn: {w_smooth:.2f} rad^2 | Điểm: {len(w_path)}')

    # 2. HMCTD-A* (với corridor)
    print('-> [2/3] Đang chạy HMCTD-A* (Phiên bản gốc)...')
    corridor_mask = (slopes / 1.2 < 5.0)
    corridor_mask[start[0], start[1]] = True
    corridor_mask[goal[0], goal[1]] = True
    
    t0 = time.perf_counter()
    fast_base = FastHMCTDPlanner(
        heights=heights, slopes=slopes, t_classes=soils, grid_size=grid_size,
        corridor_mask=corridor_mask, ssf=1.2, smooth_radius=2,
        max_step_cost=350.0, max_total_cost=10000.0,
        heuristic_weight=3.0, min_step_cost=2.0
    )
    res_orig = fast_base._search_core(start, goal, use_corridor=False)
    t_hmctd = res_orig["planning_time_ms"]
    orig_path = np.array(res_orig["raw_path"])
    hmctd_energy, hmctd_smooth = evaluate_metrics(orig_path, heights)
    print(f'   + Thời gian: {t_hmctd:.2f} ms | Năng lượng: {hmctd_energy:.1f} J | Độ mịn: {hmctd_smooth:.2f} rad^2 | Điểm: {len(orig_path)}')

    # 3. Fast-HMCTD-A* (Đề xuất mới)
    print('-> [3/3] Đang chạy Fast-HMCTD-A* (APH + 2.5D Surface Shortcutter)...')
    res_fast = fast_base.plan_fast(start, goal)
    fast_path = np.array(res_fast["path"])
    print(f'   + Thời gian: {res_fast["planning_time_ms"]:.2f} ms | Nodes mở rộng: {res_fast["expanded_nodes"]} | Năng lượng: {res_fast["energy_joules"]:.1f} J | Độ mịn: {res_fast["smoothness"]:.2f} rad^2 | Điểm: {res_fast["num_waypoints"]}')

    # Lưu log JSON
    summary = {
        "weighted_a_star": {
            "time_ms": t_weighted,
            "energy_j": w_energy,
            "smoothness": w_smooth,
            "waypoints": len(w_path)
        },
        "hmctd_a_star": {
            "time_ms": t_hmctd,
            "energy_j": hmctd_energy,
            "smoothness": hmctd_smooth,
            "waypoints": len(orig_path)
        },
        "fast_hmctd_a_star": {
            "time_ms": res_fast["planning_time_ms"],
            "expanded_nodes": res_fast["expanded_nodes"],
            "energy_j": res_fast["energy_joules"],
            "smoothness": res_fast["smoothness"],
            "waypoints": res_fast["num_waypoints"]
        }
    }
    output_path = "/home/ali/ros2_ws/src/survey_pathplanning/survey_PathPlanning_2.5D_template/4. logs/fast_planner_benchmark.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f'\n-> Đã lưu kết quả tóm tắt vào: {output_path}')

    # Vẽ đồ thị so sánh trực quan
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Quỹ đạo
    ax = axes[0]
    im = ax.imshow(heights, cmap='terrain', alpha=0.6, origin='upper')
    ax.plot(w_path[:, 1], w_path[:, 0], 'r--', linewidth=2, label=f'Weighted A* ({t_weighted:.1f}ms)')
    ax.plot(orig_path[:, 1], orig_path[:, 0], 'b-.', linewidth=2, label=f'HMCTD-A* ({t_hmctd:.1f}ms)')
    ax.plot(fast_path[:, 1], fast_path[:, 0], 'g-', linewidth=2.8, label=f'Fast-HMCTD-A* ({res_fast["planning_time_ms"]:.1f}ms)')
    ax.scatter([start[1]], [start[0]], color='yellow', s=80, zorder=5, label='Start')
    ax.scatter([goal[1]], [goal[0]], color='red', s=80, zorder=5, label='Goal')
    ax.set_title('(A) So sánh Quỹ đạo trên Bản đồ Độ cao 2.5D', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # 2. Thời gian
    ax = axes[1]
    planners = ['Weighted A*', 'HMCTD-A*', 'Fast-HMCTD-A*\n(Proposed)']
    times = [t_weighted, t_hmctd, res_fast["planning_time_ms"]]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax.bar(planners, times, color=colors, width=0.5, edgecolor='black')
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.1f} ms', xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    ax.set_title('(B) Thời gian Quy hoạch (Planning Time)', fontweight='bold')
    ax.set_ylabel('Thời gian (ms)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Độ mịn
    ax = axes[2]
    smooth_vals = [w_smooth, hmctd_smooth, res_fast["smoothness"]]
    bars = ax.bar(planners, smooth_vals, color=['#95a5a6', '#f39c12', '#27ae60'], width=0.5, edgecolor='black')
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.2f} rad²', xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    ax.set_title('(C) Độ mịn Quỹ đạo (Smoothness - Càng nhỏ càng tốt)', fontweight='bold')
    ax.set_ylabel('Tổng bình phương góc bẻ lái (rad²)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = "/home/ali/ros2_ws/src/survey_pathplanning/survey_PathPlanning_2.5D_template/1. Report/figures/fast_planner_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'-> Đã lưu ảnh so sánh trực quan vào: {plot_path}')

    print('\n====================================================================')
    print('                      BẢNG KẾT QUẢ TỔNG HỢP                        ')
    print('====================================================================')
    print('{:<25} {:<15} {:<15} {:<15} {:<12}'.format(
        "Thuật toán", "Thời gian (ms)", "Năng lượng (J)", "Độ mịn (rad²)", "Waypoints"
    ))
    print('-' * 85)
    print('{:<25} {:<15.2f} {:<15.1f} {:<15.2f} {:<12}'.format("Weighted A*", t_weighted, w_energy, w_smooth, len(w_path)))
    print('{:<25} {:<15.2f} {:<15.1f} {:<15.2f} {:<12}'.format("HMCTD-A*", t_hmctd, hmctd_energy, hmctd_smooth, len(orig_path)))
    print('{:<25} {:<15.2f} {:<15.1f} {:<15.2f} {:<12}'.format("Fast-HMCTD-A* (Đề xuất)", res_fast["planning_time_ms"], res_fast["energy_joules"], res_fast["smoothness"], res_fast["num_waypoints"]))
    print('=' * 85)

if __name__ == '__main__':
    run_planner_benchmark()
