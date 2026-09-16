# ==============================================================================
# BENCHMARK SO SÁNH TRỰC DIỆN: TRADITIONAL vs HYBRID MLP vs EDGE-HYBRID NMPC
# Đánh giá 5 tiêu chí: Thời gian giải (ms), Tần số (Hz), RMSE (m), CTE cực đại (m), Năng lượng (J)
# ==============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import os
import torch

from a_star_cost import AStarPlanner
from edge_hybrid_nmpc import run_simulation

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

def run_all_benchmarks():
    print('====================================================================')
    print('   BẮT ĐẦU BENCHMARK CÁC BỘ ĐIỀU KHIỂN NMPC TRÊN BẢN ĐỒ 2.5D       ')
    print('====================================================================')
    
    heights, slopes, soils = load_map_000_025()
    grid_size = heights.shape[0]

    planner = AStarPlanner(
        heights=heights, slopes=slopes, t_classes=soils, grid_size=grid_size,
        ssf=1.2, smooth_radius=2, max_step_cost=350.0, max_total_cost=10000.0,
        heuristic_weight=3.0, min_step_cost=2.0
    )
    start = (33, 8)
    goal = (38, 58)
    path, _ = planner.plan(start, goal)
    if path is None:
        path, _ = planner.plan((10, 10), (50, 50))
    path = np.array(path)

    modes = ["traditional", "hybrid_mlp", "edge_hybrid"]
    results = {}

    print('\n--- 1. THỬ NGHIỆM ĐIỀU KIỆN TIÊU CHUẨN (NOMINAL SLIP) ---')
    for m in modes:
        print(f'-> Đang chạy kiểm thử chế độ: {m}...')
        res = run_simulation(path, heights, slopes, soils, mode=m, disturbance=False)
        results[m] = res
        print(f'   + RMSE: {res["rmse"]:.4f} m | Max CTE: {res["max_cte"]:.4f} m | Solve Time TB: {res["mean_solve_time_ms"]:.2f} ms | Freq: {res["control_frequency_hz"]:.1f} Hz')

    print('\n--- 2. THỬ NGHIỆM ĐIỀU KIỆN NHIỄU BÙN ĐỘT NGỘT (OUT-OF-DISTRIBUTION DISTURBANCE) ---')
    results_disturbed = {}
    for m in modes:
        print(f'-> Đang chạy kiểm thử có nhiễu đột ngột: {m}...')
        res = run_simulation(path, heights, slopes, soils, mode=m, disturbance=True)
        results_disturbed[m] = res
        print(f'   + RMSE: {res["rmse"]:.4f} m | Max CTE: {res["max_cte"]:.4f} m | Thành công: {res["success"]}')

    # Lưu log JSON
    output_dir = "/home/ali/ros2_ws/src/survey_pathplanning/survey_PathPlanning_2.5D_template/4. logs"
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "edge_nmpc_benchmark_results.json")
    
    clean_json = {
        "nominal": {
            m: {
                "rmse": results[m]["rmse"],
                "max_cte": results[m]["max_cte"],
                "mean_solve_time_ms": results[m]["mean_solve_time_ms"],
                "max_solve_time_ms": results[m]["max_solve_time_ms"],
                "control_frequency_hz": results[m]["control_frequency_hz"],
                "energy_joules": results[m]["energy_joules"],
                "success": results[m]["success"]
            } for m in modes
        },
        "disturbed": {
            m: {
                "rmse": results_disturbed[m]["rmse"],
                "max_cte": results_disturbed[m]["max_cte"],
                "mean_solve_time_ms": results_disturbed[m]["mean_solve_time_ms"],
                "control_frequency_hz": results_disturbed[m]["control_frequency_hz"],
                "energy_joules": results_disturbed[m]["energy_joules"],
                "success": results_disturbed[m]["success"]
            } for m in modes
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(clean_json, f, indent=4)
    print(f'\n-> Đã lưu kết quả tóm tắt JSON vào: {summary_path}')

    # Vẽ đồ thị trực quan so sánh
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Quỹ đạo bám đường
    ax = axes[0, 0]
    im = ax.imshow(heights, cmap='terrain', alpha=0.6, origin='upper')
    ax.plot(path[:, 1], path[:, 0], 'k--', linewidth=2.5, label='A* Reference Path')
    ax.plot(results["traditional"]["history_x"], results["traditional"]["history_y"], 'r-.', linewidth=1.8, label='Traditional NMPC')
    ax.plot(results["hybrid_mlp"]["history_x"], results["hybrid_mlp"]["history_y"], 'b:', linewidth=2.0, label='Hybrid NMPC (MLP)')
    ax.plot(results["edge_hybrid"]["history_x"], results["edge_hybrid"]["history_y"], 'g-', linewidth=2.2, label='Edge-Hybrid NMPC (Proposed)')
    ax.set_title('(A) Quỹ đạo bám đường trên bản đồ độ cao 2.5D', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='lower right', fontsize=9)
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # 2. Sai số bám đường theo thời gian
    ax = axes[0, 1]
    steps_arr = np.arange(len(results["traditional"]["history_errors"])) * 0.1
    ax.plot(steps_arr, results["traditional"]["history_errors"], 'r-.', label='Traditional (RMSE: {:.3f}m)'.format(results["traditional"]["rmse"]))
    ax.plot(steps_arr, results["hybrid_mlp"]["history_errors"], 'b:', label='Hybrid MLP (RMSE: {:.3f}m)'.format(results["hybrid_mlp"]["rmse"]))
    ax.plot(steps_arr, results["edge_hybrid"]["history_errors"], 'g-', label='Edge-Hybrid (RMSE: {:.3f}m)'.format(results["edge_hybrid"]["rmse"]))
    ax.set_title('(B) Sai số lệch đường (Tracking Error) theo thời gian', fontsize=12, fontweight='bold')
    ax.set_xlabel('Thời gian (giây)')
    ax.set_ylabel('Sai số khoảng cách (m)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=9)

    # 3. Thời gian giải (Solve Time)
    ax = axes[1, 0]
    names = ['Traditional\nNMPC', 'Hybrid\n(PyTorch MLP)', 'Edge-Hybrid\n(Slip-LUT + EKF)']
    solve_means = [results[m]["mean_solve_time_ms"] for m in modes]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax.bar(names, solve_means, color=colors, width=0.5, edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f} ms', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    ax.set_title('(C) Thời gian giải trung bình mỗi chu kỳ (Solve Time)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Thời gian (ms)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Hiệu năng khi gặp bùn lạ đột ngột
    ax = axes[1, 1]
    dist_rmses = [results_disturbed[m]["rmse"] for m in modes]
    dist_maxes = [results_disturbed[m]["max_cte"] for m in modes]
    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, dist_rmses, width, label='RMSE (m)', color='#f39c12', edgecolor='black')
    ax.bar(x_pos + width/2, dist_maxes, width, label='Max CTE (m)', color='#d35400', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title('(D) Khả năng chống chịu khi gặp nhiễu bùn đất lạ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sai số lệch (m)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = "/home/ali/ros2_ws/src/survey_pathplanning/survey_PathPlanning_2.5D_template/1. Report/figures/edge_nmpc_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'-> Đã lưu biểu đồ so sánh vào: {plot_path}')

    print('\n====================================================================')
    print('                      BẢNG KẾT QUẢ TỔNG HỢP                        ')
    print('====================================================================')
    print('{:<25} {:<12} {:<12} {:<15} {:<12} {:<10}'.format(
        "Bộ điều khiển", "RMSE (m)", "Max CTE (m)", "Solve Time (ms)", "Tần số (Hz)", "Năng lượng"
    ))
    print('-' * 90)
    for m in modes:
        r = results[m]
        print('{:<25} {:<12.4f} {:<12.4f} {:<15.2f} {:<12.1f} {:<10.2f}'.format(
            m, r["rmse"], r["max_cte"], r["mean_solve_time_ms"], r["control_frequency_hz"], r["energy_joules"]
        ))
    print('=' * 90)

if __name__ == '__main__':
    run_all_benchmarks()
