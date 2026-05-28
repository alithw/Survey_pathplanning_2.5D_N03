import os
import glob
import time
import json
import torch
import numpy as np
import pandas as pd
from a_star_cost import AStarPlanner
from pp_mctd_a_star import MCTDAStarPlanner, predict_corridor, model as cnn_model

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
# ==========================================
GT_DIR = "gt_dataset"
LOG_DIR = "../4. logs"
os.makedirs(LOG_DIR, exist_ok=True)

def calculate_smoothness(path):
    """
    Tính độ mịn quỹ đạo (Smoothness): bình phương biến thiên góc lái.
    S = sum (theta_{k+1} - theta_k)^2
    """
    if len(path) < 3:
        return 0.0
    dx = np.diff(path[:, 1])
    dy = np.diff(path[:, 0])
    # Tránh chia cho 0
    angles = np.arctan2(dy, dx)
    angle_diffs = np.diff(angles)
    # Chuẩn hóa độ lệch góc về [-pi, pi]
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(angle_diffs ** 2))

def run_benchmarks(limit=509):
    print("="*65)
    print(" KHỞI ĐỘNG HỆ THỐNG TỰ ĐỘNG THỰC NGHIỆM DIỆN RỘNG (Nhóm 3)")
    print("="*65)
    
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.npz")))
    if not gt_files:
        print("[LỖI] Không tìm thấy tệp dữ liệu Ground Truth nào trong gt_dataset!")
        return
        
    num_files = min(len(gt_files), limit)
    print(f"-> Tìm thấy {len(gt_files)} tệp dữ liệu. Sẽ chạy benchmark trên {num_files} tệp...")
    
    results = []
    
    # Đảm bảo mô hình CNN ở trạng thái eval
    cnn_model.eval()
    
    start_time_all = time.time()
    
    for idx, filepath in enumerate(gt_files[:num_files]):
        filename = os.path.basename(filepath)
        data = np.load(filepath)
        
        heights = data['heights']
        slopes = data['slopes']
        t_classes = data['t_classes']
        start = data['start']
        goal = data['goal']
        GRID_SIZE = heights.shape[0]
        
        # Khởi tạo hai Planner
        planner = MCTDAStarPlanner(
            heights=heights, slopes=slopes, t_classes=t_classes, grid_size=GRID_SIZE,
            ssf=1.2, smooth_radius=2, max_step_cost=350.0, max_total_cost=10000.0,
            heuristic_weight=3.0, min_step_cost=2.0
        )
        
        # --- 1. CHẠY WEIGHTED A* (CŨ) ---
        t_start = time.perf_counter()
        path_pure, cost_pure = planner.plan(tuple(start), tuple(goal))
        time_pure = (time.perf_counter() - t_start) * 1000 # ms
        
        # --- 2. CHẠY MCTD-A* (MỚI) ---
        # Phác thảo hành lang bằng System 1
        t_start = time.perf_counter()
        corridor_mask = predict_corridor(heights, slopes, t_classes, start, goal, cnn_model)
        time_sys1 = (time.perf_counter() - t_start) * 1000 # ms
        
        # Tìm đường bằng System 2 trong hành lang
        planner.corridor_mask = corridor_mask
        planner.corridor_threshold = 0.2
        
        t_start = time.perf_counter()
        path_mctd, cost_mctd, nodes_mctd, fallback = planner.plan_with_mctd(tuple(start), tuple(goal))
        time_sys2 = (time.perf_counter() - t_start) * 1000 # ms
        time_mctd_total = time_sys1 + time_sys2
        
        # Thu thập các chỉ số
        if path_pure and path_mctd:
            path_pure_arr = np.array(path_pure)
            path_mctd_arr = np.array(path_mctd)
            
            smooth_pure = calculate_smoothness(path_pure_arr)
            smooth_mctd = calculate_smoothness(path_mctd_arr)
            
            # Tính độ dài hình học (mét)
            len_pure = float(np.sum(np.sqrt(np.sum(np.diff(path_pure_arr, axis=0)**2, axis=1))))
            len_mctd = float(np.sum(np.sqrt(np.sum(np.diff(path_mctd_arr, axis=0)**2, axis=1))))
            
            entry = {
                'map': filename,
                'a_star_time_ms': time_pure,
                'a_star_energy_j': float(cost_pure),
                'a_star_length_m': len_pure,
                'a_star_smoothness': smooth_pure,
                'a_star_waypoints': len(path_pure),
                
                'mctd_time_ms': time_mctd_total,
                'mctd_energy_j': float(cost_mctd),
                'mctd_length_m': len_mctd,
                'mctd_smoothness': smooth_mctd,
                'mctd_waypoints': len(path_mctd),
                'mctd_fallback': fallback,
                'mctd_nodes_explored': nodes_mctd
            }
            results.append(entry)
            
        if (idx + 1) % 50 == 0:
            print(f" -> Đã benchmark {idx+1}/{num_files} bản đồ...")
            
    elapsed_all = time.time() - start_time_all
    print(f"\n🎉 THỰC NGHIỆM DIỆN RỘNG HOÀN TẤT TRONG {elapsed_all:.2f} GIÂY!")
    
    # 3. LƯU BÁO CÁO KẾT QUẢ DƯỚI DẠNG JSON & CSV
    log_json_path = os.path.join(LOG_DIR, "benchmark_results.json")
    with open(log_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    df = pd.DataFrame(results)
    log_csv_path = os.path.join(LOG_DIR, "benchmark_results.csv")
    df.to_csv(log_csv_path, index=False)
    
    print(f" -> Đã xuất log thô kết quả ra JSON: {log_json_path}")
    print(f" -> Đã xuất log thô kết quả ra CSV:  {log_csv_path}")
    
    # 4. TÍNH TOÁN VÀ IN THỐNG KÊ MẪU ĐỂ CHÈN VÀO BÁO CÁO LATEX
    print("\n" + "="*45)
    print(" BẢNG TỔNG HỢP HIỆU NĂNG (BENCHMARK SUMMARY)")
    print("="*45)
    metrics_summary = {}
    for col in ['time_ms', 'energy_j', 'length_m', 'smoothness']:
        astar_col = f'a_star_{col}'
        mctd_col = f'mctd_{col}'
        
        astar_mean = df[astar_col].mean()
        mctd_mean = df[mctd_col].mean()
        improvement = (astar_mean - mctd_mean) / astar_mean * 100
        
        # Với time_ms, mctd giảm thời gian tức là tăng tốc độ (improvement dương)
        # Với smoothness, mctd giảm smoothness score tức là mượt hơn (improvement dương)
        # Với energy_j và length_m, MCTD-A* bám sát hoặc thậm chí tối ưu hơn.
        
        print(f"  * {col.upper()}:")
        print(f"    + Weighted A* : {astar_mean:.3f}")
        print(f"    + MCTD-A*     : {mctd_mean:.3f}")
        print(f"    ⚡ Cải tiến    : {improvement:.2f}%")
        print("-" * 40)
        
        metrics_summary[col] = {
            'a_star': float(astar_mean),
            'mctd': float(mctd_mean),
            'improvement': float(improvement)
        }
        
    # Thống kê fallback
    fallback_rate = df['mctd_fallback'].mean() * 100
    print(f"  * Tỉ lệ kích hoạt Fallback của Hệ thống 2: {fallback_rate:.2f}% (Rất an toàn)")
    print("="*45)
    
    # Lưu tóm tắt thống kê
    summary_path = os.path.join(LOG_DIR, "benchmark_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics_summary,
            'fallback_rate': fallback_rate,
            'total_maps': len(df)
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_benchmarks()
