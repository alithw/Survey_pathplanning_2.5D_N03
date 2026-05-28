"""
generate_gt_paths.py
---------------------------
Chạy Weighted A* trên toàn bộ 1000 bản đồ của BenchNav dataset01/train
để sinh dữ liệu quỹ đạo tối ưu vật lý (Ground Truth) phục vụ huấn luyện System 1.

Môi trường chạy:
    wsl bash -c "cd ~/ros2_ws/src/survey_pathplanning/survey_PathPlanning_2.5D_template/'3. code' && source ~/ros2_ws/src/benchnav_env/bin/activate && python generate_gt_paths.py"

Có hỗ trợ tham số --test để kiểm tra nhanh.
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import time
from a_star_cost import AStarPlanner

# Cấu hình đường dẫn
BASE_DIR = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
GT_DIR = "gt_dataset"
os.makedirs(GT_DIR, exist_ok=True)

def clean_tensor(tensors, name):
    """Hàm bóc tách và định dạng tensor 2D"""
    t = tensors[name].squeeze()
    if len(t.shape) > 2: 
        t = t[0]
    return t.numpy()

def generate_corridor_mask(path, grid_size=64, width=5):
    """
    Tạo mặt nạ hành lang (corridor mask) bao quanh đường đi A*.
    Điểm (r, c) = 1 nếu khoảng cách L2 đến điểm gần nhất trên quỹ đạo <= width.
    """
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    if path is None or len(path) == 0:
        return mask
        
    for r in range(grid_size):
        for c in range(grid_size):
            # Tính khoảng cách L2 tối thiểu từ (r, c) đến các điểm trên path
            dists = np.sqrt((path[:, 0] - r)**2 + (path[:, 1] - c)**2)
            if np.min(dists) <= width:
                mask[r, c] = 1.0
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Chạy test thử nghiệm trên 5 maps')
    parser.add_argument('--limit', type=int, default=1000, help='Giới hạn số lượng bản đồ xử lý')
    args = parser.parse_args()
    
    if args.test:
        args.limit = 5
        print("[TEST] Chạy thử nghiệm sinh dữ liệu trên 5 maps...")
        
    if not os.path.exists(BASE_DIR):
        print(f"[LỖI] Không tìm thấy thư mục dataset: {BASE_DIR}")
        sys.exit(1)
        
    # Lấy danh sách tất cả các file .pt trong thư mục dataset
    all_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith('.pt') and not f.startswith('env_')])
    if not all_files:
        # Thử lấy các file có tiền tố env_
        all_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith('.pt')])
        
    if not all_files:
        print("[LỖI] Thư mục dataset rỗng!")
        sys.exit(1)
        
    total_files = min(len(all_files), args.limit)
    print(f"Bắt đầu xử lý {total_files} bản đồ từ dataset...")
    
    success_count = 0
    start_time = time.time()
    
    for idx, filename in enumerate(all_files[:total_files]):
        # Tách index của bản đồ từ tên file (ví dụ: '000_025.pt' -> hash index để làm seed)
        map_idx = idx
        try:
            parts = filename.replace('.pt', '').split('_')
            map_idx = int(parts[0]) * 100 + int(parts[1])
        except:
            pass
            
        dataset_path = os.path.join(BASE_DIR, filename)
        
        # Load dữ liệu map
        try:
            data = torch.load(dataset_path, map_location='cpu', weights_only=False)
            tensors = data['tensors']
            heights = clean_tensor(tensors, 'heights')
            slopes = clean_tensor(tensors, 'slopes')
            t_classes = clean_tensor(tensors, 't_classes')
        except Exception as e:
            print(f" [{filename}] Lỗi đọc file: {e}. Bỏ qua.")
            continue
            
        GRID_SIZE = heights.shape[0]
        
        # Thiết lập seed cố định cho mỗi bản đồ để sinh start/goal nhất quán
        random.seed(map_idx)
        np.random.seed(map_idx)
        
        # Chọn điểm xuất phát & đích đến an toàn giống pp_rrt.py
        safe_spots = np.argwhere(slopes < 0.4)
        start_spots = [tuple(pt) for pt in safe_spots if pt[1] <= int(GRID_SIZE * 0.2)]
        goal_spots = [tuple(pt) for pt in safe_spots if pt[1] >= int(GRID_SIZE * 0.8)]
        
        if not start_spots or not goal_spots:
            print(f" [{filename}] Không có điểm start/goal an toàn. Bỏ qua.")
            continue
            
        # Thử nghiệm tối đa 5 cặp start/goal ngẫu nhiên để tìm được đường đi thành công
        path = None
        start_node = None
        goal_node = None
        
        for trial_num in range(5):
            start_node = random.choice(start_spots)
            goal_node = random.choice(goal_spots)
            min_dist = int(GRID_SIZE * 0.35)
            
            attempts = 0
            while np.linalg.norm(np.array(start_node) - np.array(goal_node)) < min_dist:
                goal_node = random.choice(goal_spots)
                attempts += 1
                if attempts > 50: 
                    break
                    
            # Khởi tạo Planner
            planner = AStarPlanner(
                heights=heights, slopes=slopes, t_classes=t_classes, grid_size=GRID_SIZE,
                ssf=1.2, smooth_radius=2, max_step_cost=350.0, max_total_cost=10000.0,
                heuristic_weight=3.0, min_step_cost=2.0
            )
            
            path, _ = planner.plan(start_node, goal_node)
            if path:
                break
        
        if path:
            path_arr = np.array(path)
            corridor_mask = generate_corridor_mask(path_arr, GRID_SIZE, width=5)
            
            # Lưu trữ file dữ liệu GT
            out_filename = filename.replace('.pt', '.npz')
            out_path = os.path.join(GT_DIR, out_filename)
            
            np.savez_compressed(
                out_path,
                heights=heights,
                slopes=slopes,
                t_classes=t_classes,
                start=np.array(start_node),
                goal=np.array(goal_node),
                path=path_arr,
                corridor_mask=corridor_mask
            )
            success_count += 1
            if (idx + 1) % 50 == 0 or args.test:
                print(f" -> Xử lý xong {idx+1}/{total_files} | Thành công: {success_count} maps")
        else:
            if args.test:
                print(f" -> [{filename}] Không tìm thấy đường đi sau 5 thử nghiệm cặp điểm!")
                
    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"🎉 HOÀN TẤT SINH DỮ LIỆU GROUND TRUTH!")
    print(f" - Tổng số bản đồ duyệt qua   : {total_files}")
    print(f" - Số bản đồ thành công        : {success_count}")
    print(f" - Thời gian thực hiện         : {elapsed:.2f} giây")
    print(f" - Thư mục lưu dữ liệu         : {GT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
