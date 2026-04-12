import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time

# Import Class Planner
from rrt_star_cost import RRTStarPlanner

# ==========================================
# CẤU HÌNH THÔNG SỐ
# ==========================================
ROBOT_MAX_STEP_COST = 350.0 
ROBOT_MAX_TOTAL_COST = 10000.0 
vx = 0
mx = 0

TRACK_WIDTH_T = 0.8   
COG_HEIGHT_H = 0.4    
SSF = 1.2 
SMOOTH_RADIUS = 2

# Thông số thuật toán RRT* (Tuning đặc biệt để vượt qua điểm yếu kẹt khe hẹp)
MAX_ITERATIONS = 15000  
STEP_SIZE = 4.0         # (Tăng lên 4.0) Bước nhảy dài giúp RRT* vượt qua các vùng kẹt cục bộ dễ hơn
SEARCH_RADIUS = 8.0     
GOAL_SAMPLE_RATE = 0.25 # (Tăng lên 25%) Lực hút về đích mạnh hơn để cây không mọc lan man

MANUAL_TEST = True  
START_MANUAL = (33, 8)
GOAL_MANUAL = (40, 58)

base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"

def clean_tensor(tensors, name):
    t = tensors[name].squeeze()
    if len(t.shape) > 2: t = t[0]
    return t.numpy()

def get_filename():
    global vx, mx
    t_idx, m_idx = vx, mx
    mx += 1
    if mx == 100:
        mx = 0; vx += 1
    if t_idx == 10: return None
    # filename = f"{t_idx:03d}_{m_idx:03d}.pt"
    filename = "000_025.pt"
    return filename

if __name__ == "__main__":
    if not os.path.exists(base_dir):
        print(f"[LỖI] Không tìm thấy thư mục: {base_dir}")
        exit(1)
        
    print(f"Bắt đầu quy hoạch RRT* tại: {base_dir}")
    print(f"Chú ý: RRT* yếu hơn A* trên grid map có hành lang hẹp. Đã tăng STEP_SIZE lên {STEP_SIZE} để ép thuật toán vượt rào.")
    
    map_found = False
    
    while not map_found:
        filename = get_filename()
        if not filename: 
            print(f"\n[CẢNH BÁO] Đã duyệt qua toàn bộ dataset nhưng không tìm được đường hợp lệ.")
            break

        dataset_path = os.path.join(base_dir, filename)
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(base_dir, f"env_{filename}")
            if not os.path.exists(dataset_path): 
                continue

        try:
            data = torch.load(dataset_path, map_location='cpu', weights_only=False)
            tensors = data['tensors']
            heights = clean_tensor(tensors, 'heights')
            slopes = clean_tensor(tensors, 'slopes')
            t_classes = clean_tensor(tensors, 't_classes')
            GRID_SIZE = heights.shape[0]
        except Exception as e:
            continue

        planner = RRTStarPlanner(
            heights=heights, slopes=slopes, t_classes=t_classes, grid_size=GRID_SIZE,
            ssf=SSF, smooth_radius=SMOOTH_RADIUS,
            max_step_cost=ROBOT_MAX_STEP_COST, max_total_cost=ROBOT_MAX_TOTAL_COST,
            max_iter=MAX_ITERATIONS, step_size=STEP_SIZE, 
            search_radius=SEARCH_RADIUS, goal_sample_rate=GOAL_SAMPLE_RATE
        )

        print(f"\n[{filename}] Đã load Map thành công. Bắt đầu lấy mẫu...")
        
        if MANUAL_TEST:
            trials = [(START_MANUAL, GOAL_MANUAL)]
        else:
            # Chọn điểm xuất phát cực kỳ an toàn
            safe_spots = np.argwhere(slopes < 0.4) 
            start_spots = [tuple(pt) for pt in safe_spots if pt[1] <= int(GRID_SIZE * 0.2)]
            goal_spots = [tuple(pt) for pt in safe_spots if pt[1] >= int(GRID_SIZE * 0.8)]
            
            if not start_spots or not goal_spots: 
                print(f"  -> Bỏ qua map do địa hình hai bên mép quá dốc (không có điểm xuất phát).")
                continue
            
            trials = []
            for _ in range(10):
                s_node = random.choice(start_spots)
                g_node = random.choice(goal_spots)
                # Giảm min_dist xuống còn 35% map để RRT* có cơ hội thành công cao hơn
                min_dist = int(GRID_SIZE * 0.35) 
                
                attempts = 0
                while np.linalg.norm(np.array(s_node) - np.array(g_node)) < min_dist:
                    g_node = random.choice(goal_spots)
                    attempts += 1
                    if attempts > 50: break
                trials.append((s_node, g_node))

        for trial_idx, (start_node, goal_node) in enumerate(trials):
            print(f"  -> Đang chạy thử nghiệm {trial_idx+1}/10 (Start: {start_node} | Goal: {goal_node})")
            
            # --- Đo đạc hiệu năng ---
            start_time = time.perf_counter()
            path, total_cost = planner.plan(start_node, goal_node)
            end_time = time.perf_counter()

            exec_time = (end_time - start_time) * 1000 

            if path:
                print("=" * 50)
                print(f"🎉 TÌM ĐƯỜNG THÀNH CÔNG: {filename}")
                print(f"  ⏱ Thời gian xử lý : {exec_time:.2f} ms")
                print(f"  ⚡ Tổng năng lượng : {total_cost:.2f}")
                print(f"  📍 Số lượng điểm   : {len(path)} node")
                print("=" * 50)

                # Visualize quỹ đạo trực quan (CẬP NHẬT 2 BIỂU ĐỒ)
                path = np.array(path)
                plt.figure(figsize=(16, 6))
                
                # BIỂU ĐỒ 1: Nền độ cao
                ax1 = plt.subplot(1, 2, 1)
                img_heights = plt.imshow(heights, cmap='terrain', origin='upper')
                plt.colorbar(img_heights, ax=ax1, label='Độ cao địa hình (z)', shrink=0.8)
                
                plt.plot(path[:, 1], path[:, 0], '-r', linewidth=2.5, label='RRT* Physics Path')
                plt.scatter(start_node[1], start_node[0], c='yellow', s=120, label='Start', edgecolors='black', zorder=5)
                plt.scatter(goal_node[1], goal_node[0], c='blue', s=120, label='Goal', edgecolors='white', zorder=5)
                
                plt.title(f"2.5D RRT* Planning (Heights)\nExecution: {exec_time:.1f}ms | Energy Cost: {total_cost:.1f}")
                plt.legend()

                # BIỂU ĐỒ 2: Loại đất (Bekker-Wong Resistance)
                ax2 = plt.subplot(1, 2, 2)
                img_classes = plt.imshow(t_classes, cmap='Set3', origin='upper')
                unique_classes = np.unique(t_classes.astype(int))
                cbar_classes = plt.colorbar(img_classes, ax=ax2, shrink=0.8, ticks=unique_classes)
                
                labels = []
                for c in unique_classes:
                    r_val = planner.soil_resistance.get(c, 2.0)
                    labels.append(f'Loại {c} (R={r_val})')
                    
                cbar_classes.set_ticklabels(labels)
                cbar_classes.set_label('Chỉ số cản của đất (Bekker-Wong)')
                
                plt.plot(path[:, 1], path[:, 0], '-r', linewidth=2.5)
                plt.scatter(start_node[1], start_node[0], c='yellow', s=120, edgecolors='black', zorder=5)
                plt.scatter(goal_node[1], goal_node[0], c='blue', s=120, edgecolors='white', zorder=5)
                plt.title("Soil Classes (Bekker-Wong Resistance)")
                
                plt.tight_layout()
                plt.show()
                
                map_found = True
                break
            else:
                print(f"     [X] Kẹt khe hẹp sau {MAX_ITERATIONS} vòng. Chuyển cặp điểm khác...")
        
        if map_found:
            break