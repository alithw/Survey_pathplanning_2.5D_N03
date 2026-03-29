import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from a_star_cost import AStarPlanner

# ==========================================
# CẤU HÌNH THÔNG SỐ ROBOT (ROBOT CAPABILITIES)
# ==========================================
ROBOT_MAX_STEP_COST = 350.0 # chi phí tối đa cho 1 bước cùa robot
ROBOT_MAX_TOTAL_COST = 10000.0 # tổng chi phí tối đa mà robot có thể hoạt động 
vx = 0
mx = 0

# Thông số Động lực học Robot (Mô hình chống lật LTR)
TRACK_WIDTH_T = 0.8   # Chiều rộng cơ sở bánh xe (m)
COG_HEIGHT_H = 0.4    # Chiều cao trọng tâm (m)
SSF = TRACK_WIDTH_T / (2 * COG_HEIGHT_H) # Chỉ số ổn định tĩnh = 1.2

# Lấy xu hướng độ dốc & đất trong bán kính 3 ô
SMOOTH_RADIUS = 2

# === CẤU HÌNH THUẬT TOÁN WEIGHTED A* ===
HEURISTIC_WEIGHT = 3.0 
MIN_STEP_COST = 2.0 

MANUAL_TEST = False
START_MANUAL = (26, 2)
GOAL_MANUAL = (14, 57)

# ==========================================
# 1. THƯ MỤC CHỨA BẢN ĐỒ
# ==========================================
base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"

def clean_tensor(tensors, name):
    t = tensors[name].squeeze()
    if len(t.shape) > 2: t = t[0]
    return t.numpy()

def get_filename():
    global vx, mx
    t_idx = vx
    m_idx = mx
    mx += 1
    if mx == 100:
        mx = 0
        vx += 1
    
    if t_idx == 10:
        print("\n--- Đã duyệt qua 1000 map ---")
        return None
        
    filename = f"{t_idx:03d}_{m_idx:03d}.pt"
    return filename

# ==========================================
# 2. VÒNG LẶP AUTO TÌM BẢN ĐỒ VÀ CHẠY
# ==========================================
if __name__ == "__main__":
    map_found = False
    while not map_found:
        try:
            filename = get_filename()
            if filename == None:
                break
        except Exception as e: 
            print(f"Lỗi: {e}")
            break

        dataset_path = os.path.join(base_dir, filename)
        if not os.path.exists(dataset_path):
            filename = f"env_{filename}"
            dataset_path = os.path.join(base_dir, filename)
            if not os.path.exists(dataset_path):
                continue

        # Load dữ liệu Map
        try:
            data = torch.load(dataset_path, map_location='cpu', weights_only=False)
            tensors = data['tensors']
            heights = clean_tensor(tensors, 'heights')
            slopes = clean_tensor(tensors, 'slopes')
            t_classes = clean_tensor(tensors, 't_classes')
            GRID_SIZE = heights.shape[0]
        except Exception as e:
            print(f"Lỗi đọc file {filename}: {e}")
            continue

        # Khởi tạo class Planner
        planner = AStarPlanner(
            heights=heights, 
            slopes=slopes, 
            t_classes=t_classes, 
            grid_size=GRID_SIZE,
            ssf=SSF, 
            smooth_radius=SMOOTH_RADIUS,
            max_step_cost=ROBOT_MAX_STEP_COST, 
            max_total_cost=ROBOT_MAX_TOTAL_COST,
            heuristic_weight=HEURISTIC_WEIGHT, 
            min_step_cost=MIN_STEP_COST
        )

        print(f"\n[{filename}] Bắt đầu phân tích map với SSF={SSF:.2f} | Size={GRID_SIZE}x{GRID_SIZE}")
        
        # Thiết lập các điểm Start/Goal
        if MANUAL_TEST:
            start_node = START_MANUAL
            goal_node = GOAL_MANUAL
            trials = [(start_node, goal_node)]
        else:
            safe_spots = np.argwhere(slopes < 1.0) 
            left_bound = int(GRID_SIZE * 0.2)
            right_bound = int(GRID_SIZE * 0.8)
            
            start_spots = [tuple(pt) for pt in safe_spots if pt[1] <= left_bound]
            goal_spots = [tuple(pt) for pt in safe_spots if pt[1] >= right_bound]
            
            if not start_spots or not goal_spots: 
                print("     -> Map có địa hình 2 bên quá gắt. Bỏ qua...")
                continue
            
            trials = []
            for _ in range(10):
                s_node = random.choice(start_spots)
                g_node = random.choice(goal_spots)
                min_dist = GRID_SIZE // 2 
                
                attempts = 0
                while np.linalg.norm(np.array(s_node) - np.array(g_node)) < min_dist:
                    g_node = random.choice(goal_spots)
                    attempts += 1
                    if attempts > 50: break
                trials.append((s_node, g_node))
                
        # Chạy các thử nghiệm Pathfinding
        for trial_idx, (start_node, goal_node) in enumerate(trials):
            print(f"  -> Thử lần {trial_idx+1} | Start: {start_node} | Goal: {goal_node}")
            
            # ==== GỌI HÀM PLAN TỪ FILE a_star_cost.py ====
            path, total_cost = planner.plan(start_node, goal_node)
            
            if path:
                print(f"=> THÀNH CÔNG! Đã tìm thấy tuyến đường tối ưu Vật lý.")
                print(f"=> Tổng chi phí tiêu hao: {total_cost:.2f} / Max: {ROBOT_MAX_TOTAL_COST}")
                
                path = np.array(path)
                plt.figure(figsize=(16, 6))
                
                # BIỂU ĐỒ 1: ĐỘ CAO VÀ RỦI RO
                ax1 = plt.subplot(1, 2, 1)
                img_heights = plt.imshow(heights, cmap='terrain')
                plt.colorbar(img_heights, ax=ax1, label='Độ cao địa hình (z)', shrink=0.8)
                plt.plot(path[:, 1], path[:, 0], color='red', linewidth=2.5, label='Safe & Energy-Optimal Path')
                plt.scatter([start_node[1]], [start_node[0]], color='yellow', s=100, label='Start', edgecolors='black', zorder=5)
                plt.scatter([goal_node[1]], [goal_node[0]], color='blue', s=100, label='Goal', edgecolors='white', zorder=5)
                plt.title(f"Terrain Heights\nTotal Energy Cost: {total_cost:.1f}")
                plt.legend()
                
                # BIỂU ĐỒ 2: LOẠI ĐẤT
                ax2 = plt.subplot(1, 2, 2)
                
                img_classes = plt.imshow(t_classes, cmap='Set3')
                unique_classes = np.unique(t_classes.astype(int))
                cbar_classes = plt.colorbar(img_classes, ax=ax2, shrink=0.8, ticks=unique_classes)
                
                labels = []
                for c in unique_classes:
                    r_val = planner.soil_resistance.get(c, 2.0)
                    labels.append(f'Loại {c} (R={r_val})')
                    
                cbar_classes.set_ticklabels(labels)
                cbar_classes.set_label('Chỉ số cản của đất (Bekker-Wong)')
                
                plt.plot(path[:, 1], path[:, 0], color='red', linewidth=2.5)
                plt.scatter([start_node[1]], [start_node[0]], color='yellow', s=100, edgecolors='black', zorder=5)
                plt.scatter([goal_node[1]], [goal_node[0]], color='blue', s=100, edgecolors='white', zorder=5)
                plt.title("Soil Classes (Bekker-Wong Resistance)")
                
                plt.tight_layout()
                plt.show()
                map_found = True
                break
            else:
                print("     Thất bại: Tuyến đường vượt ngưỡng lật xe LTR hoặc kẹt vật cản.")
                
        if not map_found and MANUAL_TEST:
             print("     => Tọa độ ghim cứng không có đường đi an toàn trên map này. Chuyển map kế tiếp...")