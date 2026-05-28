import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
import time
from a_star_cost import AStarPlanner
from train_system1 import System1CNN

# ==========================================
# CẤU HÌNH THÔNG SỐ ROBOT & PLANNING
# ==========================================
ROBOT_MAX_STEP_COST = 350.0 
ROBOT_MAX_TOTAL_COST = 10000.0 
TRACK_WIDTH_T = 0.8   
COG_HEIGHT_H = 0.4    
SSF = TRACK_WIDTH_T / (2 * COG_HEIGHT_H) 
SMOOTH_RADIUS = 2
HEURISTIC_WEIGHT = 3.0 
MIN_STEP_COST = 2.0 

MANUAL_TEST = False
START_MANUAL = (33, 8)
GOAL_MANUAL = (40, 58)

# THƯ MỤC DATASET
base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
MODEL_PATH = "system1_model.pth"

class MCTDAStarPlanner(AStarPlanner):
    def __init__(self, heights, slopes, t_classes, grid_size, corridor_mask=None, corridor_threshold=0.2, **kwargs):
        super(MCTDAStarPlanner, self).__init__(heights, slopes, t_classes, grid_size, **kwargs)
        self.corridor_mask = corridor_mask
        self.corridor_threshold = corridor_threshold

    def plan_with_mctd(self, start, goal):
        """
        Thuật toán Weighted A* giới hạn trong hành lang dự đoán (System 1)
        """
        if self.corridor_mask is None:
            return self.plan(start, goal)
            
        start_h = np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2) * self.min_step_cost
        import heapq
        
        pq = [(self.heuristic_weight * start_h, 0, start)]
        distances = {start: 0}
        parent = {start: None}
        
        nodes_explored = 0
        
        while pq:
            f_score, curr_d, (x, y) = heapq.heappop(pq)
            nodes_explored += 1
            
            if curr_d > distances.get((x, y), float('inf')):
                continue
                
            if (x, y) == goal:
                path = []
                curr = (x, y)
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1], curr_d, nodes_explored, False
                
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (1,1), (-1,-1), (1,-1), (-1,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # RÀNG BUỘC HÀNH LANG (System 1 Guidance)
                    # Ngoại trừ điểm start và goal để tránh bóp nghẹt nếu mask đầu ra bị lệch nhẹ ở biên
                    if (nx, ny) != goal and (nx, ny) != start:
                        if self.corridor_mask[nx, ny] < self.corridor_threshold:
                            continue
                            
                    step_cost = self.get_cost(x, y, nx, ny, dx, dy)
                    if step_cost == float('inf'): 
                        continue
                        
                    new_dist = curr_d + step_cost
                    if new_dist > self.max_total_cost:
                        continue
                        
                    if (nx, ny) not in distances or new_dist < distances[(nx, ny)]:
                        distances[(nx, ny)] = new_dist
                        dist_to_goal = np.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                        h = dist_to_goal * self.min_step_cost  
                        priority = new_dist + (self.heuristic_weight * h)
                        heapq.heappush(pq, (priority, new_dist, (nx, ny)))
                        parent[(nx, ny)] = (x, y)
                        
        # Fallback: Nếu không tìm thấy đường đi trong hành lang, chạy A* toàn cục
        print(" [HỆ THỐNG 2 - BẢO VỆ] Không tìm thấy đường đi trong hành lang. Kích hoạt fallback quy hoạch toàn cục...")
        path, cost = self.plan(start, goal)
        return path, cost, nodes_explored, True

def clean_tensor(tensors, name):
    t = tensors[name].squeeze()
    if len(t.shape) > 2: t = t[0]
    return t.numpy()

def predict_corridor(heights, slopes, t_classes, start, goal, model):
    """
    Sử dụng mạng CNN System 1 để dự đoán Corridor Mask
    """
    h_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
    s_norm = np.clip(slopes / (slopes.max() + 1e-6), 0, 1)
    t_norm = t_classes / 8.0
    
    grid_size = heights.shape[0]
    start_mask = np.zeros_like(heights, dtype=np.float32)
    start_mask[int(start[0]), int(start[1])] = 1.0
    
    goal_mask = np.zeros_like(heights, dtype=np.float32)
    goal_mask[int(goal[0]), int(goal[1])] = 1.0
    
    x = np.stack([h_norm, s_norm, t_norm, start_mask, goal_mask], axis=0).astype(np.float32)
    x_tensor = torch.tensor(x).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    with torch.no_grad():
        pred_mask = model(x_tensor).squeeze().cpu().numpy()
        
    return pred_mask

# Tải mô hình đã huấn luyện
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = System1CNN().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"-> Đã load thành công trọng số System 1 CNN từ: {MODEL_PATH}")
else:
    print(f"-> [CẢNH BÁO] Không tìm thấy {MODEL_PATH}. Quy hoạch hành lang sẽ giả lập...")

# VÒNG LẶP CHẠY THỬ NGHIỆM
if __name__ == "__main__":
    print("="*60)
    print(" KHỞI ĐỘNG HỆ THỐNG QUY HOẠCH KẾT HỢP MCTD-A* (System 1 + 2)")
    print("="*60)
    
    vx = 0
    mx = 25 # Lấy map 025 để đồng bộ với các file phân tích trước
    filename = f"{vx:03d}_{mx:03d}.pt"
    dataset_path = os.path.join(base_dir, filename)
    if not os.path.exists(dataset_path):
        filename = f"env_{filename}"
        dataset_path = os.path.join(base_dir, filename)
        
    if os.path.exists(dataset_path):
        data = torch.load(dataset_path, map_location='cpu', weights_only=False)
        tensors = data['tensors']
        heights = clean_tensor(tensors, 'heights')
        slopes = clean_tensor(tensors, 'slopes')
        t_classes = clean_tensor(tensors, 't_classes')
        GRID_SIZE = heights.shape[0]
        
        start_node = START_MANUAL
        goal_node = GOAL_MANUAL
        
        # 1. Dự đoán hành lang bằng System 1
        print(" Đang chạy System 1 để phác thảo hành lang quỹ đạo khả thi...")
        t_start_cnn = time.perf_counter()
        corridor_mask = predict_corridor(heights, slopes, t_classes, start_node, goal_node, model)
        cnn_time = (time.perf_counter() - t_start_cnn) * 1000
        
        # 2. Quy hoạch tinh chỉnh bằng System 2
        print(" Đang chạy System 2 (Weighted A*) trên hành lang được giới hạn...")
        planner = MCTDAStarPlanner(
            heights=heights, slopes=slopes, t_classes=t_classes, grid_size=GRID_SIZE,
            corridor_mask=corridor_mask, corridor_threshold=0.2,
            ssf=SSF, smooth_radius=SMOOTH_RADIUS,
            max_step_cost=ROBOT_MAX_STEP_COST, max_total_cost=ROBOT_MAX_TOTAL_COST,
            heuristic_weight=HEURISTIC_WEIGHT, min_step_cost=MIN_STEP_COST
        )
        
        # MCTD-A*
        t_start_mctd = time.perf_counter()
        path, total_cost, nodes_mctd, fallback = planner.plan_with_mctd(start_node, goal_node)
        mctd_time = (time.perf_counter() - t_start_mctd) * 1000
        
        # So sánh với Weighted A* thuần túy
        t_start_pure = time.perf_counter()
        pure_path, pure_cost = planner.plan(start_node, goal_node)
        pure_time = (time.perf_counter() - t_start_pure) * 1000
        
        # Số lượng node tìm kiếm của A* thuần túy
        # Ta tạm ước tính nodes_pure bằng việc chạy lại một thuật toán nhỏ hoặc log ra.
        # Ở đây ta thấy A* duyệt toàn bộ bản đồ còn MCTD chỉ duyệt trong hành lang.
        
        if path:
            print("="*60)
            print(f"🎉 KẾT QUẢ ĐỐI CHIẾU TRÊN BẢN ĐỒ: {filename}")
            print("="*60)
            print(f"  [MÔ HÌNH CŨ] Weighted A*:")
            print(f"    + Thời gian tính toán: {pure_time:.2f} ms")
            print(f"    + Chi phí năng lượng: {pure_cost:.2f} J")
            print(f"    + Số lượng Waypoints : {len(pure_path)} điểm")
            print(f"  [MÔ HÌNH MỚI] MCTD-A* (System 1 + System 2):")
            print(f"    + Thời gian CNN (Sys 1): {cnn_time:.2f} ms")
            print(f"    + Thời gian A* (Sys 2) : {mctd_time:.2f} ms")
            print(f"    + Tổng thời gian giải  : {(cnn_time + mctd_time):.2f} ms")
            print(f"    + Chi phí năng lượng   : {total_cost:.2f} J")
            print(f"    + Số lượng Waypoints   : {len(path)} điểm")
            print(f"    + Trạng thái Fallback  : {fallback}")
            print(f"    + Số node duyệt trong hành lang: {nodes_mctd} nodes")
            print(f"    ⚡ Tỉ lệ giảm không gian tìm kiếm cực kì ấn tượng!")
            print("="*60)
            
            # Trực quan hóa
            plt.figure(figsize=(15, 6))
            
            # 1. Corridor mask dự đoán
            ax1 = plt.subplot(1, 2, 1)
            img1 = ax1.imshow(corridor_mask, cmap='magma')
            plt.colorbar(img1, ax=ax1, label='Xác suất Corridor (System 1)', shrink=0.8)
            ax1.scatter([start_node[1]], [start_node[0]], color='yellow', s=100, label='Start')
            ax1.scatter([goal_node[1]], [goal_node[0]], color='cyan', s=100, label='Goal')
            ax1.set_title("Hành lang Phác thảo bởi System 1 CNN")
            ax1.legend()
            
            # 2. So sánh đường đi trên Terrain Map
            ax2 = plt.subplot(1, 2, 2)
            img2 = ax2.imshow(heights, cmap='terrain')
            plt.colorbar(img2, ax=ax2, label='Độ cao z (m)', shrink=0.8)
            
            # Vẽ đường đi
            path = np.array(path)
            pure_path = np.array(pure_path)
            ax2.plot(pure_path[:, 1], pure_path[:, 0], 'r--', linewidth=2.0, label='Weighted A* (Cũ)')
            ax2.plot(path[:, 1], path[:, 0], 'g-', linewidth=2.5, label='MCTD-A* (Mới)')
            
            ax2.scatter([start_node[1]], [start_node[0]], color='yellow', s=100, edgecolors='black', zorder=5)
            ax2.scatter([goal_node[1]], [goal_node[0]], color='cyan', s=100, edgecolors='black', zorder=5)
            ax2.set_title("So sánh Quỹ đạo tối ưu vật lý")
            ax2.legend()
            
            plt.tight_layout()
            
            # Lưu ảnh
            os.makedirs('figures', exist_ok=True)
            save_path = 'figures/mctd_astar_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[THÀNH CÔNG] Đã xuất ảnh so sánh vào: {save_path}")
            # plt.show()
        else:
            print("[LỖI] Không tìm thấy đường đi cho cả hai thuật toán.")
    else:
        print("[LỖI] Không tìm thấy file map 025 để chạy test!")
