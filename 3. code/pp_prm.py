import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import heapq
from scipy.spatial import KDTree

# ================= CẤU HÌNH =================
NUM_SAMPLES = 800      # Số điểm lấy mẫu ngẫu nhiên
CONNECTION_RADIUS = 5.0 # Bán kính nối các điểm
ROBOT_MAX_STEP_COST = 350.0
SSF = 1.2
START = (33, 8)
GOAL = (40, 58)

base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
filename = "000_025.pt"

# Hệ số Bekker-Wong
soil_res = {3: 8.0, 4: 2.0, 5: 2.0, 7: 2.0}

def get_physics_cost(node_a, node_b, heights, slopes, t_classes):
    """Hàm tính chi phí vật lý (tương tự A* và RRT*)"""
    dist = math.hypot(node_b[1] - node_a[1], node_b[0] - node_a[0])
    if dist == 0: return 0.0

    nx, ny = int(round(node_b[0])), int(round(node_b[1]))
    x, y = int(round(node_a[0])), int(round(node_a[1]))
    
    if not (0 <= nx < heights.shape[0] and 0 <= ny < heights.shape[1]):
        return float('inf')

    # 1. LTR
    LTR = slopes[nx, ny] / SSF
    if LTR >= 1.5: return float('inf')

    # 2. Bekker-Wong
    c_class = int(t_classes[nx, ny])
    c_bekker = soil_res.get(c_class, 2.0)

    # 3. Minetti (Độ dốc)
    dz = heights[nx, ny] - heights[x, y]
    s_pitch = dz / dist if dist > 0 else 0
    c_minetti = 155.4*(s_pitch**5) - 30.4*(s_pitch**4) - 43.3*(s_pitch**3) + 46.3*(s_pitch**2) + 19.5*s_pitch + 3.6
    c_minetti = max(c_minetti, 0.5)
    
    step_cost = (c_minetti * dist) + (1.5 * c_bekker) + (50 * LTR)
    
    if step_cost > ROBOT_MAX_STEP_COST: return float('inf')
    return step_cost

def run_prm():
    print("="*50)
    print(" KHỞI ĐỘNG THUẬT TOÁN PRM (Probabilistic Roadmap)")
    print("="*50)
    
    dataset_path = os.path.join(base_dir, filename)
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, f"env_{filename}")
    
    data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    heights = data['tensors']['heights'].squeeze().numpy()
    slopes = data['tensors']['slopes'].squeeze().numpy()
    t_classes = data['tensors']['t_classes'].squeeze().numpy()
    
    grid_size = heights.shape[0]
    start_time = time.perf_counter()
    
    # 1. LẤY MẪU (Sampling)
    print(f"[1/3] Đang rải {NUM_SAMPLES} điểm lấy mẫu...")
    samples = [START, GOAL]
    while len(samples) < NUM_SAMPLES + 2:
        rx = np.random.uniform(0, grid_size - 1)
        ry = np.random.uniform(0, grid_size - 1)
        # Chỉ lấy mẫu ở nơi an toàn (LTR < 1.5)
        if slopes[int(rx), int(ry)] / SSF < 1.5:
            samples.append((rx, ry))
            
    # 2. XÂY DỰNG ĐỒ THỊ (Roadmap Construction)
    print(f"[2/3] Đang xây dựng đồ thị với bán kính nối {CONNECTION_RADIUS}m...")
    kdtree = KDTree(samples)
    graph = {i: {} for i in range(len(samples))}
    
    for i, node in enumerate(samples):
        # Tìm hàng xóm trong bán kính
        neighbors = kdtree.query_ball_point(node, CONNECTION_RADIUS)
        for j in neighbors:
            if i != j:
                cost = get_physics_cost(node, samples[j], heights, slopes, t_classes)
                if cost != float('inf'):
                    graph[i][j] = cost

    # 3. TÌM ĐƯỜNG (Dijkstra)
    print(f"[3/3] Đang tìm đường đi ngắn nhất trên đồ thị PRM...")
    pq = [(0.0, 0, [0])] # (cost, current_node, path)
    visited = set()
    best_path = None
    best_cost = float('inf')

    while pq:
        cost, curr, path = heapq.heappop(pq)
        
        if curr == 1: # Index của GOAL là 1
            best_path = [samples[i] for i in path]
            best_cost = cost
            break
            
        if curr in visited: continue
        visited.add(curr)
        
        for neighbor, edge_cost in graph[curr].items():
            if neighbor not in visited:
                heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

    exec_time = (time.perf_counter() - start_time) * 1000
    
    if best_path:
        print("=> TÌM ĐƯỜNG THÀNH CÔNG (PRM)!")
        print(f"   Thời gian thực thi: {exec_time:.2f} ms")
        print(f"   Tổng chi phí      : {best_cost:.2f} J")
        print(f"   Số lượng Nodes    : {len(best_path)}")
        
        # Vẽ biểu đồ
        plt.figure(figsize=(8, 8))
        plt.imshow(heights, cmap='terrain')
        
        # Vẽ đồ thị (Roadmap) bằng đường mờ
        for i in graph:
            for j in graph[i]:
                plt.plot([samples[i][1], samples[j][1]], [samples[i][0], samples[j][0]], 'w-', alpha=0.1)
                
        # Vẽ đường đi chính
        best_path = np.array(best_path)
        plt.plot(best_path[:, 1], best_path[:, 0], 'r-', linewidth=3, label='PRM Path')
        plt.scatter(START[1], START[0], c='y', s=100, edgecolors='k', label='Start')
        plt.scatter(GOAL[1], GOAL[0], c='b', s=100, edgecolors='w', label='Goal')
        plt.title(f"PRM Planning\nExecution: {exec_time:.1f}ms | Cost: {best_cost:.1f}J")
        plt.legend()
        plt.show()
    else:
        print("=> THẤT BẠI: PRM không thể tìm được đường. Nguyên nhân: Mật độ lấy mẫu không đủ để lọt qua các khe hẹp (Narrow Corridors) trên bản đồ 2.5D, hoặc khoảng cách nối quá ngắn.")

if __name__ == "__main__":
    run_prm()