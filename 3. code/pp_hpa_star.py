import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import heapq

# ================= CẤU HÌNH HPA* =================
CLUSTER_SIZE = 16    # Kích thước mỗi cụm
SSF = 1.2
ROBOT_MAX_STEP_COST = 350.0
START = (33, 8)
GOAL = (40, 58)

base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
filename = "000_025.pt"
soil_res = {3: 8.0, 4: 2.0, 5: 2.0, 7: 2.0}

def get_physics_cost(a, b, heights, slopes, t_classes):
    dist = math.hypot(b[1] - a[1], b[0] - a[0])
    if dist == 0: return 0.0
    nx, ny = int(b[0]), int(b[1])
    if not (0 <= nx < heights.shape[0] and 0 <= ny < heights.shape[1]): return float('inf')
    
    if slopes[nx, ny] / SSF >= 1.5: return float('inf')
    c_bekker = soil_res.get(int(t_classes[nx, ny]), 2.0)
    dz = heights[nx, ny] - heights[int(a[0]), int(a[1])]
    s_pitch = dz / dist if dist > 0 else 0
    c_minetti = max(155.4*(s_pitch**5) - 30.4*(s_pitch**4) - 43.3*(s_pitch**3) + 46.3*(s_pitch**2) + 19.5*s_pitch + 3.6, 0.5)
    
    cost = (c_minetti * dist) + (1.5 * c_bekker) + (50 * (slopes[nx, ny]/SSF))
    return cost if cost <= ROBOT_MAX_STEP_COST else float('inf')

def run_hpa_star():
    print("="*50)
    print(" KHỞI ĐỘNG THUẬT TOÁN HPA* (Hierarchical A*)")
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
    
    # 1. Trừu tượng hóa (Abstraction): Tạo các tâm cụm (Cluster Centers)
    print("Đang phân rã bản đồ thành các cụm (Clusters)...")
    clusters = []
    for i in range(0, grid_size, CLUSTER_SIZE):
        for j in range(0, grid_size, CLUSTER_SIZE):
            cx, cy = i + CLUSTER_SIZE//2, j + CLUSTER_SIZE//2
            if cx < grid_size and cy < grid_size:
                if slopes[cx, cy] / SSF < 1.5:
                    clusters.append((cx, cy))
                    
    nodes = clusters + [START, GOAL]
    
    # 2. Xây dựng đồ thị vĩ mô (Macro Graph)
    print("Đang xây dựng đồ thị liên kết vĩ mô...")
    graph = {node: {} for node in nodes}
    for u in nodes:
        for v in nodes:
            if u != v and math.hypot(u[0]-v[0], u[1]-v[1]) <= CLUSTER_SIZE * 1.5:
                cost = get_physics_cost(u, v, heights, slopes, t_classes)
                if cost != float('inf'):
                    graph[u][v] = cost

    # 3. Tìm đường cấp cao (High-level A*)
    print("Đang tìm đường trên không gian cụm...")
    pq = [(0.0, START, [START])]
    visited = set()
    macro_path = None
    macro_cost = float('inf')
    
    while pq:
        cost, curr, path = heapq.heappop(pq)
        if curr == GOAL:
            macro_path = path
            macro_cost = cost
            break
        if curr in visited: continue
        visited.add(curr)
        for neighbor, edge_cost in graph[curr].items():
            if neighbor not in visited:
                heuristic = math.hypot(GOAL[0]-neighbor[0], GOAL[1]-neighbor[1])
                heapq.heappush(pq, (cost + edge_cost + heuristic, neighbor, path + [neighbor]))

    exec_time = (time.perf_counter() - start_time) * 1000

    if macro_path:
        print("=> TÌM ĐƯỜNG THÀNH CÔNG (HPA*)!")
        print(f"   Thời gian thực thi: {exec_time:.2f} ms")
        
        macro_path = np.array(macro_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(heights, cmap='terrain')
        
        # Vẽ các vùng Cluster
        for i in range(0, grid_size, CLUSTER_SIZE):
            plt.axhline(i, color='w', linestyle=':', alpha=0.5)
            plt.axvline(i, color='w', linestyle=':', alpha=0.5)
            
        plt.plot(macro_path[:, 1], macro_path[:, 0], 'r-o', linewidth=3, markersize=8, label='HPA* Macro Path')
        plt.scatter(START[1], START[0], c='y', s=150, edgecolors='k', zorder=5)
        plt.scatter(GOAL[1], GOAL[0], c='b', s=150, edgecolors='w', zorder=5)
        plt.title(f"HPA* Planning\nExecution: {exec_time:.1f}ms | Clusters: {CLUSTER_SIZE}x{CLUSTER_SIZE}")
        plt.legend()
        plt.show()
    else:
        print("=> THẤT BẠI: HPA* không thể nối các cụm với nhau do địa hình bị chia cắt mạnh bởi các vách đá (Rào cản LTR).")

if __name__ == "__main__":
    run_hpa_star()