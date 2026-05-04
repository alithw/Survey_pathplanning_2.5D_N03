import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math

# ================= CẤU HÌNH BIT* =================
BATCH_SIZE = 200
MAX_BATCHES = 5
CONNECTION_RADIUS = 5.0
SSF = 1.2
ROBOT_MAX_STEP_COST = 350.0
START = (33, 8)
GOAL = (40, 58)

base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
filename = "000_025.pt"
soil_res = {3: 8.0, 4: 2.0, 5: 2.0, 7: 2.0}

def get_physics_cost(node_a, node_b, heights, slopes, t_classes):
    dist = math.hypot(node_b[1] - node_a[1], node_b[0] - node_a[0])
    if dist == 0: return 0.0
    nx, ny = int(round(node_b[0])), int(round(node_b[1]))
    x, y = int(round(node_a[0])), int(round(node_a[1]))
    if not (0 <= nx < heights.shape[0] and 0 <= ny < heights.shape[1]): return float('inf')
    
    if slopes[nx, ny] / SSF >= 1.5: return float('inf')
    c_bekker = soil_res.get(int(t_classes[nx, ny]), 2.0)
    
    dz = heights[nx, ny] - heights[x, y]
    s_pitch = dz / dist if dist > 0 else 0
    c_minetti = max(155.4*(s_pitch**5) - 30.4*(s_pitch**4) - 43.3*(s_pitch**3) + 46.3*(s_pitch**2) + 19.5*s_pitch + 3.6, 0.5)
    
    cost = (c_minetti * dist) + (1.5 * c_bekker) + (50 * (slopes[nx, ny]/SSF))
    return cost if cost <= ROBOT_MAX_STEP_COST else float('inf')

def heuristic(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1]) * 2.0 # Uoc luong toi thieu

def run_bit_star():
    print("="*50)
    print(" KHỞI ĐỘNG THUẬT TOÁN BIT* (Batch Informed Trees)")
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
    
    V = set([START])
    E = set()
    g = {START: 0.0, GOAL: float('inf')}
    parent = {START: None}
    
    c_best = float('inf')
    
    for batch_idx in range(MAX_BATCHES):
        print(f"Đang xử lý Batch {batch_idx+1}/{MAX_BATCHES} (Lấy mẫu trong vùng elip)...")
        
        # 1. Lấy mẫu bên trong Elip
        X_samples = set()
        while len(X_samples) < BATCH_SIZE:
            rx = np.random.uniform(0, grid_size - 1)
            ry = np.random.uniform(0, grid_size - 1)
            # Kiểm tra nếu (rx,ry) nằm trong Elip khả thi
            if heuristic(START, (rx,ry)) + heuristic((rx,ry), GOAL) < c_best:
                if slopes[int(rx), int(ry)] / SSF < 1.5:
                    X_samples.add((rx, ry))
                    g[(rx, ry)] = float('inf')
        
        V_unconnected = set(X_samples)
        V_unconnected.add(GOAL)
        
        # 2. Xây dựng cây (Giản lược rẽ nhánh gần nhất)
        for v_new in V_unconnected:
            best_near = None
            min_cost = float('inf')
            
            for v_tree in V:
                if math.hypot(v_tree[0]-v_new[0], v_tree[1]-v_new[1]) <= CONNECTION_RADIUS:
                    edge_cost = get_physics_cost(v_tree, v_new, heights, slopes, t_classes)
                    if edge_cost != float('inf'):
                        if g[v_tree] + edge_cost < min_cost:
                            min_cost = g[v_tree] + edge_cost
                            best_near = v_tree
                            
            if best_near is not None:
                E.add((best_near, v_new))
                V.add(v_new)
                parent[v_new] = best_near
                g[v_new] = min_cost
                
                if v_new == GOAL and g[GOAL] < c_best:
                    c_best = g[GOAL]
                    print(f"  -> Cập nhật c_best mới: {c_best:.2f} J. Thu hẹp không gian tìm kiếm.")

    exec_time = (time.perf_counter() - start_time) * 1000

    if g[GOAL] != float('inf'):
        path = []
        curr = GOAL
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr, None)
        path.reverse()
        
        print("=> TÌM ĐƯỜNG THÀNH CÔNG (BIT*)!")
        print(f"   Thời gian thực thi: {exec_time:.2f} ms")
        print(f"   Tổng năng lượng   : {g[GOAL]:.2f} J")
        
        path = np.array(path)
        plt.figure(figsize=(8, 8))
        plt.imshow(heights, cmap='terrain')
        
        # Vẽ cây (Tree edges)
        for u, v in E:
            plt.plot([u[1], v[1]], [u[0], v[0]], 'w-', alpha=0.3)
            
        plt.plot(path[:, 1], path[:, 0], 'r-', linewidth=3, label='BIT* Path')
        plt.scatter(START[1], START[0], c='y', s=100, edgecolors='k', zorder=5)
        plt.scatter(GOAL[1], GOAL[0], c='b', s=100, edgecolors='w', zorder=5)
        plt.title(f"BIT* Planning\nExecution: {exec_time:.1f}ms | Cost: {g[GOAL]:.1f}J")
        plt.legend()
        plt.show()
    else:
        print("=> THẤT BẠI: BIT* không thể tìm đường nối tới đích trong giới hạn Batch Size. Cần chạy nhiều Batch hơn hoặc tăng bán kính nối, dẫn đến làm chậm hệ thống.")

if __name__ == "__main__":
    run_bit_star()