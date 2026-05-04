import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
import heapq

# ================= CẤU HÌNH FMM =================
SSF = 1.2
ROBOT_MAX_STEP_COST = 350.0
START = (33, 8)
GOAL = (40, 58)

base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
filename = "000_025.pt"
soil_res = {3: 8.0, 4: 2.0, 5: 2.0, 7: 2.0}

def get_physics_cost(x, y, nx, ny, heights, slopes, t_classes):
    dist = math.hypot(nx - x, ny - y)
    if dist == 0: return 0.0
    
    LTR = slopes[nx, ny] / SSF
    if LTR >= 1.5: return float('inf')

    c_class = int(t_classes[nx, ny])
    c_bekker = soil_res.get(c_class, 2.0)

    dz = heights[nx, ny] - heights[x, y]
    s_pitch = dz / dist if dist > 0 else 0
    c_minetti = 155.4*(s_pitch**5) - 30.4*(s_pitch**4) - 43.3*(s_pitch**3) + 46.3*(s_pitch**2) + 19.5*s_pitch + 3.6
    c_minetti = max(c_minetti, 0.5)
    
    cost = (c_minetti * dist) + (1.5 * c_bekker) + (50 * LTR)
    if cost > ROBOT_MAX_STEP_COST: return float('inf')
    return cost

def run_fmm():
    print("="*50)
    print(" KHỞI ĐỘNG THUẬT TOÁN FMM (Fast Marching Method)")
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
    
    # Ma trận Thời gian tới (Arrival Time / Cost)
    T = np.full((grid_size, grid_size), float('inf'))
    T[START] = 0.0
    
    # Hàng đợi (Frontier)
    pq = [(0.0, START)]
    frozen = set()
    parent = {}
    
    moves = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    print("Đang lan truyền sóng Eikonal trên bề mặt 2.5D...")
    goal_reached = False
    
    while pq:
        current_time, curr = heapq.heappop(pq)
        
        if curr in frozen: continue
        frozen.add(curr)
        
        if curr == GOAL:
            goal_reached = True
            break
            
        for dx, dy in moves:
            nx, ny = curr[0] + dx, curr[1] + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if (nx, ny) not in frozen:
                    step_cost = get_physics_cost(curr[0], curr[1], nx, ny, heights, slopes, t_classes)
                    if step_cost != float('inf'):
                        new_time = current_time + step_cost
                        if new_time < T[nx, ny]:
                            T[nx, ny] = new_time
                            parent[(nx, ny)] = curr
                            heapq.heappush(pq, (new_time, (nx, ny)))

    exec_time = (time.perf_counter() - start_time) * 1000

    if goal_reached:
        path = []
        curr = GOAL
        while curr != START:
            path.append(curr)
            curr = parent[curr]
        path.append(START)
        path.reverse()
        
        print("=> TÌM ĐƯỜNG THÀNH CÔNG (FMM)!")
        print(f"   Thời gian thực thi: {exec_time:.2f} ms")
        print(f"   Tổng năng lượng   : {T[GOAL]:.2f} J")
        print(f"   Số lượng Nodes    : {len(path)}")
        
        path = np.array(path)
        plt.figure(figsize=(8, 8))
        
        # Hiển thị bản đồ nhiệt (Wavefront)
        T_viz = np.copy(T)
        T_viz[T_viz == float('inf')] = np.nan
        plt.imshow(T_viz, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Sóng năng lượng tích lũy (Joules)')
        
        plt.plot(path[:, 1], path[:, 0], 'r-', linewidth=3, label='FMM Gradient Path')
        plt.scatter(START[1], START[0], c='y', s=100, edgecolors='k', label='Start')
        plt.scatter(GOAL[1], GOAL[0], c='b', s=100, edgecolors='w', label='Goal')
        plt.title(f"FMM Planning\nExecution: {exec_time:.1f}ms | Cost: {T[GOAL]:.1f}J")
        plt.legend()
        plt.show()
    else:
        print("=> THẤT BẠI: FMM Wavefront không thể chạm tới đích do bị chặn bởi các vùng cấm.")

if __name__ == "__main__":
    run_fmm()