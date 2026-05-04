# File: pp_aco.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math

# ================= CẤU HÌNH ACO =================
NUM_ANTS = 50        # Số lượng kiến mỗi vòng
NUM_ITERATIONS = 10  # Số vòng lặp
ALPHA = 1.0          # Trọng số Pheromone
BETA = 2.0           # Trọng số Heuristic (Chi phí vật lý)
EVAPORATION = 0.1    # Tỷ lệ bay hơi
Q = 1000             # Hệ số thưởng pheromone

START = (33, 8)
GOAL = (40, 58)
SSF = 1.2
ROBOT_MAX_STEP_COST = 350.0

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

def run_aco():
    print("="*50)
    print(" KHỞI ĐỘNG THUẬT TOÁN ACO (Ant Colony Optimization)")
    print("="*50)
    
    dataset_path = os.path.join(base_dir, filename)
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, f"env_{filename}")
    
    data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    heights = data['tensors']['heights'].squeeze().numpy()
    slopes = data['tensors']['slopes'].squeeze().numpy()
    t_classes = data['tensors']['t_classes'].squeeze().numpy()
    
    grid_size = heights.shape[0]
    
    # Ma trận Pheromone
    pheromone = np.ones((grid_size, grid_size)) * 0.1
    
    best_global_path = None
    best_global_cost = float('inf')
    
    start_time = time.perf_counter()
    
    # Các hướng di chuyển (8 hướng)
    moves = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    for it in range(NUM_ITERATIONS):
        print(f"Đang chạy Vòng lặp {it+1}/{NUM_ITERATIONS}...")
        all_paths = []
        all_costs = []
        
        for ant in range(NUM_ANTS):
            curr = START
            path = [curr]
            cost = 0.0
            visited = set([curr])
            reached_goal = False
            
            # Kiến bò tối đa 200 bước (tránh bị kẹt vòng lặp vô hạn)
            for _ in range(200):
                if curr == GOAL:
                    reached_goal = True
                    break
                    
                probs = []
                valid_moves = []
                
                for dx, dy in moves:
                    nx, ny = curr[0] + dx, curr[1] + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in visited:
                        step_cost = get_physics_cost(curr[0], curr[1], nx, ny, heights, slopes, t_classes)
                        if step_cost != float('inf'):
                            # Cục bộ hóa heuristic (ngắn nhất về đích + tốn ít năng lượng)
                            dist_to_goal = math.hypot(GOAL[0]-nx, GOAL[1]-ny)
                            eta = 1.0 / (step_cost + dist_to_goal) 
                            tau = pheromone[nx, ny]
                            probs.append((tau**ALPHA) * (eta**BETA))
                            valid_moves.append((nx, ny, step_cost))
                            
                if not valid_moves:
                    break # Kẹt cứng (Ngõ cụt)
                    
                # Quay Roulette chọn đường
                probs = np.array(probs)
                probs = probs / np.sum(probs)
                chosen_idx = np.random.choice(len(valid_moves), p=probs)
                
                next_node = valid_moves[chosen_idx]
                curr = (next_node[0], next_node[1])
                path.append(curr)
                cost += next_node[2]
                visited.add(curr)
                
            if reached_goal:
                all_paths.append(path)
                all_costs.append(cost)
                if cost < best_global_cost:
                    best_global_cost = cost
                    best_global_path = path
                    
        # Cập nhật Pheromone (Bay hơi + Thưởng)
        pheromone *= (1 - EVAPORATION)
        for p, c in zip(all_paths, all_costs):
            reward = Q / c
            for node in p:
                pheromone[node[0], node[1]] += reward

    exec_time = (time.perf_counter() - start_time) * 1000

    if best_global_path:
        print("=> TÌM ĐƯỜNG THÀNH CÔNG (ACO)!")
        print(f"   Thời gian thực thi: {exec_time:.2f} ms")
        print(f"   Tổng chi phí      : {best_global_cost:.2f} J")
        print(f"   Số lượng Nodes    : {len(best_global_path)}")
        
        best_global_path = np.array(best_global_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(heights, cmap='terrain')
        plt.plot(best_global_path[:, 1], best_global_path[:, 0], 'r-', linewidth=3, label='ACO Path')
        plt.scatter(START[1], START[0], c='y', s=100, edgecolors='k', label='Start')
        plt.scatter(GOAL[1], GOAL[0], c='b', s=100, edgecolors='w', label='Goal')
        plt.title(f"ACO Planning\nExecution: {exec_time:.1f}ms | Cost: {best_global_cost:.1f}J")
        plt.legend()
        plt.show()
    else:
        print(f"=> THẤT BẠI: Toàn bộ {NUM_ANTS} kiến đều bị chết đuối ở các vũng lầy hoặc kẹt ở rào cản LTR. Thuật toán ACO gốc không đủ sức leo dốc 2.5D nếu không có Heuristic rất mạnh dẫn đường.")

if __name__ == "__main__":
    run_aco()