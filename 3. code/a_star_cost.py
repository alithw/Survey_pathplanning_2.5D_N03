import torch
import torch.nn.functional as F
import numpy as np
import heapq

class AStarPlanner:
    def __init__(self, heights, slopes, t_classes, grid_size, ssf=1.2, smooth_radius=2, 
                 max_step_cost=350.0, max_total_cost=10000.0, 
                 heuristic_weight=3.0, min_step_cost=2.0):
        """
        Khởi tạo hệ thống Planner với các cấu hình động lực học và thuật toán.
        """
        self.heights = heights
        self.slopes = slopes
        self.t_classes = t_classes
        self.grid_size = grid_size
        
        self.ssf = ssf
        self.smooth_radius = smooth_radius
        self.max_step_cost = max_step_cost
        self.max_total_cost = max_total_cost
        self.heuristic_weight = heuristic_weight
        self.min_step_cost = min_step_cost
        
        # Bảng hệ số cản của đất
        self.soil_resistance = {0: 1.0, 1: 1.5, 2: 3.0, 3: 8.0} 
        
        # Chuẩn bị bản đồ độ cản của đất đã được làm mượt
        self.smoothed_bekker = self._prepare_soil_resistance()

    def _prepare_soil_resistance(self):
        """
        Tiền xử lý: Làm mượt cản trở đất (Bán kính cấu hình)
        """
        base_resistance = np.full_like(self.t_classes, 2.0, dtype=np.float32)
        for k, v in self.soil_resistance.items():
            base_resistance[self.t_classes == k] = v
            
        res_tensor = torch.tensor(base_resistance, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        smoothed_bekker_tensor = F.avg_pool2d(
            res_tensor, kernel_size=(self.smooth_radius*2 + 1), 
            stride=1, padding=self.smooth_radius, count_include_pad=False
        )
        return smoothed_bekker_tensor.squeeze().numpy()

    def get_cost(self, x, y, nx, ny, dx, dy):
        """
        Tính chi phí cho 1 bước di chuyển của Robot bao gồm rủi ro lật, cản đất, và năng lượng sườn dốc.
        """
        dist_1step = np.sqrt((nx-x)**2 + (ny-y)**2)
        
        # 1. RỦI RO LẬT XE (LTR)
        LTR = self.slopes[nx, ny] / self.ssf
        penalty_LTR = 5.0 * LTR
        if LTR >= 1.5: 
            penalty_LTR += 5000.0  
        
        # 2. CHỈ SỐ CẢN CỦA ĐẤT
        c_bekker = self.smoothed_bekker[nx, ny]
        
        # 3. ĐỘ DỐC TẦM XA (SMOOTH PITCH)
        fx = max(0, min(self.grid_size-1, nx + self.smooth_radius * dx))
        fy = max(0, min(self.grid_size-1, ny + self.smooth_radius * dy))
        bx = max(0, min(self.grid_size-1, x - self.smooth_radius * dx))
        by = max(0, min(self.grid_size-1, y - self.smooth_radius * dy))
        
        dz = self.heights[fx, fy] - self.heights[bx, by]
        real_dist = np.sqrt((fx - bx)**2 + (fy - by)**2)
        s_pitch = dz / real_dist if real_dist > 0 else 0
        
        # 4. NĂNG LƯỢNG TIÊU HAO (Mô hình Minetti)
        c_minetti = 155.4*(s_pitch**5) - 30.4*(s_pitch**4) - 43.3*(s_pitch**3) + 46.3*(s_pitch**2) + 19.5*s_pitch + 3.6
        c_minetti = max(c_minetti, 0.5) 
        energy_cost = c_minetti * dist_1step  
        
        step_cost = (energy_cost) + (1.5 * c_bekker) + (50*LTR)
        
        if step_cost > self.max_step_cost:
            return float('inf')
            
        return step_cost

    def plan(self, start, goal):
        """
        Thuật toán Weighted A* Pathfinding
        """
        # Khởi tạo heuristic ban đầu
        start_h = np.sqrt((start[0]-goal[0])**2 + (start[1]-goal[1])**2) * self.min_step_cost
        
        # Hàng đợi ưu tiên lưu: (priority_f_score, g_score, (x, y))
        pq = [(self.heuristic_weight * start_h, 0, start)]
        distances = {start: 0}
        parent = {start: None}
        
        while pq:
            # Pop node có priority thấp nhất
            f_score, curr_d, (x, y) = heapq.heappop(pq)
            
            # Xử lý các node lặp lại trong quá trình push vào heapq
            if curr_d > distances.get((x, y), float('inf')):
                continue
            
            # Nếu đến đích -> Truy vết đường đi
            if (x, y) == goal:
                path = []
                curr = (x, y)
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1], curr_d
                
            # Duyệt 8 hướng xung quanh
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (1,1), (-1,-1), (1,-1), (-1,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    step_cost = self.get_cost(x, y, nx, ny, dx, dy)
                    
                    if step_cost == float('inf'): 
                        continue
                    
                    new_dist = curr_d + step_cost
                    
                    if new_dist > self.max_total_cost:
                        continue
                        
                    # Nếu tìm thấy đường đi ngắn hơn đến node (nx, ny)
                    if (nx, ny) not in distances or new_dist < distances[(nx, ny)]:
                        distances[(nx, ny)] = new_dist
                        
                        # Tính Weighted Heuristic
                        dist_to_goal = np.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                        h = dist_to_goal * self.min_step_cost  
                        priority = new_dist + (self.heuristic_weight * h)
                        
                        heapq.heappush(pq, (priority, new_dist, (nx, ny)))
                        parent[(nx, ny)] = (x, y)
                        
        return None, float('inf')
