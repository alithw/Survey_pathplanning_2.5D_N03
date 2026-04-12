import torch
import torch.nn.functional as F
import numpy as np
import math
import random

class Node:
    def __init__(self, x, y):
        # Sử dụng tọa độ thực (float) để cây RRT mở rộng liên tục và mượt mà
        self.x = float(x)
        self.y = float(y)
        self.parent = None
        self.cost = 0.0

class RRTStarPlanner:
    def __init__(self, heights, slopes, t_classes, grid_size, ssf=1.2, smooth_radius=2, 
                 max_step_cost=350.0, max_total_cost=10000.0,
                 max_iter=4000, step_size=2.5, search_radius=8.0, goal_sample_rate=0.15):
        """
        Khởi tạo thuật toán RRT* với các thông số vật lý và động lực học.
        """
        self.heights = heights
        self.slopes = slopes
        self.t_classes = t_classes
        self.grid_size = grid_size
        
        self.ssf = ssf
        self.smooth_radius = smooth_radius
        self.max_step_cost = max_step_cost
        self.max_total_cost = max_total_cost
        
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        
        # Hệ số cản đất Bekker-Wong
        self.soil_resistance = {0: 1.0, 1: 1.5, 2: 3.0, 3: 8.0} 
        self.smoothed_bekker = self._prepare_soil_resistance()

    def _prepare_soil_resistance(self):
        base_resistance = np.full_like(self.t_classes, 2.0, dtype=np.float32)
        for k, v in self.soil_resistance.items():
            base_resistance[self.t_classes == k] = v
            
        res_tensor = torch.tensor(base_resistance, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        smoothed_bekker_tensor = F.avg_pool2d(
            res_tensor, kernel_size=(self.smooth_radius*2 + 1), 
            stride=1, padding=self.smooth_radius, count_include_pad=False
        )
        return smoothed_bekker_tensor.squeeze().numpy()

    def get_edge_cost(self, node_a, node_b):
        """
        Tính chi phí vật lý của một nhánh (edge) trong cây RRT.
        Ép kiểu int() an toàn để truy xuất mảng Numpy.
        """
        dist = math.hypot(node_b.x - node_a.x, node_b.y - node_a.y)
        if dist == 0: return 0.0

        nx, ny = int(round(node_b.x)), int(round(node_b.y))
        x, y = int(round(node_a.x)), int(round(node_a.y))
        
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return float('inf')

        # 1. Rủi ro lật (LTR)
        LTR = self.slopes[nx, ny] / self.ssf
        if LTR >= 1.5: 
            return float('inf') 
        
        # 2. Hệ số Bekker
        c_bekker = self.smoothed_bekker[nx, ny]
        
        # 3. Độ dốc (Pitch)
        dz = self.heights[nx, ny] - self.heights[x, y]
        s_pitch = dz / dist if dist > 0 else 0
        
        # 4. Năng lượng (Minetti)
        c_minetti = 155.4*(s_pitch**5) - 30.4*(s_pitch**4) - 43.3*(s_pitch**3) + 46.3*(s_pitch**2) + 19.5*s_pitch + 3.6
        c_minetti = max(c_minetti, 0.5) 
        energy_cost = c_minetti * dist  
        
        step_cost = energy_cost + (1.5 * c_bekker) + (50 * LTR)
        
        if step_cost > self.max_step_cost:
            return float('inf')
            
        return step_cost

    def _sample_free(self, goal):
        if random.random() < self.goal_sample_rate:
            return Node(goal[0], goal[1])
        return Node(random.uniform(0, self.grid_size - 1), random.uniform(0, self.grid_size - 1))

    def _get_nearest(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return node_list[minind]

    def _steer(self, from_node, to_node):
        new_node = Node(from_node.x, from_node.y)
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dist = math.hypot(dx, dy)

        if dist < self.step_size:
            new_node.x, new_node.y = to_node.x, to_node.y
        else:
            new_node.x = from_node.x + self.step_size * (dx / dist)
            new_node.y = from_node.y + self.step_size * (dy / dist)
            
        new_node.x = max(0.0, min(float(self.grid_size - 1), new_node.x))
        new_node.y = max(0.0, min(float(self.grid_size - 1), new_node.y))
        return new_node

    def _get_near_nodes(self, node_list, new_node):
        nnode = len(node_list) + 1
        r = min(self.search_radius * math.sqrt(math.log(nnode) / nnode), self.step_size * 4)
        near_nodes = []
        for i, node in enumerate(node_list):
            if math.hypot(node.x - new_node.x, node.y - new_node.y) <= r:
                near_nodes.append(i)
        return near_nodes

    def plan(self, start_coords, goal_coords):
        start_node = Node(start_coords[0], start_coords[1])
        goal_node = Node(goal_coords[0], goal_coords[1])
        node_list = [start_node]

        for i in range(self.max_iter):
            rnd_node = self._sample_free(goal_coords)
            nearest_node = self._get_nearest(node_list, rnd_node)
            new_node = self._steer(nearest_node, rnd_node)

            edge_cost = self.get_edge_cost(nearest_node, new_node)
            if edge_cost == float('inf'):
                continue

            near_inds = self._get_near_nodes(node_list, new_node)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + edge_cost

            # Chọn parent tốt nhất
            for near_ind in near_inds:
                near_node = node_list[near_ind]
                cost_near_to_new = self.get_edge_cost(near_node, new_node)
                if cost_near_to_new == float('inf'): continue
                
                if near_node.cost + cost_near_to_new < new_node.cost:
                    new_node.parent = near_node
                    new_node.cost = near_node.cost + cost_near_to_new

            node_list.append(new_node)

            # Rewire cây RRT*
            for near_ind in near_inds:
                near_node = node_list[near_ind]
                cost_new_to_near = self.get_edge_cost(new_node, near_node)
                if cost_new_to_near == float('inf'): continue
                
                if new_node.cost + cost_new_to_near < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_node.cost + cost_new_to_near

            # Kiểm tra xem đã tiếp cận đích chưa
            if math.hypot(new_node.x - goal_node.x, new_node.y - goal_node.y) <= self.step_size:
                final_cost = self.get_edge_cost(new_node, goal_node)
                if final_cost != float('inf'):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + final_cost
                    node_list.append(goal_node)
                    return self._generate_path(goal_node), goal_node.cost

        # Trả về None nếu duyệt hết max_iter mà không tìm thấy đường
        return None, float('inf')

    def _generate_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            # Làm tròn về int để trả về grid index vẽ lên plot
            path.append((int(round(node.x)), int(round(node.y))))
            node = node.parent
        return path[::-1]