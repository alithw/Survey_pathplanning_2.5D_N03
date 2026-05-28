#!/usr/bin/env python3
"""
pp_hmctd_a_star.py
===================
Unified implementation of the HMCTD-A* (Hierarchical Monte Carlo Tree Diffusion A*) algorithm.
Fuses Hierarchical Path-finding A* (HPA*) and MCTD-A*.

Features:
1. HPA* Macro-graph decomposition and macro-pathfinding.
2. Local window cropping with Adaptive Sizing based on Terrain Complexity Index (TCI).
3. Local System 1 CNN corridor prediction.
4. Local System 2 corridor-guided Weighted A*.
5. Multi-Level Safety Fallbacks:
   - Level 1: Local window Weighted A* search (corridor bypass).
   - Level 2: Macro-graph edge invalidation and macro re-routing.
"""

import os
import time
import math
import heapq
import numpy as np
import torch
import torch.nn.functional as F
from a_star_cost import AStarPlanner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HMCTDAStarPlanner(AStarPlanner):
    def __init__(self, heights, slopes, t_classes, grid_size, model=None, is_omb=False, **kwargs):
        """
        is_omb: True if evaluating on the Orthogonal Maze Benchmark (flat 2D binary grid).
        model: Pre-trained System 1 CNN model.
        """
        # For OMB, heights will be passed as the binary occupancy grid
        super(HMCTDAStarPlanner, self).__init__(
            heights=heights, slopes=slopes, t_classes=t_classes, grid_size=grid_size, **kwargs
        )
        self.model = model
        self.is_omb = is_omb
        if is_omb:
            self.grid = heights  # 0 = free, 1 = obstacle
            self.min_step_cost = kwargs.get('min_step_cost', 1.0)
            self.max_total_cost = kwargs.get('max_total_cost', 2000000.0)
            self.heuristic_weight = kwargs.get('heuristic_weight', 3.0)

    def get_cost(self, x, y, nx, ny, dx, dy):
        """Redefine step cost for flat 2D maze if is_omb is True."""
        if self.is_omb:
            if self.grid[nx, ny] == 1:
                return float('inf')
            return np.sqrt(dx*dx + dy*dy)
        else:
            return super(HMCTDAStarPlanner, self).get_cost(x, y, nx, ny, dx, dy)

    def check_line_collision(self, u, v):
        """Checks if the straight line between macro nodes u and v collides with obstacles (for OMB)."""
        dist = math.hypot(v[0] - u[0], v[1] - u[1])
        steps = max(int(dist), 5)
        for t in np.linspace(0, 1, steps):
            px = int(round(u[0] + t * (v[0] - u[0])))
            py = int(round(u[1] + t * (v[1] - u[1])))
            if not (0 <= px < self.grid_size and 0 <= py < self.grid_size):
                return True
            if self.is_omb and self.grid[px, py] == 1:
                return True
            elif not self.is_omb:
                # For BenchNav, verify slope safety or max step cost
                if self.slopes[px, py] / self.ssf >= 6.5:
                    return True
        return False

    def get_macro_edge_cost(self, u, v):
        """Estimates the physics cost of a macro-edge between cluster centers u and v."""
        dist = math.hypot(v[0] - u[0], v[1] - u[1])
        if self.is_omb:
            return dist
            
        # Physical estimation for BenchNav
        nx, ny = int(v[0]), int(v[1])
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return float('inf')
            
        c_bekker = self.smoothed_bekker[nx, ny]
        dz = self.heights[nx, ny] - self.heights[int(u[0]), int(u[1])]
        s_pitch = dz / dist if dist > 0 else 0
        c_minetti = max(155.4*(s_pitch**5) - 30.4*(s_pitch**4) - 43.3*(s_pitch**3) + 46.3*(s_pitch**2) + 19.5*s_pitch + 3.6, 0.5)
        
        cost = (c_minetti * dist) + (1.5 * c_bekker) + (50 * (self.slopes[nx, ny]/self.ssf))
        return cost if cost <= self.max_step_cost * dist else float('inf')

    def build_macro_graph(self, start, goal, cluster_size):
        """Builds HPA* macro-graph nodes and edges."""
        clusters = []
        for i in range(0, self.grid_size, cluster_size):
            for j in range(0, self.grid_size, cluster_size):
                cx, cy = i + cluster_size // 2, j + cluster_size // 2
                if cx < self.grid_size and cy < self.grid_size:
                    if self.is_omb:
                        if self.grid[cx, cy] == 0:
                            clusters.append((cx, cy))
                        else:
                            # Snap cluster center to nearest free cell
                            snapped = self._snap_to_free((cx, cy), radius=5)
                            if snapped:
                                clusters.append(snapped)
                    else:
                        if self.slopes[cx, cy] / self.ssf < 6.5:
                            clusters.append((cx, cy))
                            
        nodes = list(set(clusters + [start, goal]))
        graph = {node: {} for node in nodes}
        dist_thresh = cluster_size * 1.5
        
        for u in nodes:
            for v in nodes:
                if u != v and math.hypot(u[0]-v[0], u[1]-v[1]) <= dist_thresh:
                    cost = self.get_macro_edge_cost(u, v)
                    if cost != float('inf'):
                        graph[u][v] = cost

        # Ensure start and goal are not isolated
        for node in [start, goal]:
            if len(graph[node]) == 0:
                other_nodes = [n for n in nodes if n != node]
                if other_nodes:
                    nearest = min(other_nodes, key=lambda n: math.hypot(node[0]-n[0], node[1]-n[1]))
                    cost = self.get_macro_edge_cost(node, nearest)
                    if cost == float('inf'):
                        cost = math.hypot(node[0]-nearest[0], node[1]-nearest[1]) * self.min_step_cost
                    graph[node][nearest] = cost
                    if nearest in graph:
                        graph[nearest][node] = cost

        return graph, nodes

    def _snap_to_free(self, pos, radius=5):
        from collections import deque
        r, c = pos
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size and self.grid[r, c] == 0:
            return (r, c)
        q = deque([(r, c)])
        visited = {(r, c)}
        while q:
            cr, cc = q.popleft()
            if 0 <= cr < self.grid_size and 0 <= cc < self.grid_size and self.grid[cr, cc] == 0:
                return (cr, cc)
            if abs(cr - r) > radius or abs(cc - c) > radius:
                continue
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nxt = (cr+dr, cc+dc)
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return None

    def plan_macro_path(self, start, goal, graph):
        """Weighted A* at high-level to solve the HPA* macro-path."""
        pq = [(self.heuristic_weight * math.hypot(goal[0]-start[0], goal[1]-start[1]), 0.0, start, [start])]
        visited = set()
        
        while pq:
            f, g, curr, path = heapq.heappop(pq)
            if curr == goal:
                return path, g
            if curr in visited:
                continue
            visited.add(curr)
            
            for neighbor, edge_cost in graph[curr].items():
                if neighbor not in visited:
                    h = math.hypot(goal[0]-neighbor[0], goal[1]-neighbor[1])
                    heapq.heappush(pq, (g + edge_cost + self.heuristic_weight * h, g + edge_cost, neighbor, path + [neighbor]))
        return None, float('inf')

    def calculate_tci(self, r0, r1, c0, c1):
        """Computes the Terrain Complexity Index (TCI) for a submap window."""
        if self.is_omb:
            grid_crop = self.grid[r0:r1+1, c0:c1+1]
            density = np.mean(grid_crop)
            return 4.0 * density * (1.0 - density)
        else:
            slopes_crop = self.slopes[r0:r1+1, c0:c1+1]
            return np.std(slopes_crop)

    def predict_local_corridor(self, r0, r1, c0, c1, local_start, local_goal):
        """Runs pre-trained System 1 CNN on cropped local window to get corridor mask."""
        if self.model is None:
            return None
            
        h_crop = self.heights[r0:r1+1, c0:c1+1]
        s_crop = self.slopes[r0:r1+1, c0:c1+1]
        t_crop = self.t_classes[r0:r1+1, c0:c1+1]
        
        if self.is_omb:
            occ = h_crop.astype(np.float32)
            zeros = np.zeros_like(occ)
            start_mask = zeros.copy(); start_mask[int(local_start[0]), int(local_start[1])] = 1.0
            goal_mask = zeros.copy(); goal_mask[int(local_goal[0]), int(local_goal[1])] = 1.0
            x = np.stack([occ, zeros, zeros, start_mask, goal_mask], axis=0)
        else:
            h_norm = (h_crop - h_crop.min()) / (h_crop.max() - h_crop.min() + 1e-6)
            s_norm = np.clip(s_crop / (s_crop.max() + 1e-6), 0, 1)
            t_norm = t_crop / 8.0
            start_mask = np.zeros_like(h_crop, dtype=np.float32)
            start_mask[int(local_start[0]), int(local_start[1])] = 1.0
            goal_mask = np.zeros_like(h_crop, dtype=np.float32)
            goal_mask[int(local_goal[0]), int(local_goal[1])] = 1.0
            x = np.stack([h_norm, s_norm, t_norm, start_mask, goal_mask], axis=0)
            
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        x_small = F.interpolate(x_tensor, size=(64, 64), mode="bilinear", align_corners=False)
        
        with torch.no_grad():
            pred_small = self.model(x_small).squeeze(0).squeeze(0)
            
        H_c, W_c = h_crop.shape
        pred_full = F.interpolate(pred_small.unsqueeze(0).unsqueeze(0), size=(H_c, W_c), mode="bilinear", align_corners=False)
        return pred_full.squeeze().cpu().numpy()

    def local_plan(self, local_start, local_goal, r0, c0, corridor_mask, corridor_threshold=0.2):
        """Runs System 2 local Weighted A* within local window using the corridor mask."""
        pq = [(self.heuristic_weight * math.hypot(local_goal[0]-local_start[0], local_goal[1]-local_start[1]) * self.min_step_cost, 0.0, local_start)]
        dist = {local_start: 0.0}
        parent = {local_start: None}
        nodes_explored = 0
        
        H_l, W_l = corridor_mask.shape
        
        while pq:
            f, g, (x, y) = heapq.heappop(pq)
            nodes_explored += 1
            if g > dist.get((x, y), float('inf')):
                continue
            if (x, y) == local_goal:
                path = []
                curr = (x, y)
                while curr is not None:
                    path.append((curr[0] + r0, curr[1] + c0))
                    curr = parent[curr]
                return path[::-1], g, nodes_explored
                
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H_l and 0 <= ny < W_l:
                    # Corridor restriction (excluding local start and goal)
                    if (nx, ny) != local_goal and (nx, ny) != local_start:
                        if corridor_mask[nx, ny] < corridor_threshold:
                            continue
                            
                    step_cost = self.get_cost(x + r0, y + c0, nx + r0, ny + c0, dx, dy)
                    if step_cost == float('inf'):
                        continue
                        
                    new_dist = g + step_cost
                    if (nx, ny) not in dist or new_dist < dist[(nx, ny)]:
                        dist[(nx, ny)] = new_dist
                        h = math.hypot(local_goal[0]-nx, local_goal[1]-ny) * self.min_step_cost
                        heapq.heappush(pq, (new_dist + self.heuristic_weight * h, new_dist, (nx, ny)))
                        parent[(nx, ny)] = (x, y)
        return None, float('inf'), nodes_explored

    def local_plan_fallback(self, local_start, local_goal, r0, c0):
        """Level 1 Fallback: Local Weighted A* without corridor constraint."""
        pq = [(self.heuristic_weight * math.hypot(local_goal[0]-local_start[0], local_goal[1]-local_start[1]) * self.min_step_cost, 0.0, local_start)]
        dist = {local_start: 0.0}
        parent = {local_start: None}
        nodes_explored = 0
        
        H_l = int(np.abs(local_start[0] - local_goal[0]) + 100) # local search range bounds
        
        while pq:
            f, g, (x, y) = heapq.heappop(pq)
            nodes_explored += 1
            if g > dist.get((x, y), float('inf')):
                continue
            if (x, y) == local_goal:
                path = []
                curr = (x, y)
                while curr is not None:
                    path.append((curr[0] + r0, curr[1] + c0))
                    curr = parent[curr]
                return path[::-1], g, nodes_explored
                
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
                nx, ny = x + dx, y + dy
                # Check if within global grid boundaries
                gx, gy = nx + r0, ny + c0
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    step_cost = self.get_cost(x + r0, y + c0, gx, gy, dx, dy)
                    if step_cost == float('inf'):
                        continue
                    new_dist = g + step_cost
                    if (nx, ny) not in dist or new_dist < dist[(nx, ny)]:
                        dist[(nx, ny)] = new_dist
                        h = math.hypot(local_goal[0]-nx, local_goal[1]-ny) * self.min_step_cost
                        heapq.heappush(pq, (new_dist + self.heuristic_weight * h, new_dist, (nx, ny)))
                        parent[(nx, ny)] = (x, y)
        return None, float('inf'), nodes_explored

    def shortcut_path(self, path):
        """
        Applies a line-of-sight shortcutting post-processing to the global path.
        This removes jagged turns and zig-zags introduced by cluster door snapping,
        significantly reducing path length and energy costs.
        """
        if not path or len(path) < 3:
            return path
            
        smoothed = [path[0]]
        curr_idx = 0
        
        while curr_idx < len(path) - 1:
            found_shortcut = False
            # Search backwards from the end of the path to find the furthest visible node
            for test_idx in range(len(path) - 1, curr_idx + 1, -1):
                u = path[curr_idx]
                v = path[test_idx]
                
                # Check line of sight
                if not self.check_line_collision(u, v):
                    smoothed.append(v)
                    curr_idx = test_idx
                    found_shortcut = True
                    break
            
            if not found_shortcut:
                curr_idx += 1
                smoothed.append(path[curr_idx])
                
        return smoothed

    def plan_with_hmctd(self, start, goal, cluster_size=16, tci_threshold=0.15, corridor_threshold=0.2):
        """
        Executes the full HMCTD-A* algorithm.
        Returns:
            global_path: list of (row, col) coordinates.
            total_cost: total physical step cost of the path.
            nodes_explored: total nodes explored in all local searches.
            fallback_rate: percentage of segments that had to use fallback.
            fallback_triggered: True if any fallback was triggered globally.
        """
        start_time = time.perf_counter()
        
        # 1. High-level HPA* decomposition
        graph, nodes = self.build_macro_graph(start, goal, cluster_size)
        
        fallback_rate = 0.0
        fallback_triggered = False
        total_nodes = 0
        
        # We allow up to 3 macro re-routing attempts (Level 2 Fallback)
        for attempt in range(4):
            macro_path, macro_cost = self.plan_macro_path(start, goal, graph)
            if not macro_path:
                # No path possible even at macro-level
                return None, float('inf'), total_nodes, 0.0, True
                
            global_path = []
            segment_success = True
            total_cost = 0.0
            fallback_count = 0
            
            for i in range(len(macro_path) - 1):
                u, v = macro_path[i], macro_path[i+1]
                
                # Bounding box of segment
                min_r, max_r = min(u[0], v[0]), max(u[0], v[0])
                min_c, max_c = min(u[1], v[1]), max(u[1], v[1])
                
                # 2. Adaptive Window Size based on Terrain Complexity (TCI)
                tci = self.calculate_tci(min_r, max_r, min_c, max_c)
                pad = cluster_size // 4 if tci > tci_threshold else cluster_size // 2
                pad = max(pad, 8)  # Ensure minimum padding
                
                r0 = max(0, min_r - pad)
                r1 = min(self.grid_size - 1, max_r + pad)
                c0 = max(0, min_c - pad)
                c1 = min(self.grid_size - 1, max_c + pad)
                
                local_start = (u[0] - r0, u[1] - c0)
                local_goal = (v[0] - r0, v[1] - c0)
                
                # 3. Predict local corridor
                corridor_mask = self.predict_local_corridor(r0, r1, c0, c1, local_start, local_goal)
                
                # 4. Local corridor-guided search
                seg_path = None
                if corridor_mask is not None:
                    seg_path, seg_cost, seg_nodes = self.local_plan(
                        local_start, local_goal, r0, c0, corridor_mask, corridor_threshold
                    )
                    total_nodes += seg_nodes
                    
                # Level 1 Fallback: Local Window Weighted A*
                if seg_path is None:
                    fallback_count += 1
                    fallback_triggered = True
                    seg_path, seg_cost, seg_nodes = self.local_plan_fallback(local_start, local_goal, r0, c0)
                    total_nodes += seg_nodes
                    
                if seg_path is None:
                    # Level 2 Fallback: Invalidate macro edge and replan macro path
                    graph[u][v] = float('inf')
                    if u in graph[v]:
                        graph[v][u] = float('inf')
                    segment_success = False
                    break
                else:
                    total_cost += seg_cost
                    if len(global_path) > 0:
                        global_path.extend(seg_path[1:])
                    else:
                        global_path.extend(seg_path)
                        
            if segment_success:
                f_rate = (fallback_count / (len(macro_path) - 1)) * 100 if len(macro_path) > 1 else 0.0
                
                # Apply line-of-sight path shortcutting/smoothing
                smoothed_path = self.shortcut_path(global_path)
                
                # Recalculate cost of the smoothed path
                smoothed_cost = 0.0
                for k in range(len(smoothed_path) - 1):
                    pu = smoothed_path[k]
                    pv = smoothed_path[k+1]
                    step_c = self.get_cost(pu[0], pu[1], pv[0], pv[1], pv[0]-pu[0], pv[1]-pu[1])
                    if step_c == float('inf'):
                        smoothed_cost = float('inf')
                        break
                    smoothed_cost += step_c
                
                if smoothed_cost != float('inf') and len(smoothed_path) > 0:
                    return smoothed_path, smoothed_cost, total_nodes, f_rate, fallback_triggered
                else:
                    return global_path, total_cost, total_nodes, f_rate, fallback_triggered
                
        return None, float('inf'), total_nodes, 0.0, True

if __name__ == "__main__":
    print("HMCTD-A* Planner Class defined successfully!")
