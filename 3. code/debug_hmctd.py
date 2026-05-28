#!/usr/bin/env python3
import os
import glob
import numpy as np
import torch
from pp_hmctd_a_star import HMCTDAStarPlanner
from train_system1 import System1CNN

GT_DIR = "gt_dataset"
MODEL_PATH = "system1_model.pth"

def main():
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.npz")))
    if not gt_files:
        print("No files found!")
        return
    filepath = gt_files[0]
    print(f"Debugging HMCTD-A* on map: {os.path.basename(filepath)}")
    
    data = np.load(filepath)
    heights = data['heights']
    slopes = data['slopes']
    t_classes = data['t_classes']
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    GRID_SIZE = heights.shape[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = System1CNN().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    
    planner = HMCTDAStarPlanner(
        heights=heights, slopes=slopes, t_classes=t_classes, grid_size=GRID_SIZE,
        model=model, is_omb=False, ssf=1.2, smooth_radius=2, max_step_cost=350.0,
        max_total_cost=10000.0, heuristic_weight=3.0, min_step_cost=2.0
    )
    
    print(f"Start: {start}, Goal: {goal}")
    print(f"Grid size: {GRID_SIZE}")
    print(f"Heights shape: {heights.shape}, Slopes shape: {slopes.shape}")
    print(f"Planner ssf: {planner.ssf}")
    
    # Test building macro graph manually with prints
    clusters = []
    for i in range(0, GRID_SIZE, 16):
        for j in range(0, GRID_SIZE, 16):
            cx, cy = i + 8, j + 8
            if cx < GRID_SIZE and cy < GRID_SIZE:
                val = slopes[cx, cy] / planner.ssf
                print(f"cx={cx}, cy={cy}, slope={slopes[cx, cy]:.4f}, val={val:.4f}, ok={val < 1.5}")
                if val < 1.5:
                    clusters.append((cx, cy))
    
    print("Building macro graph...")
    graph, nodes = planner.build_macro_graph(start, goal, cluster_size=16)
    print(f"Total macro nodes: {len(nodes)}")
    
    # Check start and goal neighbors
    print(f"Start neighbors: {graph[start]}")
    print(f"Goal neighbors: {graph[goal]}")
    
    macro_path, macro_cost = planner.plan_macro_path(start, goal, graph)
    print(f"Macro path: {macro_path}, cost: {macro_cost}")
    
    if macro_path:
        print("Planning local segments...")
        res = planner.plan_with_hmctd(start, goal, cluster_size=16, tci_threshold=0.15, corridor_threshold=0.2)
        print(f"HMCTD Path found? {res[0] is not None}, total cost: {res[1]}, explored: {res[2]}, fallback: {res[3]}%")

if __name__ == "__main__":
    main()
