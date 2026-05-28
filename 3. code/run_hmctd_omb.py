#!/usr/bin/env python3
"""
run_hmctd_omb.py
=================
Evaluates the proposed HMCTD-A* (Hierarchical MCTD-A*) on the flat 2D maze dataset OMB.
Compares:
1. Global Weighted A*
2. Global MCTD-A*
3. Proposed HMCTD-A*
On the exact same 50 representative maze files.
"""

import os
import sys
import glob
import time
import json
import numpy as np
import pandas as pd
import torch
import random

# Add current folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pp_hmctd_a_star import HMCTDAStarPlanner
from train_system1 import System1CNN
from run_benchmarks_omb import load_maze, build_valid_filelist, OMBAStarPlanner, predict_corridor_omb, path_length, smoothness

# Configurations
DATASET_DIR = "dataset_omb/mazes"
LOG_DIR = "../4. logs"
MODEL_PATH = "system1_model.pth"
NUM_PLANNING = 50
RANDOM_SEED = 42

os.makedirs(LOG_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load System 1 CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = System1CNN().to(device)
if os.path.exists(MODEL_PATH):
    cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    cnn_model.eval()
    print(f"-> Loaded System 1 CNN from: {MODEL_PATH}")
else:
    print(f"[CẢNH BÁO] Không tìm thấy {MODEL_PATH} – dự đoán hành lang sẽ bị mô phỏng.")

def main():
    print("="*80)
    print("   EVALUATING HMCTD-A* ON THE ORTHOGONAL MAZE BENCHMARK (OMB DATASET)")
    print("="*80)
    
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.json")))
    if not all_files:
        print(f"[LỖI] Không tìm thấy file JSON nào trong: {DATASET_DIR}")
        return
        
    print(f"-> Total mazes in dataset: {len(all_files)}")
    valid_files = build_valid_filelist(all_files, verbose=True)
    
    # Choose the same 50 maps
    step_p = max(1, len(valid_files) // NUM_PLANNING)
    plan_fs = [valid_files[i * step_p] for i in range(min(NUM_PLANNING, len(valid_files)))]
    print(f"-> Selected {len(plan_fs)} representative mazes for comparison.")
    
    results = []
    n_maps = len(plan_fs)
    
    # Tuned optimal parameters for OMB (512x512)
    CLUSTER_SIZE = 64
    TCI_THRESHOLD = 0.25
    CORRIDOR_THRESHOLD = 0.2
    
    for idx, fpath in enumerate(plan_fs):
        fname = os.path.basename(fpath)
        print(f"  [{idx+1:02d}/{n_maps}] {fname} ...", end=" ")
        
        try:
            grid, start, goal, grid_size = load_maze(fpath)
        except Exception as e:
            print(f"LỖI: {e}")
            continue
            
        # 1. Global Weighted A*
        planner_pure = OMBAStarPlanner(grid, grid_size)
        t0 = time.perf_counter()
        path_pure, cost_pure, nodes_pure = planner_pure.plan(start, goal)
        t_pure = (time.perf_counter() - t0) * 1000  # ms
        
        if path_pure is None:
            print("Weighted A* failed - skipping.")
            continue
            
        # 2. Global MCTD-A*
        t0 = time.perf_counter()
        corridor = predict_corridor_omb(grid, start, goal, cnn_model)
        t_sys1 = (time.perf_counter() - t0) * 1000
        
        planner_pure.corridor_mask = corridor
        planner_pure.corridor_threshold = CORRIDOR_THRESHOLD
        
        t0 = time.perf_counter()
        path_mctd, cost_mctd, nodes_mctd, fallback_mctd = planner_pure.plan_with_corridor(start, goal)
        t_sys2 = (time.perf_counter() - t0) * 1000
        t_mctd = t_sys1 + t_sys2
        
        # 3. Proposed HMCTD-A* (Hierarchical MCTD-A*)
        slopes_dummy = np.zeros_like(grid, dtype=np.float32)
        t_dummy = np.zeros_like(grid, dtype=np.uint8)
        
        planner_hmctd = HMCTDAStarPlanner(
            heights=grid, slopes=slopes_dummy, t_classes=t_dummy, grid_size=grid_size,
            model=cnn_model, is_omb=True, ssf=1.2, smooth_radius=2, max_step_cost=500.0,
            max_total_cost=2000000.0, heuristic_weight=3.0, min_step_cost=1.0
        )
        
        t0 = time.perf_counter()
        res_hmctd = planner_hmctd.plan_with_hmctd(
            start, goal, cluster_size=CLUSTER_SIZE, tci_threshold=TCI_THRESHOLD, corridor_threshold=CORRIDOR_THRESHOLD
        )
        t_hmctd = (time.perf_counter() - t0) * 1000  # ms
        
        path_hmctd, cost_hmctd, nodes_hmctd, f_rate_hmctd, fallback_trig_hmctd = res_hmctd
        
        if path_hmctd is None:
            print("HMCTD-A* failed - skipping.")
            continue
            
        # Append entry
        entry = {
            "map": fname,
            "grid_size": grid_size,
            
            # Weighted A*
            "a_star_time_ms": round(t_pure, 3),
            "a_star_length_px": round(path_length(path_pure), 2),
            "a_star_smoothness": round(smoothness(path_pure), 4),
            "a_star_nodes": nodes_pure,
            
            # Global MCTD-A*
            "mctd_time_ms": round(t_mctd, 3),
            "mctd_length_px": round(path_length(path_mctd), 2),
            "mctd_smoothness": round(smoothness(path_mctd), 4),
            "mctd_nodes": nodes_mctd,
            "mctd_fallback": fallback_mctd,
            
            # Proposed HMCTD-A*
            "hmctd_time_ms": round(t_hmctd, 3),
            "hmctd_length_px": round(path_length(path_hmctd), 2),
            "hmctd_smoothness": round(smoothness(path_hmctd), 4),
            "hmctd_nodes": nodes_hmctd,
            "hmctd_fallback_rate": round(f_rate_hmctd, 2),
            "hmctd_fallback_triggered": fallback_trig_hmctd
        }
        results.append(entry)
        print(f"A*={t_pure:.1f}ms MCTD-A*={t_mctd:.1f}ms HMCTD-A*={t_hmctd:.1f}ms nodes↓{100*(1-nodes_hmctd/(nodes_pure+1)):.1f}%")

    df = pd.DataFrame(results)
    csv_path = os.path.join(LOG_DIR, "omb_hmctd_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Compute aggregates
    def m(c): return float(df[c].mean()) if len(df) > 0 else 0.0
    def s(c): return float(df[c].std()) if len(df) > 1 else 0.0
    
    summary = {
        "dataset": "OMB (Orthogonal Maze Benchmark)",
        "maps_evaluated": len(df),
        "a_star": {
            "avg_time_ms": m("a_star_time_ms"),
            "std_time_ms": s("a_star_time_ms"),
            "avg_length_px": m("a_star_length_px"),
            "avg_smoothness": m("a_star_smoothness"),
            "avg_nodes": m("a_star_nodes")
        },
        "mctd_a_star": {
            "avg_time_ms": m("mctd_time_ms"),
            "std_time_ms": s("mctd_time_ms"),
            "avg_length_px": m("mctd_length_px"),
            "avg_smoothness": m("mctd_smoothness"),
            "avg_nodes": m("mctd_nodes"),
            "fallback_rate": float(df["mctd_fallback"].mean()*100) if len(df) > 0 else 0.0
        },
        "hmctd_a_star": {
            "avg_time_ms": m("hmctd_time_ms"),
            "std_time_ms": s("hmctd_time_ms"),
            "avg_length_px": m("hmctd_length_px"),
            "avg_smoothness": m("hmctd_smoothness"),
            "avg_nodes": m("hmctd_nodes"),
            "avg_fallback_rate": m("hmctd_fallback_rate"),
            "fallback_rate": float(df["hmctd_fallback_triggered"].mean()*100) if len(df) > 0 else 0.0
        }
    }
    
    summary_path = os.path.join(LOG_DIR, "omb_hmctd_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*80)
    print("   CONSOLIDATED PLANNING BENCHMARK RESULTS (OMB DATASET)")
    print("="*80)
    print(f"  {'Metric':<30} {'Weighted A*':>15} {'MCTD-A*':>15} {'HMCTD-A* (Ours)':>18}")
    print("-"*80)
    print(f"  {'Avg Time (ms)':<30} {summary['a_star']['avg_time_ms']:>15.2f} {summary['mctd_a_star']['avg_time_ms']:>15.2f} {summary['hmctd_a_star']['avg_time_ms']:>18.2f}")
    print(f"  {'Avg Length (px)':<30} {summary['a_star']['avg_length_px']:>15.2f} {summary['mctd_a_star']['avg_length_px']:>15.2f} {summary['hmctd_a_star']['avg_length_px']:>18.2f}")
    print(f"  {'Avg Smoothness':<30} {summary['a_star']['avg_smoothness']:>15.4f} {summary['mctd_a_star']['avg_smoothness']:>15.4f} {summary['hmctd_a_star']['avg_smoothness']:>18.4f}")
    print(f"  {'Avg Explored Nodes':<30} {summary['a_star']['avg_nodes']:>15.0f} {summary['mctd_a_star']['avg_nodes']:>15.0f} {summary['hmctd_a_star']['avg_nodes']:>18.0f}")
    print(f"  {'Fallback Rate (%)':<30} {'—':>15} {summary['mctd_a_star']['fallback_rate']:>14.1f}% {summary['hmctd_a_star']['fallback_rate']:>17.1f}%")
    
    # Improvements
    t_a = summary['a_star']['avg_time_ms']
    t_h = summary['hmctd_a_star']['avg_time_ms']
    t_m = summary['mctd_a_star']['avg_time_ms']
    time_vs_astar = (t_a - t_h) / t_a * 100
    time_vs_mctd = (t_m - t_h) / t_m * 100
    print("-"*80)
    print(f"  ⚡ HMCTD-A* Planning Speedup vs. Weighted A*:  {time_vs_astar:.2f}%")
    print(f"  ⚡ HMCTD-A* Planning Speedup vs. MCTD-A*:      {time_vs_mctd:.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
