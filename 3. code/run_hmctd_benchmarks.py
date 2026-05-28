#!/usr/bin/env python3
"""
run_hmctd_benchmarks.py
========================
Parameter sweep and tuning script for HMCTD-A* on the BenchNav dataset.
Finds the optimal configuration by sweeping cluster size, TCI threshold, and corridor threshold.
Logs all trials to ../4. logs/hmctd_trials.json.
"""

import os
import glob
import time
import json
import torch
import numpy as np
import pandas as pd
from pp_hmctd_a_star import HMCTDAStarPlanner
from train_system1 import System1CNN

# Configuration
GT_DIR = "gt_dataset"
LOG_DIR = "../4. logs"
MODEL_PATH = "system1_model.pth"
NUM_TUNING_MAPS = 30
os.makedirs(LOG_DIR, exist_ok=True)

def calculate_smoothness(path):
    if len(path) < 3:
        return 0.0
    dx = np.diff(path[:, 1])
    dy = np.diff(path[:, 0])
    angles = np.arctan2(dy, dx)
    angle_diffs = np.diff(angles)
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(angle_diffs ** 2))

def path_length(path):
    if len(path) < 2:
        return 0.0
    arr = np.array(path)
    return float(np.sum(np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))))

def main():
    print("="*75)
    print("   HMCTD-A* PARAMETER SWEEP & OPTIMIZATION TUNING (BenchNav)")
    print("="*75)
    
    # Load CNN Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = System1CNN().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"-> Loaded System 1 CNN weights from: {MODEL_PATH}")
    else:
        print("[LỖI] Không tìm thấy system1_model.pth! Không thể thực hiện System 1 CNN corridor prediction.")
        return

    # Get tuning maps
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.npz")))
    if not gt_files:
        print("[LỖI] Không tìm thấy tệp dữ liệu Ground Truth nào trong gt_dataset!")
        return
    tuning_files = gt_files[:NUM_TUNING_MAPS]
    print(f"-> Loaded {len(tuning_files)} maps for parameter tuning sweep.")

    # Parameter ranges
    sweep_cluster_sizes = [16, 32]
    sweep_tci_thresholds = [0.10, 0.15, 0.20]
    sweep_corridor_thresholds = [0.15, 0.25]
    
    trials = []
    best_score = float('inf')
    best_params = None
    
    trial_count = 1
    total_trials = len(sweep_cluster_sizes) * len(sweep_tci_thresholds) * len(sweep_corridor_thresholds)
    
    for c_size in sweep_cluster_sizes:
        for tci_thresh in sweep_tci_thresholds:
            for corr_thresh in sweep_corridor_thresholds:
                print(f"[{trial_count}/{total_trials}] Sweeping Params: Cluster={c_size}, TCI_thresh={tci_thresh:.2f}, Corr_thresh={corr_thresh:.2f}")
                
                successes = 0
                planning_times = []
                explored_nodes_list = []
                fallback_rates = []
                path_lengths = []
                energy_costs = []
                smoothness_list = []
                
                for filepath in tuning_files:
                    data = np.load(filepath)
                    heights = data['heights']
                    slopes = data['slopes']
                    t_classes = data['t_classes']
                    start = tuple(data['start'])
                    goal = tuple(data['goal'])
                    GRID_SIZE = heights.shape[0]
                    
                    planner = HMCTDAStarPlanner(
                        heights=heights, slopes=slopes, t_classes=t_classes, grid_size=GRID_SIZE,
                        model=model, is_omb=False, ssf=1.2, smooth_radius=2, max_step_cost=350.0,
                        max_total_cost=10000.0, heuristic_weight=3.0, min_step_cost=2.0
                    )
                    
                    t0 = time.perf_counter()
                    res = planner.plan_with_hmctd(
                        start, goal, cluster_size=c_size, tci_threshold=tci_thresh, corridor_threshold=corr_thresh
                    )
                    t_plan = (time.perf_counter() - t0) * 1000  # ms
                    
                    path, cost, nodes, f_rate, fallback_trig = res
                    
                    if path is not None and len(path) > 0:
                        successes += 1
                        planning_times.append(t_plan)
                        explored_nodes_list.append(nodes)
                        fallback_rates.append(f_rate)
                        
                        path_arr = np.array(path)
                        path_lengths.append(path_length(path_arr))
                        energy_costs.append(float(cost))
                        smoothness_list.append(calculate_smoothness(path_arr))
                        
                # Compute trial statistics
                success_rate = (successes / NUM_TUNING_MAPS) * 100.0
                avg_time = np.mean(planning_times) if successes > 0 else float('inf')
                avg_nodes = np.mean(explored_nodes_list) if successes > 0 else float('inf')
                avg_fallback = np.mean(fallback_rates) if successes > 0 else 0.0
                avg_len = np.mean(path_lengths) if successes > 0 else 0.0
                avg_energy = np.mean(energy_costs) if successes > 0 else 0.0
                avg_smooth = np.mean(smoothness_list) if successes > 0 else 0.0
                
                # Multi-objective optimization score (planning_time + explored_nodes/1000)
                # Goal is to minimize this score while maintaining 100% success rate
                score = avg_time + (avg_nodes / 50.0) if success_rate == 100.0 else float('inf')
                
                trial_data = {
                    "trial_id": trial_count,
                    "cluster_size": c_size,
                    "tci_threshold": tci_thresh,
                    "corridor_threshold": corr_thresh,
                    "success_rate": success_rate,
                    "avg_time_ms": avg_time,
                    "avg_explored_nodes": avg_nodes,
                    "avg_fallback_rate": avg_fallback,
                    "avg_path_length_m": avg_len,
                    "avg_energy_j": avg_energy,
                    "avg_smoothness": avg_smooth,
                    "optimization_score": score
                }
                trials.append(trial_data)
                
                print(f"   -> Success={success_rate:.1f}%, Time={avg_time:.2f} ms, Explored={avg_nodes:.1f} nodes, Fallback={avg_fallback:.1f}%")
                
                if score < best_score:
                    best_score = score
                    best_params = trial_data
                    
                trial_count += 1
                
    print("\n" + "="*75)
    print("   OPTIMAL PARAMETERS FOUND")
    print("="*75)
    print(json.dumps(best_params, indent=2))
    print("="*75)
    
    # Save trial results
    json_path = os.path.join(LOG_DIR, "hmctd_trials.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "trials": trials,
            "best_params": best_params
        }, f, ensure_ascii=False, indent=2)
    print(f"-> Saved trial results and parameter sweeps to: {json_path}")

if __name__ == "__main__":
    main()
