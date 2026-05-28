"""
Diagnostic script: kiểm tra tại sao A* thất bại trên OMB dataset
"""
import json, os, heapq
import numpy as np

DATASET_DIR = "dataset_omb/mazes"
files = sorted(os.listdir(DATASET_DIR))

def check_maze(fname):
    fpath = os.path.join(DATASET_DIR, fname)
    with open(fpath) as f:
        d = json.load(f)
    grid_size = d["size"]
    grid      = np.array(d["grid"], dtype=np.uint8)
    start     = tuple(d["start"])
    goal      = tuple(d["goal"])

    sr, sc = start
    gr, gc = goal
    start_val = grid[sr, sc]
    goal_val  = grid[gr, gc]

    # Kiểm tra start/goal có phải obstacle không
    # Đếm free cells xung quanh start
    free_around_start = 0
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            nr, nc = sr+dr, sc+dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if grid[nr, nc] == 0:
                    free_around_start += 1

    # Đếm tổng free cells
    total_free = int(np.sum(grid == 0))
    total_obs  = int(np.sum(grid == 1))

    # Thử BFS đơn giản không giới hạn cost
    from collections import deque
    q = deque([start])
    visited = {start}
    found = False
    steps = 0
    while q and steps < 1000000:
        x, y = q.popleft()
        steps += 1
        if (x, y) == goal:
            found = True
            break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx,ny) not in visited:
                if grid[nx, ny] == 0:
                    visited.add((nx, ny))
                    q.append((nx, ny))

    return {
        "file": fname,
        "grid_size": grid_size,
        "start": start,
        "goal": goal,
        "start_cell_val": int(start_val),  # 0=free, 1=obs
        "goal_cell_val":  int(goal_val),
        "free_cells_around_start_r3": free_around_start,
        "total_free": total_free,
        "total_obstacle": total_obs,
        "bfs_found": found,
        "bfs_steps": steps,
    }

# Kiểm tra 10 file đa dạng
test_files = [files[i * 100] for i in range(10)]
print(f"{'File':<30} {'N':>5} {'s_val':>6} {'g_val':>6} {'free%':>6} {'BFS':>5} {'steps':>7}")
print("-" * 70)
for fname in test_files:
    r = check_maze(fname)
    free_pct = r['total_free'] / (r['grid_size']**2) * 100
    print(f"{r['file']:<30} {r['grid_size']:>5} {r['start_cell_val']:>6} {r['goal_cell_val']:>6} "
          f"{free_pct:>5.1f}% {'OK' if r['bfs_found'] else 'FAIL':>5} {r['bfs_steps']:>7}")
    if not r['bfs_found']:
        print(f"  -> start={r['start']}, goal={r['goal']}, free_near_start={r['free_cells_around_start_r3']}")
