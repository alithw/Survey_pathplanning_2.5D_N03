"""Quick test of snap_to_free fix"""
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Replicate helpers inline to avoid triggering full script
from collections import deque

def snap_to_free(grid, pos, grid_size, max_radius=20):
    r, c = int(pos[0]), int(pos[1])
    if 0 <= r < grid_size and 0 <= c < grid_size and grid[r, c] == 0:
        return (r, c)
    q = deque([(r, c)])
    visited = {(r, c)}
    while q:
        cr, cc = q.popleft()
        if 0 <= cr < grid_size and 0 <= cc < grid_size and grid[cr, cc] == 0:
            return (cr, cc)
        if abs(cr - r) > max_radius and abs(cc - c) > max_radius:
            break
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            nxt = (cr+dr, cc+dc)
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
    return (r, c)

def load_maze(filepath):
    with open(filepath) as f:
        d = json.load(f)
    grid_size = d["size"]
    grid_np   = np.array(d["grid"], dtype=np.uint8)
    start = snap_to_free(grid_np, tuple(d["start"]), grid_size)
    goal  = snap_to_free(grid_np, tuple(d["goal"]),  grid_size)
    return grid_np, start, goal, grid_size

# Simple A* for validation
import heapq

def astar(grid, grid_size, start, goal, max_cost=2000000.0):
    pq = [(0.0, 0.0, start)]
    dist = {start: 0.0}
    parent = {start: None}
    while pq:
        f, g, (x, y) = heapq.heappop(pq)
        if g > dist.get((x, y), float('inf')): continue
        if (x, y) == goal:
            path = []
            cur = (x, y)
            while cur: path.append(cur); cur = parent[cur]
            return path[::-1], g
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx, ny] == 0:
                sc = np.sqrt(dx*dx + dy*dy)
                ng = g + sc
                if ng <= max_cost and ((nx, ny) not in dist or ng < dist[(nx, ny)]):
                    dist[(nx, ny)] = ng
                    h  = np.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                    heapq.heappush(pq, (ng + 3.0*h, ng, (nx, ny)))
                    parent[(nx, ny)] = (x, y)
    return None, float('inf')

DATASET_DIR = "dataset_omb/mazes"
test_files = sorted(os.listdir(DATASET_DIR))
# sample 10 evenly
step = len(test_files) // 10
samples = [test_files[i*step] for i in range(10)]

print(f"{'File':<30} {'N':>5} {'s_snap':>8} {'g_snap':>8} {'found':>6} {'cost':>8}")
print("-"*72)
ok = 0
for fname in samples:
    fpath = os.path.join(DATASET_DIR, fname)
    grid, start, goal, gs = load_maze(fpath)
    raw_d = json.load(open(fpath))
    # Check if snap was needed
    s_raw = tuple(raw_d["start"])
    g_raw = tuple(raw_d["goal"])
    s_was_obs = grid[s_raw[0], s_raw[1]] == 1
    g_was_obs = grid[g_raw[0], g_raw[1]] == 1
    snap_info = ("S" if s_was_obs else ".") + ("G" if g_was_obs else ".")
    path, cost = astar(grid, gs, start, goal)
    found = path is not None
    if found: ok += 1
    print(f"{fname:<30} {gs:>5} {str(start):>8} ... snap={snap_info} {'OK' if found else 'FAIL':>6} {cost:>8.1f}")

print(f"\nSuccess: {ok}/{len(samples)}")
