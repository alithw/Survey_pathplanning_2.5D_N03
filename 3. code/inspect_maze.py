import json
import os

maze_dir = "dataset_omb/mazes"
files = sorted(os.listdir(maze_dir))
print(f"Total maze files: {len(files)}")
print(f"Sample names: {files[:5]}")

# Inspect one small maze and one large maze
for fname in [files[0], files[len(files)//2], files[-1]]:
    fpath = os.path.join(maze_dir, fname)
    with open(fpath) as f:
        d = json.load(f)
    g = d['grid']
    print(f"\n--- {fname} ---")
    print(f"  Keys: {list(d.keys())}")
    print(f"  start: {d.get('start','NA')}")
    print(f"  goal: {d.get('goal','NA')}")
    print(f"  grid rows: {len(g)}, cols: {len(g[0])}")
    print(f"  grid_size key: {'grid_size' in d}, size key: {'size' in d}")
    if 'grid_size' in d:
        print(f"  grid_size value: {d['grid_size']}")
    if 'size' in d:
        print(f"  size value: {d['size']}")
