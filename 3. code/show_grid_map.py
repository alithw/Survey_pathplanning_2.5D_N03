import torch
import numpy as np
import matplotlib.pyplot as plt
import os

dataset_path = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/001_021.pt"

if not os.path.exists(dataset_path):
    print("Không tìm thấy file.")
else:
    data = torch.load(dataset_path, map_location='cpu')
    
    tensors = data['tensors']

    # Lấy bản đồ độ cao 
    if 'heights' in tensors:
        raw_map = tensors['heights']
        
        if len(raw_map.shape) == 4:
            elevation_map = raw_map[0, 0].numpy()
        else:
            elevation_map = raw_map.numpy()

        print(f"Kích thước bản đồ: {elevation_map.shape}")
        
        # Chạy thử
        plt.figure(figsize=(10, 8))
        plt.imshow(elevation_map, cmap='terrain')
        plt.colorbar(label='Độ cao (z)')
        plt.title('2.5D Elevation Map từ Dataset')
        plt.show()
    else:
        print("Không thấy map.")