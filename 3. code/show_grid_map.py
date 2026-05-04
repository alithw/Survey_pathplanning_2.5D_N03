import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# CẤU HÌNH ĐƯỜNG DẪN DATASET
base_dir = "/home/ali/ros2_ws/src/benchnav/datasets/dataset01/train/"
filename = "000_025.pt" # Map mục tiêu 
dataset_path = os.path.join(base_dir, filename)

def clean_tensor(tensors, name):
    """Hàm làm sạch và ép kiểu tensor về mảng Numpy 2D"""
    t = tensors[name].squeeze()
    if len(t.shape) > 2: 
        t = t[0]
    return t.numpy()

def analyze_and_show_map():
    # 1. TÌM VÀ LOAD FILE DỮ LIỆU
    path_to_load = dataset_path
    if not os.path.exists(path_to_load):
        path_to_load = os.path.join(base_dir, f"env_{filename}")
        if not os.path.exists(path_to_load):
            print(f"[LỖI] Không tìm thấy file map: {filename} tại {base_dir}")
            return

    print("="*65)
    print(f" ĐANG ĐỌC VÀ BÓC TÁCH DỮ LIỆU MAP: {filename}")
    print("="*65)

    data = torch.load(path_to_load, map_location='cpu', weights_only=False)
    tensors = data['tensors']

    # 2. BÓC TÁCH CÁC LỚP TENSOR (LAYERS)
    heights = clean_tensor(tensors, 'heights')
    slopes = clean_tensor(tensors, 'slopes')
    t_classes = clean_tensor(tensors, 't_classes')

    grid_size = heights.shape
    
    soil_resistance = {
        3: 8.0,  
        4: 2.0,  
        5: 2.0,  
        7: 2.0   
    }
    
    soil_desc = {
        3: "Đất cỏ / Nền mấp mô",
        4: "Đất sét / Đường mòn",
        5: "Cát xốp / Sỏi dăm",
        7: "Bùn lầy / Đất lún sâu"
    }
    
    # 3. PHÂN TÍCH VÀ IN THỐNG KÊ RA TERMINAL ĐỂ ĐƯA VÀO REPORT
    print("\n[THÔNG SỐ CẤU TRÚC LƯỚI - GRID PARAMETERS]")
    print(f" - Kích thước lưới (Grid Size) : {grid_size[0]} x {grid_size[1]} cells")
    print(f" - Tổng số điểm dữ liệu        : {grid_size[0] * grid_size[1]} points")
    
    print("\n[CHI TIẾT CÁC LỚP DỮ LIỆU - DATA LAYERS]")
    print(f" 1. Lớp Độ cao (Heights / Elevation):")
    print(f"    + Điểm thấp nhất : {np.min(heights):.3f} m")
    print(f"    + Điểm cao nhất  : {np.max(heights):.3f} m")
    print(f"    + Độ cao trung bình: {np.mean(heights):.3f} m")
    
    print(f" 2. Lớp Độ dốc (Slopes):")
    print(f"    + Độ dốc lớn nhất  : {np.max(slopes):.3f} rad (Rủi ro LTR cao)")
    print(f"    + Độ dốc nhỏ nhất  : {np.min(slopes):.3f} rad")
    
    unique_classes = np.unique(t_classes.astype(int))
    print(f" 3. Lớp Cơ giới đất (Terrain/Soil Classes):")
    print(f"    + Các mã loại đất tồn tại : {unique_classes}")
    print(f"    + Cấu hình hệ số cản trở đất (Bekker-Wong) tương ứng:")
    for c in unique_classes:
        desc = soil_desc.get(c, "Loại đất khác")
        res = soil_resistance.get(c, 2.0)
        print(f"      * Class {c} ({desc}): Hệ số R = {res}")
    print("="*65)

    # 4. TRỰC QUAN HÓA DỮ LIỆU ĐỂ LƯU ẢNH REPORT
    print("\nĐang tạo biểu đồ trực quan...")
    plt.figure(figsize=(15, 6))

    # Biểu đồ 1: Bản đồ Độ cao
    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(heights, cmap='terrain', origin='upper')
    plt.colorbar(img1, ax=ax1, label='Độ cao địa hình (z) - mét', shrink=0.8)
    ax1.set_title(f"Bản đồ Phân tầng Độ cao (Elevation Layer)\nDataset: BenchNav | Map: {filename}")
    ax1.set_xlabel("Cột lưới (X)")
    ax1.set_ylabel("Hàng lưới (Y)")

    # Biểu đồ 2: Bản đồ Phân loại đất
    ax2 = plt.subplot(1, 2, 2)
    
    # Chỉ định các giới hạn màu sắc cố định để các mảng rời rạc hiển thị rõ
    min_class = np.min(unique_classes)
    max_class = np.max(unique_classes)
    
    img2 = ax2.imshow(t_classes, cmap='Set3', origin='upper', vmin=min_class, vmax=max_class)
    cbar = plt.colorbar(img2, ax=ax2, shrink=0.8, ticks=unique_classes)
    
    # Gắn nhãn cho colorbar có kèm hệ số đất
    labels = [f"Class {c}\n(R={soil_resistance.get(c, 2.0)})" for c in unique_classes]
    cbar.set_ticklabels(labels)
    cbar.set_label('Mã loại đất & Hệ số cản')
    
    ax2.set_title(f"Bản đồ Cơ giới đất (Soil Classes Layer)\nDataset: BenchNav | Map: {filename}")
    ax2.set_xlabel("Cột lưới (X)")
    ax2.set_ylabel("Hàng lưới (Y)")

    plt.tight_layout()
    
    # Tự động lưu ảnh vào thư mục figures để chèn vào LaTeX
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/dataset_analysis_map025.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[THÀNH CÔNG] Đã xuất ảnh minh họa sắc nét vào: {save_path}")
    
    # Hiển thị cửa sổ đồ họa
    plt.show()

if __name__ == "__main__":
    analyze_and_show_map()