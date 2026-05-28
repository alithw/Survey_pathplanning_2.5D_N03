"""
train_system1.py
----------------
Huấn luyện mạng CNN siêu nhẹ (System 1) để học trực giác địa hình từ 1000 bản đồ GT.
Mô hình nhận đầu vào (heights, slopes, soils, start_mask, goal_mask) và dự đoán Corridor Mask.

Chạy thử nghiệm hoặc huấn luyện chính thức:
    wsl bash -c "cd ~/ros2_ws/src/survey_pathplanning/survey_PathPlanning_2.5D_template/'3. code' && source ~/ros2_ws/src/benchnav_env/bin/activate && python train_system1.py"
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Thiết lập thiết bị chạy
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "system1_model.pth"
GT_DIR = "gt_dataset"

# ==============================================================
# CẤU TRÚC MẠNG NEURAL SYSTEM 1 (Lightweight Encoder-Decoder)
# ==============================================================
class System1CNN(nn.Module):
    def __init__(self):
        super(System1CNN, self).__init__()
        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2) # -> 32x32
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2) # -> 16x16
        
        # Latent Space
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # -> 32x32
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) # -> 64x64
        self.dec2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Cho xác suất [0, 1]
        )
        
    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        
        b = self.bottleneck(p2)
        
        u1 = self.up1(b)
        d1 = self.dec1(u1)
        
        u2 = self.up2(d1)
        out = self.dec2(u2)
        return out

# ==============================================================
# DATASET ĐỂ LOAD CÁC TỆP .NPZ ĐÃ SINH
# ==============================================================
class GTDataset(Dataset):
    def __init__(self, gt_dir):
        self.filepaths = glob.glob(os.path.join(gt_dir, "*.npz"))
        print(f" -> Tìm thấy {len(self.filepaths)} tệp dữ liệu Ground Truth trong thư mục '{gt_dir}'.")
        
    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        data = np.load(self.filepaths[idx])
        
        heights = data['heights']
        slopes = data['slopes']
        t_classes = data['t_classes']
        start = data['start']
        goal = data['goal']
        corridor_mask = data['corridor_mask']
        
        # Chuẩn hóa đầu vào
        h_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-6)
        s_norm = np.clip(slopes / (slopes.max() + 1e-6), 0, 1)
        t_norm = t_classes / 8.0 # class values 0-7
        
        # Khởi tạo start/goal masks
        grid_size = heights.shape[0]
        start_mask = np.zeros_like(heights, dtype=np.float32)
        start_mask[int(start[0]), int(start[1])] = 1.0
        
        goal_mask = np.zeros_like(heights, dtype=np.float32)
        goal_mask[int(goal[0]), int(goal[1])] = 1.0
        
        # Ghép thành 5 channels đầu vào
        x = np.stack([h_norm, s_norm, t_norm, start_mask, goal_mask], axis=0).astype(np.float32)
        y = np.expand_dims(corridor_mask, axis=0).astype(np.float32)
        
        return torch.tensor(x), torch.tensor(y)

# ==============================================================
# HÀM TRAINING
# ==============================================================
def train_model(epochs=15, batch_size=16, test_mode=False):
    if test_mode:
        epochs = 1
        batch_size = 2
        print("[TEST] Chạy thử nghiệm huấn luyện 1 epoch...")
        
    # Khởi tạo Dataset & DataLoader
    if not os.path.exists(GT_DIR) or len(glob.glob(os.path.join(GT_DIR, "*.npz"))) == 0:
        print(f"[CẢNH BÁO] Không có dữ liệu huấn luyện nào trong thư mục '{GT_DIR}'.")
        if test_mode:
            # Tạo dữ liệu giả lập để chạy test qua cú pháp
            os.makedirs(GT_DIR, exist_ok=True)
            np.savez_compressed(
                os.path.join(GT_DIR, "dummy.npz"),
                heights=np.random.randn(64, 64),
                slopes=np.random.rand(64, 64),
                t_classes=np.random.randint(0, 4, (64, 64)),
                start=np.array([10, 10]),
                goal=np.array([50, 50]),
                path=np.array([[10, 10], [50, 50]]),
                corridor_mask=np.ones((64, 64))
            )
            print("  -> Đã tạo dữ liệu giả lập (dummy) cho chế độ Test.")
        else:
            print("[LỖI] Hãy chạy generate_gt_paths.py trước!")
            return
            
    dataset = GTDataset(GT_DIR)
    
    # Chia train/validation set
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # Khởi tạo mô hình, loss, optimizer
    model = System1CNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    print(f"Bắt đầu huấn luyện trên thiết bị: {DEVICE}")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_set)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_set)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        scheduler.step()
        
    print("=" * 60)
    # Lưu mô hình
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"🎉 Đã lưu mô hình huấn luyện thành công vào: {MODEL_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Chạy test 1 epoch nhanh')
    args = parser.parse_args()
    
    train_model(epochs=30, batch_size=16, test_mode=args.test)
