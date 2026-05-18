"""
traversability_analysis.py
---------------------------
Phân tích Traversability đầy đủ 3 phương pháp theo Section 8 của bài survey:
  1. Geometric Methods  : Slope, Roughness, Step Hazard
  2. Physical Methods   : Terramechanics (Bekker-Wong), Friction Estimation
  3. Semantic Methods   : Mô phỏng DL-based soil classification (Mud vs Grass vs Rock)

Chạy từ thư mục '3. code':
    source ~/ros2_ws/src/benchnav_env/bin/activate
    python traversability_analysis.py

Output: ../1. Report/figures/traversability_*.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '1. Report', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'figure.dpi': 150,
})

# ==============================================================
# SINH DỮ LIỆU ĐỊA HÌNH GIẢ LẬP (không cần dataset thực)
# Mô phỏng địa hình 64x64 với đồi, thung lũng, bùn lầy
# ==============================================================
np.random.seed(42)
GRID = 64

def generate_synthetic_terrain():
    """Tạo địa hình tổng hợp thực tế hơn"""
    x = np.linspace(0, 4*np.pi, GRID)
    y = np.linspace(0, 4*np.pi, GRID)
    X, Y = np.meshgrid(x, y)

    # Height map: Kết hợp nhiều sóng để tạo địa hình phức tạp
    heights = (
        2.0 * np.sin(0.5*X) * np.cos(0.5*Y) +
        1.0 * np.sin(1.2*X + 0.3) * np.sin(0.8*Y) +
        0.5 * np.random.randn(GRID, GRID) +  # noise
        3.0 * np.exp(-((X-np.pi)**2 + (Y-np.pi)**2) / 4)  # đồi ở giữa
    )
    heights = (heights - heights.min()) / (heights.max() - heights.min()) * 5.0

    # Gradient (slope) - độ dốc theo 2 hướng
    grad_x = np.gradient(heights, axis=1)
    grad_y = np.gradient(heights, axis=0)
    slopes = np.sqrt(grad_x**2 + grad_y**2)  # magnitude

    # Roughness = local standard deviation of heights
    from scipy.ndimage import generic_filter
    roughness = generic_filter(heights, np.std, size=5)

    # Step hazard = max local height difference
    step_hazard = generic_filter(heights, lambda x: x.max()-x.min(), size=3)

    # Terrain classes (mô phỏng): 0=grass, 1=mud, 2=rock, 3=sand
    t_classes = np.zeros((GRID, GRID), dtype=int)
    t_classes[slopes > 0.6] = 2          # rock: dốc cao
    t_classes[(slopes < 0.3) & (heights < 1.5)] = 1   # mud: thấp, bằng phẳng
    t_classes[roughness > 0.4] = 3       # sand: rough
    # grass = còn lại (0)

    return heights, slopes, roughness, step_hazard, t_classes

heights, slopes, roughness, step_hazard, t_classes = generate_synthetic_terrain()

# ==============================================================
# PHƯƠNG PHÁP 1: GEOMETRIC TRAVERSABILITY
# ==============================================================
def geometric_traversability(slopes, roughness, step_hazard):
    """
    Kết hợp 3 chỉ số hình học thành điểm traversability [0=impassable, 1=free]
    Theo: Triebel et al. (Multi-Level Surface Maps)
    """
    # Chuẩn hóa từng chỉ số về [0,1]
    s_norm = np.clip(slopes / slopes.max(), 0, 1)
    r_norm = np.clip(roughness / roughness.max(), 0, 1)
    h_norm = np.clip(step_hazard / step_hazard.max(), 0, 1)

    # Hàm tổng hợp: trọng số từ tài liệu thực nghiệm
    w_s, w_r, w_h = 0.5, 0.3, 0.2
    hazard_score = w_s * s_norm + w_r * r_norm + w_h * h_norm

    # Traversability = 1 - hazard (1 = dễ đi)
    return np.clip(1.0 - hazard_score, 0, 1)

T_geo = geometric_traversability(slopes, roughness, step_hazard)

# ==============================================================
# PHƯƠNG PHÁP 2: PHYSICAL TRAVERSABILITY (Bekker-Wong + Friction)
# ==============================================================
# Bekker-Wong soil parameters (từ bảng trong thesis)
bekker_params = {
    0: {'kc': 1.5,  'kphi': 500.0, 'n': 1.0, 'name': 'Grass (Cỏ)'},     # Class 0
    1: {'kc': 0.2,  'kphi': 90.0,  'n': 0.6, 'name': 'Mud (Bùn)'},       # Class 1
    2: {'kc': 5.0,  'kphi': 1200.0,'n': 1.2, 'name': 'Rock (Đá)'},       # Class 2
    3: {'kc': 0.8,  'kphi': 200.0, 'n': 0.8, 'name': 'Sand (Cát)'},      # Class 3
}
# Hệ số ma sát tĩnh (mu1) theo loại đất
friction_mu = {0: 0.65, 1: 0.20, 2: 0.80, 3: 0.40}

def bekker_resistance(t_class_map, wheel_b=0.15, z_sinkage=0.02):
    """Tính lực cản lăn Bekker-Wong theo từng ô lưới"""
    R_c = np.zeros_like(t_class_map, dtype=float)
    for cls, params in bekker_params.items():
        mask = (t_class_map == cls)
        kc, kphi, n, b = params['kc'], params['kphi'], params['n'], wheel_b
        # p = (kc/b + kphi) * z^n
        p = (kc / b + kphi) * (z_sinkage ** n)
        # R_c ∝ pressure (simplified)
        R_c[mask] = p * z_sinkage  # [N/m^2 * m = N/m]
    return R_c

def friction_traversability(t_class_map, slopes):
    """
    Traversability dựa trên ma sát: robot trượt nếu mu < tan(slope_angle)
    """
    mu_map = np.vectorize(lambda c: friction_mu.get(c, 0.5))(t_class_map)
    slope_angle = np.arctan(slopes)  # rad
    tan_slope = np.tan(slope_angle)

    # Margin: friction_margin = mu - tan(slope) (dương = an toàn)
    friction_margin = mu_map - tan_slope
    T_phys = np.clip(friction_margin / mu_map, 0, 1)
    return T_phys, mu_map

R_c = bekker_resistance(t_classes)
T_phys, mu_map = friction_traversability(t_classes, slopes)

# ==============================================================
# PHƯƠNG PHÁP 3: SEMANTIC TRAVERSABILITY (Simulated DL Output)
# ==============================================================
def simulate_semantic_traversability(t_class_map):
    """
    Mô phỏng output của một mạng CNN phân loại đất.
    Trong thực tế: input RGB + LiDAR → CNN → confidence per class
    Ở đây ta mô phỏng bằng cách gán confidence có nhiễu Gaussian
    """
    class_traversability = {
        0: 0.90,  # Grass: dễ đi
        1: 0.20,  # Mud: nguy hiểm
        2: 0.60,  # Rock: đi được nhưng rung lắc
        3: 0.55,  # Sand: trung bình
    }
    T_sem = np.vectorize(lambda c: class_traversability.get(c, 0.5))(t_class_map)

    # Thêm uncertainty (confidence noise)
    noise = np.random.randn(GRID, GRID) * 0.05
    T_sem = np.clip(T_sem + noise, 0, 1)
    return T_sem

T_sem = simulate_semantic_traversability(t_classes)

# ==============================================================
# FUSION: KẾT HỢP 3 PHƯƠNG PHÁP
# ==============================================================
def fuse_traversability(T_geo, T_phys, T_sem, w=(0.35, 0.35, 0.30)):
    """Bayesian/Weighted fusion"""
    return w[0]*T_geo + w[1]*T_phys + w[2]*T_sem

T_fused = fuse_traversability(T_geo, T_phys, T_sem)

# ==============================================================
# VISUALIZE - FIGURE CHÍNH
# ==============================================================
trav_cmap = LinearSegmentedColormap.from_list(
    'trav', ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71'], N=256)

soil_colors = ['#27ae60', '#8e44ad', '#7f8c8d', '#f39c12']
soil_labels = ['Class 0: Grass', 'Class 1: Mud', 'Class 2: Rock', 'Class 3: Sand']
soil_cmap = plt.matplotlib.colors.ListedColormap(soil_colors)

fig = plt.figure(figsize=(18, 10))
fig.suptitle('Traversability Analysis: Ba Phương pháp Tích hợp (Nhóm 3 - Section 8)',
             fontsize=14, fontweight='bold')

axes_layout = [
    (2, 5, 1), (2, 5, 2), (2, 5, 3), (2, 5, 4), (2, 5, 5),
    (2, 5, 6), (2, 5, 7), (2, 5, 8), (2, 5, 9), (2, 5, 10),
]

# Row 1
ax1 = fig.add_subplot(2, 5, 1)
im = ax1.imshow(heights, cmap='terrain', origin='upper')
plt.colorbar(im, ax=ax1, shrink=0.7)
ax1.set_title('Địa hình (Heights)', fontsize=9)
ax1.axis('off')

ax2 = fig.add_subplot(2, 5, 2)
im = ax2.imshow(slopes, cmap='hot_r', origin='upper')
plt.colorbar(im, ax=ax2, shrink=0.7)
ax2.set_title('Độ dốc (Slopes)', fontsize=9)
ax2.axis('off')

ax3 = fig.add_subplot(2, 5, 3)
im = ax3.imshow(t_classes, cmap=soil_cmap, origin='upper', vmin=-0.5, vmax=3.5)
cbar = plt.colorbar(im, ax=ax3, shrink=0.7, ticks=[0,1,2,3])
cbar.set_ticklabels(['Grass', 'Mud', 'Rock', 'Sand'])
ax3.set_title('Phân loại Đất (Terrain Class)', fontsize=9)
ax3.axis('off')

ax4 = fig.add_subplot(2, 5, 4)
im = ax4.imshow(mu_map, cmap='RdYlGn', origin='upper', vmin=0, vmax=1)
plt.colorbar(im, ax=ax4, shrink=0.7, label='μ')
ax4.set_title('Hệ số Ma sát (μ)', fontsize=9)
ax4.axis('off')

ax5 = fig.add_subplot(2, 5, 5)
im = ax5.imshow(R_c, cmap='YlOrRd', origin='upper')
plt.colorbar(im, ax=ax5, shrink=0.7, label='N/m')
ax5.set_title('Lực cản Bekker-Wong (R_c)', fontsize=9)
ax5.axis('off')

# Row 2
ax6 = fig.add_subplot(2, 5, 6)
im = ax6.imshow(T_geo, cmap=trav_cmap, origin='upper', vmin=0, vmax=1)
plt.colorbar(im, ax=ax6, shrink=0.7)
ax6.set_title('① Geometric Traversability\n(Slope+Roughness+Step)', fontsize=9)
ax6.axis('off')

ax7 = fig.add_subplot(2, 5, 7)
im = ax7.imshow(T_phys, cmap=trav_cmap, origin='upper', vmin=0, vmax=1)
plt.colorbar(im, ax=ax7, shrink=0.7)
ax7.set_title('② Physical Traversability\n(Friction+Bekker-Wong)', fontsize=9)
ax7.axis('off')

ax8 = fig.add_subplot(2, 5, 8)
im = ax8.imshow(T_sem, cmap=trav_cmap, origin='upper', vmin=0, vmax=1)
plt.colorbar(im, ax=ax8, shrink=0.7)
ax8.set_title('③ Semantic Traversability\n(CNN Soil Classification)', fontsize=9)
ax8.axis('off')

ax9 = fig.add_subplot(2, 5, 9)
im = ax9.imshow(T_fused, cmap=trav_cmap, origin='upper', vmin=0, vmax=1)
plt.colorbar(im, ax=ax9, shrink=0.7)
ax9.set_title('④ FUSED Traversability\n(Weighted Fusion)', fontsize=9, fontweight='bold', color='#2c3e50')
ax9.axis('off')

# Colorbar giải thích
ax10 = fig.add_subplot(2, 5, 10)
ax10.axis('off')
gradient = np.linspace(0, 1, 256).reshape(256, 1)
ax10.imshow(gradient, cmap=trav_cmap, aspect='auto',
            extent=[0, 1, 0, 1], origin='lower')
ax10.set_xlim(0, 2)
ax10.text(1.1, 1.0, '1.0\n(Free / Dễ đi)', va='top', fontsize=8, color='#27ae60', fontweight='bold')
ax10.text(1.1, 0.5, '0.5\n(Khó khăn)', va='center', fontsize=8, color='#f39c12')
ax10.text(1.1, 0.0, '0.0\n(Blocked)', va='bottom', fontsize=8, color='#e74c3c', fontweight='bold')
ax10.set_title('Thang\nTraversability', fontsize=9)
ax10.set_xticks([]); ax10.set_yticks([])

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'traversability_analysis.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"[OK] Đã lưu: {save_path}")
plt.close()

# ==============================================================
# FIGURE PHỤ: So sánh 3 phương pháp theo profile cắt ngang
# ==============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
fig2.suptitle('Profile Cắt ngang tại hàng Y=32: So sánh 3 Phương pháp Traversability',
              fontsize=12, fontweight='bold')

row = GRID // 2
x_axis = np.arange(GRID)

ax = axes2[0]
ax.fill_between(x_axis, T_geo[row], alpha=0.3, color='#2980b9')
ax.plot(x_axis, T_geo[row], color='#2980b9', linewidth=2, label='Geometric')
ax.fill_between(x_axis, T_phys[row], alpha=0.3, color='#27ae60')
ax.plot(x_axis, T_phys[row], color='#27ae60', linewidth=2, label='Physical')
ax.fill_between(x_axis, T_sem[row], alpha=0.3, color='#e67e22')
ax.plot(x_axis, T_sem[row], color='#e67e22', linewidth=2, label='Semantic')
ax.set_xlabel('Vị trí lưới (X)'); ax.set_ylabel('Traversability Score [0-1]')
ax.set_title('3 Phương pháp riêng lẻ'); ax.legend(fontsize=9)
ax.axhline(0.3, color='r', linestyle='--', alpha=0.6, label='Ngưỡng nguy hiểm')
ax.set_ylim(0, 1.1); ax.grid(alpha=0.3)

ax = axes2[1]
ax.fill_between(x_axis, T_fused[row], where=(T_fused[row] >= 0.5),
                alpha=0.4, color='#27ae60', label='Khu vực đi được')
ax.fill_between(x_axis, T_fused[row], where=(T_fused[row] < 0.5),
                alpha=0.4, color='#e74c3c', label='Khu vực nguy hiểm')
ax.plot(x_axis, T_fused[row], color='#2c3e50', linewidth=2.5)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Ngưỡng 0.5')
ax.set_xlabel('Vị trí lưới (X)'); ax.set_title('Fused Traversability (Weighted)')
ax.legend(fontsize=9); ax.set_ylim(0, 1.1); ax.grid(alpha=0.3)

ax = axes2[2]
# Scatter: Geometric vs Physical vs Semantic
sc = ax.scatter(T_geo[::2, ::2].flatten(), T_sem[::2, ::2].flatten(),
                c=T_fused[::2, ::2].flatten(), cmap=trav_cmap,
                s=20, alpha=0.7, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label='Fused Score')
ax.set_xlabel('Geometric Traversability'); ax.set_ylabel('Semantic Traversability')
ax.set_title('Tương quan: Geometric vs Semantic')
ax.plot([0,1], [0,1], 'k--', alpha=0.4, label='y=x')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
save_path2 = os.path.join(OUTPUT_DIR, 'traversability_profile.png')
plt.savefig(save_path2, dpi=200, bbox_inches='tight')
print(f"[OK] Đã lưu: {save_path2}")
plt.close()

print("=" * 55)
print(" TRAVERSABILITY ANALYSIS HOÀN TẤT!")
print("=" * 55)

# In thống kê
print(f"\n[THỐNG KÊ] Phân tích địa hình {GRID}x{GRID}")
print(f"  Geometric  - Mean: {T_geo.mean():.3f} | Min: {T_geo.min():.3f} | Vùng nguy (< 0.3): {(T_geo < 0.3).sum()} cells")
print(f"  Physical   - Mean: {T_phys.mean():.3f} | Min: {T_phys.min():.3f} | Vùng nguy (< 0.3): {(T_phys < 0.3).sum()} cells")
print(f"  Semantic   - Mean: {T_sem.mean():.3f} | Min: {T_sem.min():.3f} | Vùng nguy (< 0.3): {(T_sem < 0.3).sum()} cells")
print(f"  FUSED      - Mean: {T_fused.mean():.3f} | Min: {T_fused.min():.3f} | Vùng nguy (< 0.3): {(T_fused < 0.3).sum()} cells")
print(f"\n  Tỉ lệ khu vực có thể đi được (>= 0.5): {(T_fused >= 0.5).mean()*100:.1f}%")
