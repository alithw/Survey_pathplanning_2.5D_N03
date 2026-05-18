"""
gen_comparison_figure.py
------------------------
Sinh tự động 2 hình tổng hợp từ dữ liệu thực nghiệm đã đo được:
  1. figures/algo_comparison_bar.png  — Biểu đồ cột so sánh 5 thuật toán
  2. figures/cost_function_demo.png   — Minh họa hàm chi phí Minetti theo độ dốc

Chạy từ thư mục '3. code':
    python gen_comparison_figure.py

Ảnh được lưu vào '../1. Report/figures/' để LaTeX/Beamer dùng trực tiếp.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ==============================================================
# CẤU HÌNH LƯU ẢNH
# ==============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '1. Report', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

# ==============================================================
# HÌNH 1: BIỂU ĐỒ CỘT SO SÁNH THUẬT TOÁN
#   Dữ liệu từ log thực nghiệm trên Map 000_025.pt
# ==============================================================
def plot_algorithm_comparison():
    algorithms = ['PRM\n(Sampling)', 'BIT*\n(Informed)', 'ACO\n(Bio-Inspired)',
                  'A*\n(Grid-Search)', 'RRT*\n(Optimal)']

    # Kết quả từ log thực tế (NaN = không tìm được đường)
    exec_times  = [np.nan,  np.nan,  np.nan,  89.05,   222.82]   # ms
    energy_costs = [np.nan, np.nan,  np.nan,  7781.99, 1121.10]  # J
    waypoints   = [np.nan,  np.nan,  np.nan,  54,      20]

    # Màu sắc: đỏ = thất bại, xanh đậm = A*, xanh lá = RRT*
    colors_time   = ['#e74c3c','#e74c3c','#e74c3c','#2980b9','#27ae60']
    colors_energy = ['#e74c3c','#e74c3c','#e74c3c','#2980b9','#27ae60']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('So sánh Hiệu năng Thuật toán Quy hoạch trên Map 000_025.pt (BenchNav)',
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(algorithms))
    bar_w = 0.55

    # --- Subplot 1: Thời gian ---
    ax = axes[0]
    bars = ax.bar(x, [t if not np.isnan(t) else 0 for t in exec_times],
                  color=colors_time, width=bar_w, edgecolor='white', linewidth=1.2)
    for i, (bar, val) in enumerate(zip(bars, exec_times)):
        if np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, 5, 'FAILED\n(không\ntìm được)', 
                    ha='center', va='bottom', fontsize=8.5, color='#e74c3c', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f} ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(algorithms, fontsize=9)
    ax.set_ylabel('Thời gian xử lý (ms)'); ax.set_title('Tốc độ tính toán')
    ax.set_ylim(0, 300); ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # --- Subplot 2: Năng lượng ---
    ax = axes[1]
    bars = ax.bar(x, [e if not np.isnan(e) else 0 for e in energy_costs],
                  color=colors_energy, width=bar_w, edgecolor='white', linewidth=1.2)
    for i, (bar, val) in enumerate(zip(bars, energy_costs)):
        if np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, 150, 'FAILED', 
                    ha='center', va='bottom', fontsize=8.5, color='#e74c3c', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                    f'{val:.0f} J', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(algorithms, fontsize=9)
    ax.set_ylabel('Tổng năng lượng tiêu hao (J)'); ax.set_title('Hiệu quả năng lượng')
    ax.set_ylim(0, 9500); ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # --- Subplot 3: Số Waypoints ---
    ax = axes[2]
    bars = ax.bar(x, [w if not np.isnan(w) else 0 for w in waypoints],
                  color=colors_time, width=bar_w, edgecolor='white', linewidth=1.2)
    for i, (bar, val) in enumerate(zip(bars, waypoints)):
        if np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, 0.5, 'FAILED', 
                    ha='center', va='bottom', fontsize=8.5, color='#e74c3c', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{int(val)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(algorithms, fontsize=9)
    ax.set_ylabel('Số lượng Waypoints'); ax.set_title('Độ mịn quỹ đạo')
    ax.set_ylim(0, 70); ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color='#e74c3c', label='Thất bại (không tìm được đường)'),
        mpatches.Patch(color='#2980b9', label='Weighted A* ✓ (Được chọn)'),
        mpatches.Patch(color='#27ae60', label='Physics-Aware RRT* ✓'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=10, framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'algo_comparison_bar.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[OK] Đã lưu: {save_path}")
    plt.close()


# ==============================================================
# HÌNH 2: MINH HỌA HÀM CHI PHÍ MINETTI & TỔ HỢP CHI PHÍ
# ==============================================================
def plot_cost_function_demo():
    s = np.linspace(-0.5, 0.5, 300)

    # Hàm Minetti
    C_minetti = 155.4*s**5 - 30.4*s**4 - 43.3*s**3 + 46.3*s**2 + 19.5*s + 3.6
    C_minetti = np.maximum(C_minetti, 0.5)

    # Tổng chi phí bước (Minetti + Bekker hằng + LTR tăng dần)
    bekker_const = 3.0   # Đất bùn class 3
    LTR = np.clip(np.abs(s) * 2, 0, 1)  # LTR xấp xỉ tỉ lệ với |s|
    step_cost = C_minetti + 1.5 * bekker_const + 50 * LTR

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Hàm Chi phí Vật lý Tích hợp trong Weighted A*', fontsize=13, fontweight='bold')

    # --- Subplot 1: Minetti ---
    ax = axes[0]
    ax.plot(s * 100, C_minetti, color='#e67e22', linewidth=2.5, label='C(s) - Minetti')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(3.6, color='gray', linestyle=':', alpha=0.5, label='Bằng phẳng (s=0): 3.6 J/kg·m')
    ax.fill_between(s * 100, C_minetti, 0.5, where=(s > 0), alpha=0.15, color='#e74c3c', label='Leo dốc (tốn năng lượng)')
    ax.fill_between(s * 100, C_minetti, 0.5, where=(s < 0), alpha=0.15, color='#2980b9', label='Xuống dốc (tiết kiệm)')
    ax.set_xlabel('Độ dốc s_pitch (%)')
    ax.set_ylabel('Chi phí năng lượng C (J/kg·m)')
    ax.set_title('Mô hình Minetti (2002)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # --- Subplot 2: Tổng chi phí Step ---
    ax = axes[1]
    ax.plot(s * 100, C_minetti, '--', color='#e67e22', linewidth=1.8, label='Minetti (năng lượng)')
    ax.plot(s * 100, 50 * LTR, '--', color='#e74c3c', linewidth=1.8, label='Phạt LTR (rủi ro lật)')
    ax.plot(s * 100, np.full_like(s, 1.5 * bekker_const), '--', color='#8e44ad', linewidth=1.8, label=f'Bekker-Wong (đất bùn R={bekker_const})')
    ax.plot(s * 100, step_cost, '-', color='#2c3e50', linewidth=3, label='StepCost tổng hợp')

    # Vùng phạt vô cực (LTR >= 1.5·SSF)
    ax.axvline(37, color='#e74c3c', linestyle=':', alpha=0.8)
    ax.axvline(-37, color='#e74c3c', linestyle=':', alpha=0.8)
    ax.text(38, 80, '∞ penalty\n(LTR≥1.5)', color='#e74c3c', fontsize=8)

    ax.set_xlabel('Độ dốc s_pitch (%)')
    ax.set_ylabel('Chi phí bước (StepCost)')
    ax.set_title('Hàm Chi phí Tổng hợp = Minetti + Bekker + LTR')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim(0, 120)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'cost_function_demo.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[OK] Đã lưu: {save_path}")
    plt.close()


# ==============================================================
# HÌNH 3: MINH HỌA KIẾN TRÚC HỆ THỐNG (Flowchart đơn giản)
# ==============================================================
def plot_system_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')
    ax.set_title('Kiến trúc Hệ thống A* + NMPC trên BenchNav 2.5D', fontsize=13, fontweight='bold')

    boxes = [
        (0.5,  1.5, 1.8, 1.0, '#2c3e50', 'BenchNav\nDataset\n(000_025.pt)', 'white'),
        (2.8,  1.5, 1.8, 1.0, '#2980b9', 'Weighted A*\nGlobal Planner\n(89 ms)', 'white'),
        (5.1,  1.5, 1.8, 1.0, '#27ae60', 'NMPC\nLocal Controller\n(CasADi/IPOPT)', 'white'),
        (7.4,  1.5, 1.8, 1.0, '#e67e22', 'Robot\nActuation\ncmd_vel', 'white'),
    ]

    for (x, y, w, h, color, label, fc) in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, 
                                        boxstyle="round,pad=0.08", 
                                        facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                color=fc, fontsize=9.5, fontweight='bold', linespacing=1.4)

    # Mũi tên
    arrow_props = dict(arrowstyle='->', color='#2c3e50', lw=2.5)
    for x_start in [2.3, 4.6, 6.9]:
        ax.annotate('', xy=(x_start+0.5, 2.0), xytext=(x_start, 2.0),
                    arrowprops=arrow_props)

    # Nhãn mũi tên
    ax.text(2.55, 2.2, 'heights\nslopes\nt_classes', ha='center', fontsize=7.5, color='gray')
    ax.text(4.85, 2.2, 'nav_msgs\n/Path', ha='center', fontsize=7.5, color='gray')
    ax.text(7.15, 2.2, 'Twist\nv, ω', ha='center', fontsize=7.5, color='gray')

    # Annotation kết quả
    ax.text(3.7, 0.9, '✓ 54 waypoints | 89 ms', ha='center', fontsize=8.5,
            color='#2980b9', style='italic')
    ax.text(6.0, 0.9, '✓ Tracking error < 0.22 m\n   (10% slip)', ha='center', fontsize=8.5,
            color='#27ae60', style='italic')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'system_architecture.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[OK] Đã lưu: {save_path}")
    plt.close()


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    print("=" * 55)
    print(" SINH TỰ ĐỘNG FIGURE CHO SLIDE & BÁO CÁO (Nhóm 3)")
    print("=" * 55)
    plot_algorithm_comparison()
    plot_cost_function_demo()
    plot_system_architecture()
    print("=" * 55)
    print(" HOÀN TẤT! Kiểm tra thư mục '1. Report/figures/'")
    print("=" * 55)
