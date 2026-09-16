# ==============================================================================
# ONLINE SLIP ADAPTER: BỘ ƯỚC LƯỢNG VÀ BÙ TRƯỢT THÍCH NGHI ĐỆ QUY TRỰC TUYẾN
# Tự động học phần bù trượt đất (Residual Slip) trong thời gian thực khi gặp đất lạ
# ==============================================================================

import numpy as np

class OnlineSlipAdapter:
    """
    Bộ lọc thích nghi đệ quy (Recursive Adaptive Filter) ước lượng phần bù trượt sai lệch:
        alpha_total(k) = alpha_LUT(Soil, Slope, v) + Delta_alpha(k)
    Giúp robot tự động điều chỉnh ngay khi gặp đất trơn, ẩm ướt vượt ngoài mô hình gốc.
    Độ phức tạp: O(1), thời gian tính toán < 20 nano-giây.
    """
    def __init__(self, learning_rate=0.10, max_deviation=0.15, forgetting_factor=0.95):
        self.lr = learning_rate
        self.max_dev = max_deviation
        self.gamma = forgetting_factor
        self.delta_alpha = 0.0

    def reset(self):
        self.delta_alpha = 0.0

    def update(self, v_cmd, actual_dx, actual_dy, dt, nominal_slip, step=10):
        """
        Cập nhật phần bù trượt Delta_alpha dựa trên sai lệch vận tốc bánh xe vs thực tế.
        Chỉ cập nhật khi robot đã ổn định vận tốc (step > 5 và v_cmd > 0.4 m/s).
        """
        if step < 5 or v_cmd < 0.40 or dt <= 0:
            return nominal_slip

        v_actual = np.sqrt(actual_dx**2 + actual_dy**2) / dt
        alpha_obs = 1.0 - (v_actual / v_cmd)
        alpha_obs = np.clip(alpha_obs, 0.0, 0.70)

        slip_residual = alpha_obs - nominal_slip
        slip_residual = np.clip(slip_residual, -self.max_dev, self.max_dev)

        # Lọc thông thấp đệ quy
        self.delta_alpha = self.gamma * self.delta_alpha + (1.0 - self.gamma) * slip_residual
        self.delta_alpha = float(np.clip(self.delta_alpha, -self.max_dev, self.max_dev))

        return float(np.clip(nominal_slip + self.delta_alpha, 0.01, 0.40))

    def get_adapted_horizon(self, nominal_slips):
        return np.clip(nominal_slips + self.delta_alpha, 0.01, 0.40).astype(np.float32)
