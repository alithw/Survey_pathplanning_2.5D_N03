"""HGB-A* (Hierarchical Geodesic-Guided Bidirectional A*) for 2.5D off-road DEMs.

4-tier classical multi-resolution graph planner (C++17 SIMD kernel with 1:1
vectorized Python parity; legacy aliases HMCTDAStarV2Planner / HMCTDUltraPlanner
preserved for backward compatibility):
  Tier 1: Coarse bidirectional Dijkstra geodesic potential + Gibbs probability & block TCI.
  Tier 2: Saddle-point portal graph (BSxBS) + PUCT-guided Gaussian soft corridor tube.
  Tier 3: Pohl cardinality-balanced bidirectional Soft-APH A* with auto-CBF escape.
  Tier 4: Anchor compression, 2.5D Geodesic Visibility DP (z_clearance), Safe Chaikin.
"""

import os
import math
import time
import heapq
import ctypes
import subprocess
import numpy as np


def _load_hgb_library():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(curr_dir, "libhgb_a_star.so")
    cpp_path = os.path.join(curr_dir, "hgb_a_star_core.cpp")

    needs_compile = (not os.path.exists(so_path)) or (
        os.path.exists(cpp_path) and os.path.getmtime(cpp_path) > os.path.getmtime(so_path)
    )
    if needs_compile:
        if not os.path.exists(cpp_path):
            raise FileNotFoundError(
                f"Neither compiled library '{so_path}' nor C++ source '{cpp_path}' was found."
            )
        subprocess.run(
            [
                "g++", "-O3", "-march=native", "-ffast-math", "-funroll-loops",
                "-shared", "-fPIC", cpp_path, "-o", so_path,
            ],
            check=True,
        )

    lib = ctypes.CDLL(so_path)
    c_dbl_p = ctypes.POINTER(ctypes.c_double)
    c_i32_p = ctypes.POINTER(ctypes.c_int32)
    c_u8_p = ctypes.POINTER(ctypes.c_uint8)
    c_int_p = ctypes.POINTER(ctypes.c_int)

    lib.hgb_plan_kernel.argtypes = [
        c_dbl_p, c_dbl_p, c_i32_p, c_dbl_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_int,
        c_dbl_p, c_u8_p, c_dbl_p, ctypes.c_int,
        c_int_p, c_int_p, c_int_p, c_int_p,
        c_dbl_p, ctypes.c_int, c_int_p,
    ]
    lib.hgb_plan_kernel.restype = ctypes.c_int

    lib.prepare_and_solve_riccati_rti.argtypes = [
        ctypes.c_int, ctypes.c_double,
        c_dbl_p, c_dbl_p, c_dbl_p, c_dbl_p, ctypes.c_double,
        c_dbl_p, c_dbl_p, c_dbl_p, c_dbl_p, c_dbl_p, c_dbl_p,
    ]
    lib.prepare_and_solve_riccati_rti.restype = ctypes.c_int

    lib.hgb_eval_metrics_kernel.argtypes = [
        c_dbl_p, c_i32_p, ctypes.c_int, ctypes.c_int,
        c_dbl_p, ctypes.c_int, c_dbl_p, c_dbl_p,
    ]
    lib.hgb_eval_metrics_kernel.restype = None
    return lib


_HGB_LIB = _load_hgb_library()
_V2_LIB = _HGB_LIB
_ULTRA_LIB = _HGB_LIB


class HGBAStarPlanner:
    """
    Thuật toán Quy hoạch Phân cấp HGB-A* (Hierarchical Geodesic-Guided Bidirectional A*)
    trên bản đồ độ cao 2.5D dã chiến phi cấu trúc:
    1. Tầng 1 (Coarse Bidirectional Dijkstra Geodesic Potential): Gom cụm cực tiểu trở kháng cơ học
       (Min-Impedance Pooling) với bước gộp tỉ lệ động cs = max(4, max_dim / 32) để bảo toàn các khe đèo
       hẹp 1-2 ô và tính trường thế trắc địa lưỡng hướng Gibbs P = exp(-Δφ/τ) cùng Block TCI.
    2. Tầng 2 (Saddle Boundary Portals & PUCT Soft Corridor Tube): Trích xuất các cổng yên ngựa tại cực
       tiểu trở kháng trên biên khối BSxBS và tô dải ống dẫn mềm Gaussian σ(TCI) qua đồ thị cổng.
    3. Tầng 3 (Cardinality-Balanced Bidirectional Soft-APH A*): Mở rộng song phương cân bằng lực lượng
       theo quy tắc Pohl với trọng số dẫn hướng mềm w_eff(n) = w_base * (1 + lambda_corr * (1 - P_corr(n))^2),
       loại bỏ hiện tượng đứt gãy hành lang cứng (0% Fallback) và khóa cứng vùng nước sâu (soil=8) / lật tĩnh.
    4. Tầng 4 (Multi-Scale 2.5D Geodesic Visibility DP & Locally-Verified Safe Chaikin): Giải quy hoạch
       động trên đồ thị tầm nhìn 2.5D cực tiểu hóa đồng thời cơ năng Minetti bậc 5 và năng lượng bẻ góc
       dưới ràng buộc ổn định chống lật LTR <= ltr_limit và khoảng sáng gầm DEM |z_DEM - z_chord| <= z_clearance.
    """

    def __init__(
        self,
        heights,
        slopes,
        t_classes,
        grid_size=None,
        ssf=1.118,
        ltr_limit=0.735,
        z_clearance=0.18,
        w_base=1.18,
        lambda_corr=1.35,
        puct_sims=48,
        use_native=True,
        cnn_model=None,
    ):
        self.heights = np.ascontiguousarray(heights, dtype=np.float64)
        self.slopes = np.ascontiguousarray(slopes, dtype=np.float64)
        self.t_classes = np.ascontiguousarray(t_classes, dtype=np.int32)
        self.H, self.W = self.heights.shape
        self.grid_size = grid_size if grid_size is not None else self.H
        self.ssf = float(ssf)
        self.ltr_limit = float(ltr_limit)
        self.z_clearance = float(z_clearance)
        self.w_base = float(w_base)
        self.lambda_corr = float(lambda_corr)
        self.puct_sims = int(puct_sims)
        self.use_native = bool(use_native)
        self.cnn_model = cnn_model

    def _compute_calibrated_cnn_prior(self, start, goal):
        """
        Chuẩn hóa bất biến vật lý cho mạng System1CNN (khi kích hoạt chế độ lai nơ-ron):
        Thay chuẩn hóa cực trị cục bộ s / s.max() bằng tỷ số ổn định tĩnh LTR = theta_rad / (0.72 * SSF)
        nhằm ngăn mạng dự đoán hành lang xuyên qua vách đứng trên các tập dữ liệu ngoài phân phối (GOOSE/TartanDrive).
        """
        if self.cnn_model is None:
            return None
        import torch
        h_min, h_max = float(self.heights.min()), float(self.heights.max())
        h_norm = ((self.heights - h_min) / (h_max - h_min + 1e-6)).astype(np.float32)
        ltr_norm = np.clip((np.radians(self.slopes) / self.ssf) / self.ltr_limit, 0.0, 1.0).astype(np.float32)
        t_norm = (self.t_classes.astype(np.float32)) / 8.0
        s_mask = np.zeros((self.H, self.W), dtype=np.float32)
        g_mask = np.zeros((self.H, self.W), dtype=np.float32)
        s_mask[int(start[0]), int(start[1])] = 1.0
        g_mask[int(goal[0]), int(goal[1])] = 1.0
        x_in = torch.from_numpy(np.stack([h_norm, ltr_norm, t_norm, s_mask, g_mask], axis=0)).unsqueeze(0)
        with torch.inference_mode():
            pred = self.cnn_model(x_in).squeeze().cpu().numpy().astype(np.float64)
        pred[ltr_norm >= 1.0] = 0.0
        return np.ascontiguousarray(pred, dtype=np.float64)

    def evaluate_path_metrics(self, path):
        """
        Định lượng cơ năng Minetti 2.5D và tích phân độ cong rời rạc theo đúng công thức đối chuẩn chung
        của hệ thống benchmark để bảo đảm tính khách quan tuyệt đối khi so sánh chéo các thuật toán.
        """
        if path is None or len(path) < 2:
            return float("inf"), float("inf")

        pts = np.ascontiguousarray(path, dtype=np.float64)
        out_e = ctypes.c_double(0.0)
        out_s = ctypes.c_double(0.0)
        _HGB_LIB.hgb_eval_metrics_kernel(
            self.heights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.t_classes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(self.H),
            ctypes.c_int(self.W),
            pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(pts)),
            ctypes.byref(out_e),
            ctypes.byref(out_s),
        )
        return float(out_e.value), float(out_s.value)

    def plan(self, start, goal, capture_diagnostics=False):
        return self.plan_hgb(start, goal, capture_diagnostics=capture_diagnostics)

    def plan_v2(self, start, goal, capture_diagnostics=False):
        return self.plan_hgb(start, goal, capture_diagnostics=capture_diagnostics)

    def plan_ultra(self, start, goal, capture_diagnostics=False):
        return self.plan_hgb(start, goal, capture_diagnostics=capture_diagnostics)

    def plan_hgb(self, start, goal, capture_diagnostics=False):
        if not self.use_native:
            return self._plan_python_vectorized(start, goal, capture_diagnostics=capture_diagnostics)

        t0 = time.perf_counter()
        ext_prior = self._compute_calibrated_cnn_prior(start, goal)
        ext_ptr = (
            ext_prior.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if ext_prior is not None
            else ctypes.cast(0, ctypes.POINTER(ctypes.c_double))
        )

        max_pts = max(512, (self.H + self.W) * 4)
        max_portals = max(128, (self.H + self.W) * 2)
        out_path = np.empty((max_pts, 2), dtype=np.float64)
        out_num_pts = ctypes.c_int(0)
        out_raw_len = ctypes.c_int(0)
        out_exp = ctypes.c_int(0)
        out_p_cnt = ctypes.c_int(0)
        out_cbf = ctypes.c_int(0)

        if capture_diagnostics:
            out_corr = np.empty((self.H, self.W), dtype=np.float64)
            out_vis = np.empty((self.H, self.W), dtype=np.uint8)
            out_portals = np.empty((max_portals, 2), dtype=np.float64)
            corr_ptr = out_corr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            vis_ptr = out_vis.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            port_ptr = out_portals.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            out_corr = None
            out_vis = None
            out_portals = None
            corr_ptr = ctypes.cast(0, ctypes.POINTER(ctypes.c_double))
            vis_ptr = ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8))
            port_ptr = ctypes.cast(0, ctypes.POINTER(ctypes.c_double))

        status = _HGB_LIB.hgb_plan_kernel(
            self.heights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.slopes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.t_classes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ext_ptr,
            ctypes.c_int(self.H),
            ctypes.c_int(self.W),
            ctypes.c_int(int(start[0])),
            ctypes.c_int(int(start[1])),
            ctypes.c_int(int(goal[0])),
            ctypes.c_int(int(goal[1])),
            ctypes.c_double(self.ssf),
            ctypes.c_double(self.ltr_limit),
            ctypes.c_double(self.z_clearance),
            ctypes.c_double(self.w_base),
            ctypes.c_double(self.lambda_corr),
            ctypes.c_int(self.puct_sims),
            corr_ptr,
            vis_ptr,
            out_path.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(max_pts),
            ctypes.byref(out_num_pts),
            ctypes.byref(out_raw_len),
            ctypes.byref(out_exp),
            ctypes.byref(out_p_cnt),
            port_ptr,
            ctypes.c_int(max_portals),
            ctypes.byref(out_cbf),
        )
        t_plan_ms = (time.perf_counter() - t0) * 1000.0

        if status != 1 or out_num_pts.value == 0:
            return {
                "path": None,
                "planning_time_ms": t_plan_ms,
                "expanded_nodes": int(out_exp.value),
                "success": False,
                "fallback_triggered": False,
            }

        n_pts = int(out_num_pts.value)
        path_list = [(float(out_path[i, 0]), float(out_path[i, 1])) for i in range(n_pts)]
        energy, smoothness = self.evaluate_path_metrics(path_list)

        res = {
            "path": path_list,
            "planning_time_ms": t_plan_ms,
            "expanded_nodes": int(out_exp.value),
            "raw_len": int(out_raw_len.value),
            "energy_joules": energy,
            "smoothness": smoothness,
            "num_waypoints": n_pts,
            "portal_count": int(out_p_cnt.value),
            "fallback_triggered": False,
            "used_cbf_escape": bool(out_cbf.value),
            "success": True,
        }
        if capture_diagnostics:
            p_copy = min(int(out_p_cnt.value), max_portals)
            res["corridor_field"] = out_corr
            res["visited_mask"] = out_vis
            res["portals"] = [(float(out_portals[i, 0]), float(out_portals[i, 1])) for i in range(p_copy)]
        return res

    _DIRS_8 = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (1, 1, 1.41421356237), (-1, -1, 1.41421356237),
        (1, -1, 1.41421356237), (-1, 1, 1.41421356237),
    )
    _OPP_DIR_8 = (1, 0, 3, 2, 5, 4, 7, 6)
    _SOIL_LUT = np.array([1.0, 1.2, 2.2, 8.5, 4.0, 2.5, 6.0, 1.5, 45.0, 2.0], dtype=np.float64)

    @staticmethod
    def _minetti_poly(sp):
        """Đa thức bậc 5 Minetti biểu diễn cơ năng theo độ dốc dọc mặt tiếp xúc s_pitch."""
        sp2 = sp * sp
        sp3 = sp2 * sp
        sp4 = sp2 * sp2
        sp5 = sp4 * sp
        return 155.4 * sp5 - 30.4 * sp4 - 43.3 * sp3 + 46.3 * sp2 + 19.5 * sp + 3.6

    @staticmethod
    def _llround_scalar(x):
        return int(math.floor(x + 0.5)) if x >= 0.0 else int(math.ceil(x - 0.5))

    @staticmethod
    def _llround_arr(x):
        # Match std::llround (half away from zero) so grid ray discretization agrees 1:1 with C++
        return np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int32)

    @staticmethod
    def _box_sum_div_1d(arr, axis, radius=2):
        # Tổng dịch trực tiếp 2*radius+1 lát để tránh sai số triệt tiêu khi tích lũy cumsum qua mảng lớn
        n = arr.shape[axis]
        acc = np.zeros_like(arr, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        for d in range(-radius, radius + 1):
            src_lo, src_hi = max(0, d), min(n, n + d)
            dst_lo, dst_hi = max(0, -d), min(n, n - d)
            if axis == 1:
                acc[:, dst_lo:dst_hi] += arr[:, src_lo:src_hi]
            else:
                acc[dst_lo:dst_hi, :] += arr[src_lo:src_hi, :]
            cnt[dst_lo:dst_hi] += 1.0
        return (acc / cnt[None, :]) if axis == 1 else (acc / cnt[:, None])

    def _compute_soil_and_ltr_fields(self):
        # Khóa cứng ô ngập nước sâu (t_classes == 8) vượt ltr_limit để ngăn rốn trũng thủy văn ở cả 4 tầng
        ltr_map = np.radians(self.slopes) / self.ssf
        ltr_map = np.where(self.t_classes == 8, np.maximum(ltr_map, self.ltr_limit + 0.35), ltr_map)

        cls_clipped = np.where((self.t_classes >= 0) & (self.t_classes <= 8), self.t_classes, 9)
        bekker_raw = self._SOIL_LUT[cls_clipped]
        tmp_h = self._box_sum_div_1d(bekker_raw, axis=1, radius=2)
        bekker_smooth = self._box_sum_div_1d(tmp_h, axis=0, radius=2)
        sf_map = np.where(self.t_classes == 3, 1.6, 1.0)
        return ltr_map, bekker_smooth, sf_map

    def _tier1_geodesic_diffusion_py(self, sr, sc, gr, gc, ltr_map, bekker_smooth, sf_map, ext_prior):
        H, W = self.H, self.W
        max_dim = max(H, W)
        cs = 4 if max_dim <= 64 else max(4, max_dim // 32)
        CH = (H + cs - 1) // cs
        CW = (W + cs - 1) // cs

        pad_h = CH * cs - H
        pad_w = CW * cs - W
        valid = np.zeros((CH * cs, CW * cs), dtype=bool)
        valid[:H, :W] = True
        ltr_pad = np.pad(ltr_map, ((0, pad_h), (0, pad_w)), mode="constant")
        h_pad = np.pad(self.heights, ((0, pad_h), (0, pad_w)), mode="constant")
        sf_pad = np.pad(sf_map, ((0, pad_h), (0, pad_w)), mode="constant")
        bk_pad = np.pad(bekker_smooth, ((0, pad_h), (0, pad_w)), mode="constant")

        def _blk_reshape(arr):
            return arr.reshape(CH, cs, CW, cs).transpose(0, 2, 1, 3).reshape(CH, CW, cs * cs)

        valid_b = _blk_reshape(valid)
        ltr_b = _blk_reshape(ltr_pad)
        h_b = _blk_reshape(h_pad)
        sf_b = _blk_reshape(sf_pad)
        bk_b = _blk_reshape(bk_pad)

        cnt = valid_b.sum(axis=2)
        sum_l = np.where(valid_b, ltr_b, 0.0).sum(axis=2)
        sum_l2 = np.where(valid_b, ltr_b * ltr_b, 0.0).sum(axis=2)
        mean_l = np.where(cnt > 0, sum_l / np.maximum(cnt, 1), 0.0)
        var_l = np.maximum(0.0, np.where(cnt > 1, sum_l2 / np.maximum(cnt, 1) - mean_l * mean_l, 0.0))
        c_tci = np.where(cnt > 1, np.sqrt(var_l), 0.0)

        feas_b = valid_b & (ltr_b <= self.ltr_limit)
        imp_cell = np.where(feas_b, 3.6 * sf_b + 0.35 * bk_b + 8.0 * ltr_b * ltr_b, 1e12)
        best_idx = np.argmin(imp_cell, axis=2)
        cr_g, cc_g = np.meshgrid(np.arange(CH), np.arange(CW), indexing="ij")
        c_imp = imp_cell[cr_g, cc_g, best_idx].copy()
        fallback_h = self.heights[np.minimum(H - 1, cr_g * cs), np.minimum(W - 1, cc_g * cs)]
        c_h = np.where(np.any(feas_b, axis=2), h_b[cr_g, cc_g, best_idx], fallback_h)

        s_cr, s_cc = sr // cs, sc // cs
        g_cr, g_cc = gr // cs, gc // cs
        c_imp[s_cr, s_cc] = min(c_imp[s_cr, s_cc], 4.0)
        c_imp[g_cr, g_cc] = min(c_imp[g_cr, g_cc], 4.0)

        def coarse_dijkstra(root_rc, reverse=False):
            dist = np.full((CH, CW), 1e15, dtype=np.float64)
            dist[root_rc] = 0.0
            pq = [(0.0, root_rc[0], root_rc[1])]
            while pq:
                d_u, ur, uc = heapq.heappop(pq)
                if d_u > dist[ur, uc]:
                    continue
                for dr, dc, ds0 in self._DIRS_8:
                    vr, vc = ur + dr, uc + dc
                    if 0 <= vr < CH and 0 <= vc < CW and c_imp[vr, vc] < 1e11:
                        ds = ds0 * cs
                        dz = (c_h[ur, uc] - c_h[vr, vc]) if reverse else (c_h[vr, vc] - c_h[ur, uc])
                        sp = max(-0.35, min(0.35, float(dz / ds)))
                        cm = max(0.5, min(30.0, float(self._minetti_poly(sp))))
                        imp_ref = float(c_imp[ur, uc] if reverse else c_imp[vr, vc])
                        nd = d_u + (cm + 0.5 * imp_ref) * ds
                        if nd < dist[vr, vc]:
                            dist[vr, vc] = nd
                            heapq.heappush(pq, (nd, vr, vc))
            return dist

        df_c = coarse_dijkstra((s_cr, s_cc), reverse=False)
        db_c = coarse_dijkstra((g_cr, g_cc), reverse=True)
        both_ok = (df_c < 1e14) & (db_c < 1e14)
        phi_c = np.where(both_ok, df_c + db_c, 1e15)
        phi_min = float(np.min(phi_c))

        sys1_prob = np.zeros((H, W), dtype=np.float64)
        if phi_min < 1e14:
            scale_tau = 1.0 if max_dim <= 64 else (1.0 / math.sqrt(max_dim / 64.0))
            tau = max(35.0, 0.18 * scale_tau * phi_min)
            phi_clamped = np.minimum(phi_c, phi_min + 6.0 * tau)

            gy = (np.arange(H, dtype=np.float64) + 0.5) / cs - 0.5
            r0 = np.clip(np.floor(gy).astype(np.int32), 0, CH - 1)
            r1 = np.minimum(CH - 1, r0 + 1)
            wy = np.clip(gy - r0, 0.0, 1.0)[:, None]

            gx = (np.arange(W, dtype=np.float64) + 0.5) / cs - 0.5
            c0 = np.clip(np.floor(gx).astype(np.int32), 0, CW - 1)
            c1 = np.minimum(CW - 1, c0 + 1)
            wx = np.clip(gx - c0, 0.0, 1.0)[None, :]

            if max_dim <= 64:
                p00 = phi_clamped[r0[:, None], c0[None, :]]
                p10 = phi_clamped[r1[:, None], c0[None, :]]
                p01 = phi_clamped[r0[:, None], c1[None, :]]
                p11 = phi_clamped[r1[:, None], c1[None, :]]
                phi_interp = (1.0 - wy) * ((1.0 - wx) * p00 + wx * p01) + wy * ((1.0 - wx) * p10 + wx * p11)
                prob = np.exp(-np.maximum(0.0, phi_interp - phi_min) / tau)
            else:
                prob_c = np.exp(-np.maximum(0.0, phi_clamped - phi_min) / tau)
                prob = (
                    (1.0 - wy) * ((1.0 - wx) * prob_c[r0[:, None], c0[None, :]] + wx * prob_c[r0[:, None], c1[None, :]])
                    + wy * ((1.0 - wx) * prob_c[r1[:, None], c0[None, :]] + wx * prob_c[r1[:, None], c1[None, :]])
                )

            if ext_prior is not None:
                prob = 0.75 * prob + 0.25 * np.clip(ext_prior, 0.0, 1.0)

            term_mask = np.zeros((H, W), dtype=bool)
            term_mask[sr, sc] = True
            term_mask[gr, gc] = True
            sys1_prob = np.where((ltr_map <= self.ltr_limit) | term_mask, prob, 0.0)

        return sys1_prob, df_c, db_c, c_tci, cs, CH, CW

    def _tier2_saddle_portals_puct_py(self, sr, sc, gr, gc, ltr_map, bekker_smooth, sf_map, sys1_prob, c_tci, cs, CH, CW):
        H, W = self.H, self.W
        max_dim = max(H, W)
        BS = 16 if max_dim <= 64 else max(16, max_dim // 6)
        cell_imp = np.where(
            ltr_map <= self.ltr_limit,
            3.6 * sf_map + 0.5 * bekker_smooth + 25.0 * ltr_map * ltr_map,
            1e12,
        )

        portals = [(sr, sc), (gr, gc)]
        # Trích xuất cổng yên ngựa tại cực tiểu trở kháng trên các đoạn liên thông qua biên khối BS
        for br in range(BS, H, BS):
            avg_imp = 0.5 * (cell_imp[br, :] + cell_imp[br - 1, :])
            feas = (ltr_map[br, :] <= self.ltr_limit) & (ltr_map[br - 1, :] <= self.ltr_limit)
            for bc in range(0, W, BS):
                c_end = min(W, bc + BS)
                seg_start = -1
                for c in range(bc, c_end + 1):
                    ok = (c < c_end) and bool(feas[c])
                    if ok and seg_start < 0:
                        seg_start = c
                    elif not ok and seg_start >= 0:
                        best_c = seg_start + int(np.argmin(avg_imp[seg_start:c]))
                        portals.append((br, best_c))
                        seg_start = -1

        for bc in range(BS, W, BS):
            avg_imp = 0.5 * (cell_imp[:, bc] + cell_imp[:, bc - 1])
            feas = (ltr_map[:, bc] <= self.ltr_limit) & (ltr_map[:, bc - 1] <= self.ltr_limit)
            for br in range(0, H, BS):
                r_end = min(H, br + BS)
                seg_start = -1
                for r in range(br, r_end + 1):
                    ok = (r < r_end) and bool(feas[r])
                    if ok and seg_start < 0:
                        seg_start = r
                    elif not ok and seg_start >= 0:
                        best_r = seg_start + int(np.argmin(avg_imp[seg_start:r]))
                        portals.append((best_r, bc))
                        seg_start = -1

        P = len(portals)
        p_adj = [[] for _ in range(P)]
        max_p_dist = BS * 1.55
        for i, (ri, ci) in enumerate(portals):
            for j, (rj, cj) in enumerate(portals):
                if i == j:
                    continue
                d = math.hypot(rj - ri, cj - ci)
                if d > max_p_dist and i >= 2 and j >= 2:
                    continue
                if d > BS * 2.2:
                    continue

                steps = max(int(d), 4)
                ts = np.arange(1, steps + 1, dtype=np.float64) / steps
                rr = self._llround_arr(ri + ts * (rj - ri))
                cc = self._llround_arr(ci + ts * (cj - ci))
                if steps > 1 and np.any(ltr_map[rr[:-1], cc[:-1]] > self.ltr_limit):
                    continue

                p_acc = float(np.sum(sys1_prob[rr, cc]))
                rr_prev = np.empty(steps, dtype=np.int32)
                rr_prev[0] = ri
                rr_prev[1:] = rr[:-1]
                cc_prev = np.empty(steps, dtype=np.int32)
                cc_prev[0] = ci
                cc_prev[1:] = cc[:-1]
                sd = np.hypot(rr - rr_prev, cc - cc_prev)
                mv = sd > 1e-4
                if np.any(mv):
                    dz = self.heights[rr[mv], cc[mv]] - self.heights[rr_prev[mv], cc_prev[mv]]
                    sp = np.clip(dz / sd[mv], -0.35, 0.35)
                    cm = np.clip(self._minetti_poly(sp), 0.5, 30.0)
                    e_acc = float(np.sum((cm * sf_map[rr[mv], cc[mv]] + 0.35 * bekker_smooth[rr[mv], cc[mv]]) * sd[mv]))
                else:
                    e_acc = 0.0
                p_adj[i].append([j, e_acc, max(0.05, p_acc / steps)])

            sum_pr = sum(e[2] for e in p_adj[i])
            if sum_pr > 1e-9:
                for e in p_adj[i]:
                    e[2] /= sum_pr

        p_cost_to_goal = [1e15] * P
        if max_dim > 64 and P > 2:
            p_rev = [[] for _ in range(P)]
            for u in range(P):
                for to, cost, _ in p_adj[u]:
                    p_rev[to].append((u, cost))
            p_cost_to_goal[1] = 0.0
            php = [(0.0, 1)]
            while php:
                d_u, u = heapq.heappop(php)
                if d_u > p_cost_to_goal[u]:
                    continue
                for v, c_edge in p_rev[u]:
                    nd = d_u + c_edge
                    if nd < p_cost_to_goal[v]:
                        p_cost_to_goal[v] = nd
                        heapq.heappush(php, (nd, v))

        def portal_leaf_est(u):
            if max_dim > 64 and p_cost_to_goal[u] < 1e14:
                return p_cost_to_goal[u]
            ru, cu = portals[u]
            d = math.hypot(gr - ru, gc - cu)
            if d < 1e-4:
                return 0.0
            dz = float(self.heights[gr, gc] - self.heights[ru, cu])
            cm = max(0.5, min(30.0, float(self._minetti_poly(max(-0.35, min(0.35, dz / d))))))
            return d * cm

        visit_state = [0] * P
        visit_edge = [[0] * len(p_adj[i]) for i in range(P)]
        q_edge = [[e[1] + portal_leaf_est(e[0]) for e in p_adj[i]] for i in range(P)]

        c_puct = 35.0
        best_puct_path = []
        best_puct_cost = 1e15
        max_puct_depth = 24 if max_dim <= 64 else max(24, ((H // BS) + (W // BS)) * 3)
        n_sims = max(16, self.puct_sims)

        for _ in range(n_sims):
            tr_u = []
            tr_a = []
            in_path = bytearray(P)
            cur = 0
            in_path[cur] = 1
            g_acc = 0.0
            reached = False

            for _depth in range(max_puct_depth):
                if cur == 1:
                    reached = True
                    break
                if not p_adj[cur]:
                    break
                sqrt_ns = math.sqrt(visit_state[cur] + 1.0)
                best_ucb = -1e18
                best_act = -1
                for a, e in enumerate(p_adj[cur]):
                    if in_path[e[0]]:
                        continue
                    ucb = -q_edge[cur][a] + c_puct * e[2] * (sqrt_ns / (1.0 + visit_edge[cur][a]))
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_act = a
                if best_act < 0:
                    break
                tr_u.append(cur)
                tr_a.append(best_act)
                g_acc += p_adj[cur][best_act][1]
                cur = p_adj[cur][best_act][0]
                in_path[cur] = 1

            leaf_val = 0.0 if (reached or cur == 1) else (portal_leaf_est(cur) + (400.0 if not p_adj[cur] else 40.0))
            if reached and g_acc < best_puct_cost:
                best_puct_cost = g_acc
                best_puct_path = tr_u + [1]

            ret_cost = leaf_val
            for t in range(len(tr_u) - 1, -1, -1):
                u, a = tr_u[t], tr_a[t]
                ret_cost += p_adj[u][a][1]
                visit_state[u] += 1
                visit_edge[u][a] += 1
                q_edge[u][a] += (ret_cost - q_edge[u][a]) / float(visit_edge[u][a])

        corr_field = sys1_prob.copy()
        if best_puct_path:
            scale_bs = 1.0 if max_dim <= 64 else math.sqrt(BS / 16.0)
            term_mask = np.zeros((H, W), dtype=bool)
            term_mask[sr, sc] = True
            term_mask[gr, gc] = True
            feas_mask = (ltr_map <= self.ltr_limit) | term_mask

            for k in range(len(best_puct_path) - 1):
                r0, c0 = portals[best_puct_path[k]]
                r1, c1 = portals[best_puct_path[k + 1]]
                cr_mid = min(CH - 1, ((r0 + r1) // 2) // cs)
                cc_mid = min(CW - 1, ((c0 + c1) // 2) // cs)
                local_tci = float(c_tci[cr_mid, cc_mid])
                sigma = (4.5 + 14.0 * min(0.4, local_tci)) * scale_bs
                inv_2s2 = 1.0 / (2.0 * sigma * sigma)
                pad_box = int(sigma * 2.0)

                min_r = max(0, min(r0, r1) - pad_box)
                max_r = min(H - 1, max(r0, r1) + pad_box)
                min_c = max(0, min(c0, c1) - pad_box)
                max_c = min(W - 1, max(c0, c1) + pad_box)

                seg_dr = float(r1 - r0)
                seg_dc = float(c1 - c0)
                seg_l2 = seg_dr * seg_dr + seg_dc * seg_dc + 1e-9

                rr_sub = np.arange(min_r, max_r + 1, dtype=np.float64)[:, None]
                cc_sub = np.arange(min_c, max_c + 1, dtype=np.float64)[None, :]
                t_proj = np.clip(((rr_sub - r0) * seg_dr + (cc_sub - c0) * seg_dc) / seg_l2, 0.0, 1.0)
                d2 = (rr_sub - (r0 + t_proj * seg_dr)) ** 2 + (cc_sub - (c0 + t_proj * seg_dc)) ** 2
                tube = np.where(feas_mask[min_r:max_r + 1, min_c:max_c + 1], np.exp(-d2 * inv_2s2), 0.0)
                sub_view = corr_field[min_r:max_r + 1, min_c:max_c + 1]
                np.maximum(sub_view, tube, out=sub_view)

        return corr_field, portals

    def _tier3_bidirectional_soft_aph_py(
        self, sr, sc, gr, gc, ltr_map, bekker_smooth, sf_map, corr_field, df_c, db_c, cs, CH, CW, capture_diagnostics
    ):
        H, W = self.H, self.W
        N = H * W
        max_dim = max(H, W)
        eff_lam = self.lambda_corr if max_dim <= 64 else self.lambda_corr * (1.0 + 0.95 * math.log2(max_dim / 64.0))
        w_eff_arr = (self.w_base * (1.0 + eff_lam * (1.0 - corr_field) ** 2)).ravel()

        term_mask = np.zeros((H, W), dtype=bool)
        term_mask[sr, sc] = True
        term_mask[gr, gc] = True
        rr, cc = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
        r_blk = np.minimum(CH - 1, np.arange(H) // cs)
        c_blk = np.minimum(CW - 1, np.arange(W) // cs)

        # Tiền tính chi phí 8 hướng theo bản đồ để khử gọi lại hàm trên từng bước nhảy của A* song phương
        normal_ec = np.full((8, H, W), np.inf, dtype=np.float64)
        cbf_ec = np.full((8, H, W), np.inf, dtype=np.float64)
        for k, (dr, dc, ds) in enumerate(self._DIRS_8):
            r0_s, r0_e = max(0, -dr), H - max(0, dr)
            c0_s, c0_e = max(0, -dc), W - max(0, dc)
            r1_s, r1_e = max(0, dr), H + min(0, dr)
            c1_s, c1_e = max(0, dc), W + min(0, dc)

            dz = self.heights[r1_s:r1_e, c1_s:c1_e] - self.heights[r0_s:r0_e, c0_s:c0_e]
            sp = np.clip(dz / ds, -0.35, 0.35)
            cm = np.clip(self._minetti_poly(sp), 0.5, 30.0)
            ltr_dst = ltr_map[r1_s:r1_e, c1_s:c1_e]
            raw_c = (cm * sf_map[r1_s:r1_e, c1_s:c1_e] + 0.45 * bekker_smooth[r1_s:r1_e, c1_s:c1_e] + 18.0 * ltr_dst * ltr_dst) * ds
            feas = (ltr_dst <= self.ltr_limit) | term_mask[r1_s:r1_e, c1_s:c1_e]
            normal_ec[k, r0_s:r0_e, c0_s:c0_e] = np.where(feas, raw_c, np.inf)

            excess = np.maximum(0.0, ltr_dst - self.ltr_limit)
            pen_c = (2500.0 + 18000.0 * excess * excess) * ds
            cbf_ec[k, r0_s:r0_e, c0_s:c0_e] = np.where(feas, raw_c, pen_c)

        def compute_aph(tr, tc, d_coarse, allow_cbf_escape):
            d = np.hypot(tr - rr, tc - cc)
            dz = self.heights[tr, tc] - self.heights
            sp = np.clip(dz / np.maximum(d, 1e-5), -0.35, 0.35)
            cm = np.clip(self._minetti_poly(sp), 1.0, 15.0)
            unit = np.maximum(2.0, cm + 0.35 * bekker_smooth)
            h_euc = np.where(d < 1e-5, 0.0, d * unit)
            if max_dim > 64 and not allow_cbf_escape:
                h_geo = d_coarse[r_blk[:, None], c_blk[None, :]]
                h_euc = np.where(h_geo < 1e13, np.maximum(h_euc, 0.82 * np.maximum(0.0, h_geo - cs * 2.2)), h_euc)
            return h_euc.ravel()

        strides = [dr * W + dc for dr, dc, _ in self._DIRS_8]
        s_idx = sr * W + sc
        g_idx = gr * W + gc
        visited_mask = np.zeros((H, W), dtype=np.uint8) if capture_diagnostics else None
        vis_flat = visited_mask.ravel() if visited_mask is not None else None

        def run_bidirectional(allow_cbf_escape):
            ec_flat = (cbf_ec if allow_cbf_escape else normal_ec).reshape(8, N)
            aph_f = compute_aph(gr, gc, db_c, allow_cbf_escape)
            aph_b = compute_aph(sr, sc, df_c, allow_cbf_escape)

            gf = [1e18] * N
            gb = [1e18] * N
            pf = [-1] * N
            pb = [-1] * N
            cf = bytearray(N)
            cb = bytearray(N)
            gf[s_idx] = 0.0
            gb[g_idx] = 0.0
            pq_f = [(self.w_base * float(aph_f[s_idx]), 0.0, s_idx)]
            pq_b = [(self.w_base * float(aph_b[g_idx]), 0.0, g_idx)]

            mu_best = 1e18
            meet_idx = -1
            waves_collided = False
            post_meet_pops = 0
            exp_cnt = 0

            while pq_f and pq_b:
                if waves_collided:
                    post_meet_pops += 1
                    if post_meet_pops > 24 or (pq_f[0][0] + pq_b[0][0]) * 0.5 >= mu_best * 0.98:
                        break
                if len(pq_f) <= len(pq_b):
                    _, d_u, u = heapq.heappop(pq_f)
                    if cf[u] or d_u > gf[u]:
                        continue
                    cf[u] = 1
                    if vis_flat is not None:
                        vis_flat[u] |= 1
                    exp_cnt += 1
                    if cb[u]:
                        waves_collided = True
                        if gf[u] + gb[u] <= mu_best:
                            mu_best = gf[u] + gb[u]
                            meet_idx = u
                    uc = u % W
                    for k in range(8):
                        dc = self._DIRS_8[k][1]
                        if (uc == 0 and dc < 0) or (uc == W - 1 and dc > 0):
                            continue
                        v = u + strides[k]
                        if 0 <= v < N and not cf[v]:
                            ec = float(ec_flat[k, u])
                            if ec < 1e17:
                                nd = d_u + ec
                                if nd < gf[v]:
                                    gf[v] = nd
                                    pf[v] = u
                                    if gb[v] < 1e17 and nd + gb[v] < mu_best:
                                        mu_best = nd + gb[v]
                                        meet_idx = v
                                    heapq.heappush(pq_f, (nd + float(w_eff_arr[v] * aph_f[v]) * 1.001, nd, v))
                else:
                    _, d_u, u = heapq.heappop(pq_b)
                    if cb[u] or d_u > gb[u]:
                        continue
                    cb[u] = 1
                    if vis_flat is not None:
                        vis_flat[u] |= 2
                    exp_cnt += 1
                    if cf[u]:
                        waves_collided = True
                        if gf[u] + d_u <= mu_best:
                            mu_best = gf[u] + d_u
                            meet_idx = u
                    uc = u % W
                    for k in range(8):
                        dc = self._DIRS_8[k][1]
                        if (uc == 0 and dc < 0) or (uc == W - 1 and dc > 0):
                            continue
                        v = u + strides[k]
                        if 0 <= v < N and not cb[v]:
                            ec = float(ec_flat[self._OPP_DIR_8[k], v])
                            if ec < 1e17:
                                nd = d_u + ec
                                if nd < gb[v]:
                                    gb[v] = nd
                                    pb[v] = u
                                    if gf[v] < 1e17 and gf[v] + nd < mu_best:
                                        mu_best = gf[v] + nd
                                        meet_idx = v
                                    heapq.heappush(pq_b, (nd + float(w_eff_arr[v] * aph_b[v]) * 1.001, nd, v))

            if meet_idx < 0:
                return None, exp_cnt
            raw_nodes = []
            c = meet_idx
            while c != -1:
                raw_nodes.append(c)
                c = pf[c]
            raw_nodes.reverse()
            c = pb[meet_idx]
            while c != -1:
                raw_nodes.append(c)
                c = pb[c]
            return (raw_nodes if raw_nodes else None), exp_cnt

        raw_nodes, exp_nodes = run_bidirectional(False)
        used_cbf = False
        if raw_nodes is None:
            raw_nodes, exp_cbf = run_bidirectional(True)
            exp_nodes += exp_cbf
            used_cbf = True
        return raw_nodes, exp_nodes, used_cbf, visited_mask

    @classmethod
    def _is_segment_safe_25d_py(cls, heights, ltr_map, H, W, r1, c1, r2, c2, sr, sc, gr, gc, ltr_limit, z_clearance):
        d2d = math.hypot(r2 - r1, c2 - c1)
        if d2d < 1e-6:
            if not (0 <= r1 < H and 0 <= c1 < W):
                return False
            return (r1 == sr and c1 == sc) or (r1 == gr and c1 == gc) or (ltr_map[r1, c1] <= ltr_limit)

        steps = max(int(math.ceil(d2d * 2.5)), 3)
        ts = np.arange(steps + 1, dtype=np.float64) / steps
        rr = cls._llround_arr(r1 + ts * (r2 - r1))
        cc = cls._llround_arr(c1 + ts * (c2 - c1))
        if np.any((rr < 0) | (rr >= H) | (cc < 0) | (cc >= W)):
            return False
        is_term = ((rr == sr) & (cc == sc)) | ((rr == gr) & (cc == gc))
        if np.any((~is_term) & (ltr_map[rr, cc] > ltr_limit)):
            return False
        z_exp = heights[r1, c1] + ts * (heights[r2, c2] - heights[r1, c1])
        return bool(np.all(np.abs(heights[rr, cc] - z_exp) <= z_clearance))

    @classmethod
    def _batch_segments_safe_25d_py(cls, heights, ltr_map, H, W, r1s, c1s, r2, c2, sr, sc, gr, gc, ltr_limit, z_clearance):
        """Kiểm định đồng thời tập đoạn (i -> j) với số bước chuẩn hóa ceil(d*2.5) theo từng cạnh."""
        M = len(r1s)
        d2d = np.hypot(r2 - r1s, c2 - c1s)
        safe = np.zeros(M, dtype=bool)
        zero_mask = d2d < 1e-6
        if np.any(zero_mask):
            zr = r1s[zero_mask]
            zc = c1s[zero_mask]
            z_term = ((zr == sr) & (zc == sc)) | ((zr == gr) & (zc == gc))
            safe[zero_mask] = z_term | (ltr_map[zr, zc] <= ltr_limit)

        nz = ~zero_mask
        if not np.any(nz):
            return safe

        r1_nz = r1s[nz].astype(np.float64)[:, None]
        c1_nz = c1s[nz].astype(np.float64)[:, None]
        steps_vec = np.maximum(np.ceil(d2d[nz] * 2.5).astype(np.int32), 3)
        s_max = int(steps_vec.max())
        k_grid = np.arange(s_max + 1, dtype=np.float64)[None, :]
        active = k_grid <= steps_vec[:, None]
        t_mat = np.minimum(1.0, k_grid / steps_vec[:, None])

        rr = np.clip(cls._llround_arr(r1_nz + t_mat * (r2 - r1_nz)), 0, H - 1)
        cc = np.clip(cls._llround_arr(c1_nz + t_mat * (c2 - c1_nz)), 0, W - 1)
        is_term = ((rr == sr) & (cc == sc)) | ((rr == gr) & (cc == gc))
        ltr_bad = active & (~is_term) & (ltr_map[rr, cc] > ltr_limit)
        z1_nz = heights[r1s[nz], c1s[nz]][:, None]
        z_exp = z1_nz + t_mat * (heights[r2, c2] - z1_nz)
        z_bad = active & (np.abs(heights[rr, cc] - z_exp) > z_clearance)
        safe[nz] = ~(np.any(ltr_bad, axis=1) | np.any(z_bad, axis=1))
        return safe

    @classmethod
    def _batch_segments_exact_energy_py(cls, heights, t_classes, H, W, r1s, c1s, r2, c2):
        d2d = np.hypot(r2 - r1s, c2 - c1s)
        steps_vec = np.maximum((d2d * 2.0).astype(np.int32), 2)
        s_max = int(steps_vec.max())
        k_grid = np.arange(s_max, dtype=np.float64)[None, :]
        inv_s = 1.0 / (steps_vec[:, None] - 1.0)
        t_mat = np.minimum(1.0, k_grid * inv_s)

        r1_f = r1s.astype(np.float64)[:, None]
        c1_f = c1s.astype(np.float64)[:, None]
        rr = np.clip(cls._llround_arr(r1_f + t_mat * (r2 - r1_f)), 0, H - 1)
        cc = np.clip(cls._llround_arr(c1_f + t_mat * (c2 - c1_f)), 0, W - 1)

        dr = rr[:, 1:] - rr[:, :-1]
        dc = cc[:, 1:] - cc[:, :-1]
        sd = np.hypot(dr, dc)
        step_active = (np.arange(1, s_max, dtype=np.int32)[None, :] < steps_vec[:, None]) & (sd >= 1e-4)

        rr_cur, cc_cur = rr[:, 1:], cc[:, 1:]
        rr_prv, cc_prv = rr[:, :-1], cc[:, :-1]
        dz = heights[rr_cur, cc_cur] - heights[rr_prv, cc_prv]
        sp = np.where(step_active, dz / np.maximum(sd, 1e-9), 0.0)
        cm = np.maximum(0.5, cls._minetti_poly(sp))
        sf = np.where(t_classes[rr_cur, cc_cur] == 3, 1.6, 1.0)
        return np.sum(np.where(step_active, cm * sd * sf, 0.0), axis=1)

    def _tier4_geodesic_dp_and_chaikin_py(self, raw_nodes, sr, sc, gr, gc, ltr_map, eff_ltr_lim):
        H, W = self.H, self.W
        k_raw = len(raw_nodes)
        if k_raw <= 160:
            dp_nodes = raw_nodes
        else:
            stride_dp = max(2, k_raw // 120)
            dp_nodes = [raw_nodes[0]]
            last_idx = 0
            for idx in range(1, k_raw - 1):
                rp, cp = divmod(raw_nodes[idx - 1], W)
                rc, cc = divmod(raw_nodes[idx], W)
                rn, cn = divmod(raw_nodes[idx + 1], W)
                dir_changed = ((rc - rp) != (rn - rc)) or ((cc - cp) != (cn - cc))
                if (idx - last_idx >= stride_dp) or (dir_changed and idx - last_idx >= 3):
                    r0, c0 = divmod(raw_nodes[last_idx], W)
                    if self._is_segment_safe_25d_py(
                        self.heights, ltr_map, H, W, r0, c0, rc, cc, sr, sc, gr, gc, eff_ltr_lim, self.z_clearance
                    ):
                        dp_nodes.append(raw_nodes[idx])
                    else:
                        dp_nodes.extend(raw_nodes[last_idx + 1:idx + 1])
                    last_idx = idx
            r0, c0 = divmod(raw_nodes[last_idx], W)
            rend, cend = divmod(raw_nodes[-1], W)
            if not self._is_segment_safe_25d_py(
                self.heights, ltr_map, H, W, r0, c0, rend, cend, sr, sc, gr, gc, eff_ltr_lim, self.z_clearance
            ):
                dp_nodes.extend(raw_nodes[last_idx + 1:k_raw - 1])
            dp_nodes.append(raw_nodes[-1])

        K = len(dp_nodes)
        dp_r = np.array([p // W for p in dp_nodes], dtype=np.int32)
        dp_c = np.array([p % W for p in dp_nodes], dtype=np.int32)
        dp_cost = np.full(K, 1e18, dtype=np.float64)
        dp_prev = np.full(K, -1, dtype=np.int32)
        dp_heading = np.zeros(K, dtype=np.float64)
        dp_cost[0] = 0.0

        for j in range(1, K):
            win_start = 0 if K <= 180 else max(0, j - 96)
            cand_mask = np.zeros(j, dtype=bool)
            cand_mask[win_start:j] = True
            if win_start > 0:
                cand_mask[:win_start:16] = True
            cand_mask &= dp_cost[:j] < 1e17
            cand_is = np.nonzero(cand_mask)[0]
            if len(cand_is) == 0:
                continue

            rj, cj = int(dp_r[j]), int(dp_c[j])
            seg_ok = self._batch_segments_safe_25d_py(
                self.heights, ltr_map, H, W, dp_r[cand_is], dp_c[cand_is], rj, cj,
                sr, sc, gr, gc, eff_ltr_lim, self.z_clearance,
            )
            seg_ok |= (cand_is == j - 1)
            valid_is = cand_is[seg_ok]
            if len(valid_is) == 0:
                continue

            r_val, c_val = dp_r[valid_is], dp_c[valid_is]
            e_seg = self._batch_segments_exact_energy_py(self.heights, self.t_classes, H, W, r_val, c_val, rj, cj)
            hdg = np.arctan2((cj - c_val).astype(np.float64), (rj - r_val).astype(np.float64))
            dth = np.abs(hdg - dp_heading[valid_is])
            dth = np.where(dth > math.pi, 2.0 * math.pi - dth, dth)
            turn_pen = np.where(valid_is > 0, 8.0 * dth * dth, 0.0)
            cands = dp_cost[valid_is] + e_seg + turn_pen + 0.15
            best_pos = int(np.argmin(cands))
            dp_cost[j] = float(cands[best_pos])
            dp_prev[j] = int(valid_is[best_pos])
            dp_heading[j] = float(hdg[best_pos])

        poly = []
        cur = K - 1
        while cur != -1:
            poly.append((float(dp_r[cur]), float(dp_c[cur])))
            cur = int(dp_prev[cur])
        poly.reverse()

        for _ in range(2):
            if len(poly) < 3:
                break
            nxt = [poly[0]]
            for i in range(1, len(poly) - 1):
                p_prev = nxt[-1]
                p_curr = poly[i]
                p_next = poly[i + 1]
                smoothed = False
                for a in (0.25, 0.15, 0.08):
                    q0_r = (1.0 - a) * p_curr[0] + a * p_prev[0]
                    q0_c = (1.0 - a) * p_curr[1] + a * p_prev[1]
                    q1_r = (1.0 - a) * p_curr[0] + a * p_next[0]
                    q1_c = (1.0 - a) * p_curr[1] + a * p_next[1]
                    r0i, c0i = self._llround_scalar(q0_r), self._llround_scalar(q0_c)
                    r1i, c1i = self._llround_scalar(q1_r), self._llround_scalar(q1_c)
                    if self._is_segment_safe_25d_py(
                        self.heights, ltr_map, H, W, r0i, c0i, r1i, c1i,
                        sr, sc, gr, gc, eff_ltr_lim, self.z_clearance + 0.04,
                    ):
                        nxt.append((q0_r, q0_c))
                        nxt.append((q1_r, q1_c))
                        smoothed = True
                        break
                if not smoothed:
                    nxt.append(p_curr)
            nxt.append(poly[-1])
            poly = nxt

        return poly

    def _plan_python_vectorized(self, start, goal, capture_diagnostics=False):
        """
        Bản cài đặt thuần Python + Vectorized NumPy tương đương 1:1 với 4 tầng HMCTD-A*_v2,
        dùng để tách bạch đóng góp thuật toán khỏi tăng tốc C++17 SIMD.
        """
        t0 = time.perf_counter()
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])
        ext_prior = self._compute_calibrated_cnn_prior(start, goal)

        ltr_map, bekker_smooth, sf_map = self._compute_soil_and_ltr_fields()
        sys1_prob, df_c, db_c, c_tci, cs, CH, CW = self._tier1_geodesic_diffusion_py(
            sr, sc, gr, gc, ltr_map, bekker_smooth, sf_map, ext_prior
        )
        corr_field, portals = self._tier2_saddle_portals_puct_py(
            sr, sc, gr, gc, ltr_map, bekker_smooth, sf_map, sys1_prob, c_tci, cs, CH, CW
        )
        raw_nodes, exp_nodes, used_cbf, visited_mask = self._tier3_bidirectional_soft_aph_py(
            sr, sc, gr, gc, ltr_map, bekker_smooth, sf_map, corr_field, df_c, db_c, cs, CH, CW, capture_diagnostics
        )
        if not raw_nodes:
            return {
                "path": None,
                "planning_time_ms": (time.perf_counter() - t0) * 1000.0,
                "expanded_nodes": int(exp_nodes),
                "success": False,
                "fallback_triggered": False,
            }

        eff_ltr_lim = 1.50 if used_cbf else self.ltr_limit
        poly = self._tier4_geodesic_dp_and_chaikin_py(raw_nodes, sr, sc, gr, gc, ltr_map, eff_ltr_lim)
        t_plan_ms = (time.perf_counter() - t0) * 1000.0
        energy, smoothness = self.evaluate_path_metrics(poly)

        res = {
            "path": poly,
            "planning_time_ms": t_plan_ms,
            "expanded_nodes": int(exp_nodes),
            "raw_len": len(raw_nodes),
            "energy_joules": energy,
            "smoothness": smoothness,
            "num_waypoints": len(poly),
            "portal_count": len(portals),
            "fallback_triggered": False,
            "used_cbf_escape": bool(used_cbf),
            "success": True,
        }
        if capture_diagnostics:
            max_portals = max(128, (self.H + self.W) * 2)
            res["corridor_field"] = corr_field
            res["visited_mask"] = visited_mask
            res["portals"] = [(float(pr), float(pc)) for pr, pc in portals[:max_portals]]
        return res


HMCTDAStarV2Planner = HGBAStarPlanner
HMCTDUltraPlanner = HGBAStarPlanner
