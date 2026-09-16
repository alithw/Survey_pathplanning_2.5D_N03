# ==============================================================================
# EDGE-HYBRID NMPC: BỘ ĐIỀU KHIỂN TỐI ƯU NHÚNG CHO ROBOT TỰ HÀNH 2.5D
# ==============================================================================

import casadi as ca
import numpy as np
import time
import os
import torch
from scipy.interpolate import interp1d

from slip_lut import SlipLUT
from online_slip_adapter import OnlineSlipAdapter

N = 20          
dt = 0.1        
V_MAX = 2.0     
V_MIN = 0.0     
W_MAX = 1.0     
W_MIN = -1.0

Q_x = 40.0
Q_y = 40.0
Q_theta = 4.0
R_v = 1.0
R_w = 0.5

class EdgeHybridNMPCController:
    def __init__(self):
        self.lut = SlipLUT()
        self.adapter = OnlineSlipAdapter(learning_rate=0.10, max_deviation=0.15, forgetting_factor=0.95)
        self._build_solver()

    def _build_solver(self):
        self.opti = ca.Opti()
        self.X = self.opti.variable(3, N+1) 
        self.U = self.opti.variable(2, N)   
        self.X0 = self.opti.parameter(3)          
        self.X_ref = self.opti.parameter(3, N+1)  
        self.slip_ratio = self.opti.parameter(N)   

        for k in range(N):
            x_next = self.X[0, k] + dt * self.U[0, k] * (1.0 - self.slip_ratio[k]) * ca.cos(self.X[2, k])
            y_next = self.X[1, k] + dt * self.U[0, k] * (1.0 - self.slip_ratio[k]) * ca.sin(self.X[2, k])
            theta_next = self.X[2, k] + dt * self.U[1, k]

            self.opti.subject_to(self.X[0, k+1] == x_next)
            self.opti.subject_to(self.X[1, k+1] == y_next)
            self.opti.subject_to(self.X[2, k+1] == theta_next)

        self.opti.subject_to(self.X[:, 0] == self.X0)
        self.opti.subject_to(self.opti.bounded(V_MIN, self.U[0, :], V_MAX))
        self.opti.subject_to(self.opti.bounded(W_MIN, self.U[1, :], W_MAX))

        cost = 0
        Q = ca.diag([Q_x, Q_y, Q_theta])
        R = ca.diag([R_v, R_w])

        for k in range(N):
            state_error = self.X[:, k] - self.X_ref[:, k]
            cost += ca.mtimes([state_error.T, Q, state_error])  
            cost += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])          

        term_error = self.X[:, N] - self.X_ref[:, N]
        cost += ca.mtimes([term_error.T, Q * 5.0, term_error])

        self.opti.minimize(cost)

        p_opts = {"expand": True, "print_time": False}
        s_opts = {"max_iter": 80, "print_level": 0, "sb": "yes", "tol": 1e-3}
        self.opti.solver("ipopt", p_opts, s_opts)

    def solve(self, current_state, ref_horizon, slip_horizon, warm_u=None, warm_x=None):
        self.opti.set_value(self.X0, current_state)
        self.opti.set_value(self.X_ref, ref_horizon)
        self.opti.set_value(self.slip_ratio, slip_horizon)

        if warm_u is not None:
            self.opti.set_initial(self.U, warm_u)
        if warm_x is not None:
            self.opti.set_initial(self.X, warm_x)

        sol = self.opti.solve()
        u_opt = sol.value(self.U)
        x_opt = sol.value(self.X)
        return u_opt, x_opt

def get_actual_slip(soil_class, slope, velocity, terrain_disturbance=0.0):
    base_slip = {0: 0.05, 1: 0.08, 2: 0.15, 3: 0.40, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10}.get(int(soil_class), 0.10)
    slope_rad = np.radians(slope)
    slope_effect = 0.35 * np.sin(slope_rad)
    vel_effect = 0.05 * velocity
    actual = base_slip + slope_effect + vel_effect + terrain_disturbance
    return np.clip(actual, 0.01, 0.85)

def run_simulation(astar_path, heights_map, slopes_map, t_classes_map, mode="edge_hybrid", disturbance=False):
    controller = EdgeHybridNMPCController()
    GRID_SIZE = heights_map.shape[0]

    x_astar = astar_path[:, 1]
    y_astar = astar_path[:, 0]
    dx = np.diff(x_astar)
    dy = np.diff(y_astar)
    dist = np.sqrt(dx**2 + dy**2)
    cum_dist = np.insert(np.cumsum(dist), 0, 0)
    _, unique_idx = np.unique(cum_dist, return_index=True)
    unique_idx.sort()

    V_TARGET = 1.0
    t_astar = cum_dist[unique_idx] / V_TARGET
    total_time = t_astar[-1]
    t_ref = np.arange(0, total_time, dt)

    fx = interp1d(t_astar, x_astar[unique_idx], kind='linear', fill_value="extrapolate")
    fy = interp1d(t_astar, y_astar[unique_idx], kind='linear', fill_value="extrapolate")

    x_ref_global = fx(t_ref)
    y_ref_global = fy(t_ref)
    dx_dt = np.gradient(x_ref_global)
    dy_dt = np.gradient(y_ref_global)
    theta_ref_global = np.arctan2(dy_dt, dx_dt)

    x_ref_global = np.pad(x_ref_global, (0, N+1), 'edge')
    y_ref_global = np.pad(y_ref_global, (0, N+1), 'edge')
    theta_ref_global = np.pad(theta_ref_global, (0, N+1), 'edge')

    current_state = np.array([x_ref_global[0], y_ref_global[0], theta_ref_global[0]], dtype=float)
    history_x = [current_state[0]]
    history_y = [current_state[1]]
    history_solve_time = []
    history_tracking_errors = []
    control_energy = 0.0

    last_ref_idx = 0
    max_steps = len(t_ref) + 30
    warm_u = np.zeros((2, N))
    warm_x = np.zeros((3, N+1))

    mlp_model = None
    if mode == "hybrid_mlp":
        from pp_hybrid_nmpc import SlipMLP, train_slip_mlp
        mlp_model = SlipMLP()
        train_slip_mlp(mlp_model, epochs=10)
        mlp_model.eval()

    prev_cmd_v = 1.0
    prev_pos = current_state[:2].copy()
    goal_pt = np.array([x_ref_global[len(t_ref)-1], y_ref_global[len(t_ref)-1]])

    for step in range(max_steps):
        x_curr, y_curr = current_state[0], current_state[1]

        # Kiểm tra đến đích
        dist_to_goal = np.sqrt((x_curr - goal_pt[0])**2 + (y_curr - goal_pt[1])**2)
        if dist_to_goal < 0.80 and step >= len(t_ref) - 5:
            break

        search_start = max(0, last_ref_idx)
        search_end = min(len(t_ref), last_ref_idx + 15)
        if search_end > search_start:
            dists = (x_ref_global[search_start:search_end] - x_curr)**2 + (y_ref_global[search_start:search_end] - y_curr)**2
            closest_idx = search_start + np.argmin(dists)
        else:
            closest_idx = last_ref_idx
        last_ref_idx = closest_idx

        err = np.sqrt((x_curr - x_ref_global[closest_idx])**2 + (y_curr - y_ref_global[closest_idx])**2)
        history_tracking_errors.append(err)

        x_ref_h = x_ref_global[closest_idx : closest_idx + N + 1]
        y_ref_h = y_ref_global[closest_idx : closest_idx + N + 1]
        th_ref_h = theta_ref_global[closest_idx : closest_idx + N + 1]
        ref_matrix = np.vstack((x_ref_h, y_ref_h, th_ref_h))

        r_now = int(np.clip(y_curr, 0, GRID_SIZE-1))
        c_now = int(np.clip(x_curr, 0, GRID_SIZE-1))
        soil_now = t_classes_map[r_now, c_now]
        slope_now = slopes_map[r_now, c_now]

        disturb_val = 0.15 if (disturbance and 25 <= step <= 55) else 0.0

        if mode == "traditional":
            slips_horizon = np.full(N, 0.10, dtype=np.float32)
        elif mode == "hybrid_mlp":
            slips_horizon = np.zeros(N, dtype=np.float32)
            for k in range(N):
                rk = int(np.clip(y_ref_h[k], 0, GRID_SIZE-1))
                ck = int(np.clip(x_ref_h[k], 0, GRID_SIZE-1))
                s_k = t_classes_map[rk, ck]
                sl_k = slopes_map[rk, ck]
                in_tensor = torch.tensor([[s_k / 7.0, np.radians(sl_k), 1.0 / 2.0]], dtype=torch.float32)
                with torch.no_grad():
                    est = mlp_model(in_tensor).item()
                slips_horizon[k] = min(est, 0.40)
        elif mode == "edge_hybrid":
            actual_dx = current_state[0] - prev_pos[0]
            actual_dy = current_state[1] - prev_pos[1]
            nom_now = controller.lut.lookup_scalar(soil_now, slope_now, prev_cmd_v)
            controller.adapter.update(prev_cmd_v, actual_dx, actual_dy, dt, nom_now, step=step)

            soils_h = [t_classes_map[int(np.clip(y_ref_h[k], 0, GRID_SIZE-1)), int(np.clip(x_ref_h[k], 0, GRID_SIZE-1))] for k in range(N)]
            slopes_h = [slopes_map[int(np.clip(y_ref_h[k], 0, GRID_SIZE-1)), int(np.clip(x_ref_h[k], 0, GRID_SIZE-1))] for k in range(N)]
            vels_h = [1.0] * N
            lut_slips = controller.lut.lookup_horizon(soils_h, slopes_h, vels_h)
            slips_horizon = controller.adapter.get_adapted_horizon(lut_slips)

        t_start = time.perf_counter()
        try:
            u_opt, x_opt = controller.solve(current_state, ref_matrix, slips_horizon, warm_u, warm_x)
            t_solve = (time.perf_counter() - t_start) * 1000.0
            cmd_v = float(u_opt[0, 0])
            cmd_w = float(u_opt[1, 0])

            warm_u[:, :-1] = u_opt[:, 1:]
            warm_u[:, -1] = u_opt[:, -1]
            warm_x[:, :-1] = x_opt[:, 1:]
            warm_x[:, -1] = x_opt[:, -1]
        except Exception:
            t_solve = (time.perf_counter() - t_start) * 1000.0
            cmd_v = 0.5
            cmd_w = 0.0

        history_solve_time.append(t_solve)
        control_energy += (cmd_v**2 + 0.5 * cmd_w**2) * dt

        actual_slip = get_actual_slip(soil_now, slope_now, cmd_v, disturb_val)
        prev_pos = current_state[:2].copy()
        prev_cmd_v = cmd_v

        current_state[0] += dt * cmd_v * (1.0 - actual_slip) * np.cos(current_state[2])
        current_state[1] += dt * cmd_v * (1.0 - actual_slip) * np.sin(current_state[2])
        current_state[2] += dt * cmd_w

        history_x.append(current_state[0])
        history_y.append(current_state[1])

    final_dist = np.sqrt((current_state[0] - goal_pt[0])**2 + (current_state[1] - goal_pt[1])**2)
    success = bool(final_dist < 1.2)
    rmse = float(np.sqrt(np.mean(np.array(history_tracking_errors)**2)))
    max_cte = float(np.max(history_tracking_errors))
    mean_solve_time = float(np.mean(history_solve_time))
    max_solve_time = float(np.max(history_solve_time))

    return {
        "mode": mode,
        "rmse": rmse,
        "max_cte": max_cte,
        "mean_solve_time_ms": mean_solve_time,
        "max_solve_time_ms": max_solve_time,
        "control_frequency_hz": 1000.0 / mean_solve_time if mean_solve_time > 0 else 0,
        "energy_joules": control_energy,
        "success": success,
        "final_dist": float(final_dist),
        "history_x": history_x,
        "history_y": history_y,
        "history_errors": history_tracking_errors,
        "history_solve_time": history_solve_time
    }
