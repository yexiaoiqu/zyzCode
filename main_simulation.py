"""
无人机吊载系统Lyapunov约束自适应模型预测控制
主仿真程序

包含完整的系统建模、控制器实现、仿真和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import solve_continuous_are
import time
import warnings
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Callable
import json

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================================
# 配置和参数
# ================================================================================

@dataclass
class SystemParams:
    """系统物理参数"""
    # 无人机参数
    m_q: float = 2.0              # 无人机质量 [kg]
    g: float = 9.81               # 重力加速度 [m/s^2]
    I_xx: float = 0.0347          # x轴转动惯量 [kg*m^2]
    I_yy: float = 0.0347          # y轴转动惯量 [kg*m^2]
    I_zz: float = 0.0665          # z轴转动惯量 [kg*m^2]

    # 负载参数
    m_L: float = 0.5              # 负载质量 [kg]

    # 缆绳参数
    L: float = 1.0                # 标称缆绳长度 [m]
    k_c: float = 1000.0           # 缆绳刚度 [N/m]
    c_c: float = 10.0             # 缆绳阻尼 [Ns/m]

    # 气动阻力系数
    k_dx: float = 0.01            # x方向阻力系数
    k_dy: float = 0.01            # y方向阻力系数
    k_dz: float = 0.01            # z方向阻力系数

    # 旋翼参数
    k_T: float = 1.0              # 推力系数
    k_tau: float = 0.1            # 扭矩系数

    def to_dict(self) -> Dict:
        """将参数转换为字典"""
        return {
            'uav_mass': self.m_q,
            'payload_mass': self.m_L,
            'cable_length': self.L,
            'cable_stiffness': self.k_c,
            'cable_damping': self.c_c
        }


@dataclass
class ControllerParams:
    """MPC控制器参数"""
    # MPC参数
    dt: float = 0.05              # 采样时间 [s]
    N: int = 20                   # 预测时域
    gamma: float = 0.3            # Lyapunov衰减率

    # 权重矩阵 - 基础值
    Q_pos: np.ndarray = None      # 位置跟踪权重
    Q_vel: np.ndarray = None      # 速度权重
    R_control: np.ndarray = None  # 控制作用权重
    S_delta: np.ndarray = None    # 控制增量权重
    W_swing: float = 50.0         # 摆动惩罚权重

    # 自适应权重参数
    alpha_e: float = 2.0          # 位置误差缩放
    alpha_theta: float = 3.0      # 摆角缩放
    epsilon_e: float = 0.1        # 位置平滑
    epsilon_theta: float = 5.0    # 角度平滑 [度]

    # 约束参数
    u_min: np.ndarray = None      # 控制输入下界
    u_max: np.ndarray = None      # 控制输入上界
    alpha_max: float = 30.0       # 最大摆角 [度]
    beta_max: float = 30.0        # 最大摆角 [度]
    rho_soft: float = 1e5         # 软约束惩罚

    # 观测器参数
    L_obs: np.ndarray = None      # 观测器增益矩阵

    def __post_init__(self):
        if self.Q_pos is None:
            self.Q_pos = np.diag([10.0, 10.0, 10.0])
        if self.Q_vel is None:
            self.Q_vel = np.diag([1.0, 1.0, 1.0])
        if self.R_control is None:
            self.R_control = np.diag([0.1, 0.01, 0.01, 0.01])
        if self.S_delta is None:
            self.S_delta = np.diag([1.0, 10.0, 10.0, 10.0])
        if self.u_min is None:
            self.u_min = np.array([5.0, -5.0, -5.0, -5.0])
        if self.u_max is None:
            self.u_max = np.array([30.0, 5.0, 5.0, 5.0])
        if self.L_obs is None:
            self.L_obs = np.eye(10) * 0.1


class UAVSlungLoadSystem:
    """UAV slung-load system dynamics"""

    def __init__(self, params: SystemParams):
        self.params = params

    def dynamics(self, x: np.ndarray, u: np.ndarray,
                wind_disturbance: Callable = None,
                t: float = 0.0) -> np.ndarray:
        """
        Compute system dynamics dx/dt = f(x, u)

        State vector x:
        [p_x, p_y, p_z, v_x, v_y, v_z, phi, psi, omega_phi, omega_psi]
        where:
            p: UAV position
            v: UAV velocity
            phi, psi: payload swing angles (spherical coordinates)
            omega_phi, omega_psi: swing angular velocities
        """
        p = x[0:3]
        v = x[3:6]
        phi = x[6]
        psi = x[7]
        omega_phi = x[8]
        omega_psi = x[9]

        pL = self.payload_position(p, phi, psi)
        vL = self.payload_velocity(p, v, phi, psi, omega_phi, omega_psi)

        m_q = self.params.m_q
        m_L = self.params.m_L
        L = self.params.L
        g = self.params.g

        catenation = np.cos(phi) * np.cos(psi)
        denom = m_q + m_L * (1 - catenation**2)

        T = self.cable_tension(p, v, pL, vL, phi, psi)

        accel = np.zeros(4)

        a_x = (u[0] * np.cos(phi) * np.cos(psi) - m_L * L *
                (omega_phi**2 * np.sin(phi) +
                 omega_psi**2 * np.cos(phi)**2 * np.sin(phi) * np.cos(phi) +
                 2 * omega_phi * omega_psi * np.sin(psi) * np.cos(phi) / np.cos(psi))
        a_x -= T * np.sin(phi) * np.cos(psi) / m_q

        a_y = (u[0] * np.sin(psi) - m_L * L *
               (omega_psi**2 * np.sin(psi) * np.cos(psi) +
                omega_phi * omega_psi * np.sin(phi) * (1 + np.cos(psi)**2) / np.cos(psi))
        a_y -= T * np.sin(psi) / m_q

        a_z = -m_q * g + u[1] - T * np.cos(phi) * np.cos(psi) / m_q

        dphi = omega_phi
        dpsi = omega_psi

        dot_omega_phi = -(g / L) * np.cos(phi) * np.sin(psi) ** 2 / np.cos(psi) - \
            2 * omega_phi * omega_psi * np.sin(psi) / np.cos(psi)
        dot_omega_psi = (g / L) * np.sin(phi) - omega_phi**2 * np.sin(psi) / np.cos(psi)

        dxdt = np.array([
            v[0], v[1], v[2],
            a_x, a_y, a_z,
            dphi, dpsi,
            dot_omega_phi, dot_omega_psi
        ])

        if wind_disturbance is not None:
            wind = wind_disturbance(t)
            dxdt[3:6] += wind / (m_q + m_L)

        return dxdt

    def payload_position(self, p: np.ndarray, phi: float, psi: float) -> np.ndarray:
        """Compute payload position from UAV position and swing angles"""
        L = self.params.L
        px = p[0] + L * np.sin(phi) * np.cos(psi)
        py = p[1] + L * np.sin(psi)
        pz = p[2] - L * np.cos(phi) * np.cos(psi)
        return np.array([px, py, pz])

    def payload_velocity(self, p: np.ndarray, v: np.ndarray, phi: float, psi: float,
                       omega_phi: float, omega_psi: float) -> np.ndarray:
        """Compute payload velocity"""
        L = self.params.L
        vx = v[0] + L * omega_phi * np.cos(phi) * np.cos(psi) - \
             L * omega_psi * np.sin(phi) * np.sin(psi)
        vy = v[1] + L * omega_psi * np.cos(psi)
        vz = v[2] + L * omega_phi * np.sin(phi) * np.cos(psi)
        return np.array([vx, vy, vz])

    def cable_tension(self, p: np.ndarray, v: np.ndarray, pL: np.ndarray, vL: np.ndarray,
                     phi: float, psi: float) -> float:
        """Compute cable tension from elastic cable model"""
        L_actual = np.linalg.norm(pL - p)
        T = self.params.k_c * (L_actual - self.params.L) + \
             self.params.c_c * ((pL - p).T @ (vL - v)) / L_actual
        return max(T, 0.1)


class ConstrainedAdaptiveMPC:
    """Lyapunov-Constrained Adaptive Model Predictive Control"""

    def __init__(self, system: UAVSlungLoadSystem, params: ControllerParams):
        self.system = system
        self.p = params
        self.x_current = None
        self.u_prev = None
        self.kappa_e_history = []
        self.kappa_theta_history = []

    def reset(self):
        """Reset controller state"""
        self.x_current = None
        self.u_prev = None
        self.kappa_e_history.clear()
        self.kappa_theta_history.clear()

    def compute_adaptive_weights(self, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Compute adaptive weighting matrices based on current error"""
        p = x[0:3]
        phi = x[6]
        psi = x[7]

        e_norm = np.linalg.norm(p)
        theta_norm = np.rad2deg(np.sqrt(phi**2 + psi**2))

        kappa_e = 1.0 + self.p.alpha_e * (e_norm / (e_norm + self.p.epsilon_e))
        kappa_theta = 1.0 + self.p.alpha_theta * (theta_norm / (theta_norm + self.p.epsilon_theta))

        self.kappa_e_history.append(kappa_e)
        self.kappa_theta_history.append(kappa_theta)

        Q_adaptive = np.block([
            [kappa_e * self.p.Q_pos, np.zeros((3, 7))],
            [np.zeros((7, 3)), kappa_theta * np.diag([self.p.Q_vel[0,0], 1.0, 1.0, 1.0, W, W])]
        ])

        return Q_adaptive, kappa_e, kappa_theta

    def cost_function(self, u_seq: np.ndarray, x0: np.ndarray, Q: np.ndarray, R: np.ndarray, S: np.ndarray) -> float:
        """
        MPC cost function: J = sum_{k=0}^{N-1} x_k^T Q x_k + u_k^T R u_k + delta u_k^T S delta u_k
        """
        N = self.p.N
        cost = 0.0
        x = x0.copy()
        u_prev = self.u_prev if self.u_prev is not None else np.zeros(4)

        for i in range(N):
            u = u_seq[i*4:(i+1)*4]
            cost += x.T @ Q @ x + u.T @ R @ u + (u - u_prev).T @ S @ (u - u_prev)

            sol = solve_ivp(lambda t, x: self.system.dynamics(x, u), [0, self.p.dt], x)
            x = sol.y[:, -1]
            u_prev = u

        cost += x.T @ Q @ x
        return cost

    def lyapunov_constraint(self, u_seq: np.ndarray, x0: np.ndarray) -> float:
        """
        Lyapunov stability constraint: V(x_N) <= V(x_0)
        """
        N = self.p.N
        x = x0.copy()

        for i in range(N):
            u = u_seq[i*4:(i+1)*4]
            sol = solve_ivp(lambda t, x: self.system.dynamics(x, u), [0, self.p.dt], x)
            x = sol.y[:, -1]

        V0 = self.lyapunov_function(x0)
        Vf = self.lyapunov_function(x)
        return Vf - self.p.gamma * V0

    def lyapunov_function(self, x: np.ndarray) -> float:
        """Lyapunov candidate function based on energy"""
        p = x[0:3]
        v = x[3:6]
        phi = x[6]
        psi = x[7]
        omega_phi = x[8]
        omega_psi = x[9]

        m_q = self.system.params.m_q
        m_L = self.system.params.m_L
        L = self.system.params.L
        g = self.system.params.g

        KE = 0.5 * m_q * (v[0]**2 + v[1]**2 + v[2]**2)
        KE += 0.5 * m_L * (
            (v[0] + L * omega_phi * np.cos(phi) * np.cos(psi) -
             L * omega_psi * np.sin(phi) * np.sin(psi))**2 +
            (v[1] + L * omega_psi * np.cos(psi))**2 +
            (v[2] + L * omega_phi * np.sin(phi) * np.cos(psi))**2
        )

        PE = m_L * g * (p[2] - L * np.cos(phi) * np.cos(psi))

        return KE + PE

    def solve(self, x0: np.ndarray) -> np.ndarray:
        """Solve MPC optimization problem and return first control input"""
        Q, kappa_e, kappa_theta = self.compute_adaptive_weights(x0)
        R = self.p.R_control
        S = self.p.S_delta

        if self.u_prev is None:
            u_guess = np.tile(np.array([self.system.params.m_q * self.system.params.g, 0, 0, 0]), self.p.N)
        else:
            u_guess = np.tile(self.u_prev, self.p.N)

        bounds = [ (self.p.u_min[i], self.p.u_max[i]) for i in range(4) ] * self.p.N

        nlc = NonlinearConstraint(
            lambda u: self.lyapunov_constraint(u, x0),
            -np.inf, 0.0
        )

        result = minimize(
            lambda u: self.cost_function(u, x0, Q, R, S),
            u_guess,
            bounds=bounds,
            constraints=[nlc],
            method='SLSQP',
            options={'maxiter': 100, 'disp': False}
        )

        if not result.success:
            print(f"MPC optimization warning: {result.message}")

        u_opt = result.x
        u_first = u_opt[0:4]
        self.u_prev = u_first
        self.x_current = x0

        return u_first


class Simulator:
    """Closed-loop simulator"""

    def __init__(self, system: UAVSlungLoadSystem, controller: ConstrainedAdaptiveMPC,
                 params: ControllerParams):
        self.system = system
        self.controller = controller
        self.p = params

    def simulate(self, T_sim: float,
                trajectory_type: str = 'step',
                trajectory_params: Dict = None,
                wind_disturbance: Callable = None,
                mass_change_event: Tuple[float, float] = None) -> Dict:
        """
        Run closed-loop simulation

        Parameters:
        -----------
        T_sim : float
            Total simulation time [s]
        trajectory_type : str
            'step' or 'circle' or 'figure-8'
        trajectory_params : dict
            Parameters for reference trajectory
        wind_disturbance : callable
            Function returning wind vector at time t: wind_disturbance(t) -> [wx, wy, wz]
        mass_change_event : tuple (t_change, m_L_new)
            Sudden mass change event at time t_change to new mass

        Returns:
        --------
        results : dict
            Dictionary containing time, state, control, reference, solve time
        """
        N_steps = int(T_sim / self.p.dt)
        t = np.linspace(0, T_sim, N_steps + 1)
        x = np.zeros((N_steps + 1, 10))
        u = np.zeros((N_steps, 4))
        x_ref = np.zeros((N_steps + 1, 3))
        solve_time = np.zeros(N_steps)

        x0 = self.get_initial_state(trajectory_type, trajectory_params)
        x[0] = x0

        if mass_change_event is not None:
            t_change, m_L_new = mass_change_event
            idx_change = np.argmin(np.abs(t - t_change))

        for k in range(N_steps):
            ref = self.reference_trajectory(t[k], trajectory_type, trajectory_params)
            x_ref[k] = ref[0:3]

            x_current = x[k].copy()
            x_current[0:3] -= ref[0:3]

            t_start = time.time()
            u[k] = self.controller.solve(x_current)
            solve_time[k] = time.time() - t_start

            t_span = [t[k], t[k+1]]
            sol = solve_ivp(
                lambda ti, xi: self.system.dynamics(
                    xi, u[k],
                    wind_disturbance if wind_disturbance is not None else None,
                    ti
                ),
                t_span, x[k]
            )

            x[k+1] = sol.y[:, -1]

            if mass_change_event is not None and k == idx_change:
                self.system.params.m_L = m_L_new

        x_ref[-1] = self.reference_trajectory(t[-1], trajectory_type, trajectory_params)[0:3]

        controller_metrics = {
            'kappa_e_history': self.controller.kappa_e_history,
            'kappa_theta_history': self.controller.kappa_theta_history
        }

        return {
            'time': t,
            'state': x,
            'control': u,
            'reference': x_ref,
            'solve_time': solve_time,
            'controller_metrics': controller_metrics
        }

    def get_initial_state(self, trajectory_type: str, trajectory_params: Dict) -> np.ndarray:
        """Get initial state"""
        if trajectory_type == 'step':
            p_initial = trajectory_params.get('p_initial', np.zeros(3))
            return np.array([
                p_initial[0], p_initial[1], p_initial[2],
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ])
        else:
            p0 = self.reference_trajectory(0.0, trajectory_type, trajectory_params)[0:3]
            return np.array([
                p0[0], p0[1], p0[2],
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ])

    def reference_trajectory(self, t: float, trajectory_type: str, params: Dict) -> np.ndarray:
        """Generate reference trajectory"""
        if trajectory_type == 'step':
            p_initial = params.get('p_initial', np.array([0.0, 0.0, 5.0]))
            p_final = params.get('p_final', np.array([5.0, 3.0, 8.0]))
            t_step = params.get('t_step', 2.0)

            if t < t_step:
                return np.array([
                    p_initial[0] + (p_final[0] - p_initial[0]) * (t / t_step),
                    p_initial[1] + (p_final[1] - p_initial[1]) * (t / t_step),
                    p_initial[2] + (p_final[2] - p_initial[2]) * (t / t_step),
                    (p_final[0] - p_initial[0]) / t_step,
                    (p_final[1] - p_initial[1]) / t_step,
                    (p_final[2] - p_initial[2]) / t_step
                ])
            else:
                return np.array([
                    p_final[0], p_final[1], p_final[2], 0.0, 0.0, 0.0
                ])

        elif trajectory_type == 'circle':
            radius = params.get('radius', 4.0)
            omega = params.get('omega', 0.2)
            x = radius * np.cos(omega * t)
            y = radius * np.sin(omega * t)
            z = 5.0
            vx = -radius * omega * np.sin(omega * t)
            vy = radius * omega * np.cos(omega * t)
            vz = 0.0
            return np.array([x, y, z, vx, vy, vz])

        elif trajectory_type == 'figure-8':
            A = params.get('amplitude', 4.0)
            freq = params.get('frequency', 0.2)
            x = A * np.sin(2 * np.pi * freq * t)
            y = A * np.sin(4 * np.pi * freq * t) / 2
            z = 5.0 + 0.5 * np.sin(2 * np.pi * freq * t)
            vx = 2 * np.pi * freq * A * np.cos(2 * np.pi * freq * t)
            vy = 2 * np.pi * freq * 2 * A * np.cos(4 * np.pi * freq * t) / 2
            vz = 2 * np.pi * freq * 0.5 * np.cos(2 * np.pi * freq * t)
            return np.array([x, y, z, vx, vy, vz])

        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")


class PerformanceAnalyzer:
    """Performance analysis and visualization"""

    @staticmethod
    def calculate_metrics(results: Dict) -> Dict:
        """Calculate performance metrics from simulation results"""
        time = results['time']
        x = results['state']
        u = results['control']
        x_ref = results['reference']

        e_pos = x[:, 0:3] - x_ref
        rmse_tracking = np.sqrt(np.mean(np.sum(e_pos**2, axis=1)))
        max_tracking_error = np.max(np.linalg.norm(e_pos, axis=1))

        phi = x[:, 6]
        psi = x[:, 7]
        max_swing_angle = np.max(np.rad2deg(np.maximum(np.abs(phi), np.abs(psi))))

        settled_idx = np.where(np.linalg.norm(e_pos, axis=1) < 0.1)[0]
        settling_time = time[settled_idx[0]] if len(settled_idx) > 0 else time[-1]

        control_effort = np.sum(np.sum(u**2, axis=1)) * results['control'].shape[0] * self.p.dt if 'p' in locals() else np.sum(np.sum(u**2, axis=1))
        avg_solve_time = np.mean(results['solve_time']) * 1000
        max_solve_time = np.max(results['solve_time']) * 1000

        feasible = np.all([np.all(u[:, 0] >= self.p.u_min[0]),
                          np.all(u[:, 0] <= self.p.u_max[0]),
                          np.all(u[:, 1] >= self.p.u_min[1]),
                          np.all(u[:, 1] <= self.p.u_max[1]),
                          np.all(u[:, 2] >= self.p.u_min[2]),
                          np.all(u[:, 2] <= self.p.u_max[2]),
                          np.all(u[:, 3] >= self.p.u_min[3]),
                          np.all(u[:, 3] <= self.p.u_max[3])])
        feasibility_rate = 100.0 if feasible else np.sum(
            np.all((u >= self.p.u_min[None, :]) & (u <= self.p.u_max[None, :]), axis=1)
        ) / u.shape[0] * 100.0

        return {
            'RMSE_tracking': rmse_tracking,
            'Max_tracking_error': max_tracking_error,
            'Max_swing_angle': max_swing_angle,
            'Settling_time': settling_time,
            'Control_effort': control_effort,
            'Avg_solve_time_ms': avg_solve_time,
            'Max_solve_time_ms': max_solve_time,
            'Feasibility_rate': feasibility_rate
        }

    @staticmethod
    def print_metrics(metrics: Dict, title: str = "Performance Metrics"):
        """Pretty print performance metrics"""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}")
        print(f"  RMSE:             {metrics['RMSE_tracking']:.4f} m")
        print(f"  Max tracking error:   {metrics['Max_tracking_error']:.4f} m")
        print(f"  Max swing angle:    {metrics['Max_swing_angle']:.2f}°")
        print(f"  Settling time:        {metrics['Settling_time']:.2f} s")
        print(f"  Control effort:       {metrics['Control_effort']:.2f}")
        print(f"  Avg solve time:       {metrics['Avg_solve_time_ms']:.2f} ms")
        print(f"  Max solve time:       {metrics['Max_solve_time_ms']:.2f} ms")
        print(f"  Feasibility rate:    {metrics['Feasibility_rate']:.1f}%")
        print(f"{'='*80}\n")

    @staticmethod
    def plot_comprehensive(results: Dict, save_path: str = None):
        """
        Create comprehensive visualization of simulation results
        """
        time = results['time']
        x = results['state']
        u = results['control']
        x_ref = results['reference']

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # ===== First row: Position and tracking error =====
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(time, x[:, 0], 'b-', linewidth=2, label='$x$')
        ax1.plot(time, x[:, 1], 'r-', linewidth=2, label='$y$')
        ax1.plot(time, x[:, 2], 'g-', linewidth=2, label='$z$')
        ax1.plot(time, x_ref[:, 0], 'b--', alpha=0.5, linewidth=1)
        ax1.plot(time, x_ref[:, 1], 'r--', alpha=0.5, linewidth=1)
        ax1.plot(time, x_ref[:, 2], 'g--', alpha=0.5, linewidth=1)
        ax1.set_ylabel('Position [m]', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Time [s]', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Position Tracking', fontsize=12, fontweight='bold')

        ax2 = fig.add_subplot(gs[0, 2])
        e_p = np.linalg.norm(x[:, 0:3] - x_ref, axis=1)
        ax2.plot(time, e_p, 'k-', linewidth=2)
        ax2.fill_between(time, 0, e_p, alpha=0.3, color='red')
        ax2.set_ylabel('Error [m]', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time [s]', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Tracking Error', fontsize=12, fontweight='bold')

        # ===== Second row: Swing angles =====
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(time, np.rad2deg(x[:, 6]), 'b-', linewidth=2, label=r'$\phi$')
        ax3.plot(time, np.rad2deg(x[:, 7]), 'r-', linewidth=2, label=r'$\psi$')
        ax3.axhline(y=30, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax3.axhline(y=-30, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax3.set_ylabel('Swing Angle [deg]', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Time [s]', fontsize=11)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Payload Swing Angles', fontsize=12, fontweight='bold')

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(time, np.rad2deg(x[:, 8]), 'b-', linewidth=2, label=r'$\dot{\phi}$')
        ax4.plot(time, np.rad2deg(x[:, 9]), 'r-', linewidth=2, label=r'$\dot{\psi}$')
        ax4.set_ylabel('Angular Velocity [deg/s]', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Time [s]', fontsize=11)
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Swing Angular Velocity', fontsize=12, fontweight='bold')

        # ===== Third row: Control input =====
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time, u[:, 0], 'b-', linewidth=2)
        ax5.axhline(y=self.p.u_min[0], color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax5.axhline(y=self.p.u_max[0], color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax5.set_ylabel('Thrust [N]', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Time [s]', fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Thrust Control', fontsize=12, fontweight='bold')

        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.plot(time, u[:, 1], 'b-', linewidth=2, label=r'$\tau_\phi$')
        ax6.plot(time, u[:, 2], 'r-', linewidth=2, label=r'$\tau_\theta$')
        ax6.plot(time, u[:, 3], 'g-', linewidth=2, label=r'$\tau_\psi$')
        ax6.axhline(y=5, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax6.axhline(y=-5, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax6.set_ylabel('Torque [Nm]', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Time [s]', fontsize=11)
        ax6.legend(loc='best', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_title('Attitude Torque', fontsize=12, fontweight='bold')

        # ===== Fourth row: Computation time and adaptive weights =====
        ax7 = fig.add_subplot(gs[3, 0])
        solve_times_ms = results['solve_time'] * 1000
        ax7.plot(time[:-1], solve_times_ms, 'k-', linewidth=1.5)
        ax7.axhline(y=50, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Sampling period')
        ax7.fill_between(time[:-1], 0, solve_times_ms, alpha=0.3, color='blue')
        ax7.set_ylabel('Solve Time [ms]', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Time [s]', fontsize=11)
        ax7.legend(loc='best', fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.set_title('Computation Time', fontsize=12, fontweight='bold')

        ax8 = fig.add_subplot(gs[3, 1:])
        if 'controller_metrics' in results:
            kappa_e = results['controller_metrics']['kappa_e_history']
            kappa_theta = results['controller_metrics']['kappa_theta_history']
            time_ctrl = time[:len(kappa_e)]
            ax8.plot(time_ctrl, kappa_e, 'b-', linewidth=2, label=r'$\kappa_e$ (position)')
            ax8.plot(time_ctrl, kappa_theta, 'r-', linewidth=2, label=r'$\kappa_\theta$ (swing)')
            ax8.set_ylabel('Scaling Factor', fontsize=11, fontweight='bold')
            ax8.set_xlabel('Time [s]', fontsize=11)
            ax8.legend(loc='best', fontsize=10)
            ax8.grid(True, alpha=0.3)
            ax8.set_title('Adaptive Weight Scaling', fontsize=12, fontweight='bold')

        plt.suptitle('Constrained Adaptive MPC: Comprehensive Results',
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return fig

    @staticmethod
    def plot_3d_trajectory(results: Dict, save_path: str = None):
        """
        Create 3D trajectory visualization
        """
        x = results['state']
        x_ref = results['reference']

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x[:, 0], x[:, 1], x[:, 2], 'b-', linewidth=2, label='Actual')
        ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], 'r--', linewidth=2, label='Reference')

        ax.scatter(x[0, 0], x[0, 1], x[0, 2], c='green', s=100, marker='o',
                  edgecolors='darkgreen', linewidths=2, label='Start', zorder=5)
        ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], c='red', s=100, marker='s',
                  edgecolors='darkred', linewidths=2, label='End', zorder=5)

        ax.set_xlabel('X [m]', fontsize=12, fontweight='bold', labelpad=12)
        ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold', labelpad=12)
        ax.set_zlabel('Z [m]', fontsize=12, fontweight='bold', labelpad=12)

        ax.tick_params(axis='both', which='major', labelsize=11)

        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#CCCCCC')
        ax.yaxis.pane.set_edgecolor('#CCCCCC')
        ax.zaxis.pane.set_edgecolor('#CCCCCC')

        ax.legend(fontsize=10)
        ax.set_title('3D Trajectory Tracking', fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D figure saved to: {save_path}")

        return fig


# ================================================================================
# 主仿真脚本
# ================================================================================

def main():
    """
    主仿真和分析脚本
    """
    print("\n" + "="*80)
    print(" UAV Slung-Load System Lyapunov-Constrained Adaptive MPC ".center(80, "="))
    print("="*80 + "\n")

    sys_params = SystemParams()
    ctrl_params = ControllerParams()

    system = UAVSlungLoadSystem(sys_params)
    controller = ConstrainedAdaptiveMPC(system, ctrl_params)
    simulator = Simulator(system, controller, ctrl_params)
    analyzer = PerformanceAnalyzer()

    print("\n" + "="*80)
    print(" Scenario 1: Nominal Step Response ".center(80))
    print("="*80)

    results_nominal = simulator.simulate(
        T_sim=20.0,
        trajectory_type='step',
        trajectory_params={
            'p_initial': np.array([0.0, 0.0, 5.0]),
            'p_final': np.array([5.0, 3.0, 8.0]),
            't_step': 2.0
        }
    )

    metrics_nominal = analyzer.calculate_metrics(results_nominal)
    analyzer.print_metrics(metrics_nominal, "Scenario 1: Nominal Performance")

    # ===== Scenario 2: Wind disturbance =====
    print("\n" + "="*80)
    print(" Scenario 2: Wind Disturbance Rejection ".center(80))
    print("="*80)

    controller.reset()

    def wind_disturbance(t):
        if t >= 5.0:
            return np.array([5.0, 0.0, 0.0])
        return np.zeros(3)

    results_wind = simulator.simulate(
        T_sim=20.0,
        trajectory_type='step',
        trajectory_params={
            'p_initial': np.array([0.0, 0.0, 5.0]),
            'p_final': np.array([5.0, 3.0, 8.0]),
            't_step': 2.0
        },
        wind_disturbance=wind_disturbance
    )

    metrics_wind = analyzer.calculate_metrics(results_wind)
    analyzer.print_metrics(metrics_wind, "Scenario 2: With Wind Disturbance")

    # ===== Scenario 3: Sudden mass change =====
    print("\n" + "="*80)
    print(" Scenario 3: Sudden Payload Mass Change ".center(80))
    print("="*80)

    controller.reset()
    system.params.m_L = 0.5

    results_mass = simulator.simulate(
        T_sim=20.0,
        trajectory_type='step',
        trajectory_params={
            'p_initial': np.array([0.0, 0.0, 5.0]),
            'p_final': np.array([5.0, 3.0, 8.0]),
            't_step': 2.0
        },
        mass_change_event=(10.0, 0.8)
    )

    metrics_mass = analyzer.calculate_metrics(results_mass)
    analyzer.print_metrics(metrics_mass, "Scenario 3: Recovery from Mass Change")

    # ===== Generate plots =====
    print("\n" + "="*80)
    print(" Generating Visualization ".center(80))
    print("="*80 + "\n")

    import os
    os.makedirs('outputs', exist_ok=True)

    analyzer.plot_comprehensive(
        results_nominal,
        save_path=os.path.join('outputs', 'results_nominal.png')
    )

    analyzer.plot_comprehensive(
        results_wind,
        save_path=os.path.join('outputs', 'results_wind.png')
    )

    analyzer.plot_comprehensive(
        results_mass,
        save_path=os.path.join('outputs', 'results_mass_change.png')
    )

    analyzer.plot_3d_trajectory(
        results_nominal,
        save_path=os.path.join('outputs', 'trajectory_3d_nominal.png')
    )

    # ===== Summary comparison =====
    print("\n" + "="*80)
    print(" Summary: Performance Comparison ".center(80))
    print("="*80)

    comparison_data = {
        'Scenario': ['Nominal', 'Wind (5 m/s)', 'Mass Change (+60%)'],
        'RMSE [m]': [
            metrics_nominal['RMSE_tracking'],
            metrics_wind['RMSE_tracking'],
            metrics_mass['RMSE_tracking']
        ],
        'Max Swing [°]': [
            metrics_nominal['Max_swing_angle'],
            metrics_wind['Max_swing_angle'],
            metrics_mass['Max_swing_angle']
        ],
        'Settling Time [s]': [
            metrics_nominal['Settling_time'],
            metrics_wind['Settling_time'],
            metrics_mass['Settling_time']
        ],
        'Avg Solve [ms]': [
            metrics_nominal['Avg_solve_time_ms'],
            metrics_wind['Avg_solve_time_ms'],
            metrics_mass['Avg_solve_time_ms']
        ]
    }

    print(f"\n{'Scenario':<25} {'RMSE':>10} {'Max Swing':>12} {'Settling':>12} {'Avg Solve':>12}")
    print("-" * 80)
    for i in range(3):
        print(f"{comparison_data['Scenario'][i]:<25} " +
              f"{comparison_data['RMSE [m]'][i]:>9.4f}m " +
              f"{comparison_data['Max Swing [°]'][i]:>11.2f}° " +
              f"{comparison_data['Settling Time [s]'][i]:>11.2f}s " +
              f"{comparison_data['Avg Solve [ms]'][i]:>11.2f}ms")

    print("\n" + "="*80)
    print(" Simulation Complete! ".center(80))
    print("="*80 + "\n")

    print("Generated files:")
    print("  - results_nominal.png")
    print("  - results_wind.png")
    print("  - results_mass_change.png")
    print("  - trajectory_3d_nominal.png")

    plt.show()



if __name__ == "__main__":
   main()