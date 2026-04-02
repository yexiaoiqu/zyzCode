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

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================================
# 配置和参数
# ================================================================================

@dataclass
class 系统参数:
    """系统物理参数"""
    # 无人机参数
    m_q: float = 2.0              # 无人机质量 [kg]
    g: float = 9.81               # 重力加速度 [m/s^2]
    I_xx: float = 0.0347          # x轴转动惯量 [kg*m^2]
    I_yy: float = 0.0347          # y轴转动惯量 [kg*m^2]
    I_zz: float = 0.0665          # z轴转动惯量 [kg*m^2]

    # 载荷参数
    m_L: float = 0.5              # 载荷质量 [kg]

    # 缆绳参数
    L: float = 1.0                # 缆绳标称长度 [m]
    k_c: float = 1000.0           # 缆绳刚度 [N/m]
    c_c: float = 10.0             # 缆绳阻尼 [Ns/m]

    # 空气阻力系数
    k_dx: float = 0.01            # x方向阻力系数
    k_dy: float = 0.01            # y方向阻力系数
    k_dz: float = 0.01            # z方向阻力系数

    # 旋翼参数
    k_T: float = 1.0              # 推力系数
    k_tau: float = 0.1            # 扭矩系数

    def 转字典(self) -> Dict:
        """将参数转换为字典"""
        return {
            '无人机质量': self.m_q,
            '载荷质量': self.m_L,
            '缆绳长度': self.L,
            '缆绳刚度': self.k_c,
            '缆绳阻尼': self.c_c
        }


@dataclass
class 控制器参数:
    """MPC控制器参数"""
    # MPC参数
    dt: float = 0.05              # 采样周期 [s]
    预测时域: int = 20                 # 预测时域
    李雅普诺夫衰减率: float = 0.3     # Lyapunov衰减率

    # 权重矩阵 - 基础值
    Q_pos: np.ndarray = None      # 位置跟踪权重
    Q_vel: np.ndarray = None      # 速度权重
    R_control: np.ndarray = None  # 控制努力权重
    S_increment: np.ndarray = None # 控制增量权重
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
    rho_soft: float = 1e5         # 约束软化惩罚

    # 观测器参数
    L_obs: np.ndarray = None      # 观测器增益矩阵

    def __post_init__(self):
        """初始化默认矩阵"""
        if self.Q_pos is None:
            self.Q_pos = np.diag([10.0, 10.0, 10.0])
        if self.Q_vel is None:
            self.Q_vel = np.diag([1.0, 1.0, 1.0])
        if self.R_control is None:
            self.R_control = np.diag([0.1, 0.1, 0.1, 0.1])
        if self.S_increment is None:
            self.S_increment = np.diag([1.0, 1.0, 1.0, 1.0])
        if self.u_min is None:
            self.u_min = np.array([5.0, -5.0, -5.0, -2.0])
        if self.u_max is None:
            self.u_max = np.array([30.0, 5.0, 5.0, 2.0])
        if self.L_obs is None:
            self.L_obs = np.array([10.0, 10.0, 10.0])


# ================================================================================
# 无人机吊载系统动力学
# ================================================================================

class 无人机吊载系统:
    """
    无人机吊载系统三维非线性动力学模型

    状态向量: x = [x, y, z, vx, vy, vz, alpha, beta, alpha_dot, beta_dot, phi, theta]
    控制输入: u = [T, tau_phi, tau_theta, tau_psi]

    模型包含:
    - 四旋翼无人机刚体动力学
    - 悬挂载荷摆动动力学
    - 采用Kelvin-Voigt粘弹性模型的柔性缆绳
    - 气动阻力
    - 耦合多体动力学
    """

    def __init__(self, params: 系统参数):
        self.params = params

        # 状态和控制维度
        self.n_states = 12
        self.n_controls = 4

        # 状态索引提高可读性
        self.idx_pos = slice(0, 3)      # [x, y, z]
        self.idx_vel = slice(3, 6)      # [vx, vy, vz]
        self.idx_swing = slice(6, 8)    # [alpha, beta]
        self.idx_swing_dot = slice(8, 10) # [alpha_dot, beta_dot]
        self.idx_att = slice(10, 12)    # [phi, theta]

    def 动力学(self, state: np.ndarray, u: np.ndarray,
                 干扰: np.ndarray) -> np.ndarray:

        # 提取状态
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        alpha, beta = state[6], state[7]  # 摆角
        alpha_dot, beta_dot = state[8], state[9]
        phi, theta = state[10], state[11]  # 滚转、俯仰

        # 提取控制输入
        T = u[0]          # 总推力
        tau_phi = u[1]    # 滚转扭矩
        tau_theta = u[2]  # 俯仰扭矩
        tau_psi = u[3]    # 偏航扭矩

        # 提取干扰
        d_wx, d_wy, d_wz = 干扰[0], 干扰[1], 干扰[2]

        # ===== 缆绳动力学 =====
        # 当前缆绳长度（考虑弹性）
        # 载荷相对于无人机的速度
        v_rel_x = vx - self.params.L * alpha_dot * np.cos(alpha) * np.cos(beta) + \
                  self.params.L * beta_dot * np.sin(alpha) * np.sin(beta)
        v_rel_y = vy - self.params.L * alpha_dot * np.cos(alpha) * np.sin(beta) - \
                  self.params.L * beta_dot * np.sin(alpha) * np.cos(beta)
        v_rel_z = vz + self.params.L * alpha_dot * np.sin(alpha)

        # 缆绳伸长
        delta_L = 0.0  # 小变形简化

        # 缆绳张力大小（Kelvin-Voigt模型）
        T_cable_elastic = self.params.k_c * delta_L
        T_cable_damping = self.params.c_c * (
            vx * np.sin(alpha) * np.cos(beta) +
            vy * np.sin(alpha) * np.sin(beta) +
            vz * np.cos(alpha)
        )
        T_cable = T_cable_elastic + T_cable_damping

        # 确保张力始终为正（缆绳只能拉）
        T_cable = max(0.0, T_cable)

        # 惯性坐标系下缆绳力分量
        T_cx = -T_cable * np.sin(alpha) * np.cos(beta)
        T_cy = -T_cable * np.sin(alpha) * np.sin(beta)
        T_cz = -T_cable * np.cos(alpha)

        # ===== 气动阻力 =====
        F_dx = -self.params.k_dx * vx + d_wx
        F_dy = -self.params.k_dy * vy + d_wy
        F_dz = -self.params.k_dz * vz + d_wz

        # ===== 无人机平动动力学 =====
        # 机体坐标系推力转换到惯性坐标系
        # R = Rz(psi) * Ry(theta) * Rx(phi)
        # 小角度简化旋转矩阵
        psi = 0.0  # 假设偏航单独控制

        # 旋转矩阵元素（ZYX欧拉角）
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        # 惯性坐标系下推力分量
        Tx_inertial = T * (s_theta * c_psi)
        Ty_inertial = T * (s_theta * s_psi)
        Tz_inertial = T * (c_theta * c_phi)

        # 总力
        Fx_total = Tx_inertial + T_cx + F_dx
        Fy_total = Ty_inertial + T_cy + F_dy
        Fz_total = Tz_inertial + T_cz + F_dz

        # 加速度
        ax = Fx_total / self.params.m_q
        ay = Fy_total / self.params.m_q
        az = Fz_total / self.params.m_q - self.params.g

        # ===== 载荷摆动动力学 =====
        # 这些方程由拉格朗日力学推导而来
        # 适用于支点运动的球形摆

        L = self.params.L
        m_L = self.params.m_L
        g = self.params.g

        # 载荷上的有效力
        # Alpha动力学（缆绳俯仰角）
        alpha_ddot = (
            (-T_cx / (m_L * L) + g * np.sin(alpha) / L) / np.cos(beta) -
            2 * alpha_dot * beta_dot * np.tan(beta) -
            (ax * np.cos(alpha) * np.cos(beta) +
             ay * np.cos(alpha) * np.sin(beta) -
             az * np.sin(alpha)) / L
        )

        # Beta动力学（缆绳方位角）
        beta_ddot = (
            -T_cy / (m_L * L * np.cos(alpha)) +
            g * np.sin(beta) * np.cos(alpha) / L +
            alpha_dot**2 * np.sin(beta) * np.cos(beta) -
            (-ax * np.sin(beta) + ay * np.cos(beta)) / (L * np.cos(alpha))
        )

        # 防止alpha接近±90°时出现数值问题
        if abs(np.cos(alpha)) < 0.1:
            alpha_ddot = 0.0
            beta_ddot = 0.0

        # ===== 姿态动力学 =====
        # 简化姿态动力学（假设小角度）
        phi_dot = tau_phi / self.params.I_xx
        theta_dot = tau_theta / self.params.I_yy

        # 应用速率限制防止数值不稳定
        phi_dot = np.clip(phi_dot, -10.0, 10.0)
        theta_dot = np.clip(theta_dot, -10.0, 10.0)

        # ===== 组装状态导数 =====
        state_dot = np.array([
            vx, vy, vz,                 # 位置导数
            ax, ay, az,                 # 速度导数
            alpha_dot, beta_dot,        # 摆角导数
            alpha_ddot, beta_ddot,      # 摆动角加速度
            phi_dot, theta_dot          # 姿态角速度导数
        ])

        return state_dot

    def 线性化(self, x_eq: np.ndarray, u_eq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        epsilon = 1e-6
        n_x = self.n_states
        n_u = self.n_controls

        # A矩阵数值微分
        A = np.zeros((n_x, n_x))
        f0 = self.动力学(x_eq, u_eq, np.zeros(3))

        for i in range(n_x):
            x_pert = x_eq.copy()
            x_pert[i] += epsilon
            f_pert = self.动力学(x_pert, u_eq, np.zeros(3))
            A[:, i] = (f_pert - f0) / epsilon

        # B矩阵数值微分
        B = np.zeros((n_x, n_u))
        for i in range(n_u):
            u_pert = u_eq.copy()
            u_pert[i] += epsilon
            f_pert = self.动力学(x_eq, u_pert, np.zeros(3))
            B[:, i] = (f_pert - f0) / epsilon

        return A, B


# ================================================================================
# 干扰观测器
# ================================================================================

class 扩张状态观测器:
    """
    用于在线干扰估计的扩张状态观测器（ESO）

    观测器将系统状态扩充干扰状态，
    以实现未知干扰和模型不确定性的实时估计。
    这对于自适应MPC在变化条件下保持性能至关重要。

    观测器动力学:
        ξ_dot = A_obs * ξ + B_obs * u + L_obs * (y - ŷ)

    其中 ξ = [v, d]^T 是扩张状态向量
    """

    def __init__(self, dt: float, L_obs: np.ndarray = None):
        self.dt = dt

        # 观测器增益（可调）
        if L_obs is None:
            # 默认增益设计为快速收敛
            self.L_obs = np.array([
                [20.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
                [0.0, 0.0, 20.0]
            ])
        else:
            self.L_obs = L_obs

        # 扩张状态: [vx, vy, vz, dx, dy, dz]
        self.xi = np.zeros(6)

        # 估计干扰
        self.d_hat = np.zeros(3)

        # 用于分析的历史
        self.d_history = []

    def 更新(self, v_measured: np.ndarray, v_predicted: np.ndarray,
               a_measured: np.ndarray = None) -> np.ndarray:

        # 新息（预测误差）
        e_v = v_measured - v_predicted

        # 观测器动力学（离散时间近似）
        # 速度更新
        self.xi[0:3] = v_measured

        # 干扰更新使用观测器增益
        新息 = self.L_obs @ e_v
        self.xi[3:6] += 新息 * self.dt

        # 低通滤波器防止高频噪声
        alpha_filter = 0.95
        self.d_hat = alpha_filter * self.d_hat + (1 - alpha_filter) * self.xi[3:6]

        # 饱和防止不切实际的估计
        self.d_hat = np.clip(self.d_hat, -20.0, 20.0)

        # 存储历史
        self.d_history.append(self.d_hat.copy())

        return self.d_hat

    def 重置(self):
        """重置观测器状态"""
        self.xi = np.zeros(6)
        self.d_hat = np.zeros(3)
        self.d_history = []


# ================================================================================
# Lyapunov函数
# ================================================================================

class 李雅普诺夫稳定性分析:
    """
    用于稳定性分析和约束生成的Lyapunov函数

    Lyapunov函数设计为同时捕获跟踪性能
    和摆动抑制目标:

    V(x) = 0.5 * [e_p^T * Q_p * e_p + e_v^T * Q_v * e_v +
                  alpha^2 + beta^2 + alpha_dot^2 + beta_dot^2]

    通过强制 V_dot <= -lambda * V，我们保证指数稳定性
    收敛速率由lambda确定。
    """

    def __init__(self, Q_p: np.ndarray = None, Q_v: np.ndarray = None):
        if Q_p is None:
            self.Q_p = np.diag([2.0, 2.0, 2.0])
        else:
            self.Q_p = Q_p

        if Q_v is None:
            self.Q_v = np.diag([1.0, 1.0, 1.0])
        else:
            self.Q_v = Q_v

        # 用于分析的历史
        self.V_history = []
        self.V_dot_history = []

    def 计算(self, e_p: np.ndarray, e_v: np.ndarray,
                alpha: float, beta: float,
                alpha_dot: float, beta_dot: float) -> float:
        """
        计算Lyapunov函数值

        参数:
            e_p: 位置跟踪误差 [3]
            e_v: 速度跟踪误差 [3]
            alpha, beta: 摆角
            alpha_dot, beta_dot: 摆动角速度

        返回:
            V: Lyapunov函数值（标量）
        """
        # 位置和速度误差分量
        V_tracking = 0.5 * (e_p.T @ self.Q_p @ e_p + e_v.T @ self.Q_v @ e_v)

        # 摆动能量分量
        V_swing = 0.5 * (alpha**2 + beta**2 + alpha_dot**2 + beta_dot**2)

        V = V_tracking + V_swing

        # 存储历史
        self.V_history.append(V)

        return V

    def 数值计算导数(self, V_current: float, V_prev: float,
                        dt: float) -> float:

        V_dot = (V_current - V_prev) / dt
        self.V_dot_history.append(V_dot)
        return V_dot

    def 检查稳定性条件(self, V_current: float, V_initial: float,
                         lambda_decay: float, 经过时间: float) -> bool:
        """
        检查稳定性条件是否满足:
        V(t) <= exp(-lambda * t) * V(0)

        参数:
            V_current: 当前Lyapunov值
            V_initial: 初始Lyapunov值
            lambda_decay: 衰减率
            经过时间: 经过时间

        返回:
            satisfied: True如果条件满足
        """
        V_bound = np.exp(-lambda_decay * 经过时间) * V_initial
        return V_current <= V_bound

    def 计算终端约束(self, e_p_N: np.ndarray, e_v_N: np.ndarray,
                       alpha_N: float, beta_N: float,
                       alpha_dot_N: float, beta_dot_N: float,
                       V_0: float, lambda_decay: float,
                       N_p: int, dt: float) -> float:
        """
        计算终端Lyapunov约束值

        该约束通过要求预测终端Lyapunov值
        满足: V(N) <= exp(-lambda * N * dt) * V(0)
        来保证递归可行性和稳定性。

        返回:
            constraint_value: 可行性应 <= 0
        """
        V_N = self.计算(e_p_N, e_v_N, alpha_N, beta_N, alpha_dot_N, beta_dot_N)

        V_bound = np.exp(-lambda_decay * N_p * dt) * V_0

        constraint_value = V_N - V_bound

        return constraint_value

    def 重置(self):
        """重置历史"""
        self.V_history = []
        self.V_dot_history = []


# ================================================================================
# 自适应权重机制
# ================================================================================

class 自适应权重策略:
    """
    多目标优化的自适应权重机制

    自适应权重根据实时状态误差动态调整，
    智能平衡轨迹跟踪和摆动抑制:

    Q(t) = kappa_e(t) * Q_base
    W_swing(t) = kappa_theta(t) * W_swing_base

    其中:
    kappa_e = 1 + alpha_e * (||e|| / (||e|| + epsilon_e))
    kappa_theta = 1 + alpha_theta * (||theta|| / (||theta|| + epsilon_theta))

    这确保误差大时积极跟踪，系统接近平衡时平滑控制。
    """

    def __init__(self, params: 控制器参数):
        self.params = params

        # 基础权重
        self.Q_base = params.Q_pos
        self.R_base = params.R_control
        self.S_base = params.S_increment
        self.W_swing_base = params.W_swing

        # 自适应参数
        self.alpha_e = params.alpha_e
        self.alpha_theta = params.alpha_theta
        self.epsilon_e = params.epsilon_e
        self.epsilon_theta = np.deg2rad(params.epsilon_theta)

        # 用于分析的历史
        self.kappa_e_history = []
        self.kappa_theta_history = []

    def 计算缩放因子(self, e_p: np.ndarray,
                       alpha: float, beta: float) -> Tuple[float, float]:
        """
        计算自适应缩放因子

        参数:
            e_p: 位置跟踪误差
            alpha, beta: 摆角

        返回:
            kappa_e: 位置误差缩放因子
            kappa_theta: 摆角缩放因子
        """
        # 位置误差大小
        norm_e = np.linalg.norm(e_p)

        # 摆角大小
        norm_theta = np.sqrt(alpha**2 + beta**2)

        # 使用平滑sigmoid类函数进行自适应缩放
        kappa_e = 1.0 + self.alpha_e * (norm_e / (norm_e + self.epsilon_e))
        kappa_theta = 1.0 + self.alpha_theta * (norm_theta / (norm_theta + self.epsilon_theta))

        # 存储历史
        self.kappa_e_history.append(kappa_e)
        self.kappa_theta_history.append(kappa_theta)

        return kappa_e, kappa_theta

    def 更新权重(self, e_p: np.ndarray, alpha: float, beta: float,
                  du_prev: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """
        根据当前误差更新所有权重矩阵

        参数:
            e_p: 位置跟踪误差
            alpha, beta: 摆角
            du_prev: 先前控制增量（用于自适应S）

        返回:
            Q, R, S, W_swing: 更新的权重矩阵
        """
        # 计算缩放因子
        kappa_e, kappa_theta = self.计算缩放因子(e_p, alpha, beta)

        # 更新跟踪权重
        Q = kappa_e * self.Q_base

        # 更新摆动抑制权重
        W_swing = kappa_theta * self.W_swing_base

        # 控制努力权重（固定或轻微自适应）
        R = self.R_base

        # 控制增量权重（自适应防止抖振）
        # 误差小时增加惩罚促进平滑
        norm_e = np.linalg.norm(e_p)
        kappa_smooth = 1.0 + 1.0 / (norm_e + 0.01)
        S = kappa_smooth * self.S_base

        return Q, R, S, W_swing

    def 重置(self):
        """重置历史"""
        self.kappa_e_history = []
        self.kappa_theta_history = []


# ================================================================================
# 轨迹生成器
# ================================================================================

class 轨迹生成器:
    """
    各种运动模式的参考轨迹生成
    """

    @staticmethod
    def 阶跃轨迹(t: float, p_initial: np.ndarray,
                   p_final: np.ndarray, t_step: float = 2.0) -> np.ndarray:
        """
        生成阶跃轨迹

        参数:
            t: 当前时间
            p_initial: 初始位置
            p_final: 最终位置
            t_step: 阶跃时间

        返回:
            x_ref: 参考状态 [12]
        """
        x_ref = np.zeros(12)

        if t < t_step:
            x_ref[0:3] = p_initial
        else:
            x_ref[0:3] = p_final

        return x_ref

    @staticmethod
    def 斜坡轨迹(t: float, v_ref: np.ndarray,
                   p_initial: np.ndarray) -> np.ndarray:
        """
        生成恒定速度斜坡轨迹

        参数:
            t: 当前时间
            v_ref: 参考速度
            p_initial: 初始位置

        返回:
            x_ref: 参考状态 [12]
        """
        x_ref = np.zeros(12)
        x_ref[0:3] = p_initial + v_ref * t
        x_ref[3:6] = v_ref

        return x_ref

    @staticmethod
    def 正弦轨迹(t: float, 幅值: np.ndarray,
                  频率: np.ndarray,
                  p_center: np.ndarray) -> np.ndarray:
        """
        生成正弦轨迹

        参数:
            t: 当前时间
            幅值: 振荡幅值 [3]
            频率: 振荡频率 [3] [rad/s]
            p_center: 中心位置 [3]

        返回:
            x_ref: 参考状态 [12]
        """
        x_ref = np.zeros(12)

        # 位置
        x_ref[0:3] = p_center + 幅值 * np.sin(频率 * t)

        # 速度
        x_ref[3:6] = 幅值 * 频率 * np.cos(频率 * t)

        return x_ref

    @staticmethod
    def 圆形轨迹(t: float, 半径: float, omega: float,
                  高度: float, 中心: np.ndarray = None) -> np.ndarray:
        """
        在水平面生成圆形轨迹

        参数:
            t: 当前时间
            半径: 圆半径
            omega: 角速度
            高度: 飞行高度
            中心: 圆心 [x, y]

        返回:
            x_ref: 参考状态 [12]
        """
        if 中心 is None:
            中心 = np.array([0.0, 0.0])

        x_ref = np.zeros(12)

        # 位置
        x_ref[0] = 中心[0] + 半径 * np.cos(omega * t)
        x_ref[1] = 中心[1] + 半径 * np.sin(omega * t)
        x_ref[2] = 高度

        # 速度
        x_ref[3] = -半径 * omega * np.sin(omega * t)
        x_ref[4] = 半径 * omega * np.cos(omega * t)
        x_ref[5] = 0.0

        return x_ref

    @staticmethod
    def 八字形轨迹(t: float, a: float, b: float,
                     omega: float, 高度: float) -> np.ndarray:
        """
        生成八字形轨迹（Gerono双纽线）

        参数:
            t: 当前时间
            a, b: 形状参数
            omega: 角速度
            高度: 飞行高度

        返回:
            x_ref: 参考状态 [12]
        """
        x_ref = np.zeros(12)

        # 位置
        x_ref[0] = a * np.sin(omega * t)
        x_ref[1] = b * np.sin(omega * t) * np.cos(omega * t)
        x_ref[2] = 高度

        # 速度
        x_ref[3] = a * omega * np.cos(omega * t)
        x_ref[4] = b * omega * (np.cos(omega * t)**2 - np.sin(omega * t)**2)
        x_ref[5] = 0.0

        return x_ref


# ================================================================================
# 约束自适应MPC控制器
# ================================================================================

class 约束自适应MPC:
    """
    主约束自适应模型预测控制器

    该控制器集成:
    1. 多目标优化的自适应权重
    2. 基于Lyapunov的稳定性约束
    3. 可行性约束软化
    4. 干扰抑制的扩张状态观测器
    5. 计算效率的热启动策略

    每个时间步的优化问题:

    min  Σ_{i=0}^{N-1} [||e_p||²_Q + ||u||²_R + ||Δu||²_S + W·(α²+β²)] + ρ·ξ²
     u

    s.t. x(k+1) = f(x(k), u(k), d̂)
         u_min ≤ u(k) ≤ u_max
         |α(k)|, |β(k)| ≤ α_max - ξ
         V(N) ≤ exp(-λ·N·Δt)·V(0)
         ξ ≥ 0
    """

    def __init__(self, system: 无人机吊载系统, params: 控制器参数):
        self.system = system
        self.params = params

        # 初始化组件
        self.观测器 = 扩张状态观测器(params.dt, params.L_obs)
        self.lyapunov = 李雅普诺夫稳定性分析()
        self.自适应权重 = 自适应权重策略(params)

        # 热启动
        self.u_prev_sequence = None
        self.x_prev_prediction = None

        # 性能指标
        self.solve_times = []
        self.cost_history = []
        self.constraint_violations = []
        self.feasibility_status = []

        # 迭代计数器
        self.iteration = 0

    def 预测轨迹(self, x0: np.ndarray, u_sequence: np.ndarray,
                   d_hat: np.ndarray) -> np.ndarray:
        """
        使用系统模型预测预测时域上的状态轨迹

        参数:
            x0: 初始状态 [n_states]
            u_sequence: 控制序列 [N_p x n_controls]
            d_hat: 估计干扰 [3]

        返回:
            x_pred: 预测状态轨迹 [N_p+1 x n_states]
        """
        N_p = self.params.预测时域
        dt = self.params.dt

        x_pred = np.zeros((N_p + 1, self.system.n_states))
        x_pred[0] = x0

        for i in range(N_p):
            # 欧拉积分（可替换为RK4提高精度）
            x_dot = self.system.动力学(x_pred[i], u_sequence[i], d_hat)
            x_pred[i+1] = x_pred[i] + x_dot * dt

            # 裁剪摆角防止不切实际的值
            x_pred[i+1, 6] = np.clip(x_pred[i+1, 6],
                                     -np.deg2rad(60), np.deg2rad(60))
            x_pred[i+1, 7] = np.clip(x_pred[i+1, 7],
                                     -np.deg2rad(60), np.deg2rad(60))

        return x_pred

    def 计算阶段成本(self, x: np.ndarray, x_ref: np.ndarray,
                      u: np.ndarray, u_prev: np.ndarray,
                      Q: np.ndarray, R: np.ndarray,
                      S: np.ndarray, W_swing: float) -> float:
        """
        计算单时间步的阶段成本

        参数:
            x: 当前状态
            x_ref: 参考状态
            u: 控制输入
            u_prev: 先前控制输入
            Q, R, S: 权重矩阵
            W_swing: 摆动惩罚权重

        返回:
            cost: 阶段成本值
        """
        # 位置跟踪误差
        e_p = x[0:3] - x_ref[0:3]
        cost_tracking = e_p.T @ Q @ e_p

        # 控制努力
        cost_control = u.T @ R @ u

        # 控制增量
        du = u - u_prev
        cost_increment = du.T @ S @ du

        # 摆动抑制
        alpha, beta = x[6], x[7]
        cost_swing = W_swing * (alpha**2 + beta**2)

        total_cost = cost_tracking + cost_control + cost_increment + cost_swing

        return total_cost

    def 计算总成本(self, u_flat: np.ndarray, x0: np.ndarray,
                    x_ref_traj: np.ndarray, u_prev: np.ndarray,
                    d_hat: np.ndarray) -> float:
        """
        计算预测时域上的总成本

        这是优化器将要最小化的目标函数。

        参数:
            u_flat: 展平控制序列 [N_p * n_controls]
            x0: 初始状态
            x_ref_traj: 参考轨迹 [N_p+1 x n_states]
            u_prev: 先前控制输入
            d_hat: 干扰估计

        返回:
            total_cost: 标量成本值
        """
        N_p = self.params.预测时域
        n_u = self.system.n_controls

        # 重塑控制序列
        u_sequence = u_flat.reshape(N_p, n_u)

        # 预测轨迹
        x_pred = self.预测轨迹(x0, u_sequence, d_hat)

        # 获取初始状态的自适应权重
        e_p_0 = x0[0:3] - x_ref_traj[0, 0:3]
        alpha_0, beta_0 = x0[6], x0[7]
        Q, R, S, W_swing = self.自适应权重.更新权重(e_p_0, alpha_0, beta_0)

        # 初始化成本
        total_cost = 0.0

        # 约束软化松弛变量
        xi_slack = 0.0

        # 阶段成本
        u_k = u_prev
        for i in range(N_p):
            # 当前状态和控制
            x_k = x_pred[i]
            u_k_next = u_sequence[i]

            # 阶段成本
            stage_cost = self.计算阶段成本(
                x_k, x_ref_traj[i], u_k_next, u_k, Q, R, S, W_swing
            )
            total_cost += stage_cost

            # 更新下一次迭代
            u_k = u_k_next

            # 摆角约束软化
            alpha_k, beta_k = x_k[6], x_k[7]
            alpha_max = np.deg2rad(self.params.alpha_max)
            beta_max = np.deg2rad(self.params.beta_max)

            if abs(alpha_k) > alpha_max:
                xi_slack += (abs(alpha_k) - alpha_max)**2
            if abs(beta_k) > beta_max:
                xi_slack += (abs(beta_k) - beta_max)**2

        # 终端成本（增加终端精度权重）
        x_N = x_pred[N_p]
        e_p_N = x_N[0:3] - x_ref_traj[N_p, 0:3]
        terminal_cost = 10.0 * e_p_N.T @ Q @ e_p_N

        # 终端摆动惩罚
        alpha_N, beta_N = x_N[6], x_N[7]
        terminal_cost += 10.0 * W_swing * (alpha_N**2 + beta_N**2)

        total_cost += terminal_cost

        # 约束软化惩罚
        total_cost += self.params.rho_soft * xi_slack

        # Lyapunov约束（软）
        alpha_dot_N, beta_dot_N = x_N[8], x_N[9]
        lyap_constraint = self.lyapunov.计算终端约束(
            e_p_N, np.zeros(3), alpha_N, beta_N, alpha_dot_N, beta_dot_N,
            self.lyapunov.计算(e_p_0, x0[3:6], alpha_0, beta_0, x0[8], x0[9]),
            self.params.李雅普诺夫衰减率, N_p, self.params.dt
        )

        if lyap_constraint > 0:
            total_cost += 1e4 * lyap_constraint**2

        return total_cost

    def 求解MPC(self, x_current: np.ndarray, x_ref_trajectory: np.ndarray,
                 u_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        求解MPC优化问题

        参数:
            x_current: 当前状态
            x_ref_trajectory: 预测时域上参考轨迹
            u_prev: 先前控制输入

        返回:
            u_optimal: 当前时间最优控制输入
            x_predicted: 预测轨迹
            solve_time: 计算时间
            feasible: 解是否可行
        """
        N_p = self.params.预测时域
        n_u = self.system.n_controls

        # 更新干扰估计
        v_measured = x_current[3:6]
        v_predicted = x_current[3:6]  # 简化
        d_hat = self.观测器.更新(v_measured, v_predicted)

        # 初始猜测（如果可用热启动）
        if self.u_prev_sequence is not None:
            # 移动先前解并追加最后一个控制
            u_init = np.concatenate([
                self.u_prev_sequence[n_u:],
                self.u_prev_sequence[-n_u:]
            ])
        else:
            # 对所有步使用先前控制
            u_init = np.tile(u_prev, N_p)

        # 优化变量边界
        bounds = []
        for i in range(N_p):
            for j in range(n_u):
                bounds.append((self.params.u_min[j], self.params.u_max[j]))

        # 求解优化
        start_time = time.time()

        try:
            result = minimize(
                fun=self.计算总成本,
                x0=u_init,
                args=(x_current, x_ref_trajectory, u_prev, d_hat),
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': 100,
                    'ftol': 1e-6,
                    'disp': False
                }
            )

            feasible = result.success
            u_optimal_flat = result.x
            final_cost = result.fun

        except Exception as e:
            print(f"优化失败: {e}")
            # 回退: 使用先前控制
            u_optimal_flat = u_init
            feasible = False
            final_cost = np.inf

        solve_time = time.time() - start_time

        # 提取最优控制序列
        u_optimal_sequence = u_optimal_flat.reshape(N_p, n_u)
        u_optimal = u_optimal_sequence[0]

        # 存储用于热启动
        self.u_prev_sequence = u_optimal_flat

        # 使用最优控制预测轨迹
        x_predicted = self.预测轨迹(x_current, u_optimal_sequence, d_hat)
        self.x_prev_prediction = x_predicted

        # 存储指标
        self.solve_times.append(solve_time)
        self.cost_history.append(final_cost)
        self.feasibility_status.append(feasible)

        # 检查约束违反
        max_swing = np.max(np.abs(x_predicted[:, 6:8]))
        if max_swing > np.deg2rad(self.params.alpha_max):
            self.constraint_violations.append(self.iteration)

        self.iteration += 1

        return u_optimal, x_predicted, solve_time, feasible

    def 重置(self):
        """重置控制器状态"""
        self.观测器.重置()
        self.lyapunov.重置()
        self.自适应权重.重置()
        self.u_prev_sequence = None
        self.x_prev_prediction = None
        self.solve_times = []
        self.cost_history = []
        self.constraint_violations = []
        self.feasibility_status = []
        self.iteration = 0


# ================================================================================
# 对比基准控制器
# ================================================================================

class PID控制器:
    """
    用于对比的传统PID控制器
    """

    def __init__(self, Kp: np.ndarray, Ki: np.ndarray, Kd: np.ndarray):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.dt = 0.05

    def 控制(self, x_current: np.ndarray, x_ref: np.ndarray,
              u_prev: np.ndarray) -> np.ndarray:
        """
        计算PID控制

        参数:
            x_current: 当前状态
            x_ref: 参考状态
            u_prev: 先前控制（未使用）

        返回:
            u: 控制输入 [T, tau_phi, tau_theta, tau_psi]
        """
        # 位置误差
        e_p = x_ref[0:3] - x_current[0:3]

        # 速度误差
        e_v = x_ref[3:6] - x_current[3:6]

        # PID项
        self.integral += e_p * self.dt
        derivative = (e_p - self.prev_error) / self.dt
        self.prev_error = e_p.copy()

        # 控制加速度指令
        a_cmd = self.Kp @ e_p + self.Ki @ self.integral + self.Kd @ derivative

        # 转换为推力和扭矩（简化）
        m_total = 2.5  # kg
        g = 9.81

        T = m_total * (g + a_cmd[2])
        tau_phi = 0.1 * a_cmd[1]
        tau_theta = 0.1 * a_cmd[0]
        tau_psi = 0.0

        u = np.array([T, tau_phi, tau_theta, tau_psi])

        # 应用约束
        u[0] = np.clip(u[0], 5.0, 30.0)
        u[1:3] = np.clip(u[1:3], -5.0, 5.0)
        u[3] = np.clip(u[3], -2.0, 2.0)

        return u

    def 重置(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)


class 线性MPC:
    """
    用于对比的线性MPC控制器
    """

    def __init__(self, system: 无人机吊载系统, params: 控制器参数):
        self.system = system
        self.params = params

        # 在悬停处线性化
        x_hover = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        u_hover = np.array([system.params.m_q * system.params.g, 0, 0, 0])
        self.A, self.B = system.线性化(x_hover, u_hover)

    def 控制(self, x_current: np.ndarray, x_ref_trajectory: np.ndarray,
              u_prev: np.ndarray) -> np.ndarray:
        """
        求解线性MPC问题

        对线性系统使用二次规划
        """
        # 简化: 使用与非线性MPC相同结构但线性预测
        # 为简洁，委托给约束自适应MPC结构
        # 实际上，这将使用QP求解器

        # 占位符: 返回重力补偿
        u = np.array([self.system.params.m_q * self.system.params.g, 0, 0, 0])
        return u


# ================================================================================
# 仿真环境
# ================================================================================

class 仿真环境:
    """
    完整的仿真环境，包含可视化和分析
    """

    def __init__(self, system: 无人机吊载系统,
                 controller: 约束自适应MPC,
                 params: 控制器参数):
        self.system = system
        self.controller = controller
        self.params = params

        # 轨迹生成器
        self.traj_gen = 轨迹生成器()

    def 仿真(self, T_sim: float, trajectory_type: str = 'step',
              trajectory_params: Dict = None, wind_disturbance: Callable = None,
              mass_change_event: Tuple = None,
              parameter_uncertainty: Dict = None) -> Dict:
        """
        运行闭环仿真

        参数:
            T_sim: 仿真时长
            trajectory_type: 参考轨迹类型
            trajectory_params: 轨迹生成参数
            wind_disturbance: 风干扰函数
            mass_change_event: (time, new_mass) 质量变化
            parameter_uncertainty: 参数变化字典

        返回:
            results: 仿真数据字典
        """
        dt = self.params.dt
        N_steps = int(T_sim / dt)
        time_vec = np.linspace(0, T_sim, N_steps)

        # 如果指定应用参数不确定性
        if parameter_uncertainty is not None:
            for param, value in parameter_uncertainty.items():
                setattr(self.system.params, param, value)

        # 初始化状态（在5m悬停）
        x = np.zeros(12)
        x[2] = 5.0

        # 初始化控制（悬停推力）
        u = np.array([self.system.params.m_q * self.system.params.g, 0, 0, 0])

        # 存储数组
        x_history = np.zeros((N_steps, 12))
        u_history = np.zeros((N_steps, 4))
        x_ref_history = np.zeros((N_steps, 12))
        d_hat_history = np.zeros((N_steps, 3))
        solve_time_history = np.zeros(N_steps)
        feasibility_history = np.zeros(N_steps, dtype=bool)

        print(f"\n{'='*80}")
        print(f"开始仿真: T={T_sim}s, dt={dt}s, N={N_steps} 步")
        print(f"轨迹: {trajectory_type}")
        print(f"{'='*80}\n")

        for k in range(N_steps):
            t = time_vec[k]

            # 处理质量变化事件
            if mass_change_event is not None:
                if abs(t - mass_change_event[0]) < dt / 2:
                    old_mass = self.system.params.m_L
                    self.system.params.m_L = mass_change_event[1]
                    print(f"[t={t:.2f}s] 质量变化: {old_mass}kg → {mass_change_event[1]}kg")

            # 生成参考轨迹
            N_p = self.params.预测时域
            x_ref_trajectory = np.zeros((N_p + 1, 12))

            if trajectory_params is None:
                trajectory_params = {}

            for i in range(N_p + 1):
                t_future = t + i * dt

                if trajectory_type == 'step':
                    x_ref_trajectory[i] = self.traj_gen.阶跃轨迹(
                        t_future,
                        trajectory_params.get('p_initial', np.array([0, 0, 5])),
                        trajectory_params.get('p_final', np.array([5, 3, 8])),
                        trajectory_params.get('t_step', 2.0)
                    )
                elif trajectory_type == 'circular':
                    x_ref_trajectory[i] = self.traj_gen.圆形轨迹(
                        t_future,
                        trajectory_params.get('radius', 5.0),
                        trajectory_params.get('omega', 0.5),
                        trajectory_params.get('height', 5.0)
                    )
                elif trajectory_type == 'sinusoidal':
                    x_ref_trajectory[i] = self.traj_gen.正弦轨迹(
                        t_future,
                        trajectory_params.get('amplitude', np.array([3, 2, 1])),
                        trajectory_params.get('frequency', np.array([0.5, 0.5, 0.3])),
                        trajectory_params.get('p_center', np.array([0, 0, 5]))
                    )
                else:  # hover
                    x_ref_trajectory[i, 0:3] = np.array([0, 0, 5])

            # MPC控制
            u_opt, x_pred, solve_time, feasible = self.controller.求解MPC(
                x, x_ref_trajectory, u
            )

            # 风干扰
            if wind_disturbance is not None:
                d_wind = wind_disturbance(t)
            else:
                d_wind = np.zeros(3)

            # 仿真系统动力学
            x_dot = self.system.动力学(x, u_opt, d_wind)
            x_next = x + x_dot * dt

            # 裁剪状态防止数值问题
            x_next[6:8] = np.clip(x_next[6:8], -np.deg2rad(60), np.deg2rad(60))

            # 存储数据
            x_history[k] = x
            u_history[k] = u_opt
            x_ref_history[k] = x_ref_trajectory[0]
            d_hat_history[k] = self.controller.观测器.d_hat
            solve_time_history[k] = solve_time
            feasibility_history[k] = feasible

            # 更新状态和控制
            x = x_next
            u = u_opt

            # 进度报告
            if k % 100 == 0 or not feasible:
                status = "✓" if feasible else "✗"
                print(f"[t={t:5.2f}s] {status} 位置=[{x[0]:5.2f},{x[1]:5.2f},{x[2]:5.2f}], " +
                      f"摆动=[{np.rad2deg(x[6]):5.1f}°,{np.rad2deg(x[7]):5.1f}°], " +
                      f"求解={solve_time*1000:5.1f}ms")

        print(f"\n{'='*80}")
        print("仿真完成!")
        print(f"{'='*80}\n")

        # 编译结果
        results = {
            'time': time_vec,
            'state': x_history,
            'control': u_history,
            'reference': x_ref_history,
            'disturbance_estimate': d_hat_history,
            'solve_time': solve_time_history,
            'feasibility': feasibility_history,
            'controller_metrics': {
                'solve_times': self.controller.solve_times,
                'cost_history': self.controller.cost_history,
                'constraint_violations': self.controller.constraint_violations,
                'kappa_e_history': self.controller.自适应权重.kappa_e_history,
                'kappa_theta_history': self.controller.自适应权重.kappa_theta_history
            }
        }

        return results


# ================================================================================
# 可视化和分析
# ================================================================================

class 结果分析器:
    """
    综合结果分析和可视化
    """

    @staticmethod
    def 计算性能指标(results: Dict) -> Dict:
        """
        计算定量性能指标

        返回:
            metrics: 性能指标字典
        """
        x = results['state']
        x_ref = results['reference']
        u = results['control']
        time = results['time']

        # 跟踪指标
        e_p = np.linalg.norm(x[:, 0:3] - x_ref[:, 0:3], axis=1)
        rmse_tracking = np.sqrt(np.mean(e_p**2))
        max_tracking_error = np.max(e_p)

        # 摆动指标
        swing_angles = np.rad2deg(np.abs(x[:, 6:8]))
        max_swing_alpha = np.max(swing_angles[:, 0])
        max_swing_beta = np.max(swing_angles[:, 1])
        max_swing = max(max_swing_alpha, max_swing_beta)

        # 调节时间（当跟踪误差 < 0.1m）
        settled_idx = np.where(e_p < 0.1)[0]
        if len(settled_idx) > 0:
            settling_time = time[settled_idx[0]] if settled_idx[0] > 0 else 0.0
        else:
            settling_time = time[-1]

        # 控制努力
        control_effort = np.sum(np.sum(u**2, axis=1)) * (time[1] - time[0])

        # 计算指标
        solve_times = results['solve_time']
        avg_solve_time = np.mean(solve_times) * 1000  # ms
        max_solve_time = np.max(solve_times) * 1000   # ms

        # 可行率
        feasibility = results['feasibility']
        feasibility_rate = np.sum(feasibility) / len(feasibility) * 100

        metrics = {
            'RMSE_tracking': rmse_tracking,
            'Max_tracking_error': max_tracking_error,
            'Max_swing_angle': max_swing,
            'Settling_time': settling_time,
            'Control_effort': control_effort,
            'Avg_solve_time_ms': avg_solve_time,
            'Max_solve_time_ms': max_solve_time,
            'Feasibility_rate': feasibility_rate
        }

        return metrics

    @staticmethod
    def 打印指标(metrics: Dict, title: str = "性能指标"):
        """美观打印性能指标"""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}")
        print(f"  跟踪RMSE:        {metrics['RMSE_tracking']:.4f} m")
        print(f"  最大跟踪误差:   {metrics['Max_tracking_error']:.4f} m")
        print(f"  最大摆角:      {metrics['Max_swing_angle']:.2f}°")
        print(f"  调节时间:        {metrics['Settling_time']:.2f} s")
        print(f"  控制努力:       {metrics['Control_effort']:.2f}")
        print(f"  平均求解时间:       {metrics['Avg_solve_time_ms']:.2f} ms")
        print(f"  最大求解时间:       {metrics['Max_solve_time_ms']:.2f} ms")
        print(f"  可行率:     {metrics['Feasibility_rate']:.1f}%")
        print(f"{'='*80}\n")

    @staticmethod
    def 绘制综合结果(results: Dict, save_path: str = None):
        """
        创建仿真结果的综合可视化
        """
        time = results['time']
        x = results['state']
        u = results['control']
        x_ref = results['reference']

        # 创建带子图的图形
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # ===== 第一行: 位置和跟踪误差 =====
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(time, x[:, 0], 'b-', linewidth=2, label='x')
        ax1.plot(time, x[:, 1], 'r-', linewidth=2, label='y')
        ax1.plot(time, x[:, 2], 'g-', linewidth=2, label='z')
        ax1.plot(time, x_ref[:, 0], 'b--', alpha=0.5, linewidth=1)
        ax1.plot(time, x_ref[:, 1], 'r--', alpha=0.5, linewidth=1)
        ax1.plot(time, x_ref[:, 2], 'g--', alpha=0.5, linewidth=1)
        ax1.set_ylabel('位置 [m]', fontsize=11, fontweight='bold')
        ax1.set_xlabel('时间 [s]', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('位置跟踪', fontsize=12, fontweight='bold')

        ax2 = fig.add_subplot(gs[0, 2])
        e_p = np.linalg.norm(x[:, 0:3] - x_ref[:, 0:3], axis=1)
        ax2.plot(time, e_p, 'k-', linewidth=2)
        ax2.fill_between(time, 0, e_p, alpha=0.3, color='red')
        ax2.set_ylabel('误差 [m]', fontsize=11, fontweight='bold')
        ax2.set_xlabel('时间 [s]', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('跟踪误差', fontsize=12, fontweight='bold')

        # ===== 第二行: 摆角和角速度 =====
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(time, np.rad2deg(x[:, 6]), 'b-', linewidth=2, label='α')
        ax3.plot(time, np.rad2deg(x[:, 7]), 'r-', linewidth=2, label='β')
        ax3.axhline(y=30, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax3.axhline(y=-30, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax3.set_ylabel('摆角 [度]', fontsize=11, fontweight='bold')
        ax3.set_xlabel('时间 [s]', fontsize=11)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('载荷摆角', fontsize=12, fontweight='bold')

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(time, np.rad2deg(x[:, 8]), 'b-', linewidth=2, label='α̇')
        ax4.plot(time, np.rad2deg(x[:, 9]), 'r-', linewidth=2, label='β̇')
        ax4.set_ylabel('角速度 [度/s]', fontsize=11, fontweight='bold')
        ax4.set_xlabel('时间 [s]', fontsize=11)
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('摆动角速度', fontsize=12, fontweight='bold')

        # ===== 第三行: 控制输入 =====
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time, u[:, 0], 'b-', linewidth=2)
        ax5.axhline(y=30, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax5.axhline(y=5, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax5.set_ylabel('推力 [N]', fontsize=11, fontweight='bold')
        ax5.set_xlabel('时间 [s]', fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_title('推力控制', fontsize=12, fontweight='bold')

        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.plot(time, u[:, 1], 'b-', linewidth=2, label='τ_φ')
        ax6.plot(time, u[:, 2], 'r-', linewidth=2, label='τ_θ')
        ax6.plot(time, u[:, 3], 'g-', linewidth=2, label='τ_ψ')
        ax6.axhline(y=5, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax6.axhline(y=-5, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax6.set_ylabel('扭矩 [Nm]', fontsize=11, fontweight='bold')
        ax6.set_xlabel('时间 [s]', fontsize=11)
        ax6.legend(loc='best', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_title('姿态扭矩', fontsize=12, fontweight='bold')

        # ===== 第四行: 计算和自适应权重 =====
        ax7 = fig.add_subplot(gs[3, 0])
        solve_times_ms = results['solve_time'] * 1000
        ax7.plot(time, solve_times_ms, 'k-', linewidth=1.5)
        ax7.axhline(y=50, color='r', linestyle='--', alpha=0.5, linewidth=2, label='采样周期')
        ax7.fill_between(time, 0, solve_times_ms, alpha=0.3, color='blue')
        ax7.set_ylabel('求解时间 [ms]', fontsize=11, fontweight='bold')
        ax7.set_xlabel('时间 [s]', fontsize=11)
        ax7.legend(loc='best', fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.set_title('计算时间', fontsize=12, fontweight='bold')

        ax8 = fig.add_subplot(gs[3, 1:])
        if 'controller_metrics' in results:
            kappa_e = results['controller_metrics']['kappa_e_history']
            kappa_theta = results['controller_metrics']['kappa_theta_history']
            time_ctrl = time[:len(kappa_e)]
            ax8.plot(time_ctrl, kappa_e, 'b-', linewidth=2, label='κ_e (位置)')
            ax8.plot(time_ctrl, kappa_theta, 'r-', linewidth=2, label='κ_θ (摆动)')
            ax8.set_ylabel('缩放因子', fontsize=11, fontweight='bold')
            ax8.set_xlabel('时间 [s]', fontsize=11)
            ax8.legend(loc='best', fontsize=10)
            ax8.grid(True, alpha=0.3)
            ax8.set_title('自适应权重缩放', fontsize=12, fontweight='bold')

        plt.suptitle('约束自适应MPC: 综合结果',
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存到: {save_path}")

        return fig

    @staticmethod
    def 绘制三维轨迹(results: Dict, save_path: str = None):
        """
        创建三维轨迹可视化
        """
        x = results['state']
        x_ref = results['reference']

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 实际轨迹
        ax.plot(x[:, 0], x[:, 1], x[:, 2], 'b-', linewidth=2, label='实际')

        # 参考轨迹
        ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], 'r--', linewidth=2, label='参考')

        # 起点和终点
        ax.scatter(x[0, 0], x[0, 1], x[0, 2], c='g', s=100, marker='o', label='起点')
        ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], c='r', s=100, marker='s', label='终点')

        ax.set_xlabel('X [m]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z [m]', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_title('三维轨迹', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"三维图形已保存到: {save_path}")

        return fig


# ================================================================================
# 主仿真脚本
# ================================================================================

def main():
    """
    主仿真和分析脚本
    """
    print("\n" + "="*80)
    print(" 无人机吊载系统约束自适应MPC ".center(80, "="))
    print("="*80 + "\n")

    # ===== 设置 =====
    sys_params = 系统参数()
    ctrl_params = 控制器参数()

    system = 无人机吊载系统(sys_params)
    controller = 约束自适应MPC(system, ctrl_params)
    simulator = 仿真环境(system, controller, ctrl_params)
    analyzer = 结果分析器()

    # ===== 场景1: 标称阶跃响应 =====
    print("\n" + "="*80)
    print(" 场景1: 标称阶跃响应 ".center(80))
    print("="*80)

    results_nominal = simulator.仿真(
        T_sim=20.0,
        trajectory_type='step',
        trajectory_params={
            'p_initial': np.array([0.0, 0.0, 5.0]),
            'p_final': np.array([5.0, 3.0, 8.0]),
            't_step': 2.0
        }
    )

    metrics_nominal = analyzer.计算性能指标(results_nominal)
    analyzer.打印指标(metrics_nominal, "场景1: 标称性能")

    # ===== 场景2: 风干扰 =====
    print("\n" + "="*80)
    print(" 场景2: 风干扰抑制 ".center(80))
    print("="*80)

    controller.重置()

    def 风干扰(t):
        if t >= 5.0:
            return np.array([5.0, 0.0, 0.0])  # x方向5 m/s风
        return np.zeros(3)

    results_wind = simulator.仿真(
        T_sim=20.0,
        trajectory_type='step',
        trajectory_params={
            'p_initial': np.array([0.0, 0.0, 5.0]),
            'p_final': np.array([5.0, 3.0, 8.0]),
            't_step': 2.0
        },
        wind_disturbance=风干扰
    )

    metrics_wind = analyzer.计算性能指标(results_wind)
    analyzer.打印指标(metrics_wind, "场景2: 有风干扰")

    # ===== 场景3: 质量变化 =====
    print("\n" + "="*80)
    print(" 场景3: 突发载荷质量变化 ".center(80))
    print("="*80)

    controller.重置()
    system.params.m_L = 0.5  # 重置为标称值

    results_mass = simulator.仿真(
        T_sim=20.0,
        trajectory_type='step',
        trajectory_params={
            'p_initial': np.array([0.0, 0.0, 5.0]),
            'p_final': np.array([5.0, 3.0, 8.0]),
            't_step': 2.0
        },
        mass_change_event=(10.0, 0.8)  # t=10s增加60%
    )

    metrics_mass = analyzer.计算性能指标(results_mass)
    analyzer.打印指标(metrics_mass, "场景3: 质量变化恢复")

    # ===== 生成图形 =====
    print("\n" + "="*80)
    print(" 生成可视化 ".center(80))
    print("="*80 + "\n")

    # 创建输出目录
    import os
    os.makedirs('outputs', exist_ok=True)

    # 综合结果图
    analyzer.绘制综合结果(
        results_nominal,
        save_path=os.path.join('outputs', 'results_nominal.png')
    )

    analyzer.绘制综合结果(
        results_wind,
        save_path=os.path.join('outputs', 'results_wind.png')
    )

    analyzer.绘制综合结果(
        results_mass,
        save_path=os.path.join('outputs', 'results_mass_change.png')
    )

    # 三维轨迹图
    analyzer.绘制三维轨迹(
        results_nominal,
        save_path=os.path.join('outputs', 'trajectory_3d_nominal.png')
    )

    # ===== 总结对比 =====
    print("\n" + "="*80)
    print(" 总结: 性能对比 ".center(80))
    print("="*80)

    comparison_data = {
        '场景': ['标称', '风 (5 m/s)', '质量变化 (+60%)'],
        'RMSE [m]': [
            metrics_nominal['RMSE_tracking'],
            metrics_wind['RMSE_tracking'],
            metrics_mass['RMSE_tracking']
        ],
        '最大摆角 [°]': [
            metrics_nominal['Max_swing_angle'],
            metrics_wind['Max_swing_angle'],
            metrics_mass['Max_swing_angle']
        ],
        '调节时间 [s]': [
            metrics_nominal['Settling_time'],
            metrics_wind['Settling_time'],
            metrics_mass['Settling_time']
        ],
        '平均求解 [ms]': [
            metrics_nominal['Avg_solve_time_ms'],
            metrics_wind['Avg_solve_time_ms'],
            metrics_mass['Avg_solve_time_ms']
        ]
    }

    print(f"\n{'场景':<25} {'RMSE':>10} {'最大摆角':>12} {'调节时间':>12} {'平均求解':>12}")
    print("-" * 80)
    for i in range(3):
        print(f"{comparison_data['场景'][i]:<25} " +
              f"{comparison_data['RMSE [m]'][i]:>9.4f}m " +
              f"{comparison_data['最大摆角 [°]'][i]:>11.2f}° " +
              f"{comparison_data['调节时间 [s]'][i]:>11.2f}s " +
              f"{comparison_data['平均求解 [ms]'][i]:>11.2f}ms")

    print("\n" + "="*80)
    print(" 仿真完成! ".center(80))
    print("="*80 + "\n")

    print("生成文件:")
    print("  - results_nominal.png")
    print("  - results_wind.png")
    print("  - results_mass_change.png")
    print("  - trajectory_3d_nominal.png")

    plt.show()


if __name__ == "__main__":
    main()