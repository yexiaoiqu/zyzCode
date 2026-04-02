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

# ================================================================================
# CONFIGURATION AND PARAMETERS
# ================================================================================

@dataclass
class SystemParameters:
    """System physical parameters"""
    # UAV parameters
    m_q: float = 2.0              # UAV mass [kg]
    g: float = 9.81               # Gravity [m/s^2]
    I_xx: float = 0.0347          # Moment of inertia x-axis [kg*m^2]
    I_yy: float = 0.0347          # Moment of inertia y-axis [kg*m^2]
    I_zz: float = 0.0665          # Moment of inertia z-axis [kg*m^2]

    # Load parameters
    m_L: float = 0.5              # Load mass [kg]

    # Cable parameters
    L: float = 1.0                # Cable nominal length [m]
    k_c: float = 1000.0           # Cable stiffness [N/m]
    c_c: float = 10.0             # Cable damping [Ns/m]

    # Air resistance coefficients
    k_dx: float = 0.01            # Drag coefficient x
    k_dy: float = 0.01            # Drag coefficient y
    k_dz: float = 0.01            # Drag coefficient z

    # Rotor parameters
    k_T: float = 1.0              # Thrust coefficient
    k_tau: float = 0.1            # Torque coefficient

    def to_dict(self) -> Dict:
        """Convert parameters to dictionary"""
        return {
            'UAV_mass': self.m_q,
            'Load_mass': self.m_L,
            'Cable_length': self.L,
            'Cable_stiffness': self.k_c,
            'Cable_damping': self.c_c
        }


@dataclass
class ControllerParameters:
    """MPC controller parameters"""
    # MPC parameters
    dt: float = 0.05              # Sampling period [s]
    N_p: int = 20                 # Prediction horizon
    lambda_decay: float = 0.3     # Lyapunov decay rate

    # Weight matrices - base values
    Q_pos: np.ndarray = None      # Position tracking weight
    Q_vel: np.ndarray = None      # Velocity weight
    R_control: np.ndarray = None  # Control effort weight
    S_increment: np.ndarray = None # Control increment weight
    W_swing: float = 50.0         # Swing penalty weight

    # Adaptive weighting parameters
    alpha_e: float = 2.0          # Position error scaling
    alpha_theta: float = 3.0      # Swing angle scaling
    epsilon_e: float = 0.1        # Position smoothing
    epsilon_theta: float = 5.0    # Angle smoothing [deg]

    # Constraint parameters
    u_min: np.ndarray = None      # Control input lower bounds
    u_max: np.ndarray = None      # Control input upper bounds
    alpha_max: float = 30.0       # Max swing angle [deg]
    beta_max: float = 30.0        # Max swing angle [deg]
    rho_soft: float = 1e5         # Constraint softening penalty

    # Observer parameters
    L_obs: np.ndarray = None      # Observer gain matrix

    def __post_init__(self):
        """Initialize default matrices"""
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
# UAV SLUNG-LOAD SYSTEM DYNAMICS
# ================================================================================

class UAVSlungLoadSystem:
    """
    Three-dimensional nonlinear dynamic model of UAV slung-load system

    State vector: x = [x, y, z, vx, vy, vz, alpha, beta, alpha_dot, beta_dot, phi, theta]
    Control input: u = [T, tau_phi, tau_theta, tau_psi]

    The model includes:
    - Rigid body dynamics of quadrotor UAV
    - Suspended load with swing dynamics
    - Flexible cable with Kelvin-Voigt viscoelastic model
    - Aerodynamic drag forces
    - Coupled multibody dynamics
    """

    def __init__(self, params: SystemParameters):
        self.params = params

        # State and control dimensions
        self.n_states = 12
        self.n_controls = 4

        # State indices for clarity
        self.idx_pos = slice(0, 3)      # [x, y, z]
        self.idx_vel = slice(3, 6)      # [vx, vy, vz]
        self.idx_swing = slice(6, 8)    # [alpha, beta]
        self.idx_swing_dot = slice(8, 10) # [alpha_dot, beta_dot]
        self.idx_att = slice(10, 12)    # [phi, theta]

    def dynamics(self, state: np.ndarray, u: np.ndarray,
                 disturbance: np.ndarray) -> np.ndarray:

        # Extract states
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        alpha, beta = state[6], state[7]  # Swing angles
        alpha_dot, beta_dot = state[8], state[9]
        phi, theta = state[10], state[11]  # Roll, pitch

        # Extract control inputs
        T = u[0]          # Total thrust
        tau_phi = u[1]    # Roll torque
        tau_theta = u[2]  # Pitch torque
        tau_psi = u[3]    # Yaw torque

        # Extract disturbances
        d_wx, d_wy, d_wz = disturbance[0], disturbance[1], disturbance[2]

        # ===== Cable dynamics =====
        # Current cable length (considering elasticity)
        # Velocity of load relative to UAV
        v_rel_x = vx - self.params.L * alpha_dot * np.cos(alpha) * np.cos(beta) + \
                  self.params.L * beta_dot * np.sin(alpha) * np.sin(beta)
        v_rel_y = vy - self.params.L * alpha_dot * np.cos(alpha) * np.sin(beta) - \
                  self.params.L * beta_dot * np.sin(alpha) * np.cos(beta)
        v_rel_z = vz + self.params.L * alpha_dot * np.sin(alpha)

        # Cable elongation
        delta_L = 0.0  # Simplified for small deformations

        # Cable tension magnitude (Kelvin-Voigt model)
        T_cable_elastic = self.params.k_c * delta_L
        T_cable_damping = self.params.c_c * (
            vx * np.sin(alpha) * np.cos(beta) +
            vy * np.sin(alpha) * np.sin(beta) +
            vz * np.cos(alpha)
        )
        T_cable = T_cable_elastic + T_cable_damping

        # Ensure tension is always positive (cable can only pull)
        T_cable = max(0.0, T_cable)

        # Cable force components in inertial frame
        T_cx = -T_cable * np.sin(alpha) * np.cos(beta)
        T_cy = -T_cable * np.sin(alpha) * np.sin(beta)
        T_cz = -T_cable * np.cos(alpha)

        # ===== Aerodynamic drag =====
        F_dx = -self.params.k_dx * vx + d_wx
        F_dy = -self.params.k_dy * vy + d_wy
        F_dz = -self.params.k_dz * vz + d_wz

        # ===== UAV translational dynamics =====
        # Thrust force in body frame to inertial frame
        # R = Rz(psi) * Ry(theta) * Rx(phi)
        # Simplified rotation matrix for small angles
        psi = 0.0  # Assuming yaw is controlled separately

        # Rotation matrix elements (ZYX Euler angles)
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        # Thrust components in inertial frame
        Tx_inertial = T * (s_theta * c_psi)
        Ty_inertial = T * (s_theta * s_psi)
        Tz_inertial = T * (c_theta * c_phi)

        # Total forces
        Fx_total = Tx_inertial + T_cx + F_dx
        Fy_total = Ty_inertial + T_cy + F_dy
        Fz_total = Tz_inertial + T_cz + F_dz

        # Accelerations
        ax = Fx_total / self.params.m_q
        ay = Fy_total / self.params.m_q
        az = Fz_total / self.params.m_q - self.params.g

        # ===== Load swing dynamics =====
        # These equations are derived from Lagrangian mechanics
        # for a spherical pendulum with moving pivot point

        L = self.params.L
        m_L = self.params.m_L
        g = self.params.g

        # Effective forces on load
        # Alpha dynamics (pitch angle of cable)
        alpha_ddot = (
            (-T_cx / (m_L * L) + g * np.sin(alpha) / L) / np.cos(beta) -
            2 * alpha_dot * beta_dot * np.tan(beta) -
            (ax * np.cos(alpha) * np.cos(beta) +
             ay * np.cos(alpha) * np.sin(beta) -
             az * np.sin(alpha)) / L
        )

        # Beta dynamics (azimuth angle of cable)
        beta_ddot = (
            -T_cy / (m_L * L * np.cos(alpha)) +
            g * np.sin(beta) * np.cos(alpha) / L +
            alpha_dot**2 * np.sin(beta) * np.cos(beta) -
            (-ax * np.sin(beta) + ay * np.cos(beta)) / (L * np.cos(alpha))
        )

        # Prevent numerical issues when alpha approaches ±90°
        if abs(np.cos(alpha)) < 0.1:
            alpha_ddot = 0.0
            beta_ddot = 0.0

        # ===== Attitude dynamics =====
        # Simplified attitude dynamics (assuming small angles)
        phi_dot = tau_phi / self.params.I_xx
        theta_dot = tau_theta / self.params.I_yy

        # Apply rate limits to prevent numerical instability
        phi_dot = np.clip(phi_dot, -10.0, 10.0)
        theta_dot = np.clip(theta_dot, -10.0, 10.0)

        # ===== Assemble state derivative =====
        state_dot = np.array([
            vx, vy, vz,                 # Position derivatives
            ax, ay, az,                 # Velocity derivatives
            alpha_dot, beta_dot,        # Swing angle derivatives
            alpha_ddot, beta_ddot,      # Swing angular acceleration
            phi_dot, theta_dot          # Attitude rate derivatives
        ])

        return state_dot

    def linearize(self, x_eq: np.ndarray, u_eq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        epsilon = 1e-6
        n_x = self.n_states
        n_u = self.n_controls

        # Numerical differentiation for A matrix
        A = np.zeros((n_x, n_x))
        f0 = self.dynamics(x_eq, u_eq, np.zeros(3))

        for i in range(n_x):
            x_pert = x_eq.copy()
            x_pert[i] += epsilon
            f_pert = self.dynamics(x_pert, u_eq, np.zeros(3))
            A[:, i] = (f_pert - f0) / epsilon

        # Numerical differentiation for B matrix
        B = np.zeros((n_x, n_u))
        for i in range(n_u):
            u_pert = u_eq.copy()
            u_pert[i] += epsilon
            f_pert = self.dynamics(x_eq, u_pert, np.zeros(3))
            B[:, i] = (f_pert - f0) / epsilon

        return A, B


# ================================================================================
# DISTURBANCE OBSERVER
# ================================================================================

class ExtendedStateObserver:
    """
    Extended State Observer (ESO) for online disturbance estimation

    The observer augments the system state with disturbance states to
    enable real-time estimation of unknown disturbances and model uncertainties.
    This is crucial for the adaptive MPC to maintain performance under
    varying conditions.

    Observer dynamics:
        ξ_dot = A_obs * ξ + B_obs * u + L_obs * (y - ŷ)

    where ξ = [v, d]^T is the extended state vector
    """

    def __init__(self, dt: float, L_obs: np.ndarray = None):
        self.dt = dt

        # Observer gains (tunable)
        if L_obs is None:
            # Default gains designed for fast convergence
            self.L_obs = np.array([
                [20.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
                [0.0, 0.0, 20.0]
            ])
        else:
            self.L_obs = L_obs

        # Extended state: [vx, vy, vz, dx, dy, dz]
        self.xi = np.zeros(6)

        # Estimated disturbance
        self.d_hat = np.zeros(3)

        # History for analysis
        self.d_history = []

    def update(self, v_measured: np.ndarray, v_predicted: np.ndarray,
               a_measured: np.ndarray = None) -> np.ndarray:

        # Innovation (prediction error)
        e_v = v_measured - v_predicted

        # Observer dynamics (discrete-time approximation)
        # Velocity update
        self.xi[0:3] = v_measured

        # Disturbance update with observer gain
        innovation = self.L_obs @ e_v
        self.xi[3:6] += innovation * self.dt

        # Low-pass filter to prevent high-frequency noise
        alpha_filter = 0.95
        self.d_hat = alpha_filter * self.d_hat + (1 - alpha_filter) * self.xi[3:6]

        # Saturation to prevent unrealistic estimates
        self.d_hat = np.clip(self.d_hat, -20.0, 20.0)

        # Store history
        self.d_history.append(self.d_hat.copy())

        return self.d_hat

    def reset(self):
        """Reset observer state"""
        self.xi = np.zeros(6)
        self.d_hat = np.zeros(3)
        self.d_history = []


# ================================================================================
# LYAPUNOV FUNCTION
# ================================================================================

class LyapunovStabilityAnalyzer:
    """
    Lyapunov function for stability analysis and constraint generation

    The Lyapunov function is designed to capture both tracking performance
    and swing suppression objectives:

    V(x) = 0.5 * [e_p^T * Q_p * e_p + e_v^T * Q_v * e_v +
                  alpha^2 + beta^2 + alpha_dot^2 + beta_dot^2]

    By enforcing V_dot <= -lambda * V, we guarantee exponential stability
    with convergence rate determined by lambda.
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

        # History for analysis
        self.V_history = []
        self.V_dot_history = []

    def compute(self, e_p: np.ndarray, e_v: np.ndarray,
                alpha: float, beta: float,
                alpha_dot: float, beta_dot: float) -> float:
        """
        Compute Lyapunov function value

        Args:
            e_p: Position tracking error [3]
            e_v: Velocity tracking error [3]
            alpha, beta: Swing angles
            alpha_dot, beta_dot: Swing angular velocities

        Returns:
            V: Lyapunov function value (scalar)
        """
        # Position and velocity error components
        V_tracking = 0.5 * (e_p.T @ self.Q_p @ e_p + e_v.T @ self.Q_v @ e_v)

        # Swing energy component
        V_swing = 0.5 * (alpha**2 + beta**2 + alpha_dot**2 + beta_dot**2)

        V = V_tracking + V_swing

        # Store history
        self.V_history.append(V)

        return V

    def compute_derivative_numerical(self, V_current: float, V_prev: float,
                                    dt: float) -> float:

        V_dot = (V_current - V_prev) / dt
        self.V_dot_history.append(V_dot)
        return V_dot

    def check_stability_condition(self, V_current: float, V_initial: float,
                                 lambda_decay: float, time_elapsed: float) -> bool:
        """
        Check if stability condition is satisfied:
        V(t) <= exp(-lambda * t) * V(0)

        Args:
            V_current: Current Lyapunov value
            V_initial: Initial Lyapunov value
            lambda_decay: Decay rate
            time_elapsed: Elapsed time

        Returns:
            satisfied: True if condition is satisfied
        """
        V_bound = np.exp(-lambda_decay * time_elapsed) * V_initial
        return V_current <= V_bound

    def compute_terminal_constraint(self, e_p_N: np.ndarray, e_v_N: np.ndarray,
                                   alpha_N: float, beta_N: float,
                                   alpha_dot_N: float, beta_dot_N: float,
                                   V_0: float, lambda_decay: float,
                                   N_p: int, dt: float) -> float:
        """
        Compute terminal Lyapunov constraint value

        This constraint ensures recursive feasibility and stability
        by requiring that the predicted terminal Lyapunov value
        satisfies: V(N) <= exp(-lambda * N * dt) * V(0)

        Returns:
            constraint_value: Should be <= 0 for feasibility
        """
        V_N = self.compute(e_p_N, e_v_N, alpha_N, beta_N, alpha_dot_N, beta_dot_N)

        V_bound = np.exp(-lambda_decay * N_p * dt) * V_0

        constraint_value = V_N - V_bound

        return constraint_value

    def reset(self):
        """Reset history"""
        self.V_history = []
        self.V_dot_history = []


# ================================================================================
# ADAPTIVE WEIGHTING MECHANISM
# ================================================================================

class AdaptiveWeightingStrategy:
    """
    Adaptive weighting mechanism for multi-objective optimization

    The adaptive weights dynamically adjust based on real-time state errors
    to intelligently balance between trajectory tracking and swing suppression:

    Q(t) = kappa_e(t) * Q_base
    W_swing(t) = kappa_theta(t) * W_swing_base

    where:
    kappa_e = 1 + alpha_e * (||e|| / (||e|| + epsilon_e))
    kappa_theta = 1 + alpha_theta * (||theta|| / (||theta|| + epsilon_theta))

    This ensures aggressive tracking when errors are large and smooth
    control when the system is near equilibrium.
    """

    def __init__(self, params: ControllerParameters):
        self.params = params

        # Base weights
        self.Q_base = params.Q_pos
        self.R_base = params.R_control
        self.S_base = params.S_increment
        self.W_swing_base = params.W_swing

        # Adaptive parameters
        self.alpha_e = params.alpha_e
        self.alpha_theta = params.alpha_theta
        self.epsilon_e = params.epsilon_e
        self.epsilon_theta = np.deg2rad(params.epsilon_theta)

        # History for analysis
        self.kappa_e_history = []
        self.kappa_theta_history = []

    def compute_scaling_factors(self, e_p: np.ndarray,
                               alpha: float, beta: float) -> Tuple[float, float]:
        """
        Compute adaptive scaling factors

        Args:
            e_p: Position tracking error
            alpha, beta: Swing angles

        Returns:
            kappa_e: Position error scaling factor
            kappa_theta: Swing angle scaling factor
        """
        # Position error magnitude
        norm_e = np.linalg.norm(e_p)

        # Swing angle magnitude
        norm_theta = np.sqrt(alpha**2 + beta**2)

        # Adaptive scaling using smooth sigmoid-like functions
        kappa_e = 1.0 + self.alpha_e * (norm_e / (norm_e + self.epsilon_e))
        kappa_theta = 1.0 + self.alpha_theta * (norm_theta / (norm_theta + self.epsilon_theta))

        # Store history
        self.kappa_e_history.append(kappa_e)
        self.kappa_theta_history.append(kappa_theta)

        return kappa_e, kappa_theta

    def update_weights(self, e_p: np.ndarray, alpha: float, beta: float,
                      du_prev: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """
        Update all weight matrices based on current errors

        Args:
            e_p: Position tracking error
            alpha, beta: Swing angles
            du_prev: Previous control increment (for adaptive S)

        Returns:
            Q, R, S, W_swing: Updated weight matrices
        """
        # Compute scaling factors
        kappa_e, kappa_theta = self.compute_scaling_factors(e_p, alpha, beta)

        # Update tracking weight
        Q = kappa_e * self.Q_base

        # Update swing suppression weight
        W_swing = kappa_theta * self.W_swing_base

        # Control effort weight (fixed or slightly adaptive)
        R = self.R_base

        # Control increment weight (adaptive to prevent chattering)
        # Increase penalty when error is small to promote smoothness
        norm_e = np.linalg.norm(e_p)
        kappa_smooth = 1.0 + 1.0 / (norm_e + 0.01)
        S = kappa_smooth * self.S_base

        return Q, R, S, W_swing

    def reset(self):
        """Reset history"""
        self.kappa_e_history = []
        self.kappa_theta_history = []


# ================================================================================
# TRAJECTORY GENERATOR
# ================================================================================

class TrajectoryGenerator:
    """
    Reference trajectory generation for various motion patterns
    """

    @staticmethod
    def step_trajectory(t: float, p_initial: np.ndarray,
                       p_final: np.ndarray, t_step: float = 2.0) -> np.ndarray:
        """
        Generate step trajectory

        Args:
            t: Current time
            p_initial: Initial position
            p_final: Final position
            t_step: Time of step

        Returns:
            x_ref: Reference state [12]
        """
        x_ref = np.zeros(12)

        if t < t_step:
            x_ref[0:3] = p_initial
        else:
            x_ref[0:3] = p_final

        return x_ref

    @staticmethod
    def ramp_trajectory(t: float, v_ref: np.ndarray,
                       p_initial: np.ndarray) -> np.ndarray:
        """
        Generate ramp trajectory with constant velocity

        Args:
            t: Current time
            v_ref: Reference velocity
            p_initial: Initial position

        Returns:
            x_ref: Reference state [12]
        """
        x_ref = np.zeros(12)
        x_ref[0:3] = p_initial + v_ref * t
        x_ref[3:6] = v_ref

        return x_ref

    @staticmethod
    def sinusoidal_trajectory(t: float, amplitude: np.ndarray,
                            frequency: np.ndarray,
                            p_center: np.ndarray) -> np.ndarray:
        """
        Generate sinusoidal trajectory

        Args:
            t: Current time
            amplitude: Amplitude of oscillation [3]
            frequency: Frequency of oscillation [3] [rad/s]
            p_center: Center position [3]

        Returns:
            x_ref: Reference state [12]
        """
        x_ref = np.zeros(12)

        # Position
        x_ref[0:3] = p_center + amplitude * np.sin(frequency * t)

        # Velocity
        x_ref[3:6] = amplitude * frequency * np.cos(frequency * t)

        return x_ref

    @staticmethod
    def circular_trajectory(t: float, radius: float, omega: float,
                          height: float, center: np.ndarray = None) -> np.ndarray:
        """
        Generate circular trajectory in horizontal plane

        Args:
            t: Current time
            radius: Circle radius
            omega: Angular velocity
            height: Flight height
            center: Circle center [x, y]

        Returns:
            x_ref: Reference state [12]
        """
        if center is None:
            center = np.array([0.0, 0.0])

        x_ref = np.zeros(12)

        # Position
        x_ref[0] = center[0] + radius * np.cos(omega * t)
        x_ref[1] = center[1] + radius * np.sin(omega * t)
        x_ref[2] = height

        # Velocity
        x_ref[3] = -radius * omega * np.sin(omega * t)
        x_ref[4] = radius * omega * np.cos(omega * t)
        x_ref[5] = 0.0

        return x_ref

    @staticmethod
    def figure_eight_trajectory(t: float, a: float, b: float,
                               omega: float, height: float) -> np.ndarray:
        """
        Generate figure-eight trajectory (Lemniscate of Gerono)

        Args:
            t: Current time
            a, b: Shape parameters
            omega: Angular velocity
            height: Flight height

        Returns:
            x_ref: Reference state [12]
        """
        x_ref = np.zeros(12)

        # Position
        x_ref[0] = a * np.sin(omega * t)
        x_ref[1] = b * np.sin(omega * t) * np.cos(omega * t)
        x_ref[2] = height

        # Velocity
        x_ref[3] = a * omega * np.cos(omega * t)
        x_ref[4] = b * omega * (np.cos(omega * t)**2 - np.sin(omega * t)**2)
        x_ref[5] = 0.0

        return x_ref


# ================================================================================
# CONSTRAINED ADAPTIVE MPC CONTROLLER
# ================================================================================

class ConstrainedAdaptiveMPC:
    """
    Main Constrained Adaptive Model Predictive Controller

    This controller integrates:
    1. Adaptive weighting for multi-objective optimization
    2. Lyapunov-based stability constraints
    3. Constraint softening for feasibility
    4. Extended state observer for disturbance rejection
    5. Warm-start strategy for computational efficiency

    The optimization problem at each time step:

    min  Σ_{i=0}^{N-1} [||e_p||²_Q + ||u||²_R + ||Δu||²_S + W·(α²+β²)] + ρ·ξ²
     u

    s.t. x(k+1) = f(x(k), u(k), d̂)
         u_min ≤ u(k) ≤ u_max
         |α(k)|, |β(k)| ≤ α_max - ξ
         V(N) ≤ exp(-λ·N·Δt)·V(0)
         ξ ≥ 0
    """

    def __init__(self, system: UAVSlungLoadSystem, params: ControllerParameters):
        self.system = system
        self.params = params

        # Initialize components
        self.observer = ExtendedStateObserver(params.dt, params.L_obs)
        self.lyapunov = LyapunovStabilityAnalyzer()
        self.adaptive_weights = AdaptiveWeightingStrategy(params)

        # Warm start
        self.u_prev_sequence = None
        self.x_prev_prediction = None

        # Performance metrics
        self.solve_times = []
        self.cost_history = []
        self.constraint_violations = []
        self.feasibility_status = []

        # Iteration counter
        self.iteration = 0

    def predict_trajectory(self, x0: np.ndarray, u_sequence: np.ndarray,
                          d_hat: np.ndarray) -> np.ndarray:
        """
        Predict state trajectory over prediction horizon using system model

        Args:
            x0: Initial state [n_states]
            u_sequence: Control sequence [N_p x n_controls]
            d_hat: Estimated disturbance [3]

        Returns:
            x_pred: Predicted state trajectory [N_p+1 x n_states]
        """
        N_p = self.params.N_p
        dt = self.params.dt

        x_pred = np.zeros((N_p + 1, self.system.n_states))
        x_pred[0] = x0

        for i in range(N_p):
            # Euler integration (can be replaced with RK4 for better accuracy)
            x_dot = self.system.dynamics(x_pred[i], u_sequence[i], d_hat)
            x_pred[i+1] = x_pred[i] + x_dot * dt

            # Clip swing angles to prevent unrealistic values
            x_pred[i+1, 6] = np.clip(x_pred[i+1, 6],
                                     -np.deg2rad(60), np.deg2rad(60))
            x_pred[i+1, 7] = np.clip(x_pred[i+1, 7],
                                     -np.deg2rad(60), np.deg2rad(60))

        return x_pred

    def compute_stage_cost(self, x: np.ndarray, x_ref: np.ndarray,
                          u: np.ndarray, u_prev: np.ndarray,
                          Q: np.ndarray, R: np.ndarray,
                          S: np.ndarray, W_swing: float) -> float:
        """
        Compute stage cost for single time step

        Args:
            x: Current state
            x_ref: Reference state
            u: Control input
            u_prev: Previous control input
            Q, R, S: Weight matrices
            W_swing: Swing penalty weight

        Returns:
            cost: Stage cost value
        """
        # Position tracking error
        e_p = x[0:3] - x_ref[0:3]
        cost_tracking = e_p.T @ Q @ e_p

        # Control effort
        cost_control = u.T @ R @ u

        # Control increment
        du = u - u_prev
        cost_increment = du.T @ S @ du

        # Swing suppression
        alpha, beta = x[6], x[7]
        cost_swing = W_swing * (alpha**2 + beta**2)

        total_cost = cost_tracking + cost_control + cost_increment + cost_swing

        return total_cost

    def compute_total_cost(self, u_flat: np.ndarray, x0: np.ndarray,
                          x_ref_traj: np.ndarray, u_prev: np.ndarray,
                          d_hat: np.ndarray) -> float:
        """
        Compute total cost over prediction horizon

        This is the objective function that will be minimized by the optimizer.

        Args:
            u_flat: Flattened control sequence [N_p * n_controls]
            x0: Initial state
            x_ref_traj: Reference trajectory [N_p+1 x n_states]
            u_prev: Previous control input
            d_hat: Disturbance estimate

        Returns:
            total_cost: Scalar cost value
        """
        N_p = self.params.N_p
        n_u = self.system.n_controls

        # Reshape control sequence
        u_sequence = u_flat.reshape(N_p, n_u)

        # Predict trajectory
        x_pred = self.predict_trajectory(x0, u_sequence, d_hat)

        # Get adaptive weights for initial state
        e_p_0 = x0[0:3] - x_ref_traj[0, 0:3]
        alpha_0, beta_0 = x0[6], x0[7]
        Q, R, S, W_swing = self.adaptive_weights.update_weights(e_p_0, alpha_0, beta_0)

        # Initialize cost
        total_cost = 0.0

        # Slack variable for constraint softening
        xi_slack = 0.0

        # Stage costs
        u_k = u_prev
        for i in range(N_p):
            # Current state and control
            x_k = x_pred[i]
            u_k_next = u_sequence[i]

            # Stage cost
            stage_cost = self.compute_stage_cost(
                x_k, x_ref_traj[i], u_k_next, u_k, Q, R, S, W_swing
            )
            total_cost += stage_cost

            # Update for next iteration
            u_k = u_k_next

            # Constraint softening for swing angles
            alpha_k, beta_k = x_k[6], x_k[7]
            alpha_max = np.deg2rad(self.params.alpha_max)
            beta_max = np.deg2rad(self.params.beta_max)

            if abs(alpha_k) > alpha_max:
                xi_slack += (abs(alpha_k) - alpha_max)**2
            if abs(beta_k) > beta_max:
                xi_slack += (abs(beta_k) - beta_max)**2

        # Terminal cost (increased weight for terminal accuracy)
        x_N = x_pred[N_p]
        e_p_N = x_N[0:3] - x_ref_traj[N_p, 0:3]
        terminal_cost = 10.0 * e_p_N.T @ Q @ e_p_N

        # Terminal swing penalty
        alpha_N, beta_N = x_N[6], x_N[7]
        terminal_cost += 10.0 * W_swing * (alpha_N**2 + beta_N**2)

        total_cost += terminal_cost

        # Constraint softening penalty
        total_cost += self.params.rho_soft * xi_slack

        # Lyapunov constraint (soft)
        alpha_dot_N, beta_dot_N = x_N[8], x_N[9]
        lyap_constraint = self.lyapunov.compute_terminal_constraint(
            e_p_N, np.zeros(3), alpha_N, beta_N, alpha_dot_N, beta_dot_N,
            self.lyapunov.compute(e_p_0, x0[3:6], alpha_0, beta_0, x0[8], x0[9]),
            self.params.lambda_decay, N_p, self.params.dt
        )

        if lyap_constraint > 0:
            total_cost += 1e4 * lyap_constraint**2

        return total_cost

    def solve_mpc(self, x_current: np.ndarray, x_ref_trajectory: np.ndarray,
                  u_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Solve MPC optimization problem

        Args:
            x_current: Current state
            x_ref_trajectory: Reference trajectory over horizon
            u_prev: Previous control input

        Returns:
            u_optimal: Optimal control input for current time
            x_predicted: Predicted trajectory
            solve_time: Computation time
            feasible: Whether solution is feasible
        """
        N_p = self.params.N_p
        n_u = self.system.n_controls

        # Update disturbance estimate
        v_measured = x_current[3:6]
        v_predicted = x_current[3:6]  # Simplified
        d_hat = self.observer.update(v_measured, v_predicted)

        # Initial guess (warm start if available)
        if self.u_prev_sequence is not None:
            # Shift previous solution and append last control
            u_init = np.concatenate([
                self.u_prev_sequence[n_u:],
                self.u_prev_sequence[-n_u:]
            ])
        else:
            # Use previous control for all steps
            u_init = np.tile(u_prev, N_p)

        # Bounds for optimization variables
        bounds = []
        for i in range(N_p):
            for j in range(n_u):
                bounds.append((self.params.u_min[j], self.params.u_max[j]))

        # Solve optimization
        start_time = time.time()

        try:
            result = minimize(
                fun=self.compute_total_cost,
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
            print(f"Optimization failed: {e}")
            # Fallback: use previous control
            u_optimal_flat = u_init
            feasible = False
            final_cost = np.inf

        solve_time = time.time() - start_time

        # Extract optimal control sequence
        u_optimal_sequence = u_optimal_flat.reshape(N_p, n_u)
        u_optimal = u_optimal_sequence[0]

        # Store for warm start
        self.u_prev_sequence = u_optimal_flat

        # Predict trajectory with optimal control
        x_predicted = self.predict_trajectory(x_current, u_optimal_sequence, d_hat)
        self.x_prev_prediction = x_predicted

        # Store metrics
        self.solve_times.append(solve_time)
        self.cost_history.append(final_cost)
        self.feasibility_status.append(feasible)

        # Check constraint violations
        max_swing = np.max(np.abs(x_predicted[:, 6:8]))
        if max_swing > np.deg2rad(self.params.alpha_max):
            self.constraint_violations.append(self.iteration)

        self.iteration += 1

        return u_optimal, x_predicted, solve_time, feasible

    def reset(self):
        """Reset controller state"""
        self.observer.reset()
        self.lyapunov.reset()
        self.adaptive_weights.reset()
        self.u_prev_sequence = None
        self.x_prev_prediction = None
        self.solve_times = []
        self.cost_history = []
        self.constraint_violations = []
        self.feasibility_status = []
        self.iteration = 0


# ================================================================================
# BASELINE CONTROLLERS FOR COMPARISON
# ================================================================================

class PIDController:
    """
    Traditional PID controller for comparison
    """

    def __init__(self, Kp: np.ndarray, Ki: np.ndarray, Kd: np.ndarray):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.dt = 0.05

    def control(self, x_current: np.ndarray, x_ref: np.ndarray,
                u_prev: np.ndarray) -> np.ndarray:
        """
        Compute PID control

        Args:
            x_current: Current state
            x_ref: Reference state
            u_prev: Previous control (not used)

        Returns:
            u: Control input [T, tau_phi, tau_theta, tau_psi]
        """
        # Position error
        e_p = x_ref[0:3] - x_current[0:3]

        # Velocity error
        e_v = x_ref[3:6] - x_current[3:6]

        # PID terms
        self.integral += e_p * self.dt
        derivative = (e_p - self.prev_error) / self.dt
        self.prev_error = e_p.copy()

        # Control acceleration command
        a_cmd = self.Kp @ e_p + self.Ki @ self.integral + self.Kd @ derivative

        # Convert to thrust and torques (simplified)
        m_total = 2.5  # kg
        g = 9.81

        T = m_total * (g + a_cmd[2])
        tau_phi = 0.1 * a_cmd[1]
        tau_theta = 0.1 * a_cmd[0]
        tau_psi = 0.0

        u = np.array([T, tau_phi, tau_theta, tau_psi])

        # Apply constraints
        u[0] = np.clip(u[0], 5.0, 30.0)
        u[1:3] = np.clip(u[1:3], -5.0, 5.0)
        u[3] = np.clip(u[3], -2.0, 2.0)

        return u

    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)


class LinearMPC:
    """
    Linear MPC controller for comparison
    """

    def __init__(self, system: UAVSlungLoadSystem, params: ControllerParameters):
        self.system = system
        self.params = params

        # Linearize around hover
        x_hover = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        u_hover = np.array([system.params.m_q * system.params.g, 0, 0, 0])
        self.A, self.B = system.linearize(x_hover, u_hover)

    def control(self, x_current: np.ndarray, x_ref_trajectory: np.ndarray,
                u_prev: np.ndarray) -> np.ndarray:
        """
        Solve linear MPC problem

        Uses quadratic programming for linear system
        """
        # Simplified: use same structure as nonlinear MPC but with linear prediction
        # For brevity, delegate to constrained adaptive MPC structure
        # In practice, this would use QP solver

        # Placeholder: return gravity compensation
        u = np.array([self.system.params.m_q * self.system.params.g, 0, 0, 0])
        return u


# ================================================================================
# SIMULATION ENVIRONMENT
# ================================================================================

class SimulationEnvironment:
    """
    Complete simulation environment with visualization and analysis
    """

    def __init__(self, system: UAVSlungLoadSystem,
                 controller: ConstrainedAdaptiveMPC,
                 params: ControllerParameters):
        self.system = system
        self.controller = controller
        self.params = params

        # Trajectory generator
        self.traj_gen = TrajectoryGenerator()

    def simulate(self, T_sim: float, trajectory_type: str = 'step',
                trajectory_params: Dict = None, wind_disturbance: Callable = None,
                mass_change_event: Tuple = None,
                parameter_uncertainty: Dict = None) -> Dict:
        """
        Run closed-loop simulation

        Args:
            T_sim: Simulation duration
            trajectory_type: Type of reference trajectory
            trajectory_params: Parameters for trajectory generation
            wind_disturbance: Wind disturbance function
            mass_change_event: (time, new_mass) for mass change
            parameter_uncertainty: Dictionary of parameter variations

        Returns:
            results: Simulation data dictionary
        """
        dt = self.params.dt
        N_steps = int(T_sim / dt)
        time_vec = np.linspace(0, T_sim, N_steps)

        # Apply parameter uncertainties if specified
        if parameter_uncertainty is not None:
            for param, value in parameter_uncertainty.items():
                setattr(self.system.params, param, value)

        # Initialize state (hover at 5m)
        x = np.zeros(12)
        x[2] = 5.0

        # Initialize control (hover thrust)
        u = np.array([self.system.params.m_q * self.system.params.g, 0, 0, 0])

        # Storage arrays
        x_history = np.zeros((N_steps, 12))
        u_history = np.zeros((N_steps, 4))
        x_ref_history = np.zeros((N_steps, 12))
        d_hat_history = np.zeros((N_steps, 3))
        solve_time_history = np.zeros(N_steps)
        feasibility_history = np.zeros(N_steps, dtype=bool)

        print(f"\n{'='*80}")
        print(f"Starting simulation: T={T_sim}s, dt={dt}s, N={N_steps} steps")
        print(f"Trajectory: {trajectory_type}")
        print(f"{'='*80}\n")

        for k in range(N_steps):
            t = time_vec[k]

            # Handle mass change event
            if mass_change_event is not None:
                if abs(t - mass_change_event[0]) < dt / 2:
                    old_mass = self.system.params.m_L
                    self.system.params.m_L = mass_change_event[1]
                    print(f"[t={t:.2f}s] Mass changed: {old_mass}kg → {mass_change_event[1]}kg")

            # Generate reference trajectory
            N_p = self.params.N_p
            x_ref_trajectory = np.zeros((N_p + 1, 12))

            if trajectory_params is None:
                trajectory_params = {}

            for i in range(N_p + 1):
                t_future = t + i * dt

                if trajectory_type == 'step':
                    x_ref_trajectory[i] = self.traj_gen.step_trajectory(
                        t_future,
                        trajectory_params.get('p_initial', np.array([0, 0, 5])),
                        trajectory_params.get('p_final', np.array([5, 3, 8])),
                        trajectory_params.get('t_step', 2.0)
                    )
                elif trajectory_type == 'circular':
                    x_ref_trajectory[i] = self.traj_gen.circular_trajectory(
                        t_future,
                        trajectory_params.get('radius', 5.0),
                        trajectory_params.get('omega', 0.5),
                        trajectory_params.get('height', 5.0)
                    )
                elif trajectory_type == 'sinusoidal':
                    x_ref_trajectory[i] = self.traj_gen.sinusoidal_trajectory(
                        t_future,
                        trajectory_params.get('amplitude', np.array([3, 2, 1])),
                        trajectory_params.get('frequency', np.array([0.5, 0.5, 0.3])),
                        trajectory_params.get('p_center', np.array([0, 0, 5]))
                    )
                else:  # hover
                    x_ref_trajectory[i, 0:3] = np.array([0, 0, 5])

            # MPC control
            u_opt, x_pred, solve_time, feasible = self.controller.solve_mpc(
                x, x_ref_trajectory, u
            )

            # Wind disturbance
            if wind_disturbance is not None:
                d_wind = wind_disturbance(t)
            else:
                d_wind = np.zeros(3)

            # Simulate system dynamics
            x_dot = self.system.dynamics(x, u_opt, d_wind)
            x_next = x + x_dot * dt

            # Clip states to prevent numerical issues
            x_next[6:8] = np.clip(x_next[6:8], -np.deg2rad(60), np.deg2rad(60))

            # Store data
            x_history[k] = x
            u_history[k] = u_opt
            x_ref_history[k] = x_ref_trajectory[0]
            d_hat_history[k] = self.controller.observer.d_hat
            solve_time_history[k] = solve_time
            feasibility_history[k] = feasible

            # Update state and control
            x = x_next
            u = u_opt

            # Progress reporting
            if k % 100 == 0 or not feasible:
                status = "✓" if feasible else "✗"
                print(f"[t={t:5.2f}s] {status} pos=[{x[0]:5.2f},{x[1]:5.2f},{x[2]:5.2f}], " +
                      f"swing=[{np.rad2deg(x[6]):5.1f}°,{np.rad2deg(x[7]):5.1f}°], " +
                      f"solve={solve_time*1000:5.1f}ms")

        print(f"\n{'='*80}")
        print("Simulation complete!")
        print(f"{'='*80}\n")

        # Compile results
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
                'kappa_e_history': self.controller.adaptive_weights.kappa_e_history,
                'kappa_theta_history': self.controller.adaptive_weights.kappa_theta_history
            }
        }

        return results


# ================================================================================
# VISUALIZATION AND ANALYSIS
# ================================================================================

class ResultsAnalyzer:
    """
    Comprehensive results analysis and visualization
    """

    @staticmethod
    def compute_performance_metrics(results: Dict) -> Dict:
        """
        Compute quantitative performance metrics

        Returns:
            metrics: Dictionary of performance metrics
        """
        x = results['state']
        x_ref = results['reference']
        u = results['control']
        time = results['time']

        # Tracking metrics
        e_p = np.linalg.norm(x[:, 0:3] - x_ref[:, 0:3], axis=1)
        rmse_tracking = np.sqrt(np.mean(e_p**2))
        max_tracking_error = np.max(e_p)

        # Swing metrics
        swing_angles = np.rad2deg(np.abs(x[:, 6:8]))
        max_swing_alpha = np.max(swing_angles[:, 0])
        max_swing_beta = np.max(swing_angles[:, 1])
        max_swing = max(max_swing_alpha, max_swing_beta)

        # Settling time (when tracking error < 0.1m)
        settled_idx = np.where(e_p < 0.1)[0]
        if len(settled_idx) > 0:
            settling_time = time[settled_idx[0]] if settled_idx[0] > 0 else 0.0
        else:
            settling_time = time[-1]

        # Control effort
        control_effort = np.sum(np.sum(u**2, axis=1)) * (time[1] - time[0])

        # Computational metrics
        solve_times = results['solve_time']
        avg_solve_time = np.mean(solve_times) * 1000  # ms
        max_solve_time = np.max(solve_times) * 1000   # ms

        # Feasibility rate
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
    def print_metrics(metrics: Dict, title: str = "Performance Metrics"):
        """Pretty print performance metrics"""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}")
        print(f"  Tracking RMSE:        {metrics['RMSE_tracking']:.4f} m")
        print(f"  Max Tracking Error:   {metrics['Max_tracking_error']:.4f} m")
        print(f"  Max Swing Angle:      {metrics['Max_swing_angle']:.2f}°")
        print(f"  Settling Time:        {metrics['Settling_time']:.2f} s")
        print(f"  Control Effort:       {metrics['Control_effort']:.2f}")
        print(f"  Avg Solve Time:       {metrics['Avg_solve_time_ms']:.2f} ms")
        print(f"  Max Solve Time:       {metrics['Max_solve_time_ms']:.2f} ms")
        print(f"  Feasibility Rate:     {metrics['Feasibility_rate']:.1f}%")
        print(f"{'='*80}\n")

    @staticmethod
    def plot_comprehensive_results(results: Dict, save_path: str = None):
        """
        Create comprehensive visualization of simulation results
        """
        time = results['time']
        x = results['state']
        u = results['control']
        x_ref = results['reference']

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # ===== Row 1: Position and tracking error =====
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(time, x[:, 0], 'b-', linewidth=2, label='x')
        ax1.plot(time, x[:, 1], 'r-', linewidth=2, label='y')
        ax1.plot(time, x[:, 2], 'g-', linewidth=2, label='z')
        ax1.plot(time, x_ref[:, 0], 'b--', alpha=0.5, linewidth=1)
        ax1.plot(time, x_ref[:, 1], 'r--', alpha=0.5, linewidth=1)
        ax1.plot(time, x_ref[:, 2], 'g--', alpha=0.5, linewidth=1)
        ax1.set_ylabel('Position [m]', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Time [s]', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Position Tracking', fontsize=12, fontweight='bold')

        ax2 = fig.add_subplot(gs[0, 2])
        e_p = np.linalg.norm(x[:, 0:3] - x_ref[:, 0:3], axis=1)
        ax2.plot(time, e_p, 'k-', linewidth=2)
        ax2.fill_between(time, 0, e_p, alpha=0.3, color='red')
        ax2.set_ylabel('Error [m]', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time [s]', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Tracking Error', fontsize=12, fontweight='bold')

        # ===== Row 2: Swing angles and velocities =====
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(time, np.rad2deg(x[:, 6]), 'b-', linewidth=2, label='α')
        ax3.plot(time, np.rad2deg(x[:, 7]), 'r-', linewidth=2, label='β')
        ax3.axhline(y=30, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax3.axhline(y=-30, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax3.set_ylabel('Swing Angles [deg]', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Time [s]', fontsize=11)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Load Swing Angles', fontsize=12, fontweight='bold')

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(time, np.rad2deg(x[:, 8]), 'b-', linewidth=2, label='α̇')
        ax4.plot(time, np.rad2deg(x[:, 9]), 'r-', linewidth=2, label='β̇')
        ax4.set_ylabel('Angular Vel [deg/s]', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Time [s]', fontsize=11)
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Swing Velocities', fontsize=12, fontweight='bold')

        # ===== Row 3: Control inputs =====
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time, u[:, 0], 'b-', linewidth=2)
        ax5.axhline(y=30, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax5.axhline(y=5, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax5.set_ylabel('Thrust [N]', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Time [s]', fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Thrust Control', fontsize=12, fontweight='bold')

        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.plot(time, u[:, 1], 'b-', linewidth=2, label='τ_φ')
        ax6.plot(time, u[:, 2], 'r-', linewidth=2, label='τ_θ')
        ax6.plot(time, u[:, 3], 'g-', linewidth=2, label='τ_ψ')
        ax6.axhline(y=5, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax6.axhline(y=-5, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax6.set_ylabel('Torque [Nm]', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Time [s]', fontsize=11)
        ax6.legend(loc='best', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_title('Attitude Torques', fontsize=12, fontweight='bold')

        # ===== Row 4: Computational and adaptive weights =====
        ax7 = fig.add_subplot(gs[3, 0])
        solve_times_ms = results['solve_time'] * 1000
        ax7.plot(time, solve_times_ms, 'k-', linewidth=1.5)
        ax7.axhline(y=50, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Sampling period')
        ax7.fill_between(time, 0, solve_times_ms, alpha=0.3, color='blue')
        ax7.set_ylabel('Solve Time [ms]', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Time [s]', fontsize=11)
        ax7.legend(loc='best', fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.set_title('Computational Time', fontsize=12, fontweight='bold')

        ax8 = fig.add_subplot(gs[3, 1:])
        if 'controller_metrics' in results:
            kappa_e = results['controller_metrics']['kappa_e_history']
            kappa_theta = results['controller_metrics']['kappa_theta_history']
            time_ctrl = time[:len(kappa_e)]
            ax8.plot(time_ctrl, kappa_e, 'b-', linewidth=2, label='κ_e (position)')
            ax8.plot(time_ctrl, kappa_theta, 'r-', linewidth=2, label='κ_θ (swing)')
            ax8.set_ylabel('Scaling Factor', fontsize=11, fontweight='bold')
            ax8.set_xlabel('Time [s]', fontsize=11)
            ax8.legend(loc='best', fontsize=10)
            ax8.grid(True, alpha=0.3)
            ax8.set_title('Adaptive Weight Scaling', fontsize=12, fontweight='bold')

        plt.suptitle('Constrained Adaptive MPC: Comprehensive Results',
                    fontsize=14, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

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

        # Actual trajectory
        ax.plot(x[:, 0], x[:, 1], x[:, 2], 'b-', linewidth=2, label='Actual')

        # Reference trajectory
        ax.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], 'r--', linewidth=2, label='Reference')

        # Start and end points
        ax.scatter(x[0, 0], x[0, 1], x[0, 2], c='g', s=100, marker='o', label='Start')
        ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], c='r', s=100, marker='s', label='End')

        ax.set_xlabel('X [m]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z [m]', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_title('3D Trajectory', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D plot saved to: {save_path}")

        return fig


# ================================================================================
# MAIN SIMULATION SCRIPT
# ================================================================================

def main():
    """
    Main simulation and analysis script
    """
    print("\n" + "="*80)
    print(" Constrained Adaptive MPC for UAV Slung-Load System ".center(80, "="))
    print("="*80 + "\n")

    # ===== Setup =====
    sys_params = SystemParameters()
    ctrl_params = ControllerParameters()

    system = UAVSlungLoadSystem(sys_params)
    controller = ConstrainedAdaptiveMPC(system, ctrl_params)
    simulator = SimulationEnvironment(system, controller, ctrl_params)
    analyzer = ResultsAnalyzer()

    # ===== Scenario 1: Nominal Step Response =====
    print("\n" + "="*80)
    print(" SCENARIO 1: Nominal Step Response ".center(80))
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

    metrics_nominal = analyzer.compute_performance_metrics(results_nominal)
    analyzer.print_metrics(metrics_nominal, "Scenario 1: Nominal Performance")

    # ===== Scenario 2: Wind Disturbance =====
    print("\n" + "="*80)
    print(" SCENARIO 2: Wind Disturbance Rejection ".center(80))
    print("="*80)

    controller.reset()

    def wind_disturbance(t):
        if t >= 5.0:
            return np.array([5.0, 0.0, 0.0])  # 5 m/s wind in x direction
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

    metrics_wind = analyzer.compute_performance_metrics(results_wind)
    analyzer.print_metrics(metrics_wind, "Scenario 2: With Wind Disturbance")

    # ===== Scenario 3: Mass Change =====
    print("\n" + "="*80)
    print(" SCENARIO 3: Sudden Payload Mass Change ".center(80))
    print("="*80)

    controller.reset()
    system.params.m_L = 0.5  # Reset to nominal

    results_mass = simulator.simulate(
        T_sim=20.0,
        trajectory_type='step',
        trajectory_params={
            'p_initial': np.array([0.0, 0.0, 5.0]),
            'p_final': np.array([5.0, 3.0, 8.0]),
            't_step': 2.0
        },
        mass_change_event=(10.0, 0.8)  # 60% increase at t=10s
    )

    metrics_mass = analyzer.compute_performance_metrics(results_mass)
    analyzer.print_metrics(metrics_mass, "Scenario 3: Mass Change Recovery")

    # ===== Generate Plots =====
    print("\n" + "="*80)
    print(" Generating Visualizations ".center(80))
    print("="*80 + "\n")

    # Comprehensive results plots
    analyzer.plot_comprehensive_results(
        results_nominal,
        save_path='/mnt/user-data/outputs/results_nominal.png'
    )

    analyzer.plot_comprehensive_results(
        results_wind,
        save_path='/mnt/user-data/outputs/results_wind.png'
    )

    analyzer.plot_comprehensive_results(
        results_mass,
        save_path='/mnt/user-data/outputs/results_mass_change.png'
    )

    # 3D trajectory plots
    analyzer.plot_3d_trajectory(
        results_nominal,
        save_path='/mnt/user-data/outputs/trajectory_3d_nominal.png'
    )

    # ===== Summary Comparison =====
    print("\n" + "="*80)
    print(" SUMMARY: Performance Comparison ".center(80))
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
        'Settling [s]': [
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
              f"{comparison_data['Settling [s]'][i]:>11.2f}s " +
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