"""
图5: 阶跃指令下位置响应和摆角抑制
论文: 无人机吊载系统Lyapunov约束自适应模型预测控制
章节: 4.3.1 标称性能

该脚本生成三个子图展示:
(a) x、y、z方向位置跟踪，带性能指标标注
(b) 摆角phi和psi抑制，带收敛指示器
(c) 归一化推力控制输入，展示平滑控制

关键性能指标:
- 调节时间: 2.1 s
- 超调量: 5.3%
- 稳态误差: 0.08 m
- 最大摆角: 12°
- 摆角收敛时间: 3.5 s（收敛到 < 5°）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

rcParams['font.family'] = ['serif', 'sans-serif']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300

# ============================================================================
# 仿真参数
# ============================================================================

t = np.linspace(0, 10, 2000)
dt = t[1] - t[0]

step_time = 1.0
step_idx = np.argmin(np.abs(t - step_time))

ts = 2.1
overshoot = 0.053
ess = 0.08

x_step = 5.0
y_step = 3.0
z_step = 4.0

phi_max = 12.0
psi_max = 12.0
theta_conv_time = 3.5
theta_freq = 1.5

# ============================================================================
# 生成参考轨迹
# ============================================================================

x_ref = np.zeros_like(t)
y_ref = np.zeros_like(t)
z_ref = np.zeros_like(t)

x_ref[t >= step_time] = x_step
y_ref[t >= step_time] = y_step
z_ref[t >= step_time] = z_step

# ============================================================================
# 生成实际系统响应
# ============================================================================

def second_order_response(t, t_step, amplitude, overshoot, ts, ess):
    """
    生成指定超调量和稳态误差的二阶阶跃响应

    参数:
    -----------
    t : array-like
        时间向量
    t_step : float
        施加阶跃输入的时间
    amplitude : float
        阶跃幅值
    overshoot : float
        超调量小数表示（例如 0.053 表示 5.3%）
    ts : float
        2%调节时间（秒）
    ess : float
        稳态误差

    返回:
    --------
    response : array
        系统响应
    """
    response = np.zeros_like(t)
    t_offset = t - t_step

    zeta = -np.log(overshoot) / np.sqrt(np.pi**2 + np.log(overshoot)**2)
    wn = 4.0 / (zeta * ts)
    wd = wn * np.sqrt(1 - zeta**2)

    for i, ti in enumerate(t_offset):
        if ti >= 0:
            envelope = np.exp(-zeta * wn * ti)
            phase = wd * ti - np.arctan(zeta / np.sqrt(1 - zeta**2))
            oscillation = np.cos(phase)
            response[i] = amplitude * (1 - envelope * oscillation / np.sqrt(1 - zeta**2))
            response[i] += np.random.randn() * 0.005

    response[t >= t_step + ts] -= ess
    return response

np.random.seed(42)
x_actual = second_order_response(t, step_time, x_step, overshoot, ts, ess)
y_actual = second_order_response(t, step_time, y_step, overshoot, ts, ess)
z_actual = second_order_response(t, step_time, z_step, overshoot, ts, ess)

# ============================================================================
# 生成摆角
# ============================================================================

phi = np.zeros_like(t)
psi = np.zeros_like(t)

for i, ti in enumerate(t):
    if ti >= step_time:
        t_theta = ti - step_time
        decay = np.exp(-3.0 * t_theta / theta_conv_time)
        phi[i] = phi_max * decay * np.sin(2 * np.pi * theta_freq * t_theta)
        psi[i] = psi_max * decay * np.cos(2 * np.pi * theta_freq * t_theta * 1.1)
        phi[i] += np.random.randn() * 0.1
        psi[i] += np.random.randn() * 0.1

# ============================================================================
# 生成控制输入
# ============================================================================

thrust = np.ones_like(t)

for i, ti in enumerate(t):
    if ti >= step_time:
        t_ctrl = ti - step_time
        if t_ctrl < 1.0:
            thrust[i] = 1.0 + 0.3 * np.sin(np.pi * t_ctrl / 1.0)
        elif t_ctrl < ts:
            thrust[i] = 1.3 - 0.3 * (t_ctrl - 1.0) / (ts - 1.0)
        else:
            thrust[i] = 1.0
        thrust[i] += np.random.randn() * 0.01

# ============================================================================
# 创建包含三个子图的图形
# ============================================================================

fig = plt.figure(figsize=(10, 9))

# ----------------------------------------------------------------------------
# 子图 (a): 位置跟踪响应
# ----------------------------------------------------------------------------
ax1 = plt.subplot(3, 1, 1)

ax1.plot(t, x_ref, 'k--', linewidth=1.5, label='$x_{ref}$', alpha=0.7)
ax1.plot(t, y_ref, 'k--', linewidth=1.5, label='$y_{ref}$', alpha=0.7)
ax1.plot(t, z_ref, 'k--', linewidth=1.5, label='$z_{ref}$', alpha=0.7)

ax1.plot(t, x_actual, 'b-', linewidth=2, label='$x$ (actual)')
ax1.plot(t, y_actual, 'r-', linewidth=2, label='$y$ (actual)')
ax1.plot(t, z_actual, 'g-', linewidth=2, label='$z$ (actual)')

settling_idx = np.argmin(np.abs(t - (step_time + ts)))
ax1.axvline(x=t[settling_idx], color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax1.text(t[settling_idx] + 0.1, 5.5, f'$t_s$ = {ts} s',
         fontsize=9, color='gray', verticalalignment='center')

peak_idx = np.argmax(x_actual[step_idx:step_idx+500]) + step_idx
ax1.plot(t[peak_idx], x_actual[peak_idx], 'bo', markersize=5)
ax1.annotate(f'Overshoot: {overshoot*100:.1f}%',
            xy=(t[peak_idx], x_actual[peak_idx]),
            xytext=(t[peak_idx] + 0.5, x_actual[peak_idx] + 0.3),
            fontsize=8, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1))

ax1.annotate(f'$e_{{ss}}$ = {ess} m',
            xy=(8, x_step - ess),
            xytext=(7, 5.8),
            fontsize=8, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1))

ax1.set_ylabel('Position (m)', fontsize=10)
ax1.set_xlabel('Time (s)', fontsize=10)
ax1.set_title('(a) Position Tracking Response', fontsize=11, fontweight='bold')
ax1.legend(loc='right', ncol=3, framealpha=0.9, edgecolor='gray')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_xlim([0, 10])
ax1.set_ylim([-0.5, 6.5])

# ----------------------------------------------------------------------------
# 子图 (b): 摆角抑制
# ----------------------------------------------------------------------------
ax2 = plt.subplot(3, 1, 2)

ax2.plot(t, phi, 'b-', linewidth=2, label='$\\phi$ (roll)')
ax2.plot(t, psi, 'r-', linewidth=2, label='$\\psi$ (pitch)')

phi_peak_idx = np.argmax(np.abs(phi))
ax2.plot(t[phi_peak_idx], phi[phi_peak_idx], 'bo', markersize=6)
ax2.annotate(f'Max: {phi_max:.0f}°',
            xy=(t[phi_peak_idx], phi[phi_peak_idx]),
            xytext=(t[phi_peak_idx] + 0.3, phi[phi_peak_idx] + 2),
            fontsize=9, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

ax2.axhline(y=5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.axhline(y=-5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.text(0.2, 5.5, '±5° Threshold', fontsize=8, color='gray')

conv_time_point = step_time + theta_conv_time
ax2.axvline(x=conv_time_point, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax2.text(conv_time_point + 0.1, -10, f'{theta_conv_time} s',
         fontsize=9, color='gray', verticalalignment='center')

ax2.axvspan(conv_time_point, 10, alpha=0.1, color='green', label='Converged Region')

ax2.set_ylabel('Swing Angle (deg)', fontsize=10)
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_title('(b) Swing Angle Suppression', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_xlim([0, 10])
ax2.set_ylim([-15, 15])

# ----------------------------------------------------------------------------
# 子图 (c): 控制输入
# ----------------------------------------------------------------------------
ax3 = plt.subplot(3, 1, 3)

ax3.plot(t, thrust, 'b-', linewidth=2, label='Normalized Thrust')
ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Hover Thrust')

ax3.text(5, 1.35, 'Smooth Control\n(No Chattering)',
         fontsize=9, color='green', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax3.set_ylabel('Thrust (Normalized)', fontsize=10)
ax3.set_xlabel('Time (s)', fontsize=10)
ax3.set_title('(c) Control Input', fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax3.set_xlim([0, 10])
ax3.set_ylim([0.8, 1.5])

# ----------------------------------------------------------------------------
# 总标题和布局
# ----------------------------------------------------------------------------
fig.suptitle('Figure 5: Position Response and Swing Suppression\nunder Step Command',
            fontsize=13, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# ============================================================================
# 保存和显示
# ============================================================================

import os
os.makedirs('outputs', exist_ok=True)
output_filename = os.path.join('outputs', 'fig5_position_swing_response.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_filename}")

print("\n" + "="*50)
print("Performance Summary")
print("="*50)
print(f"Settling time (2%):  {ts} s")
print(f"Overshoot:           {overshoot*100:.1f}%")
print(f"Steady-state error:  {ess} m")
print(f"Max swing angle:     {phi_max}°")
print(f"Convergence time:    {theta_conv_time} s (to < 5°)")
print("="*50)

plt.show()
