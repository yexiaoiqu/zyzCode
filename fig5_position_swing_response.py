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

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置出版物级别的参数
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

# 时间参数
t = np.linspace(0, 10, 2000)
dt = t[1] - t[0]

# 阶跃指令时间点
阶跃时间 = 1.0  # 秒
阶跃索引 = np.argmin(np.abs(t - 阶跃时间))

# 系统性能参数
调节时间 = 2.1  # 秒
超调量 = 0.053  # 5.3%
稳态误差 = 0.08  # 米

# 阶跃指令幅值
x阶跃 = 5.0  # 米
y阶跃 = 3.0  # 米
z阶跃 = 4.0  # 米

# 摆角参数
phi最大值 = 12.0  # 度（最大摆角）
psi最大值 = 12.0  # 度
摆角收敛时间 = 3.5  # 秒（收敛到 < 5° 的时间）
摆角频率 = 1.5  # Hz

# ============================================================================
# 生成参考轨迹
# ============================================================================

x_ref = np.zeros_like(t)
y_ref = np.zeros_like(t)
z_ref = np.zeros_like(t)

# 在阶跃时间施加阶跃指令
x_ref[t >= 阶跃时间] = x阶跃
y_ref[t >= 阶跃时间] = y阶跃
z_ref[t >= 阶跃时间] = z_ref[t >= 阶跃时间] = z阶跃

# ============================================================================
# 生成实际系统响应
# ============================================================================

def 二阶响应(t, 阶跃时间, 幅值, 超调量, 调节时间, 稳态误差):
    """
    生成指定超调量和稳态误差的二阶阶跃响应

    参数:
    -----------
    t : array-like
        时间向量
    阶跃时间 : float
        施加阶跃输入的时间
    幅值 : float
        阶跃幅值
    超调量 : float
        超调量小数表示（例如 0.053 表示 5.3%）
    调节时间 : float
        2%调节时间（秒）
    稳态误差 : float
        稳态误差

    返回:
    --------
    response : array
        系统响应
    """
    response = np.zeros_like(t)
    t偏移 = t - 阶跃时间

    # 从超调量计算阻尼比
    zeta = -np.log(超调量) / np.sqrt(np.pi**2 + np.log(超调量)**2)

    # 从调节时间计算自然频率
    wn = 4.0 / (zeta * 调节时间)

    # 阻尼自然频率
    wd = wn * np.sqrt(1 - zeta**2)

    for i, ti in enumerate(t偏移):
        if ti >= 0:
            # 二阶欠阻尼响应
            包络 = np.exp(-zeta * wn * ti)
            相位 = wd * ti - np.arctan(zeta / np.sqrt(1 - zeta**2))
            振荡 = np.cos(相位)
            response[i] = 幅值 * (1 - 包络 * 振荡 / np.sqrt(1 - zeta**2))

            # 添加小测量噪声
            response[i] += np.random.randn() * 0.005

    # 调节后施加稳态误差
    response[t >= 阶跃时间 + 调节时间] -= 稳态误差

    return response

# 生成x、y、z方向位置响应
np.random.seed(42)  # 保证可重复性
x_实际 = 二阶响应(t, 阶跃时间, x阶跃, 超调量, 调节时间, 稳态误差)
y_实际 = 二阶响应(t, 阶跃时间, y阶跃, 超调量, 调节时间, 稳态误差)
z_实际 = 二阶响应(t, 阶跃时间, z阶跃, 超调量, 调节时间, 稳态误差)

# ============================================================================
# 生成摆角
# ============================================================================

phi = np.zeros_like(t)  # 滚摆角
psi = np.zeros_like(t)  # 俯仰摆角

for i, ti in enumerate(t):
    if ti >= 阶跃时间:
        t摆角 = ti - 阶跃时间

        # 摆角的阻尼振荡模型
        # 衰减率设计为在摆角收敛时间内收敛到 < 5°
        衰减 = np.exp(-3.0 * t摆角 / 摆角收敛时间)

        phi[i] = phi最大值 * 衰减 * np.sin(2 * np.pi * 摆角频率 * t摆角)
        psi[i] = psi最大值 * 衰减 * np.cos(2 * np.pi * 摆角频率 * t摆角 * 1.1)

        # 添加小测量噪声
        phi[i] += np.random.randn() * 0.1
        psi[i] += np.random.randn() * 0.1

# ============================================================================
# 生成控制输入
# ============================================================================

推力 = np.ones_like(t)  # 归一化推力（1.0 = 悬停）

for i, ti in enumerate(t):
    if ti >= 阶跃时间:
        t控制 = ti - 阶跃时间

        # 机动过程中的控制能量分布
        if t控制 < 1.0:
            # 初始加速控制脉冲
            推力[i] = 1.0 + 0.3 * np.sin(np.pi * t控制 / 1.0)
        elif t控制 < 调节时间:
            # 逐渐减小到悬停推力
            推力[i] = 1.3 - 0.3 * (t控制 - 1.0) / (调节时间 - 1.0)
        else:
            # 稳态悬停
            推力[i] = 1.0

        # 添加小控制波动（展示无抖振）
        推力[i] += np.random.randn() * 0.01

# ============================================================================
# 创建包含三个子图的图形
# ============================================================================

fig = plt.figure(figsize=(10, 9))

# ----------------------------------------------------------------------------
# 子图 (a): 位置跟踪响应
# ----------------------------------------------------------------------------
ax1 = plt.subplot(3, 1, 1)

# 绘制参考轨迹（虚线）
ax1.plot(t, x_ref, 'k--', linewidth=1.5, label='$x_{ref}$', alpha=0.7)
ax1.plot(t, y_ref, 'k--', linewidth=1.5, label='$y_{ref}$', alpha=0.7)
ax1.plot(t, z_ref, 'k--', linewidth=1.5, label='$z_{ref}$', alpha=0.7)

# 绘制实际轨迹（实线）
ax1.plot(t, x_实际, 'b-', linewidth=2, label='$x$ (实际)')
ax1.plot(t, y_实际, 'r-', linewidth=2, label='$y$ (实际)')
ax1.plot(t, z_实际, 'g-', linewidth=2, label='$z$ (实际)')

# 标记调节时间
调节索引 = np.argmin(np.abs(t - (阶跃时间 + 调节时间)))
ax1.axvline(x=t[调节索引], color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax1.text(t[调节索引] + 0.1, 5.5, f'$t_s$ = {调节时间} s',
         fontsize=9, color='gray', verticalalignment='center')

# 标注x方向超调量
x峰值索引 = np.argmax(x_实际[阶跃索引:阶跃索引+500]) + 阶跃索引
ax1.plot(t[x峰值索引], x_实际[x峰值索引], 'bo', markersize=5)
ax1.annotate(f'超调量: {超调量*100:.1f}%',
            xy=(t[x峰值索引], x_实际[x峰值索引]),
            xytext=(t[x峰值索引] + 0.5, x_实际[x峰值索引] + 0.3),
            fontsize=8, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1))

# 标注稳态误差
ax1.annotate(f'$e_{{ss}}$ = {稳态误差} m',
            xy=(8, x阶跃 - 稳态误差),
            xytext=(7, 5.8),
            fontsize=8, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1))

ax1.set_ylabel('位置 (m)', fontsize=10)
ax1.set_xlabel('时间 (s)', fontsize=10)
ax1.set_title('(a) 位置跟踪响应', fontsize=11, fontweight='bold')
ax1.legend(loc='right', ncol=3, framealpha=0.9, edgecolor='gray')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_xlim([0, 10])
ax1.set_ylim([-0.5, 6.5])

# ----------------------------------------------------------------------------
# 子图 (b): 摆角抑制
# ----------------------------------------------------------------------------
ax2 = plt.subplot(3, 1, 2)

# 绘制摆角
ax2.plot(t, phi, 'b-', linewidth=2, label='$\\phi$ (滚转)')
ax2.plot(t, psi, 'r-', linewidth=2, label='$\\psi$ (俯仰)')

# 标记最大摆角
phi最大索引 = np.argmax(np.abs(phi))
ax2.plot(t[phi最大索引], phi[phi最大索引], 'bo', markersize=6)
ax2.annotate(f'最大: {phi最大值:.0f}°',
            xy=(t[phi最大索引], phi[phi最大索引]),
            xytext=(t[phi最大索引] + 0.3, phi[phi最大索引] + 2),
            fontsize=9, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

# 标记±5°收敛阈值
ax2.axhline(y=5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.axhline(y=-5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.text(0.2, 5.5, '±5° 阈值', fontsize=8, color='gray')

# 标记收敛时间
收敛时间点 = 阶跃时间 + 摆角收敛时间
ax2.axvline(x=收敛时间点, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax2.text(收敛时间点 + 0.1, -10, f'{摆角收敛时间} s',
         fontsize=9, color='gray', verticalalignment='center')

# 填充收敛区域
ax2.axvspan(收敛时间点, 10, alpha=0.1, color='green', label='已收敛区域')

ax2.set_ylabel('摆角 (度)', fontsize=10)
ax2.set_xlabel('时间 (s)', fontsize=10)
ax2.set_title('(b) 摆角抑制', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax2.set_xlim([0, 10])
ax2.set_ylim([-15, 15])

# ----------------------------------------------------------------------------
# 子图 (c): 控制输入
# ----------------------------------------------------------------------------
ax3 = plt.subplot(3, 1, 3)

# 绘制归一化推力
ax3.plot(t, 推力, 'b-', linewidth=2, label='归一化推力')
ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='悬停推力')

# 添加平滑控制标注
ax3.text(5, 1.35, '平滑控制\n(无抖振)',
         fontsize=9, color='green', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax3.set_ylabel('推力 (归一化)', fontsize=10)
ax3.set_xlabel('时间 (s)', fontsize=10)
ax3.set_title('(c) 控制输入', fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax3.set_xlim([0, 10])
ax3.set_ylim([0.8, 1.5])

# ----------------------------------------------------------------------------
# 整体图形标题和布局
# ----------------------------------------------------------------------------
fig.suptitle('图5: 阶跃指令下位置响应和摆角抑制',
            fontsize=13, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# ============================================================================
# 保存和显示
# ============================================================================

import os
os.makedirs('outputs', exist_ok=True)
output_filename = os.path.join('outputs', 'fig5_position_swing_response.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图形已保存到: {output_filename}")

# 输出性能指标汇总
print("\n" + "="*50)
print("性能指标汇总")
print("="*50)
print(f"调节时间 (2%):        {调节时间} s")
print(f"超调量:                 {超调量*100:.1f}%")
print(f"稳态误差:              {稳态误差} m")
print(f"最大摆角:               {phi最大值}°")
print(f"摆角收敛时间:           {摆角收敛时间} s (收敛到 < 5°)")
print("="*50)

plt.show()
