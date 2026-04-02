import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.lines as mlines

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(7, 9.5, '无人机悬挂负载系统概览图',
        fontsize=20, fontweight='bold', ha='center', va='top')

# 定义颜色
color_external = '#e8f4f8'
color_uav = '#f0f4f8'
color_cable = '#fef5e7'
color_payload = '#f5f0f0'
color_mpc = '#e8f8f5'
color_io = '#fef9e7'

# 边框颜色
edge_external = '#5a9fb8'
edge_uav = '#6b8fb3'
edge_cable = '#d4a574'
edge_payload = '#b87f7f'
edge_mpc = '#5fa88a'
edge_io = '#d4a574'

# 1. 外部干扰模块 (左上)
box1 = FancyBboxPatch((0.5, 7), 2, 1.5,
                       boxstyle="round,pad=0.1",
                       facecolor=color_external,
                       edgecolor=edge_external,
                       linewidth=2)
ax.add_patch(box1)
ax.text(1.5, 7.95, '外部干扰', fontsize=14, fontweight='bold', ha='center', va='center')
ax.text(1.5, 7.6, '• 风速干扰', fontsize=11, ha='center', va='center')
ax.text(1.5, 7.3, '• 负载质量变化', fontsize=11, ha='center', va='center')

# 2. 无人机模块 (中上)
box2 = FancyBboxPatch((3.5, 6.8), 2.5, 1.8,
                       boxstyle="round,pad=0.1",
                       facecolor=color_uav,
                       edgecolor=edge_uav,
                       linewidth=2)
ax.add_patch(box2)
ax.text(4.75, 8.15, '无人机 (UAV)', fontsize=14, fontweight='bold', ha='center', va='center')
ax.text(4.75, 7.75, '• 位置控制系统', fontsize=11, ha='center', va='center')
ax.text(4.75, 7.45, '• 姿态控制系统', fontsize=11, ha='center', va='center')
ax.text(4.75, 7.15, '• 推进系统', fontsize=11, ha='center', va='center')

# 3. 缆绳模块 (中间)
box3 = FancyBboxPatch((3.8, 4.5), 2, 1.3,
                       boxstyle="round,pad=0.1",
                       facecolor=color_cable,
                       edgecolor=edge_cable,
                       linewidth=2)
ax.add_patch(box3)
ax.text(4.8, 5.5, '缆绳', fontsize=14, fontweight='bold', ha='center', va='center')
ax.text(4.8, 5.15, '• 弹性阻尼特性', fontsize=11, ha='center', va='center')
ax.text(4.8, 4.85, '• 缆绳张力约束', fontsize=11, ha='center', va='center')

# 4. 悬挂负载模块 (中下)
box4 = FancyBboxPatch((3.5, 2.5), 2.5, 1.3,
                       boxstyle="round,pad=0.1",
                       facecolor=color_payload,
                       edgecolor=edge_payload,
                       linewidth=2)
ax.add_patch(box4)
ax.text(4.75, 3.45, '悬挂负载', fontsize=14, fontweight='bold', ha='center', va='center')
ax.text(4.75, 3.1, '• 负载摆动动态', fontsize=11, ha='center', va='center')
ax.text(4.75, 2.8, '• 质量变化', fontsize=11, ha='center', va='center')

# 5. MPC控制系统 (右中)
box5 = FancyBboxPatch((9, 4.2), 3.5, 2.2,
                       boxstyle="round,pad=0.1",
                       facecolor=color_mpc,
                       edgecolor=edge_mpc,
                       linewidth=2)
ax.add_patch(box5)
ax.text(10.75, 6.0, 'MPC控制系统', fontsize=14, fontweight='bold', ha='center', va='center')
ax.text(10.75, 5.6, '• 轨迹跟踪控制', fontsize=11, ha='center', va='center')
ax.text(10.75, 5.3, '• 摆动抑制控制', fontsize=11, ha='center', va='center')
ax.text(10.75, 5.0, '• 约束处理', fontsize=11, ha='center', va='center')
ax.text(10.75, 4.7, '• 李雅普诺夫稳定性保证', fontsize=11, ha='center', va='center')

# 6. 输入模块 (右上)
box6 = FancyBboxPatch((9.5, 7.5), 1.8, 0.8,
                       boxstyle="round,pad=0.08",
                       facecolor=color_io,
                       edgecolor=edge_io,
                       linewidth=1.5)
ax.add_patch(box6)
ax.text(10.4, 8.05, '输入', fontsize=12, fontweight='bold', ha='center', va='center')
ax.text(10.4, 7.75, '期望轨迹', fontsize=10, ha='center', va='center')

# 7. 输出模块 (右下)
box7 = FancyBboxPatch((9.3, 2.2), 2.2, 0.8,
                       boxstyle="round,pad=0.08",
                       facecolor=color_io,
                       edgecolor=edge_io,
                       linewidth=1.5)
ax.add_patch(box7)
ax.text(10.4, 2.75, '输出', fontsize=12, fontweight='bold', ha='center', va='center')
ax.text(10.4, 2.45, '轨迹跟踪与摆动抑制', fontsize=10, ha='center', va='center')

# 系统边界虚线框
boundary = FancyBboxPatch((3.2, 2.2), 3.3, 6.7,
                          boxstyle="round,pad=0.15",
                          facecolor='none',
                          edgecolor='#94a3b8',
                          linewidth=2,
                          linestyle='--',
                          alpha=0.5)
ax.add_patch(boundary)
ax.text(4.85, 1.7, '被控对象', fontsize=11, ha='center', va='center',
        color='#64748b', style='italic')

# 箭头定义函数
def draw_arrow(ax, x1, y1, x2, y2, color='#4a5568', width=2, style='solid', connectionstyle="arc3"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->',
                           color=color,
                           linewidth=width,
                           linestyle=style if style != 'solid' else '-',
                           mutation_scale=20,
                           connectionstyle=connectionstyle)
    ax.add_patch(arrow)
    return arrow

# 箭头1: 外部干扰 → 无人机 (干扰输入)
draw_arrow(ax, 2.5, 7.75, 3.5, 7.7, color='#4a5568', width=2)
ax.text(2.9, 8.0, '干扰输入', fontsize=10, ha='center', color='#4a5568')

# 箭头2: 外部干扰 → 悬挂负载 (环境影响 - 虚线曲线)
arrow2 = FancyArrowPatch((1.5, 7.0), (3.5, 3.0),
                        arrowstyle='->',
                        color='#4a5568',
                        linewidth=2,
                        linestyle='--',
                        mutation_scale=20,
                        connectionstyle="arc3,rad=.3")
ax.add_patch(arrow2)
ax.text(2.0, 4.8, '环境影响', fontsize=9, ha='center', color='#4a5568')

# 箭头3: 无人机 → 缆绳 (悬挂连接)
draw_arrow(ax, 4.75, 6.8, 4.75, 5.8, color='#4a5568', width=2.5, connectionstyle="arc3,rad=0")
ax.text(5.2, 6.3, '悬挂连接', fontsize=10, ha='left', color='#4a5568')

# 箭头4: 缆绳 → 悬挂负载 (力传递)
draw_arrow(ax, 4.75, 4.5, 4.75, 3.8, color='#4a5568', width=2.5, connectionstyle="arc3,rad=0")
ax.text(5.2, 4.15, '力传递', fontsize=10, ha='left', color='#4a5568')

# 箭头5: 悬挂负载 → 缆绳 (摆动影响 - 虚线)
draw_arrow(ax, 4.5, 3.8, 4.5, 4.5, color='#4a5568', width=2, style='--', connectionstyle="arc3,rad=0")
ax.text(4.0, 4.15, '摆动影响', fontsize=9, ha='right', color='#4a5568')

# 箭头6: 无人机 → MPC控制器 (状态反馈 - 橙色弧线)
arrow6 = FancyArrowPatch((6.0, 7.8), (9.0, 5.8),
                        arrowstyle='->',
                        color='#d97706',
                        linewidth=2.5,
                        mutation_scale=20,
                        connectionstyle="arc3,rad=.3")
ax.add_patch(arrow6)
ax.text(7.5, 7.2, '状态反馈', fontsize=10, ha='center', color='#d97706', fontweight='bold')
ax.text(7.5, 6.9, '(位置、速度、姿态)', fontsize=9, ha='center', color='#4a5568')

# 箭头7: 悬挂负载 → MPC控制器 (负载反馈 - 橙色弧线)
arrow7 = FancyArrowPatch((6.0, 3.2), (9.0, 4.7),
                        arrowstyle='->',
                        color='#d97706',
                        linewidth=2.5,
                        mutation_scale=20,
                        connectionstyle="arc3,rad=-.3")
ax.add_patch(arrow7)
ax.text(7.5, 3.5, '负载反馈', fontsize=10, ha='center', color='#d97706', fontweight='bold')
ax.text(7.5, 3.2, '(摆动角、角速度)', fontsize=9, ha='center', color='#4a5568')

# 箭头8: MPC控制器 → 无人机 (控制指令 - 绿色)
arrow8 = FancyArrowPatch((9.0, 5.3), (6.0, 7.3),
                        arrowstyle='->',
                        color='#16a34a',
                        linewidth=3,
                        mutation_scale=20,
                        connectionstyle="arc3,rad=.15")
ax.add_patch(arrow8)
ax.text(7.8, 6.5, '控制指令', fontsize=10, ha='center', color='#16a34a', fontweight='bold')
ax.text(7.8, 6.2, '(推力、姿态指令)', fontsize=9, ha='center', color='#4a5568')

# 箭头9: 输入 → MPC控制器
draw_arrow(ax, 10.4, 7.5, 10.4, 6.4, color='#4a5568', width=2, connectionstyle="arc3,rad=0")

# 箭头10: MPC控制器 → 输出
draw_arrow(ax, 10.4, 4.2, 10.4, 3.0, color='#4a5568', width=2, connectionstyle="arc3,rad=0")

# 图例
legend_x = 0.5
legend_y = 1.5
ax.text(legend_x, legend_y, '图例:', fontsize=12, fontweight='bold', color='#2d3748')

# 图例项1
ax.plot([legend_x, legend_x+0.5], [legend_y-0.3, legend_y-0.3],
        color='#4a5568', linewidth=2, marker='>', markersize=8, markerfacecolor='#4a5568')
ax.text(legend_x+0.7, legend_y-0.3, '物理连接/力传递', fontsize=10, va='center', color='#4a5568')

# 图例项2
ax.plot([legend_x, legend_x+0.5], [legend_y-0.6, legend_y-0.6],
        color='#d97706', linewidth=2, marker='>', markersize=8, markerfacecolor='#d97706')
ax.text(legend_x+0.7, legend_y-0.6, '状态反馈信号', fontsize=10, va='center', color='#4a5568')

# 图例项3
ax.plot([legend_x, legend_x+0.5], [legend_y-0.9, legend_y-0.9],
        color='#16a34a', linewidth=2, marker='>', markersize=8, markerfacecolor='#16a34a')
ax.text(legend_x+0.7, legend_y-0.9, '控制信号输出', fontsize=10, va='center', color='#4a5568')

# 图例项4
ax.plot([legend_x, legend_x+0.5], [legend_y-1.2, legend_y-1.2],
        color='#4a5568', linewidth=2, linestyle='--', marker='>', markersize=8, markerfacecolor='#4a5568')
ax.text(legend_x+0.7, legend_y-1.2, '间接影响', fontsize=10, va='center', color='#4a5568')

import os
os.makedirs('outputs', exist_ok=True)
plt.tight_layout()
plt.savefig('./outputs/uav_system_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("图片已保存到: ./outputs/uav_system_diagram.png")
plt.show()