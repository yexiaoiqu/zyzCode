"""
图1: 无人机吊载系统三维物理结构图
论文: 无人机吊载系统Lyapunov约束自适应模型预测控制
章节: 系统建模

该脚本绘制系统的三维物理配置图，展示：
- 惯性坐标系Oxyz
- 四旋翼无人机机体和旋翼
- 缆绳
- 悬挂载荷
- 摆角phi和psi定义
- 缆绳长度标注
- 风干扰箭头
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

# 3D箭头类，用于在3D坐标系中绘制箭头
class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# 添加3D箭头到坐标轴
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

# 创建图形和3D坐标轴
fig = plt.figure(figsize=(14, 11), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

# 设置观察视角
ax.view_init(elev=25, azim=-65)

# 步骤1: 绘制三维坐标系
原点 = np.array([0.5, 0.5, 0])
轴长度 = 1.8

# X轴 (红色 - 柔和色调)
_arrow3D(ax, 原点[0], 原点[1], 原点[2], 轴长度, 0, 0,
         mutation_scale=25, lw=3.5, arrowstyle='-|>', color='#C85A54')
ax.text(原点[0]+轴长度+0.25, 原点[1], 原点[2], r'$x$', fontsize=18, fontweight='bold', color='#C85A54')

# Y轴 (绿色 - 柔和色调)
_arrow3D(ax, 原点[0], 原点[1], 原点[2], 0, 轴长度, 0,
         mutation_scale=25, lw=3.5, arrowstyle='-|>', color='#5A9367')
ax.text(原点[0], 原点[1]+轴长度+0.25, 原点[2], r'$y$', fontsize=18, fontweight='bold', color='#5A9367')

# Z轴 (蓝色 - 柔和色调)
_arrow3D(ax, 原点[0], 原点[1], 原点[2], 0, 0, 轴长度,
         mutation_scale=25, lw=3.5, arrowstyle='-|>', color='#5B7C99')
ax.text(原点[0]-0.15, 原点[1], 原点[2]+轴长度+0.25, r'$z$', fontsize=18, fontweight='bold', color='#5B7C99')

# 原点标记
ax.scatter(原点[0], 原点[1], 原点[2], color='black', s=80, zorder=10)
ax.text(原点[0]-0.25, 原点[1]-0.25, 原点[2]-0.15, r'$O$', fontsize=14, fontweight='bold')

# 坐标系标签
ax.text(原点[0]+0.3, 原点[1]-0.5, 原点[2]-0.35, '惯性坐标系 {O, x, y, z}',
        fontsize=11, style='italic', color='#333333')

# 步骤2: 绘制带网格的地面平面
地面高度 = 0.2
地面起始_x = 1.5
地面起始_y = 1.5
地面尺寸 = 3.2

网格x = np.linspace(地面起始_x, 地面起始_x + 地面尺寸, 5)
网格y = np.linspace(地面起始_y, 地面起始_y + 地面尺寸, 5)

for gx in 网格x:
    ax.plot([gx, gx], [地面起始_y, 地面起始_y + 地面尺寸],
            [地面高度, 地面高度],
            color='#999999', linewidth=0.8, alpha=0.3, linestyle='--')
for gy in 网格y:
    ax.plot([地面起始_x, 地面起始_x + 地面尺寸], [gy, gy],
            [地面高度, 地面高度],
            color='#999999', linewidth=0.8, alpha=0.3, linestyle='--')

地面顶点 = np.array([
    [地面起始_x, 地面起始_y, 地面高度],
    [地面起始_x + 地面尺寸, 地面起始_y, 地面高度],
    [地面起始_x + 地面尺寸, 地面起始_y + 地面尺寸, 地面高度],
    [地面起始_x, 地面起始_y + 地面尺寸, 地面高度]
])
地面平面 = Poly3DCollection([地面顶点], alpha=0.08,
                                facecolor='#CCCCCC', edgecolor='#999999', linewidth=1.5)
ax.add_collection3d(地面平面)

# 步骤3: 绘制无人机机体和质心
无人机_x, 无人机_y, 无人机_z = 4.5, 3.5, 5.5
机体宽度, 机体深度, 机体高度 = 1.0, 0.8, 0.3

# 绘制长方体盒子函数
def 绘制盒子(ax, 中心, 宽度, 深度, 高度, 颜色='#D3D3D3', 透明度=0.8):
    cx, cy, cz = 中心
    w, d, h = 宽度/2, 深度/2, 高度/2

    顶点 = np.array([
        [cx-w, cy-d, cz-h], [cx+w, cy-d, cz-h],
        [cx+w, cy+d, cz-h], [cx-w, cy+d, cz-h],
        [cx-w, cy-d, cz+h], [cx+w, cy-d, cz+h],
        [cx+w, cy+d, cz+h], [cx-w, cy+d, cz+h]
    ])

    面 = [
        [顶点[0], 顶点[1], 顶点[5], 顶点[4]],
        [顶点[2], 顶点[3], 顶点[7], 顶点[6]],
        [顶点[0], 顶点[3], 顶点[7], 顶点[4]],
        [顶点[1], 顶点[2], 顶点[6], 顶点[5]],
        [顶点[0], 顶点[1], 顶点[2], 顶点[3]],
        [顶点[4], 顶点[5], 顶点[6], 顶点[7]]
    ]

    poly = Poly3DCollection(面, alpha=透明度, facecolor=颜色,
                           edgecolor='#333333', linewidth=2)
    ax.add_collection3d(poly)
    return 顶点

无人机顶点 = 绘制盒子(ax, [无人机_x, 无人机_y, 无人机_z], 机体宽度, 机体深度, 机体高度,
                        颜色='#D3D3D3', 透明度=0.9)

旋翼半径 = 0.22
机臂长度 = 0.5
旋翼位置 = [
    [无人机_x + 机臂长度, 无人机_y + 机臂长度, 无人机_z + 机体高度/2],
    [无人机_x - 机臂长度, 无人机_y + 机臂长度, 无人机_z + 机体高度/2],
    [无人机_x + 机臂长度, 无人机_y - 机臂长度, 无人机_z + 机体高度/2],
    [无人机_x - 机臂长度, 无人机_y - 机臂长度, 无人机_z + 机体高度/2]
]

theta = np.linspace(0, 2*np.pi, 30)
for rx, ry, rz in 旋翼位置:
    ax.plot([无人机_x, rx], [无人机_y, ry], [无人机_z, rz],
            color='#333333', linewidth=3, solid_capstyle='round')
    circle_x = rx + 旋翼半径 * np.cos(theta)
    circle_y = ry + 旋翼半径 * np.sin(theta)
    circle_z = np.ones_like(theta) * rz
    ax.plot(circle_x, circle_y, circle_z, color='#666666', linewidth=2.5)
    ax.scatter(rx, ry, rz, color='#555555', s=60, zorder=10)

# 无人机质心标记
ax.scatter(无人机_x, 无人机_y, 无人机_z, color='#A85A52', s=200,
           marker='o', edgecolors='#6B3935', linewidths=3, zorder=15)

ax.text(无人机_x-1.5, 无人机_y-0.5, 无人机_z+0.8,
        r'无人机质心  $\mathbf{p}=[x,y,z]^T$',
        fontsize=12, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5DEB3',
                  edgecolor='#8B7355', linewidth=2, alpha=0.95))

# 步骤4: 绘制三维缆绳和载荷
phi = 22 * np.pi / 180  # 摆角phi
psi = 30 * np.pi / 180  # 偏航角psi
L = 3.2  # 缆绳长度

载荷_x = 无人机_x + L * np.sin(phi) * np.cos(psi)
载荷_y = 无人机_y + L * np.sin(phi) * np.sin(psi)
载荷_z = 无人机_z - L * np.cos(phi)

# 绘制缆绳
ax.plot([无人机_x, 载荷_x], [无人机_y, 载荷_y], [无人机_z, 载荷_z],
        color='#333333', linewidth=4, solid_capstyle='round', zorder=8, label='缆绳')

# 绘制载荷
载荷尺寸 = 0.4
载荷顶点 = 绘制盒子(ax, [载荷_x, 载荷_y, 载荷_z],
                           载荷尺寸, 载荷尺寸, 载荷尺寸,
                           颜色='#8B6F47', 透明度=0.95)

# 载荷标签
ax.text(载荷_x+0.9, 载荷_y-0.8, 载荷_z,
        r'载荷 $(m_l)$',
        fontsize=12, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#DEB887',
                  edgecolor='#8B6F47', linewidth=2, alpha=0.95))

# 缆绳长度标签
中点_x = (无人机_x + 载荷_x) / 2
中点_y = (无人机_y + 载荷_y) / 2
中点_z = (无人机_z + 载荷_z) / 2
ax.text(中点_x + 0.9, 中点_y - 0.6, 中点_z,
        r'缆绳长度 $L(t)$',
        fontsize=11, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD',
                  edgecolor='#B8860B', linewidth=1.8, alpha=0.9))

# 步骤4.4: 绘制psi角的投影线
ax.plot([无人机_x, 无人机_x], [无人机_y, 无人机_y], [无人机_z, 地面高度],
        color='#5B7C99', linestyle='--', linewidth=2, alpha=0.6, zorder=5)

ax.plot([载荷_x, 载荷_x], [载荷_y, 载荷_y], [载荷_z, 地面高度],
        color='#5B7C99', linestyle='--', linewidth=2, alpha=0.6, zorder=5)

ax.plot([无人机_x, 载荷_x], [无人机_y, 载荷_y], [地面高度, 地面高度],
        color='#CD853F', linewidth=2.5, alpha=0.8, linestyle='-', zorder=6)

# 步骤5: 绘制摆角phi
垂直端点_z = 载荷_z
ax.plot([无人机_x, 无人机_x], [无人机_y, 无人机_y], [无人机_z, 垂直端点_z],
        color='#5B7C99', linestyle='--', linewidth=3, alpha=0.7, zorder=7)

圆弧半径 = 1.2
圆弧点数 = 40
圆弧_theta = np.linspace(0, phi, 圆弧点数)

圆弧_x = 无人机_x + 圆弧半径 * np.sin(圆弧_theta) * np.cos(psi)
圆弧_y = 无人机_y + 圆弧半径 * np.sin(圆弧_theta) * np.sin(psi)
圆弧_z = 无人机_z - 圆弧半径 * np.cos(圆弧_theta)
ax.plot(圆弧_x, 圆弧_y, 圆弧_z, color='#C85A54', linewidth=3.5, zorder=9)

中点_圆弧 = 圆弧点数 // 2
ax.text(圆弧_x[中点_圆弧]+0.4, 圆弧_y[中点_圆弧]+0.5, 圆弧_z[中点_圆弧]+0.3,
        r'$\phi$ (摆角)',
        fontsize=13, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#C85A54', linewidth=2.5, alpha=0.98))

# 步骤6: 在地面平面绘制偏航角psi
x参考长度 = 1.0
ax.plot([无人机_x, 无人机_x + x参考长度], [无人机_y, 无人机_y], [地面高度, 地面高度],
        color='#333333', linewidth=2.5, alpha=0.8, zorder=6)

psi圆弧半径 = 0.8
psi圆弧点数 = 35
psi圆弧_theta = np.linspace(0, psi, psi圆弧点数)
psi圆弧_x = 无人机_x + psi圆弧半径 * np.cos(psi圆弧_theta)
psi圆弧_y = 无人机_y + psi圆弧半径 * np.sin(psi圆弧_theta)
psi圆弧_z = np.ones(psi圆弧点数) * 地面高度
ax.plot(psi圆弧_x, psi圆弧_y, psi圆弧_z, color='#9370DB', linewidth=3, zorder=6)

中点_psi = psi圆弧点数 // 2
ax.text(psi圆弧_x[中点_psi]+0.3, psi圆弧_y[中点_psi]+0.3, 地面高度+0.3,
        r'$\psi$ (摆动偏角)',
        fontsize=12, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#9370DB', linewidth=2, alpha=0.95))

# 步骤7: 绘制三维风干扰箭头
风起始_x = 无人机_x - 2.2
风起始_y = 无人机_y - 1.8
风起始_z = 无人机_z - 0.3

风结束_x = 无人机_x - 0.5
风结束_y = 无人机_y - 0.4
风结束_z = 无人机_z - 0.1

风dx = 风结束_x - 风起始_x
风dy = 风结束_y - 风起始_y
风dz = 风结束_z - 风起始_z

_arrow3D(ax, 风起始_x, 风起始_y, 风起始_z,
         风dx, 风dy, 风dz,
         mutation_scale=35, lw=4, arrowstyle='-|>', color='#6495ED')

ax.text(风起始_x + 0.2, 风起始_y - 0.7, 风起始_z - 0.2,
        '风干扰',
        fontsize=11, fontweight='bold', color='#333333', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E0F2F7',
                  edgecolor='#6495ED', linewidth=2, alpha=0.95))

# 设置图形属性
ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_zlim([-0.2, 7])

ax.set_xlabel('X', fontsize=14, fontweight='bold', labelpad=12)
ax.set_ylabel('Y', fontsize=14, fontweight='bold', labelpad=12)
ax.set_zlabel('Z', fontsize=14, fontweight='bold', labelpad=12)

ax.tick_params(axis='both', which='major', labelsize=11)

ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)

# 设置透明坐标平面
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#CCCCCC')
ax.yaxis.pane.set_edgecolor('#CCCCCC')
ax.zaxis.pane.set_edgecolor('#CCCCCC')

plt.title('图1 无人机吊载系统三维物理配置\n' +
          '含惯性坐标系、摆角定义和缆绳长度',
          fontsize=13, fontweight='bold', pad=25)

plt.tight_layout()

# 保存图形到outputs文件夹
import os
os.makedirs('outputs', exist_ok=True)
output_path = os.path.join('outputs', 'fig1_system_3d_diagram.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图形已保存到: {output_path}")
plt.show()
