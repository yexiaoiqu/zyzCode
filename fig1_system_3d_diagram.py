"""
图1：无人机吊载系统三维物理示意图
论文：无人机吊载系统李雅普诺夫约束自适应模型预测控制
小节：系统建模

本脚本绘制系统的三维物理结构，展示内容：
- 惯性坐标系Oxyz
- 四旋翼无人机机身和旋翼
- 缆绳
- 悬挂载荷
- 摆角phi和psi定义
- 缆绳长度标注
- 风干扰箭头
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

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

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

fig = plt.figure(figsize=(14, 11), facecolor='white')
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=25, azim=-65)

origin = np.array([0.5, 0.5, 0])
axis_length = 1.8

_arrow3D(ax, origin[0], origin[1], origin[2], axis_length, 0, 0,
         mutation_scale=25, lw=3.5, arrowstyle='-|>', color='#C85A54')
ax.text(origin[0]+axis_length+0.25, origin[1], origin[2], r'$x$', fontsize=18, fontweight='bold', color='#C85A54')

_arrow3D(ax, origin[0], origin[1], origin[2], 0, axis_length, 0,
         mutation_scale=25, lw=3.5, arrowstyle='-|>', color='#5A9367')
ax.text(origin[0], origin[1]+axis_length+0.25, origin[2], r'$y$', fontsize=18, fontweight='bold', color='#5A9367')

_arrow3D(ax, origin[0], origin[1], origin[2], 0, 0, axis_length,
         mutation_scale=25, lw=3.5, arrowstyle='-|>', color='#5B7C99')
ax.text(origin[0]-0.15, origin[1], origin[2]+axis_length+0.25, r'$z$', fontsize=18, fontweight='bold', color='#5B7C99')

ax.scatter(origin[0], origin[1], origin[2], color='black', s=80, zorder=10)
ax.text(origin[0]-0.25, origin[1]-0.25, origin[2]-0.15, r'$O$', fontsize=14, fontweight='bold')

ax.text(origin[0]+0.3, origin[1]-0.5, origin[2]-0.35, 'Inertial Frame {O, x, y, z}',
        fontsize=11, style='italic', color='#333333')

ground_height = 0.2
ground_start_x = 1.5
ground_start_y = 1.5
ground_size = 3.2

grid_x = np.linspace(ground_start_x, ground_start_x + ground_size, 5)
grid_y = np.linspace(ground_start_y, ground_start_y + ground_size, 5)

for gx in grid_x:
    ax.plot([gx, gx], [ground_start_y, ground_start_y + ground_size],
            [ground_height, ground_height],
            color='#999999', linewidth=0.8, alpha=0.3, linestyle='--')
for gy in grid_y:
    ax.plot([ground_start_x, ground_start_x + ground_size], [gy, gy],
            [ground_height, ground_height],
            color='#999999', linewidth=0.8, alpha=0.3, linestyle='--')

ground_verts = np.array([
    [ground_start_x, ground_start_y, ground_height],
    [ground_start_x + ground_size, ground_start_y, ground_height],
    [ground_start_x + ground_size, ground_start_y + ground_size, ground_height],
    [ground_start_x, ground_start_y + ground_size, ground_height]
])
ground_poly = Poly3DCollection([ground_verts], alpha=0.08,
                                facecolor='#CCCCCC', edgecolor='#999999', linewidth=1.5)
ax.add_collection3d(ground_poly)

uav_x, uav_y, uav_z = 4.5, 3.5, 5.5
body_width, body_depth, body_height = 1.0, 0.8, 0.3

def draw_box(ax, center, width, depth, height, color='#D3D3D3', alpha=0.8):
    cx, cy, cz = center
    w, d, h = width/2, depth/2, height/2

    verts = np.array([
        [cx-w, cy-d, cz-h], [cx+w, cy-d, cz-h],
        [cx+w, cy+d, cz-h], [cx-w, cy+d, cz-h],
        [cx-w, cy-d, cz+h], [cx+w, cy-d, cz+h],
        [cx+w, cy+d, cz+h], [cx-w, cy+d, cz+h]
    ])

    faces = [
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[0], verts[3], verts[7], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]]
    ]

    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                           edgecolor='#333333', linewidth=2)
    ax.add_collection3d(poly)
    return verts

uav_verts = draw_box(ax, [uav_x, uav_y, uav_z], body_width, body_depth, body_height,
                      color='#D3D3D3', alpha=0.9)

rotor_radius = 0.22
arm_length = 0.5
rotor_positions = [
    [uav_x + arm_length, uav_y + arm_length, uav_z + body_height/2],
    [uav_x - arm_length, uav_y + arm_length, uav_z + body_height/2],
    [uav_x + arm_length, uav_y - arm_length, uav_z + body_height/2],
    [uav_x - arm_length, uav_y - arm_length, uav_z + body_height/2]
]

theta = np.linspace(0, 2*np.pi, 30)
for rx, ry, rz in rotor_positions:
    ax.plot([uav_x, rx], [uav_y, ry], [uav_z, rz],
            color='#333333', linewidth=3, solid_capstyle='round')
    circle_x = rx + rotor_radius * np.cos(theta)
    circle_y = ry + rotor_radius * np.sin(theta)
    circle_z = np.ones_like(theta) * rz
    ax.plot(circle_x, circle_y, circle_z, color='#666666', linewidth=2.5)
    ax.scatter(rx, ry, rz, color='#555555', s=60, zorder=10)

ax.scatter(uav_x, uav_y, uav_z, color='#A85A52', s=200,
           marker='o', edgecolors='#6B3935', linewidths=3, zorder=15)

ax.text(uav_x-1.5, uav_y-0.5, uav_z+0.8,
        r'UAV CoM  $\mathbf{p}=[x,y,z]^T$',
        fontsize=12, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5DEB3',
                  edgecolor='#8B7355', linewidth=2, alpha=0.95))

phi = 22 * np.pi / 180
psi = 30 * np.pi / 180
L = 3.2

payload_x = uav_x + L * np.sin(phi) * np.cos(psi)
payload_y = uav_y + L * np.sin(phi) * np.sin(psi)
payload_z = uav_z - L * np.cos(phi)

ax.plot([uav_x, payload_x], [uav_y, payload_y], [uav_z, payload_z],
        color='#333333', linewidth=4, solid_capstyle='round', zorder=8, label='Cable')

payload_size = 0.4
payload_verts = draw_box(ax, [payload_x, payload_y, payload_z],
                           payload_size, payload_size, payload_size,
                           color='#8B6F47', alpha=0.95)

ax.text(payload_x+0.9, payload_y-0.8, payload_z,
        r'Payload $(m_l)$',
        fontsize=12, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#DEB887',
                  edgecolor='#8B6F47', linewidth=2, alpha=0.95))

mid_x = (uav_x + payload_x) / 2
mid_y = (uav_y + payload_y) / 2
mid_z = (uav_z + payload_z) / 2
ax.text(mid_x + 0.9, mid_y - 0.6, mid_z,
        r'Cable Length $L(t)$',
        fontsize=11, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD',
                  edgecolor='#B8860B', linewidth=1.8, alpha=0.9))

ax.plot([uav_x, uav_x], [uav_y, uav_y], [uav_z, ground_height],
        color='#5B7C99', linestyle='--', linewidth=2, alpha=0.6, zorder=5)

ax.plot([payload_x, payload_x], [payload_y, payload_y], [payload_z, ground_height],
        color='#5B7C99', linestyle='--', linewidth=2, alpha=0.6, zorder=5)

ax.plot([uav_x, payload_x], [uav_y, payload_y], [ground_height, ground_height],
        color='#CD853F', linewidth=2.5, alpha=0.8, linestyle='-', zorder=6)

vertical_end_z = payload_z
ax.plot([uav_x, uav_x], [uav_y, uav_y], [uav_z, vertical_end_z],
        color='#5B7C99', linestyle='--', linewidth=3, alpha=0.7, zorder=7)

arc_radius = 1.2
arc_points = 40
arc_theta = np.linspace(0, phi, arc_points)

arc_x = uav_x + arc_radius * np.sin(arc_theta) * np.cos(psi)
arc_y = uav_y + arc_radius * np.sin(arc_theta) * np.sin(psi)
arc_z = uav_z - arc_radius * np.cos(arc_theta)
ax.plot(arc_x, arc_y, arc_z, color='#C85A54', linewidth=3.5, zorder=9)

mid_arc = arc_points // 2
ax.text(arc_x[mid_arc]+0.4, arc_y[mid_arc]+0.5, arc_z[mid_arc]+0.3,
        r'$\phi$ (swing angle)',
        fontsize=13, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#C85A54', linewidth=2.5, alpha=0.98))

x_ref_len = 1.0
ax.plot([uav_x, uav_x + x_ref_len], [uav_y, uav_y], [ground_height, ground_height],
        color='#333333', linewidth=2.5, alpha=0.8, zorder=6)

psi_arc_radius = 0.8
psi_arc_points = 35
psi_arc_theta = np.linspace(0, psi, psi_arc_points)
psi_arc_x = uav_x + psi_arc_radius * np.cos(psi_arc_theta)
psi_arc_y = uav_y + psi_arc_radius * np.sin(psi_arc_theta)
psi_arc_z = np.ones(psi_arc_points) * ground_height
ax.plot(psi_arc_x, psi_arc_y, psi_arc_z, color='#9370DB', linewidth=3, zorder=6)

mid_psi = psi_arc_points // 2
ax.text(psi_arc_x[mid_psi]+0.3, psi_arc_y[mid_psi]+0.3, ground_height+0.3,
        r'$\psi$ (yaw swing)',
        fontsize=12, fontweight='bold', color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#9370DB', linewidth=2, alpha=0.95))

wind_start_x = uav_x - 2.2
wind_start_y = uav_y - 1.8
wind_start_z = uav_z - 0.3

wind_end_x = uav_x - 0.5
wind_end_y = uav_y - 0.4
wind_end_z = uav_z - 0.1

wind_dx = wind_end_x - wind_start_x
wind_dy = wind_end_y - wind_start_y
wind_dz = wind_end_z - wind_start_z

_arrow3D(ax, wind_start_x, wind_start_y, wind_start_z,
         wind_dx, wind_dy, wind_dz,
         mutation_scale=35, lw=4, arrowstyle='-|>', color='#6495ED')

ax.text(wind_start_x + 0.2, wind_start_y - 0.7, wind_start_z - 0.2,
        'Wind Disturbance',
        fontsize=11, fontweight='bold', color='#333333', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E0F2F7',
                  edgecolor='#6495ED', linewidth=2, alpha=0.95))

ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_zlim([-0.2, 7])

ax.set_xlabel('X', fontsize=14, fontweight='bold', labelpad=12)
ax.set_ylabel('Y', fontsize=14, fontweight='bold', labelpad=12)
ax.set_zlabel('Z', fontsize=14, fontweight='bold', labelpad=12)

ax.tick_params(axis='both', which='major', labelsize=11)

ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#CCCCCC')
ax.yaxis.pane.set_edgecolor('#CCCCCC')
ax.zaxis.pane.set_edgecolor('#CCCCCC')

plt.title('Figure 1: 3D Physical Configuration of UAV Slung-Load System\n' +
          'With inertial frame, swing angle definitions and cable length',
          fontsize=13, fontweight='bold', pad=25)

plt.tight_layout()

import os
os.makedirs('outputs', exist_ok=True)
output_path = os.path.join('outputs', 'fig1_system_3d_diagram.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path}")
plt.show()
