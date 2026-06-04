# UAV 吊载系统模型预测控制仿真

这是一个基于**Lyapunov约束模型预测控制（Lyapunov-constrained Model Predictive Control, LC-MPC）**的四旋翼无人机吊载系统三维非线性仿真项目。

## 项目简介

本项目实现了无人机吊运悬挂载荷系统的完整动力学建模与控制仿真，包含：
- 完整的三维非线性多体动力学模型（四旋翼无人机 + 柔性绳索 + 悬挂载荷）
- 基于Lyapunov稳定性理论约束的自适应模型预测控制器
- 论文插图绘制（共9幅图）
- 所有图像自动输出到 `outputs/` 文件夹

## 系统模型

**状态维度**：12维状态向量
```
x = [x, y, z, vx, vy, vz, alpha, beta, alpha_dot, beta_dot, phi, theta]
```
- `x,y,z`: UAV 位置
- `vx,vy,vz`: UAV 速度
- `alpha,beta`: 载荷摆角
- `alpha_dot,beta_dot`: 摆角速度
- `phi,theta`: UAV 滚转/俯仰角

**控制输入**：4维
```
u = [T, tau_phi, tau_theta, tau_psi]
```
- `T`: 总推力
- `tau_phi, tau_theta, tau_psi`: 扭矩

## 控制算法特点

1. **Lyapunov终端约束**：保证递归可行性与指数稳定性
2. **自适应权重调整**：根据实时误差动态平衡轨迹跟踪与摆角抑制
3. **软约束处理**：约束松弛处理提高数值鲁棒性
4. **状态观测器**：降阶观测器估计不可测状态

## 依赖安装

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 安装依赖
pip install numpy matplotlib scipy
```

**必需依赖**：
- `numpy` - 数值计算
- `matplotlib` - 可视化绘图
- `scipy` - 优化求解 (`minimize`)、积分、线性代数、滤波

## 文件说明与运行方法

| 文件名 | 功能 | 输出图像 |
|--------|------|----------|
| `fig01_system_3d_structure.py` | 图1 - UAV吊载系统三维物理结构图 | `fig1_system_3d_diagram.png` |
| `fig02_control_block_diagram.py` | 图2 - 控制系统整体结构框图 | `fig2_control_block_diagram.png` |
| `fig03_simulation_figures.py` | 图4-9 - 仿真结果可视化（轨迹跟踪、位置响应、风干扰鲁棒性、质量变化恢复、控制策略对比） | `fig4_3d_trajectory_tracking.png`<br>`fig5_position_swing_response.png`<br>`fig6_adaptive_vs_fixed_comparison.png`<br>`fig7_wind_disturbance_robustness.png`<br>`fig8_mass_change_recovery.png`<br>`fig9_comparative_performance.png` |
| `fig04_trajectory_tracking.py` | 图4 - 标称条件下三维轨迹跟踪可视化 | `trajectory_3d_nominal.png` |
| `fig05_main_simulation.py` | 主仿真程序 - 完整闭环仿真，包含系统建模、控制器实现、轨迹跟踪与可视化 | `results_nominal.png`<br>`results_wind.png`<br>`results_mass_change.png`<br>`trajectory_3d_nominal.png`<br>`simulation_results.json` |

---

### 运行方法

```bash
# 图1 - 系统三维结构图
python fig01_system_3d_structure.py

# 图2 - 控制系统框图
python fig02_control_block_diagram.py

# 图3 - 仿真结果可视化（图4-9）
python fig03_simulation_figures.py

# 图4 - 三维轨迹跟踪
python fig04_trajectory_tracking.py

# 主仿真程序
python fig05_main_simulation.py
```

## 输出图像位置与命名

所有图像自动保存到 `outputs/` 文件夹：

| 输出文件 | 描述 |
|----------|------|
| `fig1_system_3d_diagram.png` | 图1 - UAV吊载系统三维物理配置 |
| `fig2_control_block_diagram.png` | 图2 - 控制系统整体结构框图 |
| `fig4_3d_trajectory_tracking.png` | 图4 - 标称条件下三维轨迹跟踪 |
| `fig5_position_swing_response.png` | 图5 - 阶跃响应位置和摆角 |
| `fig6_adaptive_vs_fixed_comparison.png` | 图6 - 自适应MPC vs 固定权重MPC对比 |
| `fig7_wind_disturbance_robustness.png` | 图7 - 风干扰下的鲁棒性 |
| `fig8_mass_change_recovery.png` | 图8 - 突发质量变化动态恢复 |
| `fig9_comparative_performance.png` | 图9 - 控制策略性能对比分析 |
| `results_nominal.png` | 标称场景仿真结果 |
| `results_wind.png` | 风干扰场景仿真结果 |
| `results_mass_change.png` | 质量变化场景仿真结果 |
| `trajectory_3d_nominal.png` | 标称场景三维轨迹 |
| `simulation_results.json` | 仿真结果数据（JSON格式） |

## 性能指标

| 控制器类型 | 调节时间 (s) | 跟踪RMSE (m) | 最大摆角 (°) | 恢复时间 (s) |
|------------|--------------|--------------|--------------|--------------|
| PID | 3.8 | 0.285 | 28.5 | 4.2 |
| 线性MPC | 2.9 | 0.195 | 22.3 | 3.1 |
| 固定权重MPC | 2.1 | 0.124 | 15.8 | 2.5 |
| **本文（自适应）** | **1.3** | **0.068** | **9.2** | **1.8** |

本文方法相对固定权重MPC改进：
- 调节时间减少 38%
- 跟踪RMSE减少 45%
- 最大摆角减少 42%
- 恢复时间减少 28%
