import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge
import matplotlib
from scipy.integrate import solve_ivp
matplotlib.rc("font",family='Kai')
plt.rcParams['axes.unicode_minus'] =False

# 定义Lorenz63系统的微分方程
def lorenz63(t, xyz, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# 生成Lorenz63时间序列
dt = 0.01
t_max = 100
t_eval = np.arange(0, t_max, dt)
initial_state = [1.0, 1.0, 1.0]  # 初始状态

# 使用scipy的ODE求解器生成时间序列
solution = solve_ivp(lorenz63, [0, t_max], initial_state, t_eval=t_eval, method='RK45')
t = solution.t
trajectory = solution.y.T  # 形状为 (n_points, 3)

# 丢弃前2000个点（暂态）使系统稳定在吸引子上
discard = 2000
if discard < len(trajectory):
    t = t[discard:]
    trajectory = trajectory[discard:]

# 准备训练和测试数据
train_size = int(len(trajectory) * 0.8)
X_train = trajectory[:train_size, :]
Y_train = trajectory[1:train_size+1, :]  # 预测下一个时间点的状态
X_test = trajectory[train_size:-1, :]
Y_test = trajectory[train_size+1:, :]

# 创建Reservoir（储备池）和Ridge读出层 - 为混沌系统增加节点数
reservoir = Reservoir(units=3000, spectral_radius=0.9, input_scaling=0.5, 
                     leak_rate=0.05, connectivity=0.2)
readout = Ridge(ridge=1e-6)

# 构建流水线：储备池 >> 读出层
pipeline = reservoir >> readout

# 训练模型
pipeline = pipeline.fit(X_train, Y_train)

# 测试模型 - 使用预测进行迭代
n_steps = len(X_test)
Y_pred = np.zeros((n_steps, 3))
Y_pred[0] = pipeline.run(X_test[0].reshape(1, -1))  # 第一步预测

# 自由运行预测
for i in range(1, n_steps):
    Y_pred[i] = pipeline.run(Y_pred[i-1].reshape(1, -1))

# 计算均方误差
mse = np.mean((Y_pred - Y_test) ** 2)
print(f"测试集MSE = {mse:.6f}")

# 可视化结果
fig = plt.figure(figsize=(18, 12))

# 1. 绘制Lorenz吸引子 - 3D视图
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(Y_test[:, 0], Y_test[:, 1], Y_test[:, 2], 'b-', 
         lw=1, alpha=0.8, label='真实轨迹')
ax1.plot(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], 'r--', 
         lw=1, alpha=1, label='预测轨迹')
ax1.set_title('Lorenz63吸引子轨迹对比', fontsize=14)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('Z', fontsize=12)
ax1.legend(fontsize=10)
ax1.view_init(elev=30, azim=45)  # 调整视角

# 2. 绘制时间序列对比 - X分量
ax2 = fig.add_subplot(2, 2, 2)
t_test = t[train_size+1:][:len(Y_test)]
ax2.plot(t_test, Y_test[:, 0], 'b-', label='真实X值', alpha=0.7)
ax2.plot(t_test, Y_pred[:, 0], 'r--', label='预测X值', linewidth=1.5)
ax2.set_title('X分量时间序列对比', fontsize=14)
ax2.set_xlabel('时间', fontsize=12)
ax2.set_ylabel('X值', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True)

# 3. 计算并绘制每个时间步的预测误差
errors = np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1))
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t_test, errors, 'g-', label='欧氏距离误差')
ax3.set_title(f'预测误差随时间变化 (平均MSE: {mse:.6f})', fontsize=14)
ax3.set_xlabel('时间', fontsize=12)
ax3.set_ylabel('误差', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True)

# 4. 预测效果散点图
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(Y_test[:, 0], Y_pred[:, 0], c='b', s=10, alpha=0.5, label='X分量')
ax4.scatter(Y_test[:, 1], Y_pred[:, 1], c='r', s=10, alpha=0.5, label='Y分量')
ax4.scatter(Y_test[:, 2], Y_pred[:, 2], c='g', s=10, alpha=0.5, label='Z分量')
ax4.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=1)  # 理想线
ax4.set_title('真实值 vs 预测值', fontsize=14)
ax4.set_xlabel('真实值', fontsize=12)
ax4.set_ylabel('预测值', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True)

plt.tight_layout()

# 保存图像
plt.savefig('/Users/dyy/github/QRC_project1/lorenz63_prediction.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()
