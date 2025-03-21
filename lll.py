import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图模块


def lorenz_equations(state, t, sigma, r, b):
    """定义Lorenz方程组"""
    x, y, z = state
    dx = -sigma * x + sigma * y
    dy = -x * z + r * x - y
    dz = x * y - b * z
    return np.array([dx, dy, dz])


def double_approximation_solver(initial_state, t0, tf, dt, sigma, r, b):
    """使用双近似法求解Lorenz方程组"""
    num_steps = int((tf - t0) / dt) + 1
    time_points = np.linspace(t0, tf, num_steps)
    solution = np.zeros((num_steps, 3))
    solution[0] = initial_state

    for n in range(num_steps - 1):
        # 计算X_{i(n+1)}
        X_in1 = solution[n] + lorenz_equations(solution[n], time_points[n], sigma, r, b) * dt
        # 计算X_{i((n+2))}
        X_i2n1 = X_in1 + lorenz_equations(X_in1, time_points[n] + dt, sigma, r, b) * dt
        # 计算公式(14)
        solution[n + 1] = 0.5 * (solution[n] + X_i2n1)

    return time_points, solution


# 参数设置
sigma = 10
r = 28
b = 8 / 3
t0 = 0
tf = 60
dt = 0.01
initial_state = np.array([0, 1, 0])

# 求解方程
time_points, solution = double_approximation_solver(initial_state, t0, tf, dt, sigma, r, b)

# 提取X, Y, Z的值
X = solution[:, 0]
Y = solution[:, 1]
Z = solution[:, 2]
print(X)
# 绘制Y随时间变化的图像，类似论文中的图1
plt.figure(figsize=(10, 6))
plt.plot(time_points, Y)
plt.xlabel('Time')
plt.ylabel('Y')
plt.title('Numerical Solution of the Convection Equations (Y vs Time)')
plt.grid(True)
plt.show()

# 绘制三维图像 - Lorenz吸引子
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X, Y, Z, lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor 3D Trajectory')
plt.show()