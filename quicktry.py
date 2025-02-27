import numpy as np
from pyqpanda import *
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

def generate_lorenz_data(num_steps, dt=0.01):
    """生成洛伦兹混沌系统的时间序列数据
    
    参数:
        num_steps: 时间步数
        dt: 时间步长
        
    返回:
        x, y, z: 三个维度的时间序列数据
    """
    # 洛伦兹系统参数
    σ = 10.0  # sigma,控制流体的粘性
    ρ = 28.0  # rho,与温度梯度相关
    β = 8.0/3.0  # beta,与系统几何尺寸相关
    
    # 初始化数组
    x = np.zeros(num_steps)
    y = np.zeros(num_steps) 
    z = np.zeros(num_steps)
    
    # 设置初始值
    x[0], y[0], z[0] = 1.0, 1.0, 1.0
    
    # 使用龙格-库塔方法求解微分方程
    for i in range(1, num_steps):
        # 计算导数
        dx = σ * (y[i-1] - x[i-1])
        dy = x[i-1] * (ρ - z[i-1]) - y[i-1]
        dz = x[i-1] * y[i-1] - β * z[i-1]
        
        # 更新状态
        x[i] = x[i-1] + dx * dt
        y[i] = y[i-1] + dy * dt
        z[i] = z[i-1] + dz * dt
        
    return x, y, z

# 生成洛伦兹数据
def process_lorenz_data(num_steps, n):
    x, y, z = generate_lorenz_data(num_steps)

    # 提取第 n 时间步的 x, y, z 值（确保 n 范围合理）
    if n >= num_steps:
        raise ValueError(f"n 必须小于 {num_steps} (当前为 {n})")
    x_value = x[n]
    y_value = y[n]
    z_value = z[n]

    # 将值映射到量子旋转门的角度
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    min_z, max_z = np.min(z), np.max(z)

    theta_x = 2 * np.pi * (x_value - min_x) / (max_x - min_x)
    theta_y = 2 * np.pi * (y_value - min_y) / (max_y - min_y)
    theta_z = 2 * np.pi * (z_value - min_z) / (max_z - min_z)
    
    return theta_x, theta_y, theta_z

def reservior_circle(time_step, n_qubits=10, n_cbits=10):
    qvm = CPUQVM()
    qvm.init_qvm()
    
    # 分配量子比特和经典比特
    qubits = qvm.qAlloc_many(n_qubits)  # 分配量子比特
    cbits = qvm.cAlloc_many(n_cbits)    # 分配经典比特

    # 创建量子程序
    prog = QProg()
    circuit = QCircuit()
    
    # 获取洛伦兹数据对应的旋转角度
    theta_x, theta_y, theta_z = process_lorenz_data(1000, time_step)

    circuit << RY(qubits[0], theta_x) 
    circuit << CNOT(qubits[0], qubits[1]) 
    circuit << RY(qubits[1], theta_y) 
    circuit << CNOT(qubits[1], qubits[2]) 
    circuit << RY(qubits[2], theta_z) 
    circuit << CNOT(qubits[2], qubits[3]) 
    circuit << CNOT(qubits[3], qubits[4]) 
    circuit << CNOT(qubits[4], qubits[5]) 
    circuit << CNOT(qubits[5], qubits[6]) 
    circuit << CNOT(qubits[6], qubits[7]) 
    circuit << CNOT(qubits[7], qubits[0]) 
    
    prog << circuit << measure_all(qubits, cbits)
    
    # 量子程序运行并返回测量结果
    result = qvm.run_with_configuration(prog, cbits, 10000)
    return result

if __name__ == "__main__":
    result = reservior_circle(time_step=5)