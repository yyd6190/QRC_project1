#region 导入库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='Kai')
plt.rcParams['axes.unicode_minus'] =False
from pyqpanda import *
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
#endregion

#region Lorenz工具函数
def RK45(x, func, h):
    """四阶龙格-库塔积分器"""
    K1 = func(x)
    K2 = func(x + h/2*K1)
    K3 = func(x + h/2*K2)
    K4 = func(x + h*K3)
    return x + h/6*(K1 + 2*K2 + 2*K3 + K4)
def L63_rhs(x, sigma=10.0, rho=28.0, beta=8/3):
    """定义Lorenz 63微分方程"""
    dx = np.zeros(3)
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = rho*x[0] - x[1] - x[0]*x[2]
    dx[2] = x[0]*x[1] - beta*x[2]
    return dx
def integrate_lorenz63(x0, num_steps=2200, dt=0.05):
    trajectory = np.zeros((num_steps, 3))
    trajectory[0] = x0
    for t in range(1, num_steps):
        trajectory[t] = RK45(trajectory[t-1], L63_rhs, dt)
    return trajectory
def normalize_trajectory(trajectory):
    """将轨迹归一化到[0, 1]范围"""
    # 计算每个坐标轴的极值
    mins = np.min(trajectory, axis=0)
    maxs = np.max(trajectory, axis=0)
    
    # 线性归一化公式：new = (original - min)/(max - min)
    normalized =(trajectory - mins) / (maxs - mins)
    return normalized
#endregion

#region Lorenz可视化+数据准备
# 参数设置
x0 = np.array([1.508870, -1.531271, 25.46091])  # 混沌初值
traj = integrate_lorenz63(x0)

# 应用归一化
traj_normalized = normalize_trajectory(traj)

# 划分训练集和测试集
train_data = traj_normalized[:2000]
test_data = traj_normalized[2000:2200]

# 绘制归一化后的轨迹
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj_normalized[:,0], traj_normalized[:,1], traj_normalized[:,2], lw=0.5)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel("Normalized X")
ax.set_ylabel("Normalized Y")
ax.set_zlabel("Normalized Z")
plt.title("Normalized Lorenz 63 Attractor")
plt.show()
#endregion

#region 量子计算参数设置
### 9-qubit machine (原来是8比特)
# 在循环开始前初始化列表
all_sorted_values = []
# 对于9个量子比特，可能的状态总数是2^9=512
n_states = 512  # 原来是256
# 初始化512个参数
params = [0.5*np.pi] * 512  # 原来是256个参数
# 存储量子线路的输出结果
quantum_outputs = []
# 初始化上一次的概率值为None
prev_sorted_values = None
#参数\varepsilon
varepsilon = 0.05
#量子线路概率输入的层数
ccc = 57
#q、c比特数量 shot
n_qubits = 9 
n_cbits = 9  
shots = 1000
# 参数设置 - 调整beta参数以达到目标谱半径
beta_range = (0, 2*np.pi) 
# 固定随机种子以确保结果可复现
np.random.seed(42) 
# β角度会影响量子储层的计算能力，类似于经典储层中的连接权重矩阵的谱半径
beta = np.random.uniform(*beta_range, n_qubits)
#endregion

#region 量子训练过程
for i in range(2000):
    if i % 1000 == 0:
        print(f"处理训练数据: {i}/2000")
 
    if i > 0 and len(all_sorted_values) > 0:
       # 使用上一次的sorted_values更新参数
       prev_values = all_sorted_values[-1]
       for j in range(min(len(prev_values), 512)):  # 从256改为512
           params[j] = prev_values[j] *4*np.pi

    x_norm, y_norm, z_norm = 4*np.pi*traj_normalized[i]
    
    # 初始化量子虚拟机
    qvm = CPUQVM()
    qvm.init_qvm()
    # 分配量子比特和经典比特
    qubits = qvm.qAlloc_many(n_qubits)  # 现在分配9个量子比特
    cbits = qvm.cAlloc_many(n_cbits)    # 现在分配9个经典比特
    # 构建量子程序
    prog = QProg()
    circuit = QCircuit()
    # 使用512个参数构建电路
    param_index = 0
    for layer in range(ccc):  # 32层，每层使用9个参数（原来是8个）
        for qubit in range(9):  # 从0-8，共9个比特
            if param_index < len(params):
                circuit << RY(qubits[qubit], params[param_index])
                param_index += 1
                if qubit < 8:  # 连接前8个比特与后继比特
                    circuit << CNOT(qubits[qubit], qubits[qubit+1])
        circuit << CNOT(qubits[8], qubits[0])  # 连接最后一个与第一个比特，形成环
    # 添加Lorenz坐标相关的门
    circuit << RY(qubits[0], x_norm) 
    circuit << CNOT(qubits[0], qubits[1]) 
    circuit << RY(qubits[1], y_norm) 
    circuit << CNOT(qubits[1], qubits[2]) 
    circuit << RY(qubits[2], z_norm)
    # 连接更多比特，形成更复杂的纠缠
    circuit << CNOT(qubits[2], qubits[3])
    circuit << CNOT(qubits[3], qubits[4])
    circuit << CNOT(qubits[4], qubits[5])
    circuit << CNOT(qubits[5], qubits[6])
    circuit << CNOT(qubits[6], qubits[7])
    circuit << CNOT(qubits[7], qubits[8])  # 增加对第9个比特的连接
    circuit << CNOT(qubits[8], qubits[0]) 
    for qubit in range(9):  # 应用9个beta旋转参数
        circuit << RY(qubits[qubit], beta[qubit])
        if qubit < 8:  # 连接相邻比特
            circuit << CNOT(qubits[qubit], qubits[qubit+1])
    circuit << CNOT(qubits[8], qubits[0])  # 闭合环
    prog << circuit << measure_all(qubits, cbits)
    # 运行多次量子程序，并返回测量结果
    result = qvm.run_with_configuration(prog, cbits, shots)
    # 计算概率分布
    total = shots
    probabilities = {}
    
    # 确保所有可能的状态都有值 (现在是512个可能的状态)
    for state_idx in range(n_states):
        # 将索引转换为9位二进制字符串
        state = format(state_idx, '09b')  # 从'08b'改为'09b'
        # 如果状态在结果中，使用实际概率；否则为0
        probabilities[state] = round(result.get(state, 0) / total, 7)
    
    # 确保按照状态排序，而不是按照概率值排序
    sorted_states = sorted(probabilities.keys())
    raw_sorted_values = [probabilities[state] for state in sorted_states]
        
    # 非线性化处理：新概率 = varepsilon*当前概率 + (1-varepsilon)*上一次概率
    if i > 0 and prev_values is not None:
        sorted_values = [0] * len(raw_sorted_values)
        for j in range(len(raw_sorted_values)):
            sorted_values[j] = round(varepsilon * raw_sorted_values[j] + (1 - varepsilon) * prev_values[j], 7)
    else:
        sorted_values = raw_sorted_values.copy()

    all_sorted_values.append(sorted_values)
    quantum_outputs.append(sorted_values)
    print(sorted_values)

    qvm.finalize()
#endregion

#region 模型训练
# 准备训练数据
X_train = []
Y_train = []

# 使用量子输出和当前坐标预测下一个时间步的坐标
for i in range(50,1999):
    # 特征：当前量子输出 + 当前坐标
    features = quantum_outputs[i] #+list(train_data[i])
    # 目标：下一个时间步的坐标
    target = train_data[i+1]
    X_train.append(features)
    Y_train.append(target)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# 训练岭回归模型
ridge = Ridge(alpha=0)
ridge.fit(X_train, Y_train)
print("岭回归模型训练完成")
#endregion

#region 预测过程
# 测试阶段：预测2001-2501时间步
predictions = []
current_state = train_data[-1]  # 从训练集最后一个状态开始
current_quantum_output = quantum_outputs[-1]  # 最后一个量子输出

# 初始化预测阶段的上一次概率值
prev_pred_values = current_quantum_output.copy()

for i in range(200):
    if i % 100 == 0:
        print(f"预测测试数据: {i}/500")
    
    # 特征：当前量子输出 + 当前坐标
    features = current_quantum_output #+ list(current_state)
    for j in range(min(len(features), 512)):  # 从256改为512
           params[j] = features[j] *4*np.pi
    # 预测下一个状态
    next_state = ridge.predict([features])[0]
    predictions.append(next_state)
    
    # 更新当前状态为预测的状态
    current_state = next_state

    # 使用预测的状态生成新的量子输出
    x_norm, y_norm, z_norm = 4*np.pi*current_state
    
    # 初始化量子虚拟机
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(n_qubits)  # 分配9个量子比特
    cbits = qvm.cAlloc_many(n_cbits)    # 分配9个经典比特

    # 构建量子程序
    prog = QProg()
    circuit = QCircuit()
    param_index = 0
    for layer in range(ccc): 
        for qubit in range(9):  # 使用9个比特
            if param_index < len(params):
                circuit << RY(qubits[qubit], params[param_index])
                param_index += 1
                if qubit < 8:  # 连接0-7与后继比特
                    circuit << CNOT(qubits[qubit], qubits[qubit+1])
        circuit << CNOT(qubits[8], qubits[0])  # 连接最后一个与第一个
    circuit << RY(qubits[0], x_norm) 
    circuit << CNOT(qubits[0], qubits[1]) 
    circuit << RY(qubits[1], y_norm) 
    circuit << CNOT(qubits[1], qubits[2]) 
    circuit << RY(qubits[2], z_norm)
    circuit << CNOT(qubits[2], qubits[3])
    circuit << CNOT(qubits[3], qubits[4])
    circuit << CNOT(qubits[4], qubits[5])
    circuit << CNOT(qubits[5], qubits[6])
    circuit << CNOT(qubits[6], qubits[7])
    circuit << CNOT(qubits[7], qubits[8])  # 增加的连接
    circuit << CNOT(qubits[8], qubits[0]) 
    for qubit in range(9):  # 应用9个beta值
        circuit << RY(qubits[qubit], beta[qubit])
        if qubit < 8:  # 连接前8个比特
            circuit << CNOT(qubits[qubit], qubits[qubit+1])
    circuit << CNOT(qubits[8], qubits[0])
    prog << circuit << measure_all(qubits, cbits)
    result = qvm.run_with_configuration(prog, cbits, shots)

    # 计算概率分布
    probabilities = {}
    for state_idx in range(n_states):
        state = format(state_idx, '09b')  # 9位二进制
        probabilities[state] = round(result.get(state, 0) / shots, 7)
    
    sorted_states = sorted(probabilities.keys())
    raw_quantum_output = [probabilities[state] for state in sorted_states]
    
    # 对预测阶段的概率也进行非线性化处理
    current_quantum_output = []
    for j in range(len(raw_quantum_output)):
        new_prob = round(varepsilon * raw_quantum_output[j] + (1 - varepsilon) * prev_pred_values[j], 7)
        current_quantum_output.append(new_prob)
    # 更新上一次的概率值
    prev_pred_values=current_quantum_output.copy()

    qvm.finalize()
predictions = np.array(predictions)

print(predictions)
#endregion

#region 评估和可视化结果
# 计算预测误差
mse = mean_squared_error(test_data, predictions)#检查过，引用函数与文献一致
print(f"测试集MSE: {mse}")

# 保存预测结果和实际值
np.save('/Users/dyy/github/QRC_project1/quantum_predictions.npy', predictions)
np.save('/Users/dyy/github/QRC_project1/actual_values.npy', test_data)

# 绘制对比图
fig = plt.figure(figsize=(15, 10))

# 绘制x坐标对比
ax1 = fig.add_subplot(311)
ax1.plot(range(2001, 2201), test_data[:, 0], 'b-', label='实际值')
ax1.plot(range(2001, 2201), predictions[:, 0], 'r--', label='量子预测值')
ax1.set_title('x坐标对比')
ax1.set_xlabel('时间步')
ax1.set_ylabel('归一化x值')
ax1.legend()

# 绘制y坐标对比
ax2 = fig.add_subplot(312)
ax2.plot(range(2001, 2201), test_data[:, 1], 'b-', label='实际值')
ax2.plot(range(2001, 2201), predictions[:, 1], 'r--', label='量子预测值')
ax2.set_title('y坐标对比')
ax2.set_xlabel('时间步')
ax2.set_ylabel('归一化y值')
ax2.legend()

# 绘制z坐标对比
ax3 = fig.add_subplot(313)
ax3.plot(range(2001, 2201), test_data[:, 2], 'b-', label='实际值')
ax3.plot(range(2001, 2201), predictions[:, 2], 'r--', label='量子预测值')
ax3.set_title('z坐标对比')
ax3.set_xlabel('时间步')
ax3.set_ylabel('归一化z值')
ax3.legend()
plt.savefig('/Users/dyy/github/QRC_project1/quantum_prediction_comparison.png')
plt.show()

# 绘制3D轨迹对比图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制实际轨迹
ax.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], 'b-', label='实际轨迹')

# 绘制预测轨迹
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], 'r--', label='量子预测轨迹')

ax.set_title('Lorenz系统轨迹对比 (时间步 2001-2501)')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')
ax.legend()

plt.savefig('/Users/dyy/github/QRC_project1/quantum_3d_comparison.png')
plt.show()
#endregion