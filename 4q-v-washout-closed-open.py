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


# 参数设置
T = 100
alpha = 2.11 / T
beta = 3.73 / T
gamma_val = 4.11 / T
num_steps = 650  # 总时间步数

# 初始化输入和输出序列
u = np.zeros(num_steps)
y = np.zeros(num_steps)
y[0] = 0.19
y[1] = 0.19  # 初始条件

# 生成输入信号 u_k
for k in range(num_steps):
    u[k] = 0.1 * (np.sin(2 * np.pi * alpha * k) *
                 np.sin(2 * np.pi * beta * k) *
                 np.sin(2 * np.pi * gamma_val * k) + 1)

# 计算 NARMA-2 输出
for k in range(1, num_steps - 1):
    y[k + 1] = 0.4 * y[k] + 0.4 * y[k] * y[k - 1] + 0.6 * (u[k] ** 3) + 0.1

# 创建预测数组，初始值与原始序列相同
y_predicted = y.copy()

# 保存原始序列，用于后续评估和可视化
original_data = y.copy()

### 4-qubit machine设置
# 在循环开始前初始化列表
all_sorted_values = []
n_states = 16
params = [0.5*np.pi] * 16 
quantum_outputs = []
prev_sorted_values = None
varepsilon = 0.2
ccc = 4
n_qubits = 4
n_cbits = 4
shots = 10000
beta_range = (0, 2*np.pi) 
np.random.seed(999) 
xt = np.random.rand(n_qubits)
beta = np.random.uniform(*beta_range, n_qubits)
print(f"随机初始化的beta参数: {beta}")

#############################################
### 第一阶段: Washout (时间步 0-99)
#############################################
print("开始Washout阶段 (时间步 0-99)")
washout_quantum_outputs = []

for i in range(100):
    if i % 20 == 0:
        print(f"Washout进度: {i}/100")
 
    if i > 0 and len(all_sorted_values) > 0:
       # 使用上一次的sorted_values更新参数
       prev_values = all_sorted_values[-1]
       for j in range(min(len(prev_values), 16)):
           params[j] = prev_values[j] * 4 * np.pi

    y_k = 4 * np.pi * y[i]
    
    # 初始化量子虚拟机
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(n_qubits)
    cbits = qvm.cAlloc_many(n_cbits)
    
    # 构建量子程序
    prog = QProg()
    circuit = QCircuit()
    # 使用16个参数构建电路
    param_index = 0
    for layer in range(ccc):  # 4层，每层使用4个参数
        for qubit in range(4):
            circuit << RY(qubits[qubit], params[param_index])
            param_index += 1
            if qubit < 3:
                circuit << CNOT(qubits[qubit], qubits[qubit+1])
        circuit << CNOT(qubits[3], qubits[0])
    
    # 添加与y_k相关的门
    circuit << RY(qubits[0], y_k) 
    circuit << CNOT(qubits[0], qubits[1]) 
    circuit << CNOT(qubits[1], qubits[2]) 
    circuit << CNOT(qubits[2], qubits[3])
    circuit << CNOT(qubits[3], qubits[0])
    
    for qubit in range(4):
        circuit << RY(qubits[qubit], beta[qubit])
        if qubit < 3:
            circuit << CNOT(qubits[qubit], qubits[qubit+1])
    circuit << CNOT(qubits[3], qubits[0])
    
    prog << circuit << measure_all(qubits, cbits)
    result = qvm.run_with_configuration(prog, cbits, shots)
    
    # 计算概率分布
    probabilities = {}
    for state_idx in range(n_states):
        state = format(state_idx, '04b')
        probabilities[state] = round(result.get(state, 0) / shots, 6)
    
    sorted_states = sorted(probabilities.keys())
    raw_sorted_values = [probabilities[state] for state in sorted_states]
        
    # 非线性化处理
    if i > 0 and prev_values is not None:
        sorted_values = [0] * len(raw_sorted_values)
        for j in range(len(raw_sorted_values)):
            sorted_values[j] = round(varepsilon * raw_sorted_values[j] + (1 - varepsilon) * prev_values[j], 6)
    else:
        sorted_values = raw_sorted_values.copy()

    all_sorted_values.append(sorted_values)
    washout_quantum_outputs.append(sorted_values)
    qvm.finalize()

print("Washout阶段完成")

#############################################
### 第二阶段: 闭环量子储层计算 (时间步 100-499)
#############################################
print("开始闭环量子储层计算 (时间步 100-499)")

# 准备初始训练数据 (从washout阶段提取)
X_train_initial = []
Y_train_initial = []

# 使用最后50个washout输出训练初始模型
for i in range(50, 99):
    features = washout_quantum_outputs[i]
    target = y[i+1]
    X_train_initial.append(features)
    Y_train_initial.append(target)

X_train_initial = np.array(X_train_initial)
Y_train_initial = np.array(Y_train_initial)

# 训练初始岭回归模型
ridge_model = Ridge(alpha=0.01)
ridge_model.fit(X_train_initial, Y_train_initial)
print("初始岭回归模型训练完成")

# 开始闭环量子储层计算
closed_loop_quantum_outputs = []
current_state = y[99]  # 从时间步99的状态开始
current_quantum_output = washout_quantum_outputs[-1]  # 使用最后一个washout量子输出

for i in range(100, 500):
    if (i-100) % 50 == 0:
        print(f"闭环量子储层计算进度: {i-100}/400")
    
    # 更新参数
    for j in range(min(len(current_quantum_output), 16)):
        params[j] = current_quantum_output[j] * 4 * np.pi
    
    # 使用当前状态作为量子线路输入
    y_k = 4 * np.pi * current_state
    
    # 初始化量子虚拟机
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(n_qubits)
    cbits = qvm.cAlloc_many(n_cbits)
    
    # 构建量子程序
    prog = QProg()
    circuit = QCircuit()
    param_index = 0
    for layer in range(ccc): 
        for qubit in range(4):
            circuit << RY(qubits[qubit], params[param_index])
            param_index += 1
            if qubit < 3:
                circuit << CNOT(qubits[qubit], qubits[qubit+1])
        circuit << CNOT(qubits[3], qubits[0])
    
    circuit << RY(qubits[0], y_k) 
    circuit << CNOT(qubits[0], qubits[1])  
    circuit << CNOT(qubits[1], qubits[2]) 
    circuit << CNOT(qubits[2], qubits[3])
    circuit << CNOT(qubits[3], qubits[0])
    
    for qubit in range(4):
        circuit << RY(qubits[qubit], beta[qubit])
        if qubit < 3:
            circuit << CNOT(qubits[qubit], qubits[qubit+1])
    circuit << CNOT(qubits[3], qubits[0])
    
    prog << circuit << measure_all(qubits, cbits)
    result = qvm.run_with_configuration(prog, cbits, shots)
    
    # 计算概率分布
    probabilities = {}
    for state_idx in range(n_states):
        state = format(state_idx, '04b')
        probabilities[state] = round(result.get(state, 0) / shots, 6)
    
    sorted_states = sorted(probabilities.keys())
    raw_quantum_output = [probabilities[state] for state in sorted_states]
    
    # 非线性化处理
    new_quantum_output = []
    for j in range(len(raw_quantum_output)):
        new_prob = round(varepsilon * raw_quantum_output[j] + (1 - varepsilon) * current_quantum_output[j], 6)
        new_quantum_output.append(new_prob)
    
    # 更新当前量子输出
    current_quantum_output = new_quantum_output
    closed_loop_quantum_outputs.append(current_quantum_output)
    
    # 使用岭回归模型预测下一个状态
    next_state_prediction = ridge_model.predict([current_quantum_output])[0]
    
    # 更新预测数组
    y_predicted[i] = next_state_prediction
    
    # 更新当前状态为预测值，用于下一次迭代
    current_state = next_state_prediction
    
    qvm.finalize()
    
    # 每50步更新一次模型
    if (i-100) % 50 == 0 and i > 150:
        # 收集最近100个闭环数据点
        update_X = []
        update_Y = []
        
        # 计算可用的更新样本数量
        num_samples = min(100, len(closed_loop_quantum_outputs) - 1)
        start_idx = len(closed_loop_quantum_outputs) - num_samples - 1
        
        for idx in range(num_samples):
            features = closed_loop_quantum_outputs[start_idx + idx]
            target = y_predicted[100 + start_idx + idx + 1]
            update_X.append(features)
            update_Y.append(target)
        
        if len(update_X) > 0:
            update_X = np.array(update_X)
            update_Y = np.array(update_Y)
            # 更新模型
            ridge_model.fit(update_X, update_Y)
            print(f"在时间步 {i} 更新模型，使用 {len(update_X)} 个样本")

print("闭环量子储层计算完成")

#############################################
### 第三阶段: 开环预测 (时间步 500-649)
#############################################
print("开始开环预测 (时间步 500-649)")
# 使用最后一个闭环状态和量子输出作为起点
current_state = y_predicted[499]
current_quantum_output = closed_loop_quantum_outputs[-1]
prev_pred_values = current_quantum_output.copy()

for i in range(500, 650):
    if (i-500) % 25 == 0:
        print(f"开环预测进度: {i-500}/150")
    
    # 特征：当前量子输出
    features = current_quantum_output
    for j in range(min(len(features), 16)):
        params[j] = features[j] * 4 * np.pi
        
    # 预测下一个状态
    next_state = ridge_model.predict([features])[0]
    y_predicted[i] = next_state
    
    # 更新当前状态为预测的状态
    current_state = next_state

    # 使用预测的状态生成新的量子输出
    y_k = 4 * np.pi * current_state
    
    # 初始化量子虚拟机
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(n_qubits)
    cbits = qvm.cAlloc_many(n_cbits)

    # 构建量子程序
    prog = QProg()
    circuit = QCircuit()
    param_index = 0
    for layer in range(ccc): 
        for qubit in range(4):
            circuit << RY(qubits[qubit], params[param_index])
            param_index += 1
            if qubit < 3:
                circuit << CNOT(qubits[qubit], qubits[qubit+1])
        circuit << CNOT(qubits[3], qubits[0]) 
    
    circuit << RY(qubits[0], y_k) 
    circuit << CNOT(qubits[0], qubits[1])  
    circuit << CNOT(qubits[1], qubits[2]) 
    circuit << CNOT(qubits[2], qubits[3])
    circuit << CNOT(qubits[3], qubits[0])
    
    for qubit in range(4):
        circuit << RY(qubits[qubit], beta[qubit])
        if qubit < 3:
            circuit << CNOT(qubits[qubit], qubits[qubit+1])
    circuit << CNOT(qubits[3], qubits[0])
        
    prog << circuit << measure_all(qubits, cbits)
    result = qvm.run_with_configuration(prog, cbits, shots)

    # 计算概率分布
    probabilities = {}
    for state_idx in range(n_states):
        state = format(state_idx, '04b')
        probabilities[state] = round(result.get(state, 0) / shots, 6)
    
    sorted_states = sorted(probabilities.keys())
    raw_quantum_output = [probabilities[state] for state in sorted_states]
    
    # 非线性化处理
    current_quantum_output = []
    for j in range(len(raw_quantum_output)):
        new_prob = round(varepsilon * raw_quantum_output[j] + (1 - varepsilon) * prev_pred_values[j], 6)
        current_quantum_output.append(new_prob)
        
    # 更新上一次的概率值
    prev_pred_values = current_quantum_output.copy()

    qvm.finalize()

print("开环预测完成")

#############################################
### 性能评估与可视化
#############################################

# 计算各阶段的MSE
# 闭环阶段MSE
mse_closed_loop = mean_squared_error(original_data[100:500], y_predicted[100:500])
print(f"闭环量子储层预测MSE (时间步100-499): {mse_closed_loop}")

# 开环阶段MSE
mse_open_loop = mean_squared_error(original_data[500:650], y_predicted[500:650])
print(f"开环预测MSE (时间步500-649): {mse_open_loop}")

# 可视化结果
plt.figure(figsize=(14, 7))
plt.plot(range(num_steps), original_data, label='原始NARMA-2序列', color='blue')
plt.axvspan(0, 99, alpha=0.1, color='gray', label='Washout阶段 (0-99)')
plt.plot(range(100, 500), y_predicted[100:500], label='闭环量子储层预测 (100-499)', color='red', linestyle='--')
plt.plot(range(500, 650), y_predicted[500:650], label='开环预测 (500-649)', color='green', linestyle='-.')
plt.axvspan(100, 499, alpha=0.2, color='red')
plt.axvspan(500, 649, alpha=0.2, color='green')
plt.xlabel('时间步 (k)')
plt.ylabel('值')
plt.title('NARMA-2序列的量子储层计算：Washout、闭环预测和开环预测')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('narma2_quantum_reservoir_three_phases.png', dpi=300)
plt.show()
