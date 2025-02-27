import numpy as np
from pyqpanda import *
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# 生成洛伦兹吸引子数据
def generate_lorenz_data(num_steps, dt=0.01):
    σ = 10.0
    ρ = 28.0
    β = 8.0 / 3.0
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    z = np.zeros(num_steps)
    x[0], y[0], z[0] = (1.0, 1.0, 1.0)
    for i in range(1, num_steps):
        dx = σ * (y[i-1] - x[i-1])
        dy = x[i-1] * (ρ - z[i-1]) - y[i-1]
        dz = x[i-1] * y[i-1] - β * z[i-1]
        x[i] = x[i-1] + dx * dt
        y[i] = y[i-1] + dy * dt
        z[i] = z[i-1] + dz * dt
    return x, y, z

# 数据预处理（归一化到[-π, π]）
def normalize(data, max_val=None):
    if max_val is None:
        max_val = np.max(np.abs(data))
    return data / max_val * np.pi, max_val

# 参数设置
num_steps = 4000
train_size = 3000
test_size = num_steps - train_size
num_qubits = 10

# 生成数据并归一化
x, y, z = generate_lorenz_data(num_steps)
x_scaled, x_max = normalize(x[:train_size])
y_scaled, y_max = normalize(y[:train_size])
z_scaled, z_max = normalize(z[:train_size])

# 训练阶段
def train_qrc(x_scaled, y_scaled, z_scaled):
    X_train = []
    Y_train = []
    
    # 初始化量子参数
    np.random.seed(42)
    third_layer_angles = np.random.uniform(0, 2*np.pi, size=num_qubits)
    qvm = CPUQVM()
    qvm.init_qvm()
    
    current_state = np.zeros(2**num_qubits, dtype=np.complex128)
    current_state[0] = 1.0  # 初始状态|0⟩
    
    for n in range(train_size):
        # 第一层：应用三层随机门
        qubits = qvm.qAlloc_many(num_qubits)
        prog = QProg()
        qvm.init_state(current_state)
        
        # 应用三层随机门
        for _ in range(3):
            for q in qubits:
                prog << RY(q, np.random.uniform(0, 2*np.pi))
            for i in range(num_qubits-1):
                prog << CNOT(qubits[i], qubits[i+1])
        
        # 第二层：编码坐标
        prog << RY(qubits[0], x_scaled[n]) \
            << RY(qubits[1], y_scaled[n]) \
            << RY(qubits[2], z_scaled[n])
        
        # 第三层：固定参数门
        for i, q in enumerate(qubits):
            prog << RY(q, third_layer_angles[i])
        for i in range(num_qubits-1):
            prog << CNOT(qubits[i], qubits[i+1])
        
        qvm.directly_run(prog)
        final_state = np.array(qvm.get_qstate())  # 将列表转换为NumPy数组
        
        # 计算概率
        probs = []
        for i in range(num_qubits):
            prob = np.sum(np.abs(final_state.reshape([2]*num_qubits)[tuple([0]*(i+1) + [slice(None)]*(num_qubits-i-1))])**2)
            probs.append(prob)
        
        X_train.append(probs)
        Y_train.append([x_scaled[n], y_scaled[n], z_scaled[n]])
        
        # 更新状态
        current_state = final_state
        qvm.qFree_all(qubits)
    
    # 训练岭回归模型
    model = Ridge(alpha=1e-5).fit(X_train, Y_train)
    return model, current_state

# 执行训练
model, final_state = train_qrc(x_scaled, y_scaled, z_scaled)

# 预测阶段
def predict_qrc(model, initial_state, steps):
    predictions = []
    current_state = initial_state.copy()
    np.random.seed(42)
    third_layer_angles = np.random.uniform(0, 2*np.pi, size=num_qubits)
    qvm = CPUQVM()
    qvm.init_qvm()
    
    # 初始输入使用最后一个训练数据
    current_input = [x_scaled[-1], y_scaled[-1], z_scaled[-1]]
    
    for _ in range(steps):
        qubits = qvm.qAlloc_many(num_qubits)
        prog = QProg()
        qvm.init_state(current_state)
        
        # 应用三层随机门
        for _ in range(3):
            for q in qubits:
                prog << RY(q, np.random.uniform(0, 2*np.pi))
            for i in range(num_qubits-1):
                prog << CNOT(qubits[i], qubits[i+1])
        
        # 编码预测值
        prog << RY(qubits[0], current_input[0]) \
            << RY(qubits[1], current_input[1]) \
            << RY(qubits[2], current_input[2])
        
        # 固定参数门
        for i, q in enumerate(qubits):
            prog << RY(q, third_layer_angles[i])
        for i in range(num_qubits-1):
            prog << CNOT(qubits[i], qubits[i+1])
        
        qvm.directly_run(prog)
        final_state = np.array(qvm.get_qstate())  # 将列表转换为NumPy数组
        
        # 计算概率
        probs = []
        for i in range(num_qubits):
            prob = np.sum(np.abs(final_state.reshape([2]*num_qubits)[tuple([0]*(i+1) + [slice(None)]*(num_qubits-i-1))])**2)
            probs.append(prob)
        
        # 预测下一步
        pred = model.predict([probs])[0]
        predictions.append(pred)
        current_input = pred
        current_state = final_state
        qvm.qFree_all(qubits)
    
    return np.array(predictions)

# 执行预测
predictions = predict_qrc(model, final_state, test_size)

# 反归一化
x_pred = predictions[:,0] * x_max / np.pi
y_pred = predictions[:,1] * y_max / np.pi
z_pred = predictions[:,2] * z_max / np.pi

# 获取真实数据
x_true = x[train_size:num_steps]
y_true = y[train_size:num_steps]
z_true = z[train_size:num_steps]

# 绘制对比图
plt.figure(figsize=(12, 6))
plt.plot(x_true, label='True X', alpha=0.7)
plt.plot(x_pred, label='Predicted X', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Comparison of True and Predicted X Coordinates')
plt.legend()
plt.show()
