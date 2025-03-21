import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='Kai')
plt.rcParams['axes.unicode_minus'] =False
from pyqpanda import *
from sklearn import linear_model
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# NARMA-2部分

def generate_narma2_series(num_steps, initial_conditions=(0.19, 0.19)):
    """
    生成NARMA-2时间序列。
    
    参数:
    num_steps (int): 要生成的时间步数
    initial_conditions (tuple): 初始条件，默认为(0.19, 0.19)
    
    返回:
    tuple: (输入序列u, 输出序列y, 归一化后的输出序列traj_normalized)
    """
    # 参数设置
    T = 100
    alpha = 2.11 / T
    beta = 3.73 / T
    gamma_val = 4.11 / T
    
    # 初始化输入和输出序列
    u = np.zeros(num_steps)
    y = np.zeros(num_steps)
    y[0] = initial_conditions[0]
    y[1] = initial_conditions[1]
    
    # 生成输入信号 u_k
    for k in range(num_steps):
        u[k] = 0.1 * (np.sin(2 * np.pi * alpha * k) *
                     np.sin(2 * np.pi * beta * k) *
                     np.sin(2 * np.pi * gamma_val * k) + 1)
    
    # 计算 NARMA-2 输出
    for k in range(1, num_steps - 1):
        y[k + 1] = 0.4 * y[k] + 0.4 * y[k] * y[k - 1] + 0.6 * (u[k] ** 3) + 0.1
    
    # 构建归一化输出序列
    traj_normalized = np.array([[float(val)] for val in y])
    
    return u, y, traj_normalized

#Quantum Reservoir Computing 部分

def all_probabilities(traj_normalized,varepsilon, n_qubits, n_cbits, seed, nub):

    def update_params():
        params = [0] * (2**n_qubits)
        if i > 0:
            # 使用上一次的sorted_values更新参数
            for j in range(len(sorted_values)):
                params[j] = sorted_values[j]*4*np.pi
        return params

    def create_quantum_circuit():
        """创建并返回量子线路"""
        qvm = CPUQVM()
        qvm.init_qvm()
        
        qubits = qvm.qAlloc_many(n_qubits)
        cbits = qvm.cAlloc_many(n_qubits)
        def module(Ry):
            circuit = QCircuit()
            Ry_index = 0
            ccc = (len(Ry) + n_qubits - 1) // n_qubits
            for layer in range(ccc):
                for qubit in range(n_qubits):
                    if Ry_index < len(Ry):
                        circuit << RY(qubits[qubit], Ry[Ry_index])
                        Ry_index += 1
                        if qubit < n_qubits - 1:
                            circuit << CNOT(qubits[qubit], qubits[qubit+1])
            return circuit
        # 构建量子程序
        prog = QProg()
        circuit = QCircuit()
        circuit << module(Ry=params)
        circuit << module(Ry=traj_norm)
        circuit << module(Ry=beta_gate)
        prog << circuit
        
        result = qvm.prob_run_dict(prog, qubits, -1)
        value_list = list(result.values())
        return value_list

    def nolinearize():
        """非线性化处理"""
        if i > 0:
            raw_sorted_values = [0] * len(sorted_values)
            for j in range(len(sorted_values)):
                raw_sorted_values[j] = round(varepsilon * value_list[j] + (1 - varepsilon) * sorted_values[j],nub)
        else:
            raw_sorted_values = value_list.copy()
        return raw_sorted_values

    quantum_outputs = [[0]*(2**n_qubits)]
    for i in range(len(traj_normalized)):
        params = update_params()
        traj_norm = 4*np.pi*traj_normalized[i]
        np.random.seed (seed) 
        beta_gate = np.random.uniform (0,2*np.pi,n_qubits)
        value_list = create_quantum_circuit()
        sorted_values = nolinearize()
        quantum_outputs.append(sorted_values)
    return quantum_outputs, n_qubits, n_cbits

#训练模型部分

def train_ridge_model(washout,nstop_train, alpha, loading_dataX, loading_datay):
    """训练模型"""
    # 准备训练数据
    X_train = loading_dataX[washout:nstop_train]
    y_train = loading_datay[washout:nstop_train]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # 训练模型
    ridge = Ridge(alpha)
    ridge.fit(X_train, y_train)
    return ridge, washout, nstop_train

def predict_next_state(method, loading_dataX, loading_datay):
    predictions = []
    for i in range(len(loading_datay)):  
        features = loading_dataX[i]  # 使用已有的量子输出
        all_state_pred = method.predict([features])  # 使用传入的method对象
        predictions.append(all_state_pred[0])
    predictions = np.array(predictions)
    return predictions

##可视化模块

def plot_normalized_traj(traj_normalized, title="Normalized Trajectory"):
    """
    绘制归一化后的轨迹，根据数据维度自动选择 1D、2D 或 3D 绘图。

    :param traj_normalized: 归一化后的轨迹数据，为 numpy 数组
    :param title: 图的标题
    """
    dim = traj_normalized.shape[1]  # 获取数据的维度

    if dim == 1:
        # 1D 绘图
        fig = plt.figure(figsize=(10, 6))
        plt.plot(traj_normalized[:, 0], lw=1)
        plt.xlabel("Index")
        plt.ylabel("Normalized Value")
        plt.title(title)
    elif dim == 2:
        # 2D 绘图
        fig = plt.figure(figsize=(10, 6))
        plt.plot(traj_normalized[:, 0], traj_normalized[:, 1], lw=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("Normalized X")
        plt.ylabel("Normalized Y")
        plt.title(title)
    elif dim == 3:
        # 3D 绘图
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj_normalized[:, 0], traj_normalized[:, 1], traj_normalized[:, 2], lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.set_zlabel("Normalized Z")
        plt.title(title)
    else:
        print(f"不支持绘制 {dim} 维的轨迹图。仅支持 1D、2D 或 3D 绘图。")
        return

    plt.show()

def plot_comparison(dimensions=('x', 'y', 'z'),
                    plot_3d=True,
                    plot_washout=False,
                    washout=None,
                    nstop_train=None,
                    traj_normalized=None,
                    predictions=None):
    """
    绘制 2D 对比图和 3D 轨迹对比图

    参数:
    washout (int): 清洗期的长度
    nstop_train (int): 特定时间步
    traj_normalized (np.ndarray): 实际轨迹的归一化数据
    predictions (np.ndarray): 量子预测的轨迹数据
    dimensions (tuple): 要绘制的维度标签，默认为 ('x', 'y', 'z')
    plot_3d (bool): 是否绘制 3D 轨迹对比图，默认为 True
    """
    # 检查输入数组的维度并调整
    if traj_normalized is None or predictions is None:
        print("错误：未提供轨迹数据或预测数据")
        return
    
    # 确认数组维度
    is_1d = len(traj_normalized.shape) == 1 or traj_normalized.shape[1] == 1
    
    # 对于一维数据，只使用第一个标签进行绘图
    if is_1d:
        dimensions = [dimensions[0] if isinstance(dimensions, tuple) else dimensions]
        num_dims = 1
        
        # 如果是一维列向量，转换为一维数组
        if len(traj_normalized.shape) > 1 and traj_normalized.shape[1] == 1:
            traj_normalized = traj_normalized.flatten()
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
    else:
        num_dims = min(len(dimensions), traj_normalized.shape[1])

    if num_dims > 0:
        # 创建一个包含多个子图的图形窗口
        fig_2d, axes = plt.subplots(num_dims, 1, figsize=(15, 10))
        if num_dims == 1:
            axes = [axes]

        for i, dim in enumerate(dimensions[:num_dims]):
            # 计算子图的位置
            ax = axes[i]
            if plot_washout:
                # 绘制实际值曲线 - 一维数据无需索引维度
                if is_1d:
                    ax.plot(traj_normalized[1:], 'b-', label='实际值')
                    ax.plot(predictions[:-1], 'r--', label='QRC预测')
                else:
                    ax.plot(traj_normalized[1:, i], 'b-', label='实际值')
                    ax.plot(predictions[:-1, i], 'r--', label='QRC预测')
            else:
                # 为一维数据创建nan掩码
                mask = np.ones_like(predictions)
                mask[:washout] = np.nan
                
                # 使用掩码进行绘图
                if is_1d:
                    ax.plot(traj_normalized[1:], 'b-', label='实际值')
                    ax.plot(predictions[:-1] * mask[:-1], 'r--', label='QRC预测')
                else:
                    ax.plot(traj_normalized[1:, i], 'b-', label='实际值')
                    # 对于多维数据，使用二维索引
                    masked_pred = predictions.copy()
                    masked_pred[:washout, i] = np.nan
                    ax.plot(masked_pred[:-1, i], 'r--', label='QRC预测')

            # 设置子图的标题和标签
            ax.set_title(f'{dim}坐标对比')
            ax.set_xlabel('时间步')
            ax.set_ylabel(f'归一化{dim}值')
            ax.legend()

            # 添加背景色
            ax.axvspan(0, washout, facecolor='black', alpha=0.2)
            ax.axvspan(washout, nstop_train, facecolor='blue', alpha=0.2)
            ax.axvspan(nstop_train, len(traj_normalized), facecolor='yellow', alpha=0.2)

        plt.subplots_adjust(hspace=0.6)
        plt.show()

    # 仅在数据是三维且用户要求3D图时绘制3D图
    if plot_3d and not is_1d and traj_normalized.shape[1] >= 3:
        # 创建一个 3D 图形窗口
        fig_3d = plt.figure(figsize=(12, 10))
        ax = fig_3d.add_subplot(111, projection='3d')
        # 绘制实际轨迹
        ax.plot(traj_normalized[washout:-1, 0], traj_normalized[washout:-1, 1], traj_normalized[washout:-1, 2], 'b-', label='实际轨迹')
        # 绘制预测轨迹
        ax.plot(predictions[washout + 1:, 0], predictions[washout + 1:, 1], predictions[washout + 1:, 2], 'r--', label='量子预测轨迹')
        # 设置3D图的标题和标签
        ax.set_title('系统轨迹对比')
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('Z轴')
        ax.legend()
        plt.show()

#主函数
if __name__ == "__main__":
    # 生成NARMA-2时间序列
    u, y, traj_normalized = generate_narma2_series(num_steps=650)
    print(f"生成的NARMA-2时间序列:traj_normalized={traj_normalized}")
    # 使用可视化模块的第一个函数绘制NARMA-2时间序列
    #plot_normalized_traj(traj_normalized, title="NARMA-2 Time Series")
    
    # 量子计算部分
    quantum_outputs, n_qubits, n_cbits = all_probabilities(traj_normalized,
                                         varepsilon=0.2, 
                                         n_qubits=4, 
                                         n_cbits=4, 
                                         seed=42, 
                                         nub=10)
    print(quantum_outputs)
    # 训练模型
    ridge, washout, nstop_train = train_ridge_model(washout=0, 
                                  nstop_train=500, 
                                  alpha=0, 
                                  loading_dataX=quantum_outputs,
                                  loading_datay=traj_normalized)

    # 预测
    predictions = predict_next_state(method=ridge, loading_dataX=quantum_outputs, loading_datay=traj_normalized)

    # 输出结果
    #print(predictions)
    mse = mean_squared_error(traj_normalized[nstop_train+1:], predictions[nstop_train:-1])
    print(f"测试集MSE: {mse}")
    
    # 结果可视化
    plot_comparison(
        dimensions=['x'], 
        plot_3d=False, 
        plot_washout=False,
        washout=washout,
        nstop_train=nstop_train,
        traj_normalized=traj_normalized,
        predictions=predictions
    )