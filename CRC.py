import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Tuple
from sklearn.linear_model import Ridge

# ======================== Lorenz63系统部分 ========================
def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Lorenz63系统的微分方程
    
    Args:
        t: 时间点
        state: 系统状态 [x, y, z]
        sigma, rho, beta: 系统参数
        
    Returns:
        状态导数 [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def generate_lorenz_data(initial_state=[1.0, 1.0, 1.0], 
                         t_span=(0, 100), 
                         dt=0.01,
                         sigma=10.0, 
                         rho=28.0, 
                         beta=8.0/3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成Lorenz系统的时间序列数据
    
    Args:
        initial_state: 初始状态 [x0, y0, z0]
        t_span: 时间范围 (t_start, t_end)
        dt: 时间步长
        sigma, rho, beta: 系统参数
        
    Returns:
        t: 时间点数组
        states: 对应的系统状态数组，形状为(n_points, 3)
    """
    t = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        lambda t, y: lorenz_system(t, y, sigma, rho, beta),
        t_span,
        initial_state,
        t_eval=t,
        method='RK45'
    )
    return sol.t, sol.y.T

# ======================== Classical Reservoir Computing 部分 ========================
class ClassicalReservoirComputing:
    def __init__(self, 
                 n_reservoir=200,
                 sparsity=0.05,
                 spectral_radius=0.95,
                 noise=1e-3,
                 input_scaling=0.5,
                 ridge_alpha=1e-4,
                 input_dim=3):  # 添加输入维度参数
        """
        Classical Reservoir Computing (经典回声状态网络)实现
        
        Args:
            n_reservoir: 储层中神经元数量
            sparsity: 储层连接的稀疏度（0到1之间）
            spectral_radius: 储层权重矩阵的谱半径，影响系统记忆长度
            noise: 训练时添加的噪声大小
            input_scaling: 输入缩放因子
            ridge_alpha: 岭回归正则化参数
            input_dim: 输入维度
        """
        self.n_reservoir = n_reservoir
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.noise = noise
        self.input_scaling = input_scaling
        self.ridge_alpha = ridge_alpha
        self.input_dim = input_dim
        
        # 随机初始化储层权重
        self.W_reservoir = self._initialize_reservoir()
        
        # 初始化输入权重矩阵 (将输入维度映射到储层维度)
        self.W_in = np.random.rand(n_reservoir, input_dim) * 2 - 1
        
        # 训练后获得的输出权重
        self.W_out = None
        self.intercept = None
        
    def _initialize_reservoir(self):
        """初始化储层权重矩阵"""
        # 创建随机稀疏矩阵
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # 设置稀疏性
        mask = np.random.rand(self.n_reservoir, self.n_reservoir) < self.sparsity
        W *= mask
        # 计算矩阵的最大特征值
        eigenvalues = np.linalg.eigvals(W)
        max_eigen = np.max(np.abs(eigenvalues))
        # 缩放矩阵使其谱半径为spectral_radius
        W *= self.spectral_radius / max_eigen
        return W
        
    def _update(self, state, input_signal):
        """
        更新储层状态
        
        Args:
            state: 当前储层状态，形状为(n_reservoir,)
            input_signal: 输入信号，形状为(input_dim,)
            
        Returns:
            新的储层状态
        """
        # 使用输入权重矩阵将输入信号映射到储层维度
        input_scaled = np.dot(self.W_in, input_signal) * self.input_scaling
        
        # 非线性激活函数 - 使用tanh
        return np.tanh(np.dot(self.W_reservoir, state) + 
                     input_scaled + 
                     self.noise * (np.random.rand(self.n_reservoir) - 0.5))
    
    def fit(self, X, Y):
        """
        训练储层计算模型
        
        Args:
            X: 输入时间序列，形状为(n_samples, n_features)
            Y: 目标输出，形状为(n_samples, n_outputs)
        """
        n_samples = X.shape[0]
        
        # 初始化储层状态
        state = np.zeros((self.n_reservoir,))
        states_collection = np.zeros((n_samples, self.n_reservoir))
        
        # 收集储层状态
        for i in range(n_samples):
            input_signal = X[i]
            state = self._update(state, input_signal)
            states_collection[i] = state
        
        # 使用岭回归训练输出层
        ridge = Ridge(alpha=self.ridge_alpha)
        ridge.fit(states_collection, Y)
        self.W_out = ridge.coef_
        self.intercept = ridge.intercept_
        
        return self
        
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        Args:
            X: 输入时间序列，形状为(n_samples, n_features)
            
        Returns:
            预测值，形状为(n_samples, n_outputs)
        """
        n_samples = X.shape[0]
        n_outputs = self.W_out.shape[0]
        predictions = np.zeros((n_samples, n_outputs))
        
        # 初始化状态
        state = np.zeros((self.n_reservoir,))
        
        # 逐步预测
        for i in range(n_samples):
            input_signal = X[i]
            state = self._update(state, input_signal)
            prediction = np.dot(self.W_out, state) + self.intercept
            predictions[i] = prediction
            
        return predictions
    
    def predict_future(self, X_init, steps, feedback=True):
        """
        对未来进行预测
        
        Args:
            X_init: 初始输入状态，形状为(n_features,)
            steps: 需要预测的步数
            feedback: 是否使用预测结果作为下一步的输入
            
        Returns:
            预测序列，形状为(steps, n_outputs)
        """
        n_outputs = self.W_out.shape[0]
        predictions = np.zeros((steps, n_outputs))
        
        # 初始化状态
        state = np.zeros((self.n_reservoir,))
        current_input = X_init
        
        # 逐步预测
        for i in range(steps):
            state = self._update(state, current_input)
            prediction = np.dot(self.W_out, state) + self.intercept
            predictions[i] = prediction
            
            if feedback:
                # 使用预测结果作为下一步的输入（闭环预测）
                current_input = prediction
                
        return predictions

# ======================== 可视化部分 ========================
def plot_lorenz_system(states: np.ndarray):
    """
    绘制Lorenz系统的3D吸引子
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lorenz吸引子')
    
    plt.tight_layout()
    plt.show()

def plot_prediction(original: np.ndarray, predicted: np.ndarray, variable_idx=0, variable_name='x'):
    """
    绘制原始数据和预测结果的比较图
    
    Args:
        original: 原始数据，形状为(n_samples, n_dimensions)
        predicted: 预测数据，形状为(n_samples, n_dimensions)
        variable_idx: 要绘制的变量索引
        variable_name: 变量名称，用于标题
    """
    plt.figure(figsize=(12, 6))
    
    # 获取选定变量的数据
    orig_var = original[:, variable_idx]
    pred_var = predicted[:, variable_idx]
    
    # 训练-预测分割点
    train_size = len(orig_var) - len(pred_var)
    
    # 绘制原始数据
    plt.plot(range(len(orig_var)), orig_var, 'b-', label='实际值')
    
    # 绘制预测数据
    plt.plot(range(train_size, len(orig_var)), pred_var, 'r-', label='预测值')
    
    # 添加分界线
    plt.axvline(x=train_size, color='gray', linestyle='--')
    
    plt.title(f'Lorenz系统{variable_name}分量的预测结果')
    plt.xlabel('时间步')
    plt.ylabel(f'{variable_name}值')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ======================== 主函数 ========================
if __name__ == "__main__":
    # 生成Lorenz系统数据
    t, states = generate_lorenz_data(t_span=(0, 50), dt=0.01)
    
    # 绘制Lorenz吸引子
    plot_lorenz_system(states)
    
    # 分割数据为训练集和测试集
    train_size = int(0.8 * len(states))
    train_data = states[:train_size]
    test_data = states[train_size:]
    
    # 准备训练数据（使用当前状态预测下一状态）
    X_train = train_data[:-1]
    Y_train = train_data[1:]
    
    # 初始化并训练CRC模型
    crc = ClassicalReservoirComputing(
        n_reservoir=300,
        sparsity=0.05,
        spectral_radius=0.95,
        noise=1e-3,
        input_scaling=1.0,
        ridge_alpha=1e-6,
        input_dim=3  # 指定输入维度为3 (Lorenz系统的x,y,z)
    )
    
    # 训练模型
    crc.fit(X_train, Y_train)
    
    # 使用训练好的模型进行短期预测
    n_predict = len(test_data)
    last_train = train_data[-1].reshape(1, -1)
    predictions = []
    
    # 初始输入是训练集最后一个状态
    current_input = last_train[0]
    
    # 进行多步预测
    for _ in range(n_predict):
        # 预测下一状态
        current_input = current_input.reshape(1, -1)
        next_state = crc.predict(current_input)[0]
        predictions.append(next_state)
        # 预测值作为下一步的输入
        current_input = next_state
    
    predictions = np.array(predictions)
    
    # 可视化结果
    plot_prediction(states, predictions, 0, 'x')
    plot_prediction(states, predictions, 1, 'y')
    plot_prediction(states, predictions, 2, 'z')
    
    print("\n经典回声状态网络(Classical Reservoir Computing)是一种神经网络计算方法，")
    print("其主要特点是使用固定权重的递归神经网络'储层'进行状态映射，只训练输出层权重，")
    print("这大大简化了训练过程，特别适合处理时间序列预测和混沌系统建模等任务。")
    print("CRC在预测Lorenz混沌系统上表现良好，能够捕捉系统短期动态行为。")
