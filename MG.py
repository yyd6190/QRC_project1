from reservoirpy.datasets import mackey_glass, lorenz
from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.observables import rmse, rsquare
import matplotlib.pyplot as plt
from pyqpanda import *
import numpy as np
import matplotlib
matplotlib.rc("font",family='Kai')
plt.rcParams['axes.unicode_minus'] =False

from all import *

nstop_train=2500

###Mackey-Glass
mg_series = mackey_glass(n_timesteps=10,
                         tau=17, 
                         a=0.2, 
                         b=0.1, 
                         n=10, 
                         x0=1.2, 
                         h=1.0, 
                         seed=42)
mg_norm = (mg_series - np.min(mg_series)) / (np.max(mg_series) - np.min(mg_series))
#IV.A(黄)中说明时间经离散化后，输入信号被线性放缩到[0,1]区间,故猜测为这种最大最小值归一化

reservoir = Reservoir(units=8,           #FIG.5(b)-(e) nodes[8,50,100]
                     lr=0.3,             #AppendixD:lr[0.4,0.6,0.8,1.0]
                     sr=1.25)            #AppendixD:lambda[1.0,1.25,1.5,1.75]
readout = Ridge(output_dim=1, ridge=1e-5)#AppendixD:beta=1e-5
esn = reservoir >> readout
esn.fit(mg_norm[:nstop_train], mg_norm[1:nstop_train+1], warmup=500)
predictions = esn.run(mg_norm[nstop_train+1:-1])

###Quantum

def create_quantum_circuit(n_qubits, a_in, a_fb, s_k, params):
    """创建量子线路并对每个量子比特分别测量期望值"""
    def R_ij(q, params1):
        prog = QProg()
        prog << RX(qubits[q], params1)
        prog << RX(qubits[q+1], params1) 
        prog << CNOT(qubits[q], qubits[q+1])
        prog << RZ(qubits[q+1], params1)
        prog << CNOT(qubits[q], qubits[q+1])
        return prog

    def U_res():
        prog = QProg()
        for qubit in n_qbits:
            prog << RY(qubit, np.pi/4)
        for qubit in n_qbits:
            prog << RX(qubit, np.pi/4)
        
    # 初始化量子虚拟机
    qvm = init_quantum_machine(QMachineType.CPU)
    qubits = qvm.qAlloc_many(n_qubits)
    
    # 构建量子程序
    prog = QProg()
    prog << R_ij(0, a_in*s_k)
    prog << R_ij(2, a_fb * params1[0])
    prog << R_ij(4, a_fb * params1[1])
    prog << R_ij(6, a_fb * params1[2])
    prog << R_ij(3, a_fb * params1[3])
    prog << R_ij(5, a_fb * params1[4])
    prog << R_ij(2, a_fb * params1[5])
    prog << R_ij(4, a_fb * params1[6])
    prog << R_ij(6, a_fb * params1[7]) 

    z_k = []
    for i in range(n_qubits):
        # 构建仅作用于第 i 个量子比特的 Pauli 算符
        pauli_dict = {f"Z{i}": 1}
        p = PauliOperator(pauli_dict)
        Hmt = p.to_hamiltonian(False)
        expec = qvm.get_expectation(prog, Hmt, qubits)
        z_k.append(expec)
    z_k.append(1)
    destroy_quantum_machine(qvm)
    return z_k

all_probabilities = []
z_j = np.random.uniform(0, 1, 8).tolist()
z_j.append(1)
all_probabilities.append(z_j)

for i in range(len(mg_norm)):
    z_j = create_quantum_circuit(n_qubits=8, a_in=0.01, a_fb=6,s_k=mg_norm[i], params=z_j)
    all_probabilities.append(z_j)

ridge, washout, nstop_train = train_ridge_model(washout=500, 
                                               nstop_train=2500, 
                                               alpha=0,
                                               loading_dataX=all_probabilities, 
                                               loading_datay=mg_norm)
                            
predictions = predict_next_state(method=ridge, 
                                 loading_dataX=all_probabilities, 
                                 loading_datay=mg_norm)

print(predictions)


# 创建完整轨迹对比图
plt.figure(figsize=(12, 6))
plt.plot(mg_norm[501:], 'b-', label='实际轨迹')
plt.plot(predictions[500:-1], 'r--', label='预测轨迹')
plt.xlabel('时间步')
plt.ylabel('归一化振幅')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
