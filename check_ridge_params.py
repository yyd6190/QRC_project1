from reservoirpy.nodes import Ridge
import reservoirpy

# 打印版本和帮助信息
print(f"ReservoirPy version: {reservoirpy.__version__}")
print("\nRidge class documentation:")
help(Ridge)

# 尝试不同的初始化参数
try:
    ridge1 = Ridge(ridge=1e-5)
    print("Ridge(ridge=1e-5) 成功")
except Exception as e:
    print(f"Ridge(ridge=1e-5) 失败: {e}")

try:
    ridge2 = Ridge(alpha=1e-5)
    print("Ridge(alpha=1e-5) 成功")
except Exception as e:
    print(f"Ridge(alpha=1e-5) 失败: {e}")

try:
    ridge3 = Ridge()
    print("Ridge() 成功")
except Exception as e:
    print(f"Ridge() 失败: {e}")
