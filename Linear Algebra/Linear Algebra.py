import numpy as np

"""
有关线性代数的基本操作，
知识来自花书第二章，
代码参考https://github.com/MingchaoZhu/DeepLearning/blob/master
"""

"""
2.1 标量、向量、矩阵和张量
"""

# 标量
s = 1

# 向量
v = np.array([1, 2])

# 矩阵
m = np.array([[1, 2], [3, 4]])

# 张量
t = np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
])

print("标量：" + str(s))
print("向量：" + str(v))
print("矩阵：" + str(m))
print("张量：" + str(t))

# 矩阵转置

A = np.array([[1, 2, 3], [4, 5, 6]])
A_T = A.transpose()
# A_T = A.T
print("A:", A)
print("A的转置：", A_T)

# 矩阵加法

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B
print("C:", C)

# 广播
D = A + v
print("D:", D)

"""
2.2 矩阵和向量相乘
"""

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# 矩阵乘法

E = np.dot(A, B)
print("E:", E)

# Hadamard乘积

F = np.multiply(A, B)
G = A * B
print("F:", F)
print("G:", G)

# 向量内积

v1 = np.array([1, 2])
v2 = np.array([3, 4])
vp = np.dot(v1, v2)
print("向量内积vp:", vp)

"""
2.3 单位矩阵和逆矩阵
"""

# 单位矩阵,参数为阶数

I = np.identity(4)
# I = np.eye(3)
print("I:", I)

# 矩阵求逆
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
print("A的逆矩阵：", A_inv)

"""
2.4 线性相关和生成子空间
奇异矩阵和生成子空间的定义
"""

"""
2.5 范数
"""

a = np.array([1, 3])
print("向量的2范数：", np.linalg.norm(a, ord=2))
print("向量的1范数：", np.linalg.norm(a, ord=1))
print("向量的无穷范数：", np.linalg.norm(a, ord=np.inf))
print("A的F范数：", np.linalg.norm(A, ord="fro"))

"""
2.6 特殊类型的矩阵和向量

对角矩阵（长方形） 对称矩阵 单位向量 正交矩阵 
"""

"""
2.7 特征分解
"""
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
# 计算特征值
print("特征值：", np.linalg.eigvals(A))
# 计算特征值和特征向量
eigvals, eigvectors = np.linalg.eig(A)
print("特征值：", eigvals)
print("特征向量：", eigvectors)

"""
2.10 迹运算
转置不变迹，乘法结果交换乘阵迹不变
"""
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("A的迹：",np.trace(A))
B = np.array([[1, 2, 3], [4, 5, 6]])
print("B的迹：",np.trace(B))

"""
2.11 行列式
"""
A = np.array([[1, 2, 3],
              [4, 10, 6],
              [7, 1, 9]])
print("A的行列式：",np.linalg.det(A))