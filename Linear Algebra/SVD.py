import numpy as np

"""
2.8 奇异值分解
"""
A = np.array([[1, 2, 3], [4, 5, 6]])
# A = np.array([[1,2],[3,4],[5,6]])
# 调用库函数版
U, D, V = np.linalg.svd(A)
print("库函数：")
print("U:", U)
print("D:", D)
print("V:", V)


# 把特征值和特征向量按从大到小排序
def sort_eval(eval, evct):
    # 参数不加负号-从小到大，参数加负号-从大到小
    sorted_indices = np.argsort(-eval)
    evals = eval[sorted_indices]
    topk_evecs = evct[sorted_indices]
    return evals, topk_evecs


# 手写版
lam, U = np.linalg.eig(np.dot(A, A.T))
lam1, V = np.linalg.eig(np.dot(A.T, A))
lam, U = sort_eval(lam, U)
lam1, V = sort_eval(lam1, V)

V = V.T
D = np.sqrt(lam)
print("手写：")
print("U:", U)
print("D:", D)
print("V:", V)

"""
2.9 Moore-Penrose 伪逆
"""
# 调用库函数版
A_pinv = np.linalg.pinv(A)
print("库函数：")
print("A的伪逆:", A_pinv)

# 手写版
# D的伪逆是对D的所有非0元素取倒数后转置
D_pinv = 1. / D
D_pinv = np.diag(D_pinv)
# print("D_pinv:",D_pinv)
# 补0
diff = A.shape[0] - A.shape[1]
if diff > 0:
    b = np.zeros(A.shape[1])
    for i in range(diff):
        D_pinv = np.insert(D_pinv, D_pinv.shape[0], values=b, axis=0)
elif diff < 0:
    b = np.zeros(A.shape[0])
    for i in range(abs(diff)):
        D_pinv = np.insert(D_pinv, D_pinv.shape[1], values=b, axis=1)
# print("D_pinv:",D_pinv)
D_pinv = D_pinv.T
A_pinv = np.dot(V.T, D_pinv)
A_pinv = np.dot(A_pinv, U.T)
print("手写：")
print("A的伪逆:", A_pinv)
