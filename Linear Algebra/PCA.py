import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

"""
iris数据集的中文名是安德森鸢尾花卉数据集，英文全称是Anderson’s Iris data set。
iris包含150个样本，对应数据集的每行数据。
每行数据包含每个样本的四个特征和样本的类别信息，所以iris数据集是一个150行5列的二维表。
通俗地说，iris数据集是用来给花做分类的数据集，
每个样本包含了花萼长度、花萼宽度、花瓣长度、花瓣宽度四个特征（前4列），
我们需要建立一个分类器，
分类器可以通过样本的四个特征来判断样本属于山鸢尾、变色鸢尾还是维吉尼亚鸢尾。
iris的每个样本都包含了品种信息，即目标属性（第5列，也叫target或label）。
"""

# 数据载入
iris = load_iris()
# DataFrame的行索引是index，列索引是columns
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

# 查看数据
"""
pandas 的 value_counts() 函数可以对Series里面的每个值进行计数并且排序。
"""
print(df.label.value_counts())
"""
DataFrame.tail(n)
返回最后 n 行。
对于n的负值，此函数返回除前n行之外的所有行，相当于 df[n:] 
"""
print(df.tail(5))

"""
df.iloc[]根据数据的坐标（position）获取数据
"""
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]
print("查看第一个数据：\n", X.iloc[0])
print("查看第一个标签：\n", y.iloc[0])


# 手写PCA实现
class PCA():
    def __init__(self):
        pass

    def fit(self, X, k):
        n_samples = X.shape[0]
        # 行零均值化
        mean = X.mean(axis=0)
        X_norm = X - mean
        # 协方差矩阵
        C = np.cov(X_norm.T)
        # 特征值分解
        eval, evect = np.linalg.eig(C)
        # 取前k行
        sorted_indices = np.argsort(-eval)
        P = evect.T[sorted_indices[:k]]
        X_low = np.dot(X_norm,P.T)
        return X_low


model = PCA()
Y = model.fit(X, 2)
# sklearn库调包
# sklearn_pca = sklearnPCA(n_components=2)
# Y = sklearn_pca.fit_transform(X)

principalDf = pd.DataFrame(data = np.array(Y),
                           columns=['principal component1', 'principal component2'])
Df = pd.concat([principalDf, y], axis=1)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('principal component1', fontsize=15)
ax.set_xlabel('principal component2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = Df['label'] == target
    ax.scatter(Df.loc[indicesToKeep, 'principal component1']
               , Df.loc[indicesToKeep, 'principal component2']
               , c=color)
ax.legend(targets)
ax.grid()
plt.show()

"""
问题解决：为什么手写和调包画的图旋转了180度
https://www.cnblogs.com/lpzblog/p/9519756.html
sklearn中的PCA分解时的方法是通过SVD来实现的。发现了少了一步，
在sklearn中的SVD中分解完后进行翻转特征向量符号以强制执行确定
性输出操作（svd_flip），这才是相差负号的原因。
fit中的部分关键代码
...
self.mean_ = np.mean(X, axis=0)
X -= self.mean_
U, S, V = linalg.svd(X, full_matrices=False)
U, V = svd_flip(U, V)
components_ = V
...
self.components_ = components_[:n_components]

https://zhuanlan.zhihu.com/p/85701151
SVD奇异值分解的结果是唯一的，但是分解出来的U矩阵和V矩阵的正负可以不是唯一，
只要保证它们乘起来是一致的就行。因此，sklearn为了保证svd分解结果的一致性，
它们的方案是：保证U矩阵的每一行(u_i)中，绝对值最大的元素一定是正数，
否则将u_i转成-u_i,并将相应的v_i转成-v_i已保证结果的一致。

这又是数学与工程的问题了。在数学上，几种结果都是正确的。
但是在工程上，有个很重要的特性叫幂等性(Idempotence)。

Methods can also have the property of “idempotence” in that 
(aside from error or expiration issues) the side-effects 
of N > 0 identical requests is the same as for a single request.
这是源自于HTTP规范中的一个概念，可以引申至各种分布式服务的设计当中，
即：高质量的服务，一次请求和多次请求，其副作用（结果）应当是一致的。
Scikit Learn正是通过svd_flip这个函数，把一个数学上并不幂等的操作，
转化成了幂等的服务，其设计之讲究可见一斑。
...
"""