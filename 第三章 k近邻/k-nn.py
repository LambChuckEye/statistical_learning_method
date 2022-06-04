import numpy as np
from sklearn.neighbors import KDTree, KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ================== 1. 数据集 ===========================
data = np.array([[5, 12, 1], [6, 21, 0], [14, 5, 0], [16, 10, 0], [13, 19, 0],
                 [13, 32, 1], [17, 27, 1], [18, 24, 1], [20, 20,
                                                         0], [23, 14, 1],
                 [23, 25, 1], [23, 31, 1], [26, 8, 0], [30, 17, 1],
                 [30, 26, 1], [34, 8, 0], [34, 19, 1], [37, 28, 1]])

plt.scatter(data[:, 0], data[:, 1], c=data[:, 2])
plt.show()
X_train = data[:, 0:2]
y_train = data[:, 2]

# ================== 2. sklearn 实现 ====================
# a. 使用KD树找最近点
tree = KDTree(X_train, leaf_size=2)
dist, ind = tree.query(np.array([[1, 3]]), k=1)
print(dist)
print(X_train[ind])

# b. 使用knn做分类
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)

# 画图
X0, X1 = X_train[:, 0], X_train[:, 1]
x_min, x_max = X0.min() - 1, X0.max() + 1
y_min, y_max = X1.min() - 1, X1.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(Z))])
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
plt.scatter(X0, X1, c=y_train, s=50, edgecolors='k', cmap=cmap, alpha=0.5)
plt.show()
