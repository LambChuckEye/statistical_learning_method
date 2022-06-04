import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# ================ 1.数据整理 ===============================================
# 读取鸢尾花数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# 取target为 +1 , -1 的类
df['label'] = iris.target
df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]

print(df)
# 画图
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title("data")
plt.legend()
plt.show()

# 构建数据集
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
# 标记正负样本
y = np.array([1 if i == 1 else -1 for i in y])


# ============== 2.手写实现 ========================================
class Model:
    # 初始化
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.lr = 0.1

    # 目标函数
    def forward(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 学习过程
    def fit(self, X_train, y_train):
        flag = False
        while not flag:
            wrong_count = 0
            # 一个epoch
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                # 负样本时权重更新
                if y * self.forward(X, self.w, self.b) <= 0:
                    self.w += self.lr * np.dot(y, X)
                    self.b += self.lr * y
                    wrong_count += 1
            # 无负样本时结束训练
            if wrong_count == 0:
                flag = True


# 计算
perceptron = Model()
perceptron.fit(X, y)

print(perceptron.w)
print(perceptron.b)

# 画图
x_points = np.linspace(4, 7, 10)
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
plt.plot(x_points, y_)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.title("custom")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# ============ 3.sklearn实现 =======================================
from sklearn.linear_model import Perceptron

# 定义模型
clf = Perceptron(fit_intercept=True,
                 max_iter=1000,
                 shuffle=True)
# 计算
clf.fit(X, y)
# 查看学得参数
print(clf.coef_)
print(clf.intercept_)


# 画图
x_points = np.linspace(4, 7, 10)
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_points, y_)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title("sklearn")
plt.legend()
plt.show()
