import pandas as pd
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 训练数据
wine = load_wine()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,
                                                wine.target,
                                                test_size=0.3)

# ID3
# clf = tree.DecisionTreeClassifier(criterion="entropy")
# CART
clf = tree.DecisionTreeClassifier(criterion="gini")
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)
print(score)

# 绘制决策树
feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素',
                '颜色强度', '色调', '稀释葡萄酒', '脯氨酸']

import graphviz

dot_data = tree.export_graphviz(clf,
                                out_file="myTree",
                                feature_names=feature_name,
                                class_names=["琴酒", "雪梨", "贝尔摩德"],
                                filled=True,
                                rounded=True)
graph = graphviz.Source(dot_data)

# 决策树特征
clf.feature_importances_
print([*zip(feature_name, clf.feature_importances_)])

# 剪枝参数曲线
import matplotlib.pyplot as plt

test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i + 1
                                      , criterion="entropy"
                                      , random_state=30
                                      , splitter="random"
                                      )
    clf = clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)
    test.append(score)
plt.plot(range(1, 11), test, color="red", label="max_depth")
plt.legend()
plt.show()
