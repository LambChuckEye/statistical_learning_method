# 一、感知机

二分类线性分类模型

## 1. 模型：

模型表达式为：
$$
f(x) = sign(w\cdot x + b)
$$
其中 sign 函数为：
$$
sign(x) = \begin{cases} +1, & x \geq 0 \\
	-1, & x < 0
\end{cases}
$$


## 2. 损失函数：

使用误分类点到超平面的距离作为损失函数：

任意一点到超平面的距离为：
$$
\frac{1}{||w||} |w \cdot x + b|
$$
对于误分类点，我们有：
$$
-y_i(w \cdot x + b) > 0
$$
所以我们有单个误分类点距离超平面的距离公式：
$$
-\frac{1}{||w||}y_i(w \cdot x + b)
$$
进而可得所误分类点距离超平面的距离：
$$
-\frac{1}{||w||}\sum_{x_i\in M}y_i(w \cdot x + b)
$$
在感知机中，我们去函数距离作为损失函数，即：
$$
L(w,b) = -\sum_{x_i\in M}y_i(w \cdot x + b)
$$

## 3. 学习方法：

使用随机梯度下降作为感知机的学习方法，即求偏导为：
$$
\nabla_w L(w,b) = - \sum_ {x\in M}y_i x_i \\
\nabla_b L(w,b) = - \sum_ {x\in M}y_i
$$
进而有参数更新方法：
$$
w \leftarrow w + \eta y_i x_i \\
b \leftarrow b + \eta y_i 
$$

## 4. 感知机收敛性证明：

首先对参数进行定义：
$$
\hat{w_{opt}} = (w_{opt}, b_{opt}) \\
\hat{x_i} = (x_i,1)
$$
我们可以将求得的最优平面 $ w_{opt}x_i + b_{opt} =0$ 记为 $ \hat{w_{opt}} \hat{x_i} = 0$

由于平面 $ \hat{w_{opt}} \hat{x_i} = 0$ 为最优平面，所以所有数据点均为正样本，所以有：
$$
\hat{w_{opt}} \hat{x_i}y_i > 0
$$
即 $ \hat{w_{opt}} \hat{x_i}y_i $ 有下界，则我们可以说一定存在一个极小的正数$ \gamma$，使得：
$$
\hat{w_{opt}} \hat{x_i}y_i \geq \gamma
$$
对于参数更新方法，我们可以记为：
$$
\hat{w} \leftarrow \hat{w} + \eta y_i \hat{x_i}
$$
则有递推公式：
$$
\begin{aligned}
\hat{w_k} &= \hat{w_{k-1}} + \eta y_i \hat{x_i}\\
& = \hat{w_{k-2}} + 2\eta y_i \hat{x_i}\\
& \dots \\
& = k\eta y_i \hat{x_i}

\end{aligned}
$$
由公式12和公式14，我们可以得到：
$$
\begin{aligned}
\hat{w_k}\hat{w_{opt}} & =  k\eta y_i \hat{x_i}\hat{w_{opt}}\\
&\ge k\eta\gamma 
\end{aligned}
$$
同样的，对于$ ||\hat{w_k}||^2$ 有递推公式：
$$
\begin{aligned}
||\hat{w_k}||^2 & = ||\hat{w_{k-1}}+ \eta y_i\hat{x_i}||\\
& = ||\hat{w_{k-1}}||^2 + 2\eta y_i\hat{x_i}\hat{w_{k-1}} + \eta^2 y_i^2||\hat{x_i}||^2 \\
& \because y_i\hat{x_i}\hat{w_{k-1}} > 0
\\ 
& \le ||\hat{w_{k-1}}||^2+\eta^2||\hat{x_i}||^2\\
& ...\\
& \le k\eta^2||\hat{x_i}||^2
\end{aligned}
$$
设参数 R 为 $R= \max||\hat{x_i}||^2$，则有：
$$
\begin{aligned}
||\hat{w_k}||^2  \le k\eta^2R^2
\end{aligned}
$$
由于超平面为 $ \hat{w_{opt}} \hat{x_i} = 0$，所以我们可以通过缩放，使$ ||\hat{w_{opt}} || = 1$，这样对于公式 15 计算L2距离，可以得到：
$$
||\hat{w_k}||^2\space||\hat{w_{opt}}||^2 \ge k^2\eta^2\gamma^2 \\
||\hat{w_k}||^2\ge k^2\eta^2\gamma^2 \\
$$
公式18与公式17联立，得：
$$
k\eta^2R^2 \ge||\hat{w_k}||^2\ge k^2\eta^2\gamma^2 \\
  k\eta^2R^2 \ge k^2\eta^2\gamma^2 \\
  k \le (\frac{R}{\gamma})^2
$$
迭代次数 $k$ 有上界，即感知机学习过程可以在有限次迭代后得到最优平面，算法收敛性证明完成。

