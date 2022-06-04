# 三、朴素贝叶斯

朴素贝叶斯算法是生成学习算法，通过先验概率求联合分布，再由联合分布求后验概率。

## 1. 生成模型

机器学习模型分为生成模型和判别模型，两种模型所求的目标不同。

对于联合概率分布和条件概率分布，我们有：
$$
P(Y|X) = \frac{P(X,Y)}{P(X)}
$$

- 对于生成模型，我们要根据训练集数据求出联合分布$P(X,Y)$，进而求出条件分布 $P(Y|X)$。
- 对于判别模型，我们直接学习 $P(Y|X)$ 的近似函数表达方式，不必知晓联合分布。

## 2. 贝叶斯定理

贝叶斯定理的一般形式为：
$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$
推广到分类问题上，有：
$$
\begin{aligned}
P(Y=c_k|X = x) &= \frac{P(X=x|Y=c_k)P(Y=c_k)}{P(X=x)}\\
& = \frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_kP(X=x|Y=c_k)P(Y=c_k)}
\end{aligned}
$$

## 3. 条件独立性假设

我们在计算$P(X=x|Y=c_k)$ 时，由于输入样本 X 中包含多个特征，所以其组合种类成指数形式增长，不适合计算。

为了方便计算，我们可以将输入样本 X 分割成不同的特征的组合，这样可以大大减少参数。

但自然条件下，输入样本的特征之间会有一定的联系，所以我们要进行条件独立性假设，来保证我们能进行分割。

条件独立性假设，指假设所有特征之间是独立的，即：
$$
\begin{aligned}
P(X=x|Y=c_k) &= P(X^1=x^1,...,X^n=x^n|Y=c_k)\\
&=\prod_{j=1}^{n}P(X^j = x^j|Y=c_k)
\end{aligned}
$$
朴素贝叶斯的朴素，就是指条件独立性假设是一个较强的假设。

## 4. 朴素贝叶斯公式

联立公式 28 和公式 29，我们有：
$$
\begin{aligned}
P(Y=c_k|X = x) &= \frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_kP(X=x|Y=c_k)P(Y=c_k)}\\
& = \frac{P(Y=c_k)\prod_{j=1}^{n}P(X^j = x^j|Y=c_k)}{\sum_kP(Y=c_k)\prod_{j=1}^{n}P(X^j = x^j|Y=c_k)}
\end{aligned}
$$
我们想通过朴素贝叶斯计算出后验概率最大的类别 $c_k$，所以我们可以将朴素贝叶斯分类器表示为：
$$
y = f(x) = arg\max_{c_k}\frac{P(Y=c_k)\prod_{j=1}^{n}P(X^j = x^j|Y=c_k)}{\sum_kP(Y=c_k)\prod_{j=1}^{n}P(X^j = x^j|Y=c_k)}
$$
可以发现对于所以 $c_k$ 而言，分母都是一样的，所以我们可以给出更一般的形式：
$$
y = arg\max_{c_k}P(Y=c_k)\prod_{j=1}^{n}P(X^j = x^j|Y=c_k)
$$

## 5. 后验概率最大化的意义

我们假设使用 0-1 损失函数计算期望风险：
$$
L(Y,f(X)) = 
\begin{cases}
1,&Y\ne f(X) \\
0,&Y=f(X)
\end{cases}
$$
则期望风险函数为：
$$
R_{exp}(f) = E[L(Y,f(X))]
$$
而期望是对于联合分布 $P(X,Y)$取的，所以增加 Y 后为：
$$
R_{exp}(f) = E_x\sum_{k=1}^K[L(c_k,f(X))]P(c_k|X=x)
$$
这样的，我们的期望风险最小化过程就等于：
$$
\begin{aligned}
f(x) &= arg\min_{y\in Y}\sum_{k=1}^KL(c_k,y)P(c_k|X=x)\\
& \because when\space y=c_k,L = 0\\
&= arg\min_{y\in Y}\sum_{k=1}^KP(y\ne c_k|X=x)\\
&= arg\min_{y\in Y}(1-P(y= c_k|X=x))\\
&= arg\max_{y\in Y}P(y= c_k|X=x)\\

\end{aligned}
$$
如此一来，期望风险最小化就等价于后验概率最大化了。

## 6. 参数估计

#### 1. 极大似然估计

$$
P(Y=c_k) = \frac{\sum_{i=1}^N I(y_i=c_k)}{N}\\
$$

$$
P(X^j = x^j|Y=c_k) = \frac{\sum_{i=1}^N I(x_i^j = a_j,y_i=c_k)}{\sum_{i=1}^NI(y_i = c_k)}
$$

#### 2. 贝叶斯估计

由于极大似然估计的概率可能为零，会影响接下来的计算，所以我们有贝叶斯估计：
$$
P(Y=c_k) = \frac{\sum_{i=1}^N I(y_i=c_k)+\lambda}{N+K\lambda},\space K:总类数
$$

$$
P(X^j = x^j|Y=c_k) = \frac{\sum_{i=1}^N I(x_i^j = a_j,y_i=c_k)+\lambda}{\sum_{i=1}^NI(y_i = c_k)+S_j\lambda},\space S_j:样本特征数
$$

