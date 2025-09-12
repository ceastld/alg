# 概率论基本公式

## 1. 基本概率公式

### 1.1 概率的基本性质
- **非负性**: $P(A) \geq 0$
- **规范性**: $P(\Omega) = 1$ (样本空间概率为1)
- **可列可加性**: 对于互斥事件 $A_1, A_2, \ldots$，有 $P(\bigcup_{i=1}^{\infty} A_i) = \sum_{i=1}^{\infty} P(A_i)$

### 1.2 补事件概率
$$P(\overline{A}) = 1 - P(A)$$

### 1.3 并事件概率
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

## 2. 条件概率

### 2.1 条件概率定义
$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

### 2.2 乘法公式
$$P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)$$

### 2.3 全概率公式
设 $B_1, B_2, \ldots, B_n$ 是样本空间的一个划分，则：
$$P(A) = \sum_{i=1}^{n} P(A|B_i) \cdot P(B_i)$$

### 2.4 贝叶斯定理
$$P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{\sum_{j=1}^{n} P(A|B_j) \cdot P(B_j)}$$

## 3. 独立性

### 3.1 事件独立性
事件 $A$ 和 $B$ 独立当且仅当：
$$P(A \cap B) = P(A) \cdot P(B)$$

等价地：
$$P(A|B) = P(A) \quad \text{或} \quad P(B|A) = P(B)$$

### 3.2 多个事件的独立性
$n$ 个事件 $A_1, A_2, \ldots, A_n$ 相互独立当且仅当对任意 $k$ 个事件 $A_{i_1}, A_{i_2}, \ldots, A_{i_k}$，有：
$$P(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = P(A_{i_1}) \cdot P(A_{i_2}) \cdots P(A_{i_k})$$

## 4. 随机变量

### 4.1 离散随机变量
- **概率质量函数**: $p(x) = P(X = x)$
- **累积分布函数**: $F(x) = P(X \leq x) = \sum_{k \leq x} p(k)$

### 4.2 连续随机变量
- **概率密度函数**: $f(x)$ 满足 $\int_{-\infty}^{\infty} f(x) dx = 1$
- **累积分布函数**: $F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) dt$

## 5. 期望和方差

### 5.1 期望（均值）
- **离散情况**: $E[X] = \sum_{i} x_i \cdot p(x_i)$
- **连续情况**: $E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$

### 5.2 期望的性质
- $E[aX + b] = aE[X] + b$
- $E[X + Y] = E[X] + E[Y]$
- 若 $X$ 和 $Y$ 独立，则 $E[XY] = E[X]E[Y]$

### 5.3 方差
$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

### 5.4 方差的性质
- $\text{Var}(aX + b) = a^2\text{Var}(X)$
- 若 $X$ 和 $Y$ 独立，则 $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

## 6. 常见分布

### 6.1 二项分布 $B(n, p)$
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$
- $E[X] = np$
- $\text{Var}(X) = np(1-p)$

### 6.2 泊松分布 $P(\lambda)$
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$
- $E[X] = \lambda$
- $\text{Var}(X) = \lambda$

### 6.3 正态分布 $N(\mu, \sigma^2)$
$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
- $E[X] = \mu$
- $\text{Var}(X) = \sigma^2$

### 6.4 指数分布 $Exp(\lambda)$
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$
- $E[X] = \frac{1}{\lambda}$
- $\text{Var}(X) = \frac{1}{\lambda^2}$

## 7. 中心极限定理

### 7.1 林德伯格-勒维中心极限定理（经典版本）

**条件**：
- $X_1, X_2, \ldots, X_n$ 是独立同分布的随机变量
- $E[X_i] = \mu < \infty$（期望存在且有限）
- $\text{Var}(X_i) = \sigma^2 < \infty$（方差存在且有限，$\sigma^2 > 0$）
- $n \to \infty$（样本量趋于无穷大）

**结论**：
$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

其中 $\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$ 是样本均值。

### 7.2 林德伯格中心极限定理（一般版本）

**条件**：
- $X_1, X_2, \ldots, X_n$ 是独立的随机变量（不必同分布）
- $E[X_i] = \mu_i$，$\text{Var}(X_i) = \sigma_i^2$
- 记 $s_n^2 = \sum_{i=1}^{n} \sigma_i^2$，$S_n = \sum_{i=1}^{n} X_i$
- **林德伯格条件**：对任意 $\epsilon > 0$，
  $$\lim_{n \to \infty} \frac{1}{s_n^2} \sum_{i=1}^{n} E[(X_i - \mu_i)^2 \cdot I_{|X_i - \mu_i| > \epsilon s_n}] = 0$$

**结论**：
$$\frac{S_n - \sum_{i=1}^{n} \mu_i}{s_n} \xrightarrow{d} N(0, 1)$$

### 7.3 李雅普诺夫中心极限定理

**条件**：
- $X_1, X_2, \ldots, X_n$ 是独立的随机变量
- $E[X_i] = \mu_i$，$\text{Var}(X_i) = \sigma_i^2$
- 存在 $\delta > 0$，使得 $E[|X_i - \mu_i|^{2+\delta}] < \infty$
- **李雅普诺夫条件**：
  $$\lim_{n \to \infty} \frac{\sum_{i=1}^{n} E[|X_i - \mu_i|^{2+\delta}]}{(\sum_{i=1}^{n} \sigma_i^2)^{1+\delta/2}} = 0$$

**结论**：
$$\frac{S_n - \sum_{i=1}^{n} \mu_i}{s_n} \xrightarrow{d} N(0, 1)$$

### 7.4 德莫弗-拉普拉斯中心极限定理（二项分布）

**条件**：
- $X_n \sim B(n, p)$，其中 $0 < p < 1$
- $n \to \infty$

**结论**：
$$\frac{X_n - np}{\sqrt{np(1-p)}} \xrightarrow{d} N(0, 1)$$

### 7.5 中心极限定理的收敛速度

对于经典中心极限定理，有**贝里-埃森定理**：

设 $X_1, X_2, \ldots, X_n$ 独立同分布，$E[X_i] = 0$，$\text{Var}(X_i) = 1$，$E[|X_i|^3] < \infty$，则：

$$\sup_{x} \left| P\left(\frac{S_n}{\sqrt{n}} \leq x\right) - \Phi(x) \right| \leq \frac{C E[|X_1|^3]}{\sqrt{n}}$$

其中 $C$ 是常数，$\Phi(x)$ 是标准正态分布的累积分布函数。

## 8. 大数定律

### 8.1 弱大数定律
$$\lim_{n \to \infty} P(|\bar{X} - \mu| < \epsilon) = 1$$

### 8.2 强大数定律
$$P(\lim_{n \to \infty} \bar{X} = \mu) = 1$$

## 9. 协方差和相关系数

### 9.1 协方差
$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$

### 9.2 相关系数
$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$$

### 9.3 协方差的性质
- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
- $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$
- 若 $X$ 和 $Y$ 独立，则 $\text{Cov}(X, Y) = 0$

## 10. 应用示例

### 10.1 贝叶斯定理应用
**问题**: 某地区晴天概率 0.7、雨天 0.3；若晴天穿蓝色外套的概率 0.4，雨天为 0.8。已知当天穿蓝外套，则当天是晴天的概率约为多少？

**解答**:
设 $A$ 为晴天事件，$B$ 为穿蓝外套事件。

根据贝叶斯定理：
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

其中：
- $P(A) = 0.7$
- $P(\overline{A}) = 0.3$
- $P(B|A) = 0.4$
- $P(B|\overline{A}) = 0.8$

根据全概率公式：
$$P(B) = P(B|A) \cdot P(A) + P(B|\overline{A}) \cdot P(\overline{A}) = 0.4 \times 0.7 + 0.8 \times 0.3 = 0.52$$

因此：
$$P(A|B) = \frac{0.4 \times 0.7}{0.52} = \frac{0.28}{0.52} \approx 0.538$$
