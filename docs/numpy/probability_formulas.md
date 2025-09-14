# 概率论公式与 NumPy 实现

## 目录
- [基本概率公式](#基本概率公式)
- [离散概率分布](#离散概率分布)
- [连续概率分布](#连续概率分布)
- [统计量](#统计量)
- [假设检验](#假设检验)
- [贝叶斯定理](#贝叶斯定理)

## 基本概率公式

### 概率基本性质
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 概率的基本性质
# 1. 非负性: P(A) ≥ 0
# 2. 规范性: P(Ω) = 1
# 3. 可列可加性: P(∪A_i) = ΣP(A_i) (互斥事件)

# 补事件概率
def complement_probability(p_a):
    """计算补事件概率 P(A') = 1 - P(A)"""
    return 1 - p_a

# 并事件概率（一般情况）
def union_probability(p_a, p_b, p_ab):
    """计算并事件概率 P(A∪B) = P(A) + P(B) - P(A∩B)"""
    return p_a + p_b - p_ab

# 条件概率
def conditional_probability(p_ab, p_b):
    """计算条件概率 P(A|B) = P(A∩B) / P(B)"""
    return p_ab / p_b if p_b != 0 else 0

# 示例
p_a = 0.3
p_b = 0.4
p_ab = 0.1

print(f"P(A) = {p_a}")
print(f"P(A') = {complement_probability(p_a)}")
print(f"P(A∪B) = {union_probability(p_a, p_b, p_ab)}")
print(f"P(A|B) = {conditional_probability(p_ab, p_b)}")
```

### 独立事件
```python
# 事件独立性判断
def are_independent(p_a, p_b, p_ab):
    """判断事件A和B是否独立"""
    return abs(p_ab - p_a * p_b) < 1e-10

# 独立事件的并概率
def independent_union_probability(p_a, p_b):
    """独立事件的并概率 P(A∪B) = P(A) + P(B) - P(A)P(B)"""
    return p_a + p_b - p_a * p_b

# 示例
p_a = 0.3
p_b = 0.4
p_ab = 0.12  # P(A∩B) = P(A)P(B) = 0.3 * 0.4 = 0.12

print(f"事件A和B是否独立: {are_independent(p_a, p_b, p_ab)}")
print(f"独立事件的并概率: {independent_union_probability(p_a, p_b)}")
```

## 离散概率分布

### 二项分布 (Binomial Distribution)
```python
# 二项分布: X ~ B(n, p)
# P(X = k) = C(n,k) * p^k * (1-p)^(n-k)

def binomial_probability(n, k, p):
    """计算二项分布概率"""
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binomial_mean_variance(n, p):
    """计算二项分布的均值和方差"""
    mean = n * p
    variance = n * p * (1 - p)
    return mean, variance

# 使用 NumPy 生成二项分布随机数
n, p = 10, 0.3
samples = np.random.binomial(n, p, 1000)

# 理论概率
k_values = np.arange(0, n + 1)
theoretical_probs = [binomial_probability(n, k, p) for k in k_values]

print(f"二项分布 B({n}, {p}) 的均值: {binomial_mean_variance(n, p)[0]}")
print(f"二项分布 B({n}, {p}) 的方差: {binomial_mean_variance(n, p)[1]}")
print(f"样本均值: {np.mean(samples)}")
print(f"样本方差: {np.var(samples)}")
```

### 泊松分布 (Poisson Distribution)
```python
# 泊松分布: X ~ P(λ)
# P(X = k) = (λ^k * e^(-λ)) / k!

def poisson_probability(lam, k):
    """计算泊松分布概率"""
    from math import factorial, exp
    return (lam ** k * exp(-lam)) / factorial(k)

def poisson_mean_variance(lam):
    """计算泊松分布的均值和方差"""
    return lam, lam  # 泊松分布的均值和方差都等于λ

# 使用 NumPy 生成泊松分布随机数
lam = 3
samples = np.random.poisson(lam, 1000)

# 理论概率
k_values = np.arange(0, 15)
theoretical_probs = [poisson_probability(lam, k) for k in k_values]

print(f"泊松分布 P({lam}) 的均值: {poisson_mean_variance(lam)[0]}")
print(f"泊松分布 P({lam}) 的方差: {poisson_mean_variance(lam)[1]}")
print(f"样本均值: {np.mean(samples)}")
print(f"样本方差: {np.var(samples)}")
```

### 几何分布 (Geometric Distribution)
```python
# 几何分布: X ~ Geom(p)
# P(X = k) = (1-p)^(k-1) * p

def geometric_probability(p, k):
    """计算几何分布概率"""
    return (1 - p) ** (k - 1) * p

def geometric_mean_variance(p):
    """计算几何分布的均值和方差"""
    mean = 1 / p
    variance = (1 - p) / (p ** 2)
    return mean, variance

# 使用 NumPy 生成几何分布随机数
p = 0.3
samples = np.random.geometric(p, 1000)

print(f"几何分布 Geom({p}) 的均值: {geometric_mean_variance(p)[0]}")
print(f"几何分布 Geom({p}) 的方差: {geometric_mean_variance(p)[1]}")
print(f"样本均值: {np.mean(samples)}")
```

## 连续概率分布

### 正态分布 (Normal Distribution)
```python
# 正态分布: X ~ N(μ, σ²)
# f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))

def normal_pdf(x, mu, sigma):
    """计算正态分布概率密度函数"""
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def normal_cdf(x, mu, sigma):
    """计算正态分布累积分布函数"""
    return 0.5 * (1 + np.sign(x - mu) * np.sqrt(1 - np.exp(-2 * (x - mu)**2 / (np.pi * sigma**2))))

# 使用 NumPy 生成正态分布随机数
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 1000)

# 计算理论值
x = np.linspace(-4, 4, 100)
pdf_values = normal_pdf(x, mu, sigma)

print(f"正态分布 N({mu}, {sigma}²) 的均值: {mu}")
print(f"正态分布 N({mu}, {sigma}²) 的方差: {sigma**2}")
print(f"样本均值: {np.mean(samples)}")
print(f"样本方差: {np.var(samples)}")
```

### 指数分布 (Exponential Distribution)
```python
# 指数分布: X ~ Exp(λ)
# f(x) = λ * exp(-λx), x ≥ 0

def exponential_pdf(x, lam):
    """计算指数分布概率密度函数"""
    return lam * np.exp(-lam * x) * (x >= 0)

def exponential_cdf(x, lam):
    """计算指数分布累积分布函数"""
    return 1 - np.exp(-lam * x) * (x >= 0)

def exponential_mean_variance(lam):
    """计算指数分布的均值和方差"""
    mean = 1 / lam
    variance = 1 / (lam ** 2)
    return mean, variance

# 使用 NumPy 生成指数分布随机数
lam = 2
samples = np.random.exponential(1/lam, 1000)

print(f"指数分布 Exp({lam}) 的均值: {exponential_mean_variance(lam)[0]}")
print(f"指数分布 Exp({lam}) 的方差: {exponential_mean_variance(lam)[1]}")
print(f"样本均值: {np.mean(samples)}")
```

### 均匀分布 (Uniform Distribution)
```python
# 均匀分布: X ~ U(a, b)
# f(x) = 1/(b-a), a ≤ x ≤ b

def uniform_pdf(x, a, b):
    """计算均匀分布概率密度函数"""
    return np.where((x >= a) & (x <= b), 1/(b-a), 0)

def uniform_cdf(x, a, b):
    """计算均匀分布累积分布函数"""
    return np.where(x < a, 0, np.where(x > b, 1, (x - a)/(b - a)))

def uniform_mean_variance(a, b):
    """计算均匀分布的均值和方差"""
    mean = (a + b) / 2
    variance = (b - a)**2 / 12
    return mean, variance

# 使用 NumPy 生成均匀分布随机数
a, b = 0, 1
samples = np.random.uniform(a, b, 1000)

print(f"均匀分布 U({a}, {b}) 的均值: {uniform_mean_variance(a, b)[0]}")
print(f"均匀分布 U({a}, {b}) 的方差: {uniform_mean_variance(a, b)[1]}")
print(f"样本均值: {np.mean(samples)}")
```

## 统计量

### 描述性统计
```python
def descriptive_statistics(data):
    """计算描述性统计量"""
    stats_dict = {
        'mean': np.mean(data),           # 均值
        'median': np.median(data),       # 中位数
        'mode': stats.mode(data)[0][0],  # 众数
        'std': np.std(data),             # 标准差
        'var': np.var(data),             # 方差
        'skewness': stats.skew(data),    # 偏度
        'kurtosis': stats.kurtosis(data), # 峰度
        'min': np.min(data),             # 最小值
        'max': np.max(data),             # 最大值
        'range': np.max(data) - np.min(data),  # 极差
        'q1': np.percentile(data, 25),   # 第一四分位数
        'q3': np.percentile(data, 75),   # 第三四分位数
        'iqr': np.percentile(data, 75) - np.percentile(data, 25)  # 四分位距
    }
    return stats_dict

# 示例
data = np.random.normal(0, 1, 1000)
stats_result = descriptive_statistics(data)

for key, value in stats_result.items():
    print(f"{key}: {value:.4f}")
```

### 中心极限定理
```python
def central_limit_theorem_demo(n_samples=1000, sample_size=30):
    """中心极限定理演示"""
    # 生成原始数据（非正态分布）
    original_data = np.random.exponential(1, n_samples * sample_size)
    
    # 计算样本均值
    sample_means = []
    for i in range(n_samples):
        sample = np.random.choice(original_data, sample_size, replace=False)
        sample_means.append(np.mean(sample))
    
    sample_means = np.array(sample_means)
    
    # 理论值
    original_mean = np.mean(original_data)
    original_std = np.std(original_data)
    theoretical_std = original_std / np.sqrt(sample_size)
    
    print(f"原始数据均值: {original_mean:.4f}")
    print(f"原始数据标准差: {original_std:.4f}")
    print(f"样本均值均值: {np.mean(sample_means):.4f}")
    print(f"样本均值标准差: {np.std(sample_means):.4f}")
    print(f"理论标准差: {theoretical_std:.4f}")
    
    return sample_means

# 演示中心极限定理
sample_means = central_limit_theorem_demo()
```

## 假设检验

### t检验
```python
def t_test_one_sample(data, mu0, alpha=0.05):
    """单样本t检验"""
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # 样本标准差
    
    # 计算t统计量
    t_stat = (sample_mean - mu0) / (sample_std / np.sqrt(n))
    
    # 计算p值
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    
    # 判断是否拒绝原假设
    reject_h0 = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_h0': reject_h0,
        'sample_mean': sample_mean,
        'sample_std': sample_std
    }

def t_test_two_samples(data1, data2, alpha=0.05):
    """双样本t检验"""
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    
    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # 计算t统计量
    t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2))
    
    # 计算p值
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n2 - 2))
    
    reject_h0 = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_h0': reject_h0,
        'mean1': mean1,
        'mean2': mean2
    }

# 示例
data1 = np.random.normal(100, 15, 30)
data2 = np.random.normal(105, 15, 30)

result1 = t_test_one_sample(data1, 100)
result2 = t_test_two_samples(data1, data2)

print("单样本t检验结果:")
for key, value in result1.items():
    print(f"{key}: {value}")

print("\n双样本t检验结果:")
for key, value in result2.items():
    print(f"{key}: {value}")
```

### 卡方检验
```python
def chi_square_test(observed, expected=None):
    """卡方检验"""
    if expected is None:
        # 期望频数相等
        expected = np.full_like(observed, np.sum(observed) / len(observed))
    
    # 计算卡方统计量
    chi2_stat = np.sum((observed - expected)**2 / expected)
    
    # 自由度
    df = len(observed) - 1
    
    # 计算p值
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': df
    }

# 示例
observed = np.array([20, 30, 25, 25])
result = chi_square_test(observed)
print("卡方检验结果:")
for key, value in result.items():
    print(f"{key}: {value}")
```

## 贝叶斯定理

### 贝叶斯公式
```python
def bayes_theorem(p_a, p_b_given_a, p_b):
    """贝叶斯定理: P(A|B) = P(B|A) * P(A) / P(B)"""
    return (p_b_given_a * p_a) / p_b

def bayes_theorem_with_evidence(p_a, p_b_given_a, p_b_given_not_a, p_not_a):
    """使用全概率公式的贝叶斯定理"""
    p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    return bayes_theorem(p_a, p_b_given_a, p_b)

# 示例：医学诊断
# P(疾病) = 0.01
# P(阳性|疾病) = 0.95
# P(阳性|无疾病) = 0.05

p_disease = 0.01
p_positive_given_disease = 0.95
p_positive_given_no_disease = 0.05
p_no_disease = 1 - p_disease

# 计算P(疾病|阳性)
p_disease_given_positive = bayes_theorem_with_evidence(
    p_disease, p_positive_given_disease, p_positive_given_no_disease, p_no_disease
)

print(f"P(疾病|阳性) = {p_disease_given_positive:.4f}")
```

### 贝叶斯更新
```python
def bayesian_update(prior, likelihood, evidence):
    """贝叶斯更新"""
    posterior = (likelihood * prior) / evidence
    return posterior

def sequential_bayesian_update(prior, likelihoods):
    """序列贝叶斯更新"""
    posterior = prior
    for likelihood in likelihoods:
        # 这里简化处理，实际应用中需要计算证据概率
        evidence = likelihood * posterior + (1 - likelihood) * (1 - posterior)
        posterior = bayesian_update(posterior, likelihood, evidence)
    return posterior

# 示例：硬币投掷
# 先验概率：硬币公平的概率为0.5
prior = 0.5
# 观察到连续3次正面
likelihoods = [0.5, 0.5, 0.5]  # 如果硬币公平，每次正面的概率

posterior = sequential_bayesian_update(prior, likelihoods)
print(f"观察到3次正面后，硬币公平的后验概率: {posterior:.4f}")
```

这些概率论公式和 NumPy 实现涵盖了概率论的核心概念，从基本概率公式到高级的贝叶斯推理。通过结合理论公式和实际代码实现，可以更好地理解和应用概率论知识。
