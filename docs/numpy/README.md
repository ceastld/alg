# NumPy 数学库文档

本文件夹包含了 NumPy 常用方法、概率论和线性代数相关公式的详细文档，所有内容都配有中文说明和英文代码注释。

## 📚 文档结构

### 1. [NumPy 基础方法详解](numpy_basics.md)
- **数组创建**: 基本数组、特殊数组、随机数组
- **数组操作**: 形状操作、索引切片、数组拼接
- **数学运算**: 基本运算、矩阵运算、三角函数
- **统计函数**: 描述性统计、排序、聚合函数
- **线性代数**: 矩阵运算、特征值分解、线性方程组
- **随机数生成**: 各种概率分布的随机数生成
- **实用技巧**: 广播、条件运算、数组去重

### 2. [概率论公式与 NumPy 实现](probability_formulas.md)
- **基本概率公式**: 概率性质、条件概率、独立事件
- **离散概率分布**: 二项分布、泊松分布、几何分布
- **连续概率分布**: 正态分布、指数分布、均匀分布
- **统计量**: 描述性统计、中心极限定理
- **假设检验**: t检验、卡方检验
- **贝叶斯定理**: 贝叶斯公式、贝叶斯更新

### 3. [线性代数公式与 NumPy 实现](linear_algebra_formulas.md)
- **矩阵基础**: 矩阵创建、转置、基本属性
- **矩阵运算**: 基本运算、幂运算、Hadamard积
- **行列式与逆矩阵**: 行列式计算、逆矩阵性质、伴随矩阵
- **特征值与特征向量**: 特征值分解、特征值性质
- **线性方程组**: 高斯消元法、最小二乘法
- **向量空间**: 向量运算、线性相关性、基
- **正交化**: Gram-Schmidt正交化、QR分解
- **矩阵分解**: LU分解、SVD分解、特征值分解

## 🚀 快速开始

### 环境要求
```bash
# 使用 uv 安装依赖
uv add numpy scipy matplotlib
```

### 基本使用示例
```python
import numpy as np
import matplotlib.pyplot as plt

# 创建数组
arr = np.array([1, 2, 3, 4, 5])

# 基本运算
print(f"数组: {arr}")
print(f"均值: {np.mean(arr)}")
print(f"标准差: {np.std(arr)}")

# 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])
C = A @ B  # 矩阵乘法
print(f"矩阵乘法结果:\n{C}")
```

## 📖 学习路径建议

### 初学者路径
1. **NumPy 基础** → 掌握数组创建和基本操作
2. **数学运算** → 学习向量和矩阵运算
3. **统计函数** → 了解描述性统计
4. **概率分布** → 学习常用概率分布

### 进阶路径
1. **线性代数** → 深入理解矩阵理论
2. **特征值分解** → 掌握矩阵分解技术
3. **假设检验** → 学习统计推断
4. **贝叶斯推理** → 了解概率推理

### 高级应用
1. **矩阵分解** → SVD、QR、LU分解
2. **优化算法** → 最小二乘法、梯度下降
3. **机器学习** → 特征工程、数据预处理
4. **科学计算** → 数值分析、仿真

## 🔧 实用工具函数

### 常用工具函数集合
```python
# 数据预处理
def normalize_data(data):
    """数据标准化"""
    return (data - np.mean(data)) / np.std(data)

def min_max_scale(data):
    """最小-最大缩放"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# 矩阵工具
def is_symmetric(matrix):
    """判断矩阵是否对称"""
    return np.allclose(matrix, matrix.T)

def matrix_condition_number(matrix):
    """计算矩阵条件数"""
    return np.linalg.cond(matrix)

# 统计工具
def confidence_interval(data, confidence=0.95):
    """计算置信区间"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    margin = stats.t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n)
    return mean - margin, mean + margin
```

## 📊 可视化示例

### 概率分布可视化
```python
import matplotlib.pyplot as plt

# 正态分布
x = np.linspace(-4, 4, 100)
y = (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)
plt.plot(x, y, label='Standard Normal')
plt.legend()
plt.show()
```

### 矩阵热图
```python
import seaborn as sns

# 相关性矩阵热图
correlation_matrix = np.corrcoef(data.T)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```

## 🎯 应用场景

### 数据科学
- **数据清洗**: 缺失值处理、异常值检测
- **特征工程**: 特征选择、特征变换
- **数据可视化**: 统计图表、分布图

### 机器学习
- **数据预处理**: 标准化、归一化
- **模型训练**: 梯度计算、参数更新
- **模型评估**: 性能指标、交叉验证

### 科学计算
- **数值分析**: 数值积分、微分方程
- **优化问题**: 线性规划、非线性优化
- **仿真建模**: 蒙特卡洛方法、随机过程

## 📝 注意事项

### 性能优化
- 使用向量化操作替代循环
- 合理选择数据类型（float32 vs float64）
- 利用 NumPy 的广播机制
- 避免不必要的数组复制

### 数值稳定性
- 注意浮点数精度问题
- 使用稳定的算法（如SVD而非直接求逆）
- 检查矩阵条件数
- 处理奇异矩阵情况

### 内存管理
- 使用 `inplace` 操作减少内存使用
- 及时释放不需要的大数组
- 合理使用 `copy()` 和 `view()`

## 🔗 相关资源

### 官方文档
- [NumPy 官方文档](https://numpy.org/doc/stable/)
- [SciPy 官方文档](https://docs.scipy.org/doc/scipy/)
- [Matplotlib 官方文档](https://matplotlib.org/stable/)

### 推荐书籍
- 《Python 科学计算》- 张若愚
- 《NumPy 用户指南》- 官方文档
- 《线性代数及其应用》- David C. Lay

### 在线课程
- Coursera: 机器学习课程
- edX: 数据科学微学位
- 慕课网: Python 数据分析

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这些文档：

1. 发现错误或需要改进的地方
2. 添加新的示例和用例
3. 改进代码的可读性和效率
4. 补充更多的数学理论说明

## 📄 许可证

本文档遵循 MIT 许可证，可以自由使用和修改。

---

**最后更新**: 2024年9月15日  
**维护者**: AI Assistant  
**版本**: 1.0.0
