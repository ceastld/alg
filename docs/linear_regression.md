# 线性回归完整数学推导

## 1. 线性回归模型

### 1.1 基本概念
线性回归是一种用于回归任务的监督学习算法，用于预测连续的数值输出。给定输入特征 $x \in \mathbb{R}^d$，线性回归模型预测输出 $y \in \mathbb{R}$。

### 1.2 模型定义
线性回归模型假设输出是输入的线性组合：

$$h_w(x) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_d x_d = w^T x + b$$

其中：
- $w = [w_1, w_2, \ldots, w_d]^T$ 是权重向量
- $b = w_0$ 是偏置项
- $x = [x_1, x_2, \ldots, x_d]^T$ 是输入特征向量

为了简化表示，我们通常将偏置项合并到权重向量中：
- $\tilde{w} = [w_0, w_1, w_2, \ldots, w_d]^T$
- $\tilde{x} = [1, x_1, x_2, \ldots, x_d]^T$

则模型简化为：$h_w(x) = \tilde{w}^T \tilde{x}$

## 2. 损失函数推导

### 2.1 最小二乘法
线性回归使用最小二乘法作为损失函数，目标是找到参数 $w$ 使得预测值与真实值的平方误差最小。

对于训练集 $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$，损失函数为：

$$J(w) = \frac{1}{2m} \sum_{i=1}^m (h_w(x^{(i)}) - y^{(i)})^2$$

$$= \frac{1}{2m} \sum_{i=1}^m (w^T x^{(i)} + b - y^{(i)})^2$$

### 2.2 矩阵形式
将损失函数写成矩阵形式：

$$J(w) = \frac{1}{2m} ||Xw - y||^2$$

其中：
- $X$ 是 $m \times (d+1)$ 的设计矩阵（包含偏置列）
- $w$ 是 $(d+1) \times 1$ 的权重向量
- $y$ 是 $m \times 1$ 的目标向量

## 3. 梯度推导

### 3.1 对权重的梯度
计算 $\frac{\partial J}{\partial w_j}$：

$$\frac{\partial J}{\partial w_j} = \frac{1}{2m} \sum_{i=1}^m \frac{\partial}{\partial w_j} (w^T x^{(i)} + b - y^{(i)})^2$$

$$= \frac{1}{2m} \sum_{i=1}^m 2(w^T x^{(i)} + b - y^{(i)}) \cdot \frac{\partial}{\partial w_j} (w^T x^{(i)} + b - y^{(i)})$$

$$= \frac{1}{m} \sum_{i=1}^m (w^T x^{(i)} + b - y^{(i)}) \cdot x_j^{(i)}$$

$$= \frac{1}{m} \sum_{i=1}^m (h_w(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

### 3.2 向量化梯度
将所有梯度组合成向量形式：

$$\nabla_w J = \frac{1}{m} X^T (Xw - y)$$

### 3.3 解析解（正规方程）
通过令梯度为零，可以得到解析解：

$$\nabla_w J = 0$$

$$\frac{1}{m} X^T (Xw - y) = 0$$

$$X^T Xw = X^T y$$

$$w = (X^T X)^{-1} X^T y$$

这就是正规方程（Normal Equation）。

## 4. 梯度下降算法

### 4.1 批量梯度下降
更新规则：

$$w := w - \alpha \nabla_w J$$

$$w := w - \frac{\alpha}{m} X^T (Xw - y)$$

其中 $\alpha$ 是学习率。

### 4.2 随机梯度下降
对于每个样本 $(x^{(i)}, y^{(i)})$：

$$w := w - \alpha (h_w(x^{(i)}) - y^{(i)}) x^{(i)}$$

### 4.3 小批量梯度下降
对于小批量 $B$：

$$w := w - \frac{\alpha}{|B|} \sum_{i \in B} (h_w(x^{(i)}) - y^{(i)}) x^{(i)}$$

## 5. 正则化

### 5.1 岭回归（L2正则化）
添加L2正则化项：

$$J_{ridge}(w) = \frac{1}{2m} ||Xw - y||^2 + \frac{\lambda}{2m} ||w||^2$$

梯度：

$$\nabla_w J_{ridge} = \frac{1}{m} X^T (Xw - y) + \frac{\lambda}{m} w$$

更新规则：

$$w := w - \alpha \left( \frac{1}{m} X^T (Xw - y) + \frac{\lambda}{m} w \right)$$

### 5.2 Lasso回归（L1正则化）
添加L1正则化项：

$$J_{lasso}(w) = \frac{1}{2m} ||Xw - y||^2 + \frac{\lambda}{m} ||w||_1$$

## 6. Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class LinearRegression:
    """
    Linear Regression implementation using numpy
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 regularization: str = 'none', lambda_reg: float = 0.01):
        """
        Initialize linear regression model
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            regularization: Type of regularization ('none', 'l1', 'l2')
            lambda_reg: Regularization parameter
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.cost_history = []
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias column to feature matrix"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean squared error cost"""
        m = X.shape[0]
        predictions = np.dot(X, self.weights)
        mse = (1/(2*m)) * np.sum((predictions - y)**2)
        
        # Add regularization
        if self.regularization == 'l2':
            mse += (self.lambda_reg/(2*m)) * np.sum(self.weights[1:]**2)  # Exclude bias
        elif self.regularization == 'l1':
            mse += (self.lambda_reg/m) * np.sum(np.abs(self.weights[1:]))  # Exclude bias
            
        return mse
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear regression model
        
        Args:
            X: Training features (m, n)
            y: Training labels (m,)
        """
        # Add bias column
        X_with_bias = self._add_bias(X)
        m, n = X_with_bias.shape
        
        # Initialize weights
        self.weights = np.zeros(n)
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward propagation
            predictions = np.dot(X_with_bias, self.weights)
            
            # Compute cost
            cost = self._compute_cost(X_with_bias, y)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/m) * np.dot(X_with_bias.T, (predictions - y))
            
            # Add regularization gradients
            if self.regularization == 'l2':
                dw[1:] += (self.lambda_reg/m) * self.weights[1:]  # Exclude bias
            elif self.regularization == 'l1':
                dw[1:] += (self.lambda_reg/m) * np.sign(self.weights[1:])  # Exclude bias
            
            # Update weights
            self.weights -= self.learning_rate * dw
            
            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        X_with_bias = self._add_bias(X)
        return np.dot(X_with_bias, self.weights)
    
    def plot_cost_history(self) -> None:
        """Plot the cost function history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

# Example usage
def generate_sample_data(n_samples: int = 100, n_features: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample regression data"""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with linear relationship + noise
    true_weights = np.random.randn(n_features + 1)  # +1 for bias
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    y = np.dot(X_with_bias, true_weights) + 0.1 * np.random.randn(n_samples)
    
    return X, y

def main():
    """Main function to demonstrate linear regression"""
    print("=== Linear Regression Demo ===\n")
    
    # Generate sample data
    print("1. Generating sample data...")
    X, y = generate_sample_data(n_samples=100, n_features=1)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Train the model
    print("\n2. Training the model...")
    model = LinearRegression(learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = np.mean((y - y_pred)**2)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    
    print(f"   Mean Squared Error: {mse:.4f}")
    print(f"   R-squared: {r2:.4f}")
    
    # Show final parameters
    print(f"\n4. Final parameters:")
    print(f"   Bias: {model.weights[0]:.4f}")
    print(f"   Weights: {model.weights[1:]}")
    
    # Plot results
    print("\n5. Plotting results...")
    model.plot_cost_history()
    
    # Plot data and regression line (for 1D case)
    if X.shape[1] == 1:
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.6, label='Data')
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, 'r-', label='Regression Line')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Linear Regression')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    print("\n=== Demo completed ===")

if __name__ == "__main__":
    main()
```

## 7. 总结

线性回归通过以下步骤实现回归任务：

1. **模型定义**：假设输出是输入的线性组合
2. **损失函数**：使用最小二乘法（均方误差）
3. **优化**：通过梯度下降或正规方程求解
4. **预测**：输出连续数值

线性回归具有简单、可解释性强、计算效率高的特点，是回归任务的基础算法。
