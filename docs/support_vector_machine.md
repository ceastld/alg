# 支持向量机（SVM）完整数学推导

## 1. SVM基本概念

### 1.1 核心思想
支持向量机（Support Vector Machine）是一种用于分类任务的监督学习算法。SVM的核心思想是找到一个最优的超平面，使得不同类别之间的间隔（margin）最大化。

### 1.2 数学表示
对于二分类问题，给定训练集 $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$，其中 $y^{(i)} \in \{-1, +1\}$。

SVM寻找一个超平面：
$$w^T x + b = 0$$

使得：
- 对于 $y^{(i)} = +1$ 的样本：$w^T x^{(i)} + b \geq +1$
- 对于 $y^{(i)} = -1$ 的样本：$w^T x^{(i)} + b \leq -1$

可以统一写成：
$$y^{(i)}(w^T x^{(i)} + b) \geq 1, \quad \forall i$$

## 2. 间隔最大化

### 2.1 几何间隔
点到超平面的距离为：
$$d = \frac{|w^T x + b|}{||w||}$$

对于支持向量（满足 $y^{(i)}(w^T x^{(i)} + b) = 1$ 的点），距离为：
$$d = \frac{1}{||w||}$$

### 2.2 优化目标
SVM的目标是最大化间隔，即最小化 $||w||$：

$$\min_{w,b} \frac{1}{2} ||w||^2$$

约束条件：
$$y^{(i)}(w^T x^{(i)} + b) \geq 1, \quad i = 1, 2, \ldots, m$$

## 3. 拉格朗日对偶

### 3.1 拉格朗日函数
引入拉格朗日乘子 $\alpha_i \geq 0$：

$$L(w, b, \alpha) = \frac{1}{2} ||w||^2 - \sum_{i=1}^m \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1]$$

### 3.2 对偶问题
对 $w$ 和 $b$ 求偏导并令其为零：

$$\frac{\partial L}{\partial w} = w - \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)} = 0$$

$$\frac{\partial L}{\partial b} = -\sum_{i=1}^m \alpha_i y^{(i)} = 0$$

得到：
$$w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}$$

$$\sum_{i=1}^m \alpha_i y^{(i)} = 0$$

### 3.3 对偶优化问题
将上述结果代入拉格朗日函数，得到对偶问题：

$$\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)}$$

约束条件：
$$\sum_{i=1}^m \alpha_i y^{(i)} = 0$$
$$\alpha_i \geq 0, \quad i = 1, 2, \ldots, m$$

### 3.4 KKT条件
最优解必须满足KKT条件：
$$\alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1] = 0$$

这意味着：
- 如果 $\alpha_i > 0$，则 $y^{(i)}(w^T x^{(i)} + b) = 1$（支持向量）
- 如果 $y^{(i)}(w^T x^{(i)} + b) > 1$，则 $\alpha_i = 0$（非支持向量）

## 4. 软间隔SVM

### 4.1 引入松弛变量
对于线性不可分的情况，引入松弛变量 $\xi_i \geq 0$：

$$\min_{w,b,\xi} \frac{1}{2} ||w||^2 + C \sum_{i=1}^m \xi_i$$

约束条件：
$$y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i, \quad i = 1, 2, \ldots, m$$
$$\xi_i \geq 0, \quad i = 1, 2, \ldots, m$$

其中 $C$ 是正则化参数。

### 4.2 软间隔对偶问题
$$\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)}$$

约束条件：
$$\sum_{i=1}^m \alpha_i y^{(i)} = 0$$
$$0 \leq \alpha_i \leq C, \quad i = 1, 2, \ldots, m$$

## 5. 核函数

### 5.1 核技巧
对于非线性可分问题，通过核函数将数据映射到高维空间：

$$K(x^{(i)}, x^{(j)}) = \phi(x^{(i)})^T \phi(x^{(j)})$$

### 5.2 常用核函数

#### 多项式核
$$K(x^{(i)}, x^{(j)}) = (x^{(i)} \cdot x^{(j)} + 1)^d$$

#### 高斯RBF核
$$K(x^{(i)}, x^{(j)}) = \exp\left(-\frac{||x^{(i)} - x^{(j)}||^2}{2\sigma^2}\right)$$

#### Sigmoid核
$$K(x^{(i)}, x^{(j)}) = \tanh(\gamma x^{(i)} \cdot x^{(j)} + r)$$

### 5.3 核化对偶问题
$$\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)}, x^{(j)})$$

## 6. 预测函数

### 6.1 决策函数
$$f(x) = \sum_{i=1}^m \alpha_i y^{(i)} K(x^{(i)}, x) + b$$

### 6.2 分类规则
$$\hat{y} = \text{sign}(f(x)) = \begin{cases}
+1 & \text{if } f(x) > 0 \\
-1 & \text{if } f(x) < 0
\end{cases}$$

## 7. Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable
from sklearn.datasets import make_classification

class SimpleSVM:
    """
    Simple Support Vector Machine implementation using SMO algorithm
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear', 
                 gamma: float = 1.0, degree: int = 3):
        """
        Initialize SVM model
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'poly', 'rbf')
            gamma: Kernel coefficient for 'rbf' and 'poly'
            degree: Polynomial degree for 'poly' kernel
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.alpha = None
        self.b = 0.0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
    
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Linear kernel"""
        return np.dot(X1, X2.T)
    
    def _polynomial_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Polynomial kernel"""
        return (self.gamma * np.dot(X1, X2.T) + 1) ** self.degree
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Gaussian) kernel"""
        pairwise_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + \
                        np.sum(X2**2, axis=1)[np.newaxis, :] - \
                        2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * pairwise_dists)
    
    def _kernel_function(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Apply kernel function"""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'poly':
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_bounds(self, i: int, j: int, y: np.ndarray) -> Tuple[float, float]:
        """Compute bounds for alpha values in SMO"""
        if y[i] != y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        return L, H
    
    def _compute_eta(self, i: int, j: int, X: np.ndarray) -> float:
        """Compute eta for SMO algorithm"""
        K_ii = self._kernel_function(X[i:i+1], X[i:i+1])[0, 0]
        K_jj = self._kernel_function(X[j:j+1], X[j:j+1])[0, 0]
        K_ij = self._kernel_function(X[i:i+1], X[j:j+1])[0, 0]
        return K_ii + K_jj - 2 * K_ij
    
    def _compute_error(self, i: int, X: np.ndarray, y: np.ndarray) -> float:
        """Compute prediction error for sample i"""
        prediction = self._predict_single(X[i], X, y)
        return prediction - y[i]
    
    def _predict_single(self, x: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Predict single sample"""
        if self.support_vectors is None:
            return 0.0
        
        kernel_values = self._kernel_function(x.reshape(1, -1), self.support_vectors)
        prediction = np.sum(self.support_vector_alphas * self.support_vector_labels * kernel_values[0])
        return prediction + self.b
    
    def _smo_step(self, i: int, j: int, X: np.ndarray, y: np.ndarray) -> bool:
        """Single SMO optimization step"""
        if i == j:
            return False
        
        # Compute bounds
        L, H = self._compute_bounds(i, j, y)
        if L == H:
            return False
        
        # Compute eta
        eta = self._compute_eta(i, j, X)
        if eta <= 0:
            return False
        
        # Compute errors
        E_i = self._compute_error(i, X, y)
        E_j = self._compute_error(j, X, y)
        
        # Save old alpha values
        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]
        
        # Update alpha_j
        self.alpha[j] = alpha_j_old + y[j] * (E_i - E_j) / eta
        self.alpha[j] = np.clip(self.alpha[j], L, H)
        
        # Update alpha_i
        self.alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alpha[j])
        
        # Check for convergence
        if abs(self.alpha[j] - alpha_j_old) < 1e-5:
            return False
        
        # Update bias
        b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * \
             self._kernel_function(X[i:i+1], X[i:i+1])[0, 0] - \
             y[j] * (self.alpha[j] - alpha_j_old) * \
             self._kernel_function(X[i:i+1], X[j:j+1])[0, 0]
        
        b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * \
             self._kernel_function(X[i:i+1], X[j:j+1])[0, 0] - \
             y[j] * (self.alpha[j] - alpha_j_old) * \
             self._kernel_function(X[j:j+1], X[j:j+1])[0, 0]
        
        if 0 < self.alpha[i] < self.C:
            self.b = b1
        elif 0 < self.alpha[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        return True
    
    def fit(self, X: np.ndarray, y: np.ndarray, max_iterations: int = 1000) -> None:
        """
        Train SVM using SMO algorithm
        
        Args:
            X: Training features (m, n)
            y: Training labels (m,) with values {-1, +1}
            max_iterations: Maximum number of iterations
        """
        m, n = X.shape
        
        # Initialize alpha values
        self.alpha = np.zeros(m)
        self.b = 0.0
        
        # SMO algorithm
        num_changed = 0
        examine_all = True
        
        for iteration in range(max_iterations):
            num_changed = 0
            
            if examine_all:
                # Examine all examples
                for i in range(m):
                    num_changed += self._smo_step(i, np.random.randint(m), X, y)
            else:
                # Examine only non-bound examples
                non_bound_indices = np.where((self.alpha > 1e-5) & (self.alpha < self.C - 1e-5))[0]
                for i in non_bound_indices:
                    num_changed += self._smo_step(i, np.random.randint(m), X, y)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Changed: {num_changed}")
        
        # Store support vectors
        support_vector_indices = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.support_vector_alphas = self.alpha[support_vector_indices]
        
        print(f"Number of support vectors: {len(support_vector_indices)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions (-1 or +1)
        """
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pred = self._predict_single(X[i], self.support_vectors, self.support_vector_labels)
            predictions[i] = 1 if pred > 0 else -1
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values
        
        Args:
            X: Input features
            
        Returns:
            Decision function values
        """
        values = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            values[i] = self._predict_single(X[i], self.support_vectors, self.support_vector_labels)
        return values

def generate_sample_data(n_samples: int = 100, n_features: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample binary classification data"""
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                              n_redundant=0, n_informative=2, 
                              n_clusters_per_class=1, random_state=42)
    y = 2 * y - 1  # Convert to {-1, +1}
    return X, y

def plot_decision_boundary(model: SimpleSVM, X: np.ndarray, y: np.ndarray) -> None:
    """Plot decision boundary for 2D data"""
    if X.shape[1] != 2:
        print("Can only plot decision boundary for 2D data")
        return
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Make predictions on the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(12, 8))
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], 
                linestyles=['--', '-', '--'], alpha=0.8)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.colorbar(scatter, label='True Label')
    
    # Plot support vectors
    if model.support_vectors is not None:
        plt.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], 
                   s=200, facecolors='none', edgecolors='black', linewidth=2, 
                   label='Support Vectors')
    
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main function to demonstrate SVM"""
    print("=== Support Vector Machine Demo ===\n")
    
    # Generate sample data
    print("1. Generating sample data...")
    X, y = generate_sample_data(n_samples=100, n_features=2)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Class distribution: {np.bincount((y + 1) // 2)}")
    
    # Train the model
    print("\n2. Training SVM...")
    model = SimpleSVM(C=1.0, kernel='linear')
    model.fit(X, y, max_iterations=1000)
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"   Training accuracy: {accuracy:.4f}")
    
    # Plot results
    print("\n4. Plotting results...")
    plot_decision_boundary(model, X, y)
    
    print("\n=== Demo completed ===")

if __name__ == "__main__":
    main()
```

## 8. 总结

支持向量机通过以下步骤实现分类：

1. **间隔最大化**：寻找最优超平面，最大化类别间隔
2. **对偶优化**：将原问题转化为对偶问题求解
3. **核技巧**：通过核函数处理非线性可分问题
4. **支持向量**：只有支持向量对决策边界有贡献

SVM具有泛化能力强、适合高维数据、内存效率高的特点，是分类任务中的重要算法。
