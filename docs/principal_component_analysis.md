# 主成分分析（PCA）完整数学推导

## 1. PCA基本概念

### 1.1 核心思想
主成分分析（Principal Component Analysis, PCA）是一种无监督降维技术，通过线性变换将数据投影到低维空间，同时保留数据的主要变化信息。

### 1.2 目标
PCA的目标是找到一组正交的主成分（主方向），使得：
1. 第一个主成分具有最大的方差
2. 第二个主成分具有次大的方差，且与第一个主成分正交
3. 依此类推...

### 1.3 数学表示
给定数据矩阵 $X \in \mathbb{R}^{m \times d}$，其中 $m$ 是样本数，$d$ 是特征数，PCA寻找投影矩阵 $W \in \mathbb{R}^{d \times k}$，使得：

$$Y = XW$$

其中 $Y \in \mathbb{R}^{m \times k}$ 是降维后的数据，$k < d$。

## 2. 数学推导

### 2.1 数据预处理
首先对数据进行中心化（零均值化）：

$$\bar{x} = \frac{1}{m} \sum_{i=1}^m x^{(i)}$$

$$x^{(i)}_{centered} = x^{(i)} - \bar{x}$$

中心化后的数据矩阵为：
$$\tilde{X} = X - \mathbf{1}\bar{x}^T$$

其中 $\mathbf{1}$ 是全1向量。

### 2.2 协方差矩阵
计算协方差矩阵：

$$C = \frac{1}{m-1} \tilde{X}^T \tilde{X}$$

协方差矩阵 $C$ 是 $d \times d$ 的对称矩阵，其元素为：

$$C_{ij} = \frac{1}{m-1} \sum_{k=1}^m (x^{(k)}_i - \bar{x}_i)(x^{(k)}_j - \bar{x}_j)$$

### 2.3 特征值分解
对协方差矩阵进行特征值分解：

$$C = V\Lambda V^T$$

其中：
- $V = [v_1, v_2, \ldots, v_d]$ 是特征向量矩阵
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$ 是特征值对角矩阵
- $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$

### 2.4 主成分选择
选择前 $k$ 个最大的特征值对应的特征向量作为主成分：

$$W = [v_1, v_2, \ldots, v_k]$$

### 2.5 降维变换
将数据投影到主成分空间：

$$Y = \tilde{X}W$$

## 3. 方差解释

### 3.1 主成分方差
第 $i$ 个主成分的方差为：

$$\text{Var}(Y_i) = \lambda_i$$

### 3.2 方差贡献率
第 $i$ 个主成分的方差贡献率为：

$$\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

### 3.3 累积方差贡献率
前 $k$ 个主成分的累积方差贡献率为：

$$\text{Cumulative Explained Variance Ratio}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

## 4. 优化目标

### 4.1 最大方差视角
PCA可以看作是在约束条件下最大化投影方差的问题：

$$\max_{w} \text{Var}(Xw) = \max_{w} w^T C w$$

约束条件：$||w||^2 = 1$

使用拉格朗日乘数法：

$$L(w, \lambda) = w^T C w - \lambda(w^T w - 1)$$

对 $w$ 求偏导：

$$\frac{\partial L}{\partial w} = 2Cw - 2\lambda w = 0$$

得到特征值方程：

$$Cw = \lambda w$$

### 4.2 最小重构误差视角
PCA也可以看作是最小化重构误差的问题：

$$\min_{W} ||X - XWW^T||_F^2$$

约束条件：$W^T W = I$

其中 $||\cdot||_F$ 是Frobenius范数。

## 5. 奇异值分解（SVD）方法

### 5.1 SVD分解
对中心化数据矩阵进行SVD分解：

$$\tilde{X} = U\Sigma V^T$$

其中：
- $U \in \mathbb{R}^{m \times m}$ 是左奇异向量矩阵
- $\Sigma \in \mathbb{R}^{m \times d}$ 是奇异值矩阵
- $V \in \mathbb{R}^{d \times d}$ 是右奇异向量矩阵

### 5.2 与特征值分解的关系
协方差矩阵的特征值分解与SVD的关系：

$$C = \frac{1}{m-1} \tilde{X}^T \tilde{X} = \frac{1}{m-1} V\Sigma^T U^T U\Sigma V^T = \frac{1}{m-1} V\Sigma^T \Sigma V^T$$

因此：
- 特征向量：$V$ 的列向量
- 特征值：$\frac{1}{m-1} \sigma_i^2$，其中 $\sigma_i$ 是奇异值

## 6. 算法步骤

### 6.1 标准PCA算法
1. 数据标准化：$X_{std} = \frac{X - \mu}{\sigma}$
2. 计算协方差矩阵：$C = \frac{1}{m-1} X_{std}^T X_{std}$
3. 特征值分解：$C = V\Lambda V^T$
4. 选择主成分：$W = V[:, :k]$
5. 降维变换：$Y = X_{std}W$

### 6.2 SVD算法
1. 数据标准化：$X_{std} = \frac{X - \mu}{\sigma}$
2. SVD分解：$X_{std} = U\Sigma V^T$
3. 选择主成分：$W = V[:, :k]$
4. 降维变换：$Y = X_{std}W$

## 7. Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import StandardScaler

class PCA:
    """
    Principal Component Analysis implementation
    """
    
    def __init__(self, n_components: Optional[int] = None, 
                 method: str = 'eigen', random_state: Optional[int] = None):
        """
        Initialize PCA
        
        Args:
            n_components: Number of components to keep
            method: Method for PCA ('eigen' or 'svd')
            random_state: Random seed
        """
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ratio_ = None
        self.mean_ = None
        self.scaler = StandardScaler()
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_covariance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute covariance matrix"""
        return np.cov(X.T)
    
    def _eigen_decomposition(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform PCA using eigenvalue decomposition"""
        # Compute covariance matrix
        cov_matrix = self._compute_covariance_matrix(X)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _svd_decomposition(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform PCA using SVD decomposition"""
        # SVD decomposition
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Convert singular values to eigenvalues
        eigenvalues = s ** 2 / (X.shape[0] - 1)
        
        # Transpose Vt to get eigenvectors
        eigenvectors = Vt.T
        
        return eigenvalues, eigenvectors
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA to the data
        
        Args:
            X: Input data (m, n)
            
        Returns:
            self
        """
        # Standardize the data
        X_scaled = self.scaler.fit_transform(X)
        self.mean_ = self.scaler.mean_
        
        # Determine number of components
        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1])
        
        # Perform decomposition
        if self.method == 'eigen':
            eigenvalues, eigenvectors = self._eigen_decomposition(X_scaled)
        elif self.method == 'svd':
            eigenvalues, eigenvectors = self._svd_decomposition(X_scaled)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Select components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Compute explained variance ratios
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transforming data")
        
        # Standardize the data
        X_scaled = self.scaler.transform(X)
        
        # Transform to principal component space
        return np.dot(X_scaled, self.components_)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space
        
        Args:
            X_transformed: Transformed data
            
        Returns:
            Data in original space
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before inverse transforming data")
        
        # Transform back to original space
        X_scaled = np.dot(X_transformed, self.components_.T)
        
        # Denormalize
        return self.scaler.inverse_transform(X_scaled)
    
    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """
        Compute reconstruction error
        
        Args:
            X: Original data
            
        Returns:
            Mean squared reconstruction error
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed) ** 2)

def generate_sample_data(n_samples: int = 200, n_features: int = 3) -> np.ndarray:
    """Generate sample data with correlation"""
    np.random.seed(42)
    
    # Generate correlated data
    X = np.random.randn(n_samples, n_features)
    
    # Add correlation between features
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 2] = 0.5 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * np.random.randn(n_samples)
    
    return X

def plot_pca_results(X: np.ndarray, X_transformed: np.ndarray, 
                    explained_variance_ratio: np.ndarray, 
                    cumulative_explained_variance_ratio: np.ndarray) -> None:
    """Plot PCA results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data (first 2 features)
    axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.6)
    axes[0, 0].set_title('Original Data (Features 1 vs 2)')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Transformed data
    axes[0, 1].scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6)
    axes[0, 1].set_title('PCA Transformed Data (PC1 vs PC2)')
    axes[0, 1].set_xlabel('First Principal Component')
    axes[0, 1].set_ylabel('Second Principal Component')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Explained variance ratio
    axes[1, 0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    axes[1, 0].set_title('Explained Variance Ratio')
    axes[1, 0].set_xlabel('Principal Component')
    axes[1, 0].set_ylabel('Explained Variance Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative explained variance ratio
    axes[1, 1].plot(range(1, len(cumulative_explained_variance_ratio) + 1), 
                   cumulative_explained_variance_ratio, 'bo-')
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    axes[1, 1].set_title('Cumulative Explained Variance Ratio')
    axes[1, 1].set_xlabel('Number of Components')
    axes[1, 1].set_ylabel('Cumulative Explained Variance Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_3d_data(X: np.ndarray, X_transformed: np.ndarray) -> None:
    """Plot 3D data and its 2D projection"""
    fig = plt.figure(figsize=(15, 6))
    
    # Original 3D data
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.6)
    ax1.set_title('Original 3D Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    
    # 2D projection
    ax2 = fig.add_subplot(122)
    ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6)
    ax2.set_title('2D PCA Projection')
    ax2.set_xlabel('First Principal Component')
    ax2.set_ylabel('Second Principal Component')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate PCA"""
    print("=== Principal Component Analysis Demo ===\n")
    
    # Generate sample data
    print("1. Generating sample data...")
    X = generate_sample_data(n_samples=200, n_features=3)
    print(f"   Data shape: {X.shape}")
    print(f"   Data mean: {np.mean(X, axis=0)}")
    print(f"   Data std: {np.std(X, axis=0)}")
    
    # PCA with eigenvalue decomposition
    print("\n2. PCA with eigenvalue decomposition...")
    pca_eigen = PCA(n_components=2, method='eigen')
    X_transformed_eigen = pca_eigen.fit_transform(X)
    
    print(f"   Explained variance ratio: {pca_eigen.explained_variance_ratio_}")
    print(f"   Cumulative explained variance ratio: {pca_eigen.cumulative_explained_variance_ratio_}")
    print(f"   Components shape: {pca_eigen.components_.shape}")
    
    # PCA with SVD decomposition
    print("\n3. PCA with SVD decomposition...")
    pca_svd = PCA(n_components=2, method='svd')
    X_transformed_svd = pca_svd.fit_transform(X)
    
    print(f"   Explained variance ratio: {pca_svd.explained_variance_ratio_}")
    print(f"   Cumulative explained variance ratio: {pca_svd.cumulative_explained_variance_ratio_}")
    
    # Compare methods
    print("\n4. Comparing methods...")
    print(f"   Eigen method reconstruction error: {pca_eigen.get_reconstruction_error(X):.6f}")
    print(f"   SVD method reconstruction error: {pca_svd.get_reconstruction_error(X):.6f}")
    
    # Plot results
    print("\n5. Plotting results...")
    plot_pca_results(X, X_transformed_eigen, 
                    pca_eigen.explained_variance_ratio_,
                    pca_eigen.cumulative_explained_variance_ratio_)
    
    plot_3d_data(X, X_transformed_eigen)
    
    # Iris dataset example
    print("\n6. Iris dataset example...")
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    pca_iris = PCA(n_components=2)
    X_iris_transformed = pca_iris.fit_transform(X_iris)
    
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        plt.scatter(X_iris_transformed[y_iris == i, 0], 
                   X_iris_transformed[y_iris == i, 1],
                   c=color, label=iris.target_names[i], alpha=0.7)
    
    plt.title('PCA on Iris Dataset')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"   Iris explained variance ratio: {pca_iris.explained_variance_ratio_}")
    print(f"   Iris cumulative explained variance ratio: {pca_iris.cumulative_explained_variance_ratio_}")
    
    print("\n=== Demo completed ===")

if __name__ == "__main__":
    main()
```

## 8. 总结

主成分分析通过以下步骤实现降维：

1. **数据标准化**：将数据标准化为零均值单位方差
2. **协方差计算**：计算特征间的协方差矩阵
3. **特征值分解**：对协方差矩阵进行特征值分解
4. **主成分选择**：选择前k个最大特征值对应的特征向量
5. **数据变换**：将数据投影到主成分空间

PCA具有以下特点：
- **线性变换**：保持数据的线性关系
- **方差最大化**：保留数据的主要变化信息
- **正交性**：主成分之间相互正交
- **可逆性**：可以重构原始数据（有损失）

PCA广泛应用于数据可视化、特征提取、噪声去除和降维等任务。
