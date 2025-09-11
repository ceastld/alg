# 决策树完整数学推导

## 1. 决策树基本概念

### 1.1 核心思想
决策树是一种基于树状结构的分类和回归方法。通过一系列if-then规则将数据逐步分割，最终到达叶节点进行预测。

### 1.2 树结构
决策树由以下部分组成：
- **根节点**：包含所有训练样本
- **内部节点**：包含分割条件
- **叶节点**：包含最终预测结果
- **分支**：连接节点的边

### 1.3 分割策略
决策树的核心是选择最优的分割特征和分割点，使得分割后的子集更加"纯净"。

## 2. 信息论基础

### 2.1 信息熵（Entropy）
信息熵衡量数据的不确定性：

$$H(S) = -\sum_{i=1}^c p_i \log_2 p_i$$

其中：
- $S$ 是数据集
- $c$ 是类别数
- $p_i$ 是类别 $i$ 在数据集中的比例

### 2.2 信息增益（Information Gain）
信息增益衡量通过某个特征分割后信息熵的减少：

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

其中：
- $A$ 是特征
- $Values(A)$ 是特征 $A$ 的所有可能值
- $S_v$ 是特征 $A$ 取值为 $v$ 的样本子集

### 2.3 信息增益比（Information Gain Ratio）
信息增益比解决信息增益偏向于多值特征的问题：

$$IGR(S, A) = \frac{IG(S, A)}{IV(A)}$$

其中分裂信息（Split Information）为：

$$IV(A) = -\sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

### 2.4 基尼不纯度（Gini Impurity）
基尼不纯度是另一种衡量数据不纯度的指标：

$$Gini(S) = 1 - \sum_{i=1}^c p_i^2$$

基尼增益：

$$GiniGain(S, A) = Gini(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Gini(S_v)$$

## 3. 连续特征处理

### 3.1 分割点选择
对于连续特征，需要选择最优分割点。对于特征 $A$ 和分割点 $t$：

$$IG(S, A, t) = H(S) - \frac{|S_{A \leq t}|}{|S|} H(S_{A \leq t}) - \frac{|S_{A > t}|}{|S|} H(S_{A > t})$$

最优分割点：

$$t^* = \arg\max_t IG(S, A, t)$$

### 3.2 分割点搜索
常用的分割点搜索策略：
1. 排序所有特征值
2. 取相邻值的中间点作为候选分割点
3. 计算每个候选点的信息增益
4. 选择信息增益最大的分割点

## 4. 回归树

### 4.1 回归目标
对于回归任务，使用均方误差（MSE）作为分割标准：

$$MSE(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

其中 $\bar{y}$ 是样本均值。

### 4.2 回归分割
对于特征 $A$ 和分割点 $t$：

$$MSE(S, A, t) = \frac{|S_{A \leq t}|}{|S|} MSE(S_{A \leq t}) + \frac{|S_{A > t}|}{|S|} MSE(S_{A > t})$$

最优分割点：

$$t^* = \arg\min_t MSE(S, A, t)$$

## 5. 剪枝策略

### 5.1 预剪枝（Pre-pruning）
在构建树的过程中进行剪枝：
- 设置最大深度
- 设置最小样本数
- 设置最小信息增益阈值

### 5.2 后剪枝（Post-pruning）
在树构建完成后进行剪枝：
- 计算剪枝前后的验证误差
- 如果剪枝后误差不增加，则进行剪枝

## 6. Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
from sklearn.datasets import make_classification, make_regression
from collections import Counter

class TreeNode:
    """Decision tree node"""
    
    def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None,
                 value: Optional[Union[int, float]] = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes
        self.is_leaf = value is not None

class DecisionTree:
    """
    Decision Tree implementation for classification and regression
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, criterion: str = 'gini',
                 random_state: Optional[int] = None):
        """
        Initialize decision tree
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            criterion: Splitting criterion ('gini', 'entropy', 'mse')
            random_state: Random seed
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.root = None
        self.is_classification = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        entropy = 0
        for count in counts.values():
            p = count / len(y)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        gini = 1
        for count in counts.values():
            p = count / len(y)
            gini -= p ** 2
        return gini
    
    def _calculate_mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error"""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion"""
        if self.criterion == 'entropy':
            return self._calculate_entropy(y)
        elif self.criterion == 'gini':
            return self._calculate_gini(y)
        elif self.criterion == 'mse':
            return self._calculate_mse(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best split for the given data"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        current_impurity = self._calculate_impurity(y)
        
        for feature_idx in range(X.shape[1]):
            # Get unique values for this feature
            feature_values = np.unique(X[:, feature_idx])
            
            # Try each possible threshold
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Split the data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                weighted_impurity = (np.sum(left_mask) / len(y)) * left_impurity + \
                                  (np.sum(right_mask) / len(y)) * right_impurity
                
                # Calculate gain
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _create_leaf_node(self, y: np.ndarray) -> TreeNode:
        """Create a leaf node"""
        if self.is_classification:
            # For classification, use the most common class
            value = Counter(y).most_common(1)[0][0]
        else:
            # For regression, use the mean
            value = np.mean(y)
        return TreeNode(value=value)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """Build the decision tree recursively"""
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self._create_leaf_node(y)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_feature is None or best_gain <= 0:
            return self._create_leaf_node(y)
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check if split is valid
        if (np.sum(left_mask) < self.min_samples_leaf or 
            np.sum(right_mask) < self.min_samples_leaf):
            return self._create_leaf_node(y)
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(feature_idx=best_feature, threshold=best_threshold,
                       left=left_subtree, right=right_subtree)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the decision tree
        
        Args:
            X: Training features (m, n)
            y: Training labels (m,)
        """
        # Determine if this is classification or regression
        if np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) < 20:
            self.is_classification = True
        else:
            self.is_classification = False
        
        # Build the tree
        self.root = self._build_tree(X, y)
    
    def _predict_single(self, x: np.ndarray, node: TreeNode) -> Union[int, float]:
        """Predict a single sample"""
        if node.is_leaf:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            predictions[i] = self._predict_single(X[i], self.root)
        return predictions
    
    def _get_tree_depth(self, node: TreeNode) -> int:
        """Get the depth of the tree"""
        if node.is_leaf:
            return 0
        return 1 + max(self._get_tree_depth(node.left), self._get_tree_depth(node.right))
    
    def get_depth(self) -> int:
        """Get the depth of the trained tree"""
        if self.root is None:
            return 0
        return self._get_tree_depth(self.root)
    
    def _count_nodes(self, node: TreeNode) -> int:
        """Count the number of nodes in the tree"""
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def get_node_count(self) -> int:
        """Get the number of nodes in the trained tree"""
        if self.root is None:
            return 0
        return self._count_nodes(self.root)

def generate_classification_data(n_samples: int = 200, n_features: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample classification data"""
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                              n_redundant=0, n_informative=2, 
                              n_clusters_per_class=1, random_state=42)
    return X, y

def generate_regression_data(n_samples: int = 200, n_features: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample regression data"""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          noise=0.1, random_state=42)
    return X, y

def plot_decision_boundary_classification(model: DecisionTree, X: np.ndarray, y: np.ndarray) -> None:
    """Plot decision boundary for 2D classification data"""
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
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.colorbar(scatter, label='True Label')
    
    plt.title('Decision Tree Classification Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_regression_result(model: DecisionTree, X: np.ndarray, y: np.ndarray) -> None:
    """Plot regression results for 1D data"""
    if X.shape[1] != 1:
        print("Can only plot regression results for 1D data")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Sort data for plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Plot original data
    plt.scatter(X_sorted, y_sorted, alpha=0.6, label='Data')
    
    # Plot predictions
    y_pred = model.predict(X_sorted)
    plt.plot(X_sorted, y_pred, 'r-', linewidth=2, label='Decision Tree')
    
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Decision Tree Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main function to demonstrate decision tree"""
    print("=== Decision Tree Demo ===\n")
    
    # Classification example
    print("1. Classification Example")
    print("-" * 30)
    X_cls, y_cls = generate_classification_data(n_samples=200, n_features=2)
    print(f"   Data shape: X={X_cls.shape}, y={y_cls.shape}")
    
    # Train classification tree
    print("\n2. Training classification tree...")
    tree_cls = DecisionTree(max_depth=5, criterion='gini', random_state=42)
    tree_cls.fit(X_cls, y_cls)
    
    # Make predictions
    y_pred_cls = tree_cls.predict(X_cls)
    accuracy = np.mean(y_pred_cls == y_cls)
    print(f"   Training accuracy: {accuracy:.4f}")
    print(f"   Tree depth: {tree_cls.get_depth()}")
    print(f"   Number of nodes: {tree_cls.get_node_count()}")
    
    # Plot results
    print("\n3. Plotting classification results...")
    plot_decision_boundary_classification(tree_cls, X_cls, y_cls)
    
    # Regression example
    print("\n4. Regression Example")
    print("-" * 30)
    X_reg, y_reg = generate_regression_data(n_samples=200, n_features=1)
    print(f"   Data shape: X={X_reg.shape}, y={y_reg.shape}")
    
    # Train regression tree
    print("\n5. Training regression tree...")
    tree_reg = DecisionTree(max_depth=5, criterion='mse', random_state=42)
    tree_reg.fit(X_reg, y_reg)
    
    # Make predictions
    y_pred_reg = tree_reg.predict(X_reg)
    mse = np.mean((y_reg - y_pred_reg) ** 2)
    r2 = 1 - np.sum((y_reg - y_pred_reg) ** 2) / np.sum((y_reg - np.mean(y_reg)) ** 2)
    print(f"   Mean Squared Error: {mse:.4f}")
    print(f"   R-squared: {r2:.4f}")
    print(f"   Tree depth: {tree_reg.get_depth()}")
    print(f"   Number of nodes: {tree_reg.get_node_count()}")
    
    # Plot results
    print("\n6. Plotting regression results...")
    plot_regression_result(tree_reg, X_reg, y_reg)
    
    print("\n=== Demo completed ===")

if __name__ == "__main__":
    main()
```

## 7. 总结

决策树通过以下步骤实现分类和回归：

1. **特征选择**：使用信息增益、基尼不纯度等指标选择最优分割特征
2. **分割策略**：对连续特征选择最优分割点，对离散特征选择最优分割值
3. **递归构建**：递归地构建左右子树
4. **剪枝优化**：通过预剪枝或后剪枝防止过拟合
5. **预测**：从根节点开始，根据特征值选择分支，直到到达叶节点

决策树具有可解释性强、能处理非线性关系、对异常值不敏感的特点，是机器学习中的重要算法。
