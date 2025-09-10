# 背包问题算法集合

这个模块包含了各种基础背包问题的Python实现，适用于算法学习和竞赛编程。

## 包含的背包问题类型

### 1. 01背包问题 (Zero-One Knapsack)
- **特点**: 每个物品只能选择一次
- **状态转移**: `dp[w] = max(dp[w], dp[w-weight] + value)`
- **遍历顺序**: 从大到小（避免重复选择）

```python
from alg.knapsack import solve_01_knapsack

capacity = 10
items = [(2, 3), (3, 4), (4, 5), (5, 6)]  # (weight, value)
result = solve_01_knapsack(capacity, items)
print(f"最大价值: {result}")
```

### 2. 完全背包问题 (Complete Knapsack)
- **特点**: 每个物品可以选择无限次
- **状态转移**: `dp[w] = max(dp[w], dp[w-weight] + value)`
- **遍历顺序**: 从小到大（允许重复选择）

```python
from alg.knapsack import solve_complete_knapsack

capacity = 10
items = [(2, 3), (3, 4), (4, 5), (5, 6)]  # (weight, value)
result = solve_complete_knapsack(capacity, items)
print(f"最大价值: {result}")
```

### 3. 多重背包问题 (Multiple Knapsack)
- **特点**: 每个物品有数量限制
- **优化方法**: 二进制优化、单调队列优化

```python
from alg.knapsack import solve_multiple_knapsack

capacity = 10
items = [(2, 3, 2), (3, 4, 1), (4, 5, 3)]  # (weight, value, count)
result = solve_multiple_knapsack(capacity, items)
print(f"最大价值: {result}")
```

### 4. 分组背包问题 (Group Knapsack)
- **特点**: 物品分为若干组，每组最多选择一个
- **应用**: 依赖关系物品选择

```python
from alg.knapsack import solve_group_knapsack

capacity = 10
groups = [
    [(2, 3), (3, 4)],  # 组1
    [(4, 5), (5, 6)],  # 组2
]
result = solve_group_knapsack(capacity, groups)
print(f"最大价值: {result}")
```

### 5. 二维背包问题 (2D Knapsack)
- **特点**: 有两个维度的限制条件
- **应用**: 重量+体积限制

```python
from alg.knapsack import solve_2d_knapsack

weight_capacity = 10
volume_capacity = 8
items = [(2, 1, 3), (3, 2, 4), (4, 3, 5)]  # (weight, volume, value)
result = solve_2d_knapsack(weight_capacity, volume_capacity, items)
print(f"最大价值: {result}")
```

## 使用方式

### 方式1: 使用便捷函数
```python
from alg.knapsack import solve_01_knapsack, solve_complete_knapsack

# 直接调用便捷函数
result = solve_01_knapsack(capacity, items)
```

### 方式2: 使用类方法
```python
from alg.knapsack import ZeroOneKnapsack

# 创建求解器实例
solver = ZeroOneKnapsack()
result = solver.solve(capacity, items)

# 获取选择路径（仅01背包支持）
max_value, selected_items = solver.solve_with_path(capacity, items)
```

## 算法复杂度

| 背包类型 | 时间复杂度 | 空间复杂度 | 说明 |
|---------|-----------|-----------|------|
| 01背包 | O(n×W) | O(W) | n为物品数，W为容量 |
| 完全背包 | O(n×W) | O(W) | 同上 |
| 多重背包 | O(n×W×log(C)) | O(W) | C为最大数量 |
| 分组背包 | O(n×W) | O(W) | n为总物品数 |
| 二维背包 | O(n×W×V) | O(W×V) | V为第二维容量 |

## 关键理解点

### 1. 为什么01背包要从大到小遍历？
```python
# 错误：从小到大遍历
for w in range(weight, capacity + 1):
    dp[w] = max(dp[w], dp[w - weight] + value)
# 问题：dp[w-weight]可能已经被当前物品更新，导致重复选择

# 正确：从大到小遍历
for w in range(capacity, weight - 1, -1):
    dp[w] = max(dp[w], dp[w - weight] + value)
# 解决：dp[w-weight]还是上一轮的值，避免重复选择
```

### 2. 完全背包为什么从小到大遍历？
```python
# 正确：从小到大遍历
for w in range(weight, capacity + 1):
    dp[w] = max(dp[w], dp[w - weight] + value)
# 原因：允许重复选择，dp[w-weight]可能已经包含当前物品
```

### 3. 分组背包的处理方式
```python
# 对每个组，从大到小遍历容量
for group in groups:
    for w in range(capacity, -1, -1):  # 从大到小
        for weight, value in group:
            if w >= weight:
                dp[w] = max(dp[w], dp[w - weight] + value)
# 确保每组最多选择一个物品
```

## 测试和演示

### 运行演示
```bash
cd src/alg
python knapsack_demo.py
```

### 运行测试
```bash
cd src/alg
python -m pytest test_knapsack.py -v
```

## 实际应用场景

1. **资源分配**: 在有限资源下选择最优项目组合
2. **投资组合**: 在预算限制下选择最优投资方案
3. **任务调度**: 在时间限制下选择最优任务组合
4. **物品采购**: 在预算限制下选择最优商品组合
5. **游戏设计**: 角色装备选择、技能点分配等

## 扩展阅读

- [背包问题九讲](https://www.cnblogs.com/jbelial/articles/2116074.html)
- [动态规划专题](https://oi-wiki.org/dp/)
- [算法竞赛进阶指南](https://book.douban.com/subject/30136932/)

## 注意事项

1. 所有实现都假设输入数据合法
2. 对于负价值物品，算法仍能运行但结果可能不符合预期
3. 大规模数据时注意内存使用
4. 可以根据具体问题调整状态转移方程
