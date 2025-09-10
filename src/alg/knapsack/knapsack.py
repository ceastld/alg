"""
背包问题集合 - 各种基础背包问题的Python实现

包含以下背包问题类型：
1. 01背包问题 - 每个物品只能选择一次
2. 完全背包问题 - 每个物品可以选择无限次
3. 多重背包问题 - 每个物品有数量限制
4. 分组背包问题 - 物品分为若干组，每组最多选一个
5. 二维背包问题 - 有两个维度的限制条件
"""

from typing import List, Tuple, Union
import math


class KnapsackSolver:
    """背包问题求解器基类"""
    
    def __init__(self):
        pass
    
    def solve(self, capacity: int, items: List[Tuple[int, int]]) -> int:
        """
        求解背包问题的抽象方法
        
        Args:
            capacity: 背包容量
            items: 物品列表，每个物品为(weight, value)元组
            
        Returns:
            最大价值
        """
        raise NotImplementedError


class ZeroOneKnapsack(KnapsackSolver):
    """
    01背包问题：每个物品只能选择一次
    
    状态转移方程：dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight] + value)
    空间优化：dp[w] = max(dp[w], dp[w-weight] + value) (从大到小遍历)
    """
    
    def solve(self, capacity: int, items: List[Tuple[int, int]]) -> int:
        """
        求解01背包问题
        
        Args:
            capacity: 背包容量
            items: 物品列表，每个物品为(weight, value)元组
            
        Returns:
            最大价值
        """
        # 一维DP数组，dp[w]表示容量为w时的最大价值
        dp = [0] * (capacity + 1)
        
        for weight, value in items:
            # 从大到小遍历，避免重复选择
            for w in range(capacity, weight - 1, -1):
                dp[w] = max(dp[w], dp[w - weight] + value)
        
        return dp[capacity]
    
    def solve_with_path(self, capacity: int, items: List[Tuple[int, int]]) -> Tuple[int, List[int]]:
        """
        求解01背包问题并返回选择的物品路径
        
        Returns:
            (最大价值, 选择的物品索引列表)
        """
        n = len(items)
        # 二维DP数组，dp[i][w]表示前i个物品在容量w下的最大价值
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        # 填充DP表
        for i in range(1, n + 1):
            weight, value = items[i - 1]
            for w in range(capacity + 1):
                if w < weight:
                    dp[i][w] = dp[i - 1][w]
                else:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + value)
        
        # 回溯找路径
        selected_items = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:  # 选择了第i个物品
                selected_items.append(i - 1)  # 物品索引从0开始
                w -= items[i - 1][0]
        
        return dp[n][capacity], selected_items


class CompleteKnapsack(KnapsackSolver):
    """
    完全背包问题：每个物品可以选择无限次
    
    状态转移方程：dp[i][w] = max(dp[i-1][w], dp[i][w-weight] + value)
    空间优化：dp[w] = max(dp[w], dp[w-weight] + value) (从小到大遍历)
    """
    
    def solve(self, capacity: int, items: List[Tuple[int, int]]) -> int:
        """
        求解完全背包问题
        
        Args:
            capacity: 背包容量
            items: 物品列表，每个物品为(weight, value)元组
            
        Returns:
            最大价值
        """
        dp = [0] * (capacity + 1)
        
        for weight, value in items:
            # 从小到大遍历，允许重复选择
            for w in range(weight, capacity + 1):
                dp[w] = max(dp[w], dp[w - weight] + value)
        
        return dp[capacity]


class MultipleKnapsack(KnapsackSolver):
    """
    多重背包问题：每个物品有数量限制
    
    方法1：转换为01背包问题
    方法2：二进制优化
    方法3：单调队列优化
    """
    
    def solve_binary_optimization(self, capacity: int, items: List[Tuple[int, int, int]]) -> int:
        """
        使用二进制优化求解多重背包问题
        
        Args:
            capacity: 背包容量
            items: 物品列表，每个物品为(weight, value, count)元组
            
        Returns:
            最大价值
        """
        # 二进制优化：将每个物品按二进制拆分
        optimized_items = []
        
        for weight, value, count in items:
            # 二进制拆分
            k = 1
            while k <= count:
                optimized_items.append((weight * k, value * k))
                count -= k
                k *= 2
            
            # 剩余部分
            if count > 0:
                optimized_items.append((weight * count, value * count))
        
        # 转换为01背包问题求解
        zero_one_solver = ZeroOneKnapsack()
        return zero_one_solver.solve(capacity, optimized_items)
    
    def solve_monotonic_queue(self, capacity: int, items: List[Tuple[int, int, int]]) -> int:
        """
        使用单调队列优化求解多重背包问题
        
        Args:
            capacity: 背包容量
            items: 物品列表，每个物品为(weight, value, count)元组
            
        Returns:
            最大价值
        """
        dp = [0] * (capacity + 1)
        
        for weight, value, count in items:
            # 按余数分组处理
            for r in range(weight):
                # 单调队列
                queue = []
                for k in range((capacity - r) // weight + 1):
                    w = r + k * weight
                    if w > capacity:
                        break
                    
                    # 维护单调队列
                    while queue and k - queue[0] > count:
                        queue.pop(0)
                    
                    # 计算当前状态的值
                    if queue:
                        prev_k = queue[0]
                        prev_w = r + prev_k * weight
                        current_value = dp[prev_w] + (k - prev_k) * value
                    else:
                        current_value = dp[w]
                    
                    # 更新DP值
                    dp[w] = max(dp[w], current_value)
                    
                    # 维护队列单调性
                    while queue:
                        last_k = queue[-1]
                        last_w = r + last_k * weight
                        if dp[last_w] - last_k * value <= dp[w] - k * value:
                            queue.pop()
                        else:
                            break
                    queue.append(k)
        
        return dp[capacity]


class GroupKnapsack(KnapsackSolver):
    """
    分组背包问题：物品分为若干组，每组最多选择一个物品
    
    状态转移方程：dp[i][w] = max(dp[i-1][w], max(dp[i-1][w-weight_j] + value_j))
    """
    
    def solve(self, capacity: int, groups: List[List[Tuple[int, int]]]) -> int:
        """
        求解分组背包问题
        
        Args:
            capacity: 背包容量
            groups: 分组列表，每个组包含多个(weight, value)元组
            
        Returns:
            最大价值
        """
        dp = [0] * (capacity + 1)
        
        for group in groups:
            # 从大到小遍历，确保每组最多选择一个物品
            for w in range(capacity, -1, -1):
                for weight, value in group:
                    if w >= weight:
                        dp[w] = max(dp[w], dp[w - weight] + value)
        
        return dp[capacity]


class TwoDimensionalKnapsack(KnapsackSolver):
    """
    二维背包问题：有两个维度的限制条件（如重量和体积）
    
    状态转移方程：dp[i][w][v] = max(dp[i-1][w][v], dp[i-1][w-weight][v-volume] + value)
    """
    
    def solve(self, weight_capacity: int, volume_capacity: int, 
              items: List[Tuple[int, int, int]]) -> int:
        """
        求解二维背包问题
        
        Args:
            weight_capacity: 重量限制
            volume_capacity: 体积限制
            items: 物品列表，每个物品为(weight, volume, value)元组
            
        Returns:
            最大价值
        """
        # 二维DP数组
        dp = [[0] * (volume_capacity + 1) for _ in range(weight_capacity + 1)]
        
        for weight, volume, value in items:
            # 从大到小遍历两个维度
            for w in range(weight_capacity, weight - 1, -1):
                for v in range(volume_capacity, volume - 1, -1):
                    dp[w][v] = max(dp[w][v], dp[w - weight][v - volume] + value)
        
        return dp[weight_capacity][volume_capacity]


# 便捷函数
def solve_01_knapsack(capacity: int, items: List[Tuple[int, int]]) -> int:
    """便捷函数：求解01背包问题"""
    solver = ZeroOneKnapsack()
    return solver.solve(capacity, items)


def solve_complete_knapsack(capacity: int, items: List[Tuple[int, int]]) -> int:
    """便捷函数：求解完全背包问题"""
    solver = CompleteKnapsack()
    return solver.solve(capacity, items)


def solve_multiple_knapsack(capacity: int, items: List[Tuple[int, int, int]]) -> int:
    """便捷函数：求解多重背包问题（二进制优化）"""
    solver = MultipleKnapsack()
    return solver.solve_binary_optimization(capacity, items)


def solve_group_knapsack(capacity: int, groups: List[List[Tuple[int, int]]]) -> int:
    """便捷函数：求解分组背包问题"""
    solver = GroupKnapsack()
    return solver.solve(capacity, groups)


def solve_2d_knapsack(weight_capacity: int, volume_capacity: int, 
                     items: List[Tuple[int, int, int]]) -> int:
    """便捷函数：求解二维背包问题"""
    solver = TwoDimensionalKnapsack()
    return solver.solve(weight_capacity, volume_capacity, items)
