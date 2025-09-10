"""
背包问题算法包

包含各种基础背包问题的Python实现：
- 01背包问题
- 完全背包问题  
- 多重背包问题
- 分组背包问题
- 二维背包问题
"""

from .knapsack import (
    # 求解器类
    KnapsackSolver,
    ZeroOneKnapsack,
    CompleteKnapsack,
    MultipleKnapsack,
    GroupKnapsack,
    TwoDimensionalKnapsack,
    
    # 便捷函数
    solve_01_knapsack,
    solve_complete_knapsack,
    solve_multiple_knapsack,
    solve_group_knapsack,
    solve_2d_knapsack,
)

__all__ = [
    # 求解器类
    'KnapsackSolver',
    'ZeroOneKnapsack',
    'CompleteKnapsack',
    'MultipleKnapsack',
    'GroupKnapsack',
    'TwoDimensionalKnapsack',
    
    # 便捷函数
    'solve_01_knapsack',
    'solve_complete_knapsack',
    'solve_multiple_knapsack',
    'solve_group_knapsack',
    'solve_2d_knapsack',
]

__version__ = "1.0.0"
__author__ = "Algorithm Library"
__description__ = "Comprehensive knapsack problem algorithms implementation"
