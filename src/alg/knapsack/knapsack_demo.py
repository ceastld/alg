"""
背包问题演示和测试用例

展示各种背包问题的使用方法和测试结果
"""

from .knapsack import (
    ZeroOneKnapsack, CompleteKnapsack, MultipleKnapsack, 
    GroupKnapsack, TwoDimensionalKnapsack,
    solve_01_knapsack, solve_complete_knapsack, solve_multiple_knapsack,
    solve_group_knapsack, solve_2d_knapsack
)


def demo_01_knapsack():
    """演示01背包问题"""
    print("=== 01背包问题演示 ===")
    
    # 测试用例1：基础01背包
    capacity = 10
    items = [(2, 3), (3, 4), (4, 5), (5, 6)]  # (weight, value)
    
    print(f"背包容量: {capacity}")
    print("物品列表:")
    for i, (w, v) in enumerate(items):
        print(f"  物品{i+1}: 重量={w}, 价值={v}")
    
    # 使用类方法
    solver = ZeroOneKnapsack()
    result1 = solver.solve(capacity, items)
    print(f"最大价值: {result1}")
    
    # 使用便捷函数
    result2 = solve_01_knapsack(capacity, items)
    print(f"便捷函数结果: {result2}")
    
    # 获取选择路径
    max_value, selected_items = solver.solve_with_path(capacity, items)
    print(f"选择的物品索引: {selected_items}")
    print(f"选择的物品: {[items[i] for i in selected_items]}")
    print()


def demo_complete_knapsack():
    """演示完全背包问题"""
    print("=== 完全背包问题演示 ===")
    
    capacity = 10
    items = [(2, 3), (3, 4), (4, 5), (5, 6)]  # (weight, value)
    
    print(f"背包容量: {capacity}")
    print("物品列表:")
    for i, (w, v) in enumerate(items):
        print(f"  物品{i+1}: 重量={w}, 价值={v}")
    
    result1 = CompleteKnapsack().solve(capacity, items)
    result2 = solve_complete_knapsack(capacity, items)
    
    print(f"最大价值: {result1}")
    print(f"便捷函数结果: {result2}")
    print()


def demo_multiple_knapsack():
    """演示多重背包问题"""
    print("=== 多重背包问题演示 ===")
    
    capacity = 10
    items = [(2, 3, 2), (3, 4, 1), (4, 5, 3), (5, 6, 2)]  # (weight, value, count)
    
    print(f"背包容量: {capacity}")
    print("物品列表:")
    for i, (w, v, c) in enumerate(items):
        print(f"  物品{i+1}: 重量={w}, 价值={v}, 数量={c}")
    
    # 二进制优化
    result1 = MultipleKnapsack().solve_binary_optimization(capacity, items)
    result2 = solve_multiple_knapsack(capacity, items)
    
    print(f"二进制优化结果: {result1}")
    print(f"便捷函数结果: {result2}")
    
    # 单调队列优化
    result3 = MultipleKnapsack().solve_monotonic_queue(capacity, items)
    print(f"单调队列优化结果: {result3}")
    print()


def demo_group_knapsack():
    """演示分组背包问题"""
    print("=== 分组背包问题演示 ===")
    
    capacity = 10
    groups = [
        [(2, 3), (3, 4)],  # 组1
        [(4, 5), (5, 6)],  # 组2
        [(1, 2), (2, 3), (3, 4)]  # 组3
    ]
    
    print(f"背包容量: {capacity}")
    print("分组情况:")
    for i, group in enumerate(groups):
        print(f"  组{i+1}: {group}")
    
    result1 = GroupKnapsack().solve(capacity, groups)
    result2 = solve_group_knapsack(capacity, groups)
    
    print(f"最大价值: {result1}")
    print(f"便捷函数结果: {result2}")
    print()


def demo_2d_knapsack():
    """演示二维背包问题"""
    print("=== 二维背包问题演示 ===")
    
    weight_capacity = 10
    volume_capacity = 8
    items = [(2, 1, 3), (3, 2, 4), (4, 3, 5), (5, 2, 6)]  # (weight, volume, value)
    
    print(f"重量限制: {weight_capacity}, 体积限制: {volume_capacity}")
    print("物品列表:")
    for i, (w, v, val) in enumerate(items):
        print(f"  物品{i+1}: 重量={w}, 体积={v}, 价值={val}")
    
    result1 = TwoDimensionalKnapsack().solve(weight_capacity, volume_capacity, items)
    result2 = solve_2d_knapsack(weight_capacity, volume_capacity, items)
    
    print(f"最大价值: {result1}")
    print(f"便捷函数结果: {result2}")
    print()


def compare_knapsack_types():
    """比较不同背包问题的结果"""
    print("=== 背包问题类型比较 ===")
    
    capacity = 10
    items = [(2, 3), (3, 4), (4, 5), (5, 6)]  # (weight, value)
    
    print(f"背包容量: {capacity}")
    print("物品列表:")
    for i, (w, v) in enumerate(items):
        print(f"  物品{i+1}: 重量={w}, 价值={v}")
    print()
    
    # 01背包
    result_01 = solve_01_knapsack(capacity, items)
    print(f"01背包结果: {result_01}")
    
    # 完全背包
    result_complete = solve_complete_knapsack(capacity, items)
    print(f"完全背包结果: {result_complete}")
    
    # 多重背包（每个物品最多2个）
    multiple_items = [(w, v, 2) for w, v in items]
    result_multiple = solve_multiple_knapsack(capacity, multiple_items)
    print(f"多重背包结果（每物品最多2个）: {result_multiple}")
    
    # 分组背包（每2个物品一组）
    groups = [[items[i], items[i+1]] for i in range(0, len(items), 2)]
    result_group = solve_group_knapsack(capacity, groups)
    print(f"分组背包结果（每2个物品一组）: {result_group}")
    print()


def test_edge_cases():
    """测试边界情况"""
    print("=== 边界情况测试 ===")
    
    # 空物品列表
    print("空物品列表:")
    result = solve_01_knapsack(10, [])
    print(f"结果: {result}")
    
    # 容量为0
    print("\n容量为0:")
    items = [(2, 3), (3, 4)]
    result = solve_01_knapsack(0, items)
    print(f"结果: {result}")
    
    # 所有物品重量都超过容量
    print("\n所有物品重量都超过容量:")
    items = [(15, 10), (20, 15)]
    result = solve_01_knapsack(10, items)
    print(f"结果: {result}")
    
    # 单个物品
    print("\n单个物品:")
    items = [(5, 8)]
    result = solve_01_knapsack(10, items)
    print(f"结果: {result}")
    print()


def main():
    """主函数：运行所有演示"""
    print("背包问题算法演示")
    print("=" * 50)
    
    demo_01_knapsack()
    demo_complete_knapsack()
    demo_multiple_knapsack()
    demo_group_knapsack()
    demo_2d_knapsack()
    compare_knapsack_types()
    test_edge_cases()
    
    print("演示完成！")


if __name__ == "__main__":
    main()
