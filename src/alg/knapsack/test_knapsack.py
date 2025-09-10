"""
背包问题单元测试

使用pytest框架进行测试
"""

import pytest
from .knapsack import (
    ZeroOneKnapsack, CompleteKnapsack, MultipleKnapsack,
    GroupKnapsack, TwoDimensionalKnapsack,
    solve_01_knapsack, solve_complete_knapsack, solve_multiple_knapsack,
    solve_group_knapsack, solve_2d_knapsack
)


class TestZeroOneKnapsack:
    """01背包问题测试"""
    
    def test_basic_case(self):
        """基础测试用例"""
        capacity = 10
        items = [(2, 3), (3, 4), (4, 5), (5, 6)]
        expected = 10  # 选择物品2和4: 3+4 + 4+5 = 10
        
        solver = ZeroOneKnapsack()
        result = solver.solve(capacity, items)
        assert result == expected
        
        # 测试便捷函数
        result2 = solve_01_knapsack(capacity, items)
        assert result2 == expected
    
    def test_empty_items(self):
        """空物品列表测试"""
        capacity = 10
        items = []
        expected = 0
        
        result = solve_01_knapsack(capacity, items)
        assert result == expected
    
    def test_zero_capacity(self):
        """容量为0测试"""
        capacity = 0
        items = [(2, 3), (3, 4)]
        expected = 0
        
        result = solve_01_knapsack(capacity, items)
        assert result == expected
    
    def test_single_item(self):
        """单个物品测试"""
        capacity = 10
        items = [(5, 8)]
        expected = 8
        
        result = solve_01_knapsack(capacity, items)
        assert result == expected
    
    def test_all_items_too_heavy(self):
        """所有物品都超重测试"""
        capacity = 5
        items = [(10, 20), (15, 30)]
        expected = 0
        
        result = solve_01_knapsack(capacity, items)
        assert result == expected
    
    def test_path_tracking(self):
        """路径跟踪测试"""
        capacity = 10
        items = [(2, 3), (3, 4), (4, 5), (5, 6)]
        
        solver = ZeroOneKnapsack()
        max_value, selected_items = solver.solve_with_path(capacity, items)
        
        assert max_value == 10
        assert len(selected_items) == 2
        # 验证选择的物品总重量不超过容量
        total_weight = sum(items[i][0] for i in selected_items)
        assert total_weight <= capacity


class TestCompleteKnapsack:
    """完全背包问题测试"""
    
    def test_basic_case(self):
        """基础测试用例"""
        capacity = 10
        items = [(2, 3), (3, 4), (4, 5), (5, 6)]
        expected = 15  # 选择5个物品1: 5 * 3 = 15
        
        result = solve_complete_knapsack(capacity, items)
        assert result == expected
    
    def test_single_item(self):
        """单个物品测试"""
        capacity = 10
        items = [(3, 4)]
        expected = 12  # 选择3个物品: 3 * 4 = 12
        
        result = solve_complete_knapsack(capacity, items)
        assert result == expected


class TestMultipleKnapsack:
    """多重背包问题测试"""
    
    def test_binary_optimization(self):
        """二进制优化测试"""
        capacity = 10
        items = [(2, 3, 2), (3, 4, 1), (4, 5, 3)]  # (weight, value, count)
        expected = 10  # 选择物品1的2个 + 物品2的1个 + 物品3的1个
        
        result = solve_multiple_knapsack(capacity, items)
        assert result == expected
    
    def test_monotonic_queue(self):
        """单调队列优化测试"""
        capacity = 10
        items = [(2, 3, 2), (3, 4, 1), (4, 5, 3)]
        
        solver = MultipleKnapsack()
        result1 = solver.solve_binary_optimization(capacity, items)
        result2 = solver.solve_monotonic_queue(capacity, items)
        
        # 两种方法结果应该相同
        assert result1 == result2


class TestGroupKnapsack:
    """分组背包问题测试"""
    
    def test_basic_case(self):
        """基础测试用例"""
        capacity = 10
        groups = [
            [(2, 3), (3, 4)],  # 组1
            [(4, 5), (5, 6)],  # 组2
        ]
        expected = 10  # 选择组1的物品2 + 组2的物品1
        
        result = solve_group_knapsack(capacity, groups)
        assert result == expected
    
    def test_empty_groups(self):
        """空分组测试"""
        capacity = 10
        groups = []
        expected = 0
        
        result = solve_group_knapsack(capacity, groups)
        assert result == expected


class TestTwoDimensionalKnapsack:
    """二维背包问题测试"""
    
    def test_basic_case(self):
        """基础测试用例"""
        weight_capacity = 10
        volume_capacity = 8
        items = [(2, 1, 3), (3, 2, 4), (4, 3, 5), (5, 2, 6)]
        expected = 9  # 选择物品1和物品2
        
        result = solve_2d_knapsack(weight_capacity, volume_capacity, items)
        assert result == expected
    
    def test_zero_capacity(self):
        """容量为0测试"""
        weight_capacity = 0
        volume_capacity = 0
        items = [(2, 1, 3), (3, 2, 4)]
        expected = 0
        
        result = solve_2d_knapsack(weight_capacity, volume_capacity, items)
        assert result == expected


class TestEdgeCases:
    """边界情况测试"""
    
    def test_negative_values(self):
        """负价值测试"""
        capacity = 10
        items = [(2, -3), (3, 4), (4, -5)]
        
        # 应该能处理负价值
        result = solve_01_knapsack(capacity, items)
        assert isinstance(result, int)
    
    def test_large_numbers(self):
        """大数字测试"""
        capacity = 1000
        items = [(100, 200), (200, 300), (300, 400)]
        
        result = solve_01_knapsack(capacity, items)
        assert result > 0
    
    def test_duplicate_items(self):
        """重复物品测试"""
        capacity = 10
        items = [(2, 3), (2, 3), (3, 4), (3, 4)]
        
        result = solve_01_knapsack(capacity, items)
        assert result > 0


def test_performance():
    """性能测试"""
    import time
    
    # 大规模测试
    capacity = 1000
    items = [(i, i*2) for i in range(1, 101)]  # 100个物品
    
    start_time = time.time()
    result = solve_01_knapsack(capacity, items)
    end_time = time.time()
    
    assert result > 0
    assert end_time - start_time < 1.0  # 应该在1秒内完成


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
