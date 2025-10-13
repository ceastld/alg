"""
309. 最佳买卖股票时机含冷冻期
给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

题目链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/

示例 1:
输入: prices = [1,2,3,0,2]
输出: 3
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]

示例 2:
输入: prices = [1]
输出: 0

提示：
- 1 <= prices.length <= 5000
- 0 <= prices[i] <= 1000
"""

from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    prices = [1,2,3,0,2]
    result = solution.maxProfit(prices)
    expected = 3
    assert result == expected
    
    # 测试用例2
    prices = [1]
    result = solution.maxProfit(prices)
    expected = 0
    assert result == expected
    
    # 测试用例3
    prices = [1,2,3,4,5]
    result = solution.maxProfit(prices)
    expected = 4
    assert result == expected
    
    # 测试用例4
    prices = [7,6,4,3,1]
    result = solution.maxProfit(prices)
    expected = 0
    assert result == expected
    
    # 测试用例5
    prices = [1,2,4,0,2]
    result = solution.maxProfit(prices)
    expected = 3
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
