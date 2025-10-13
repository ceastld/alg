"""
188. 买卖股票的最佳时机IV
给你一个整数数组 prices 和一个整数 k ，其中 prices[i] 是某支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

题目链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/

示例 1:
输入：k = 2, prices = [2,4,1]
输出：2
解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。

示例 2:
输入：k = 2, prices = [3,2,6,5,0,3]
输出：7
解释：在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
     随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。

提示：
- 0 <= k <= 100
- 0 <= prices.length <= 1000
- 0 <= prices[i] <= 1000
"""

from typing import List


class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    k = 2
    prices = [2,4,1]
    result = solution.maxProfit(k, prices)
    expected = 2
    assert result == expected
    
    # 测试用例2
    k = 2
    prices = [3,2,6,5,0,3]
    result = solution.maxProfit(k, prices)
    expected = 7
    assert result == expected
    
    # 测试用例3
    k = 2
    prices = [1,2,4,2,5,7,2,4,9,0]
    result = solution.maxProfit(k, prices)
    expected = 13
    assert result == expected
    
    # 测试用例4
    k = 0
    prices = [1,3,2,8,4,9]
    result = solution.maxProfit(k, prices)
    expected = 0
    assert result == expected
    
    # 测试用例5
    k = 1
    prices = [1,2,3,4,5]
    result = solution.maxProfit(k, prices)
    expected = 4
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
