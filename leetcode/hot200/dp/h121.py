"""
121. 买卖股票的最佳时机
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

题目链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/

示例 1:
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。

示例 2:
输入：prices = [7,6,4,3,1]
输出：0
解释：在这种情况下, 没有交易完成, 所以最大利润为 0。

提示：
- 1 <= prices.length <= 10^5
- 0 <= prices[i] <= 10^4
"""

from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        left = 0
        max_profit = 0
        for i in range(1, len(prices)):
            if prices[i] < prices[left]:
                left = i
            else:
                max_profit = max(max_profit, prices[i] - prices[left])
        return max_profit


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    prices = [7,1,5,3,6,4]
    result = solution.maxProfit(prices)
    expected = 5
    assert result == expected
    
    # 测试用例2
    prices = [7,6,4,3,1]
    result = solution.maxProfit(prices)
    expected = 0
    assert result == expected
    
    # 测试用例3
    prices = [1,2,3,4,5]
    result = solution.maxProfit(prices)
    expected = 4
    assert result == expected
    
    # 测试用例4
    prices = [2,4,1]
    result = solution.maxProfit(prices)
    expected = 2
    assert result == expected
    
    # 测试用例5
    prices = [1]
    result = solution.maxProfit(prices)
    expected = 0
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
