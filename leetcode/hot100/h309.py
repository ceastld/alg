"""
LeetCode 309. Best Time to Buy and Sell Stock with Cooldown

题目描述：
给定一个整数数组prices，其中第prices[i]表示第i天的股票价格。
设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）：
- 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）
- 卖出股票后，你无法在第二天买入股票（即冷冻期为1天）

示例：
prices = [1,2,3,0,2]
输出：3
解释：对应的交易状态为：[买入, 卖出, 冷冻期, 买入, 卖出]

数据范围：
- 1 <= prices.length <= 5000
- 0 <= prices[i] <= 1000
"""

class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        if not prices:
            return 0
        
        # 状态：hold(持有), sold(刚卖出), rest(休息)
        hold = -prices[0]  # 持有股票的最大利润
        sold = 0           # 刚卖出股票的最大利润
        rest = 0           # 休息状态的最大利润
        
        for i in range(1, len(prices)):
            prev_hold, prev_sold, prev_rest = hold, sold, rest
            
            # 当前状态转移
            hold = max(prev_hold, prev_rest - prices[i])  # 持有：继续持有 或 从休息买入
            sold = prev_hold + prices[i]                  # 卖出：从持有状态卖出
            rest = max(prev_rest, prev_sold)              # 休息：继续休息 或 从卖出进入休息
        
        return max(sold, rest)
