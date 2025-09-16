"""
LeetCode 121. Best Time to Buy and Sell Stock

题目描述：
给定一个数组prices，它的第i个元素prices[i]表示一支给定股票第i天的价格。
你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。设计一个算法来计算你所能获取的最大利润。
返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回0。

示例：
prices = [7,1,5,3,6,4]
输出：5
解释：在第2天（股票价格=1）的时候买入，在第5天（股票价格=6）的时候卖出，最大利润=6-1=5。

数据范围：
- 1 <= prices.length <= 10^5
- 0 <= prices[i] <= 10^4
"""

class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        min_price = float('inf')
        max_profit = 0
        
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        
        return max_profit
