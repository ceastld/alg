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
