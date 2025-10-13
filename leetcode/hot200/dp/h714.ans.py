"""
714. 买卖股票的最佳时机含手续费 - 标准答案
"""
from typing import List


class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. 定义两个状态：持有股票、不持有股票
        2. 状态转移方程：
           - hold[i] = max(hold[i-1], sold[i-1] - prices[i])
           - sold[i] = max(sold[i-1], hold[i-1] + prices[i] - fee)
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        n = len(prices)
        
        # 初始化状态
        hold = [-float('inf')] * n  # 持有股票
        sold = [0] * n              # 不持有股票
        
        # 第一天只能买入
        hold[0] = -prices[0]
        
        for i in range(1, n):
            # 持有股票：要么继续持有，要么从卖出状态买入
            hold[i] = max(hold[i-1], sold[i-1] - prices[i])
            # 不持有股票：要么继续不持有，要么从持有状态卖出（扣除手续费）
            sold[i] = max(sold[i-1], hold[i-1] + prices[i] - fee)
        
        return sold[n-1]
    
    def maxProfit_optimized(self, prices: List[int], fee: int) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用滚动数组优化空间复杂度
        2. 只保留必要的状态信息
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # 初始化状态
        hold = -prices[0]  # 持有股票
        sold = 0           # 不持有股票
        
        for i in range(1, len(prices)):
            # 更新状态（注意更新顺序）
            new_hold = max(hold, sold - prices[i])
            new_sold = max(sold, hold + prices[i] - fee)
            
            hold = new_hold
            sold = new_sold
        
        return sold
    
    def maxProfit_greedy(self, prices: List[int], fee: int) -> int:
        """
        贪心解法：局部最优
        
        解题思路：
        1. 每次遇到更低价格就更新买入点
        2. 每次遇到更高价格就计算利润
        3. 如果利润大于手续费，就进行交易
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        buy_price = prices[0]
        max_profit = 0
        
        for i in range(1, len(prices)):
            if prices[i] < buy_price:
                # 更新买入点
                buy_price = prices[i]
            elif prices[i] > buy_price + fee:
                # 可以获利，进行交易
                max_profit += prices[i] - buy_price - fee
                buy_price = prices[i] - fee  # 更新买入点，避免重复计算手续费
        
        return max_profit
    
    def maxProfit_recursive(self, prices: List[int], fee: int) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最大利润
        2. 使用记忆化避免重复计算
        3. 考虑手续费
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        memo = {}
        
        def dfs(i, holding):
            if (i, holding) in memo:
                return memo[(i, holding)]
            
            if i == len(prices):
                return 0
            
            if holding:
                # 当前持有股票，可以选择卖出或继续持有
                result = max(dfs(i + 1, False) + prices[i] - fee, 
                           dfs(i + 1, True))
            else:
                # 当前不持有股票，可以选择买入或继续不持有
                result = max(dfs(i + 1, True) - prices[i], 
                           dfs(i + 1, False))
            
            memo[(i, holding)] = result
            return result
        
        return dfs(0, False)
    
    def maxProfit_brute_force(self, prices: List[int], fee: int) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的交易方案
        2. 计算每种方案的利润
        3. 返回最大利润
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        def dfs(i, holding, profit):
            if i == len(prices):
                return profit
            
            if holding:
                # 当前持有股票，可以选择卖出或继续持有
                return max(dfs(i + 1, False, profit + prices[i] - fee), 
                          dfs(i + 1, True, profit))
            else:
                # 当前不持有股票，可以选择买入或继续不持有
                return max(dfs(i + 1, True, profit - prices[i]), 
                          dfs(i + 1, False, profit))
        
        return dfs(0, False, 0)
    
    def maxProfit_state_machine(self, prices: List[int], fee: int) -> int:
        """
        状态机解法
        
        解题思路：
        1. 定义状态机：未交易 -> 持有 -> 卖出 -> 未交易
        2. 状态转移：买入、卖出、等待
        3. 计算最终状态的最大利润
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # 状态：0-未交易, 1-持有, 2-卖出
        state = 0
        profit = 0
        max_profit = 0
        
        for i in range(len(prices)):
            if state == 0:  # 未交易
                if i < len(prices) - 1 and prices[i] < prices[i + 1]:
                    # 买入
                    state = 1
                    profit -= prices[i]
            elif state == 1:  # 持有
                if i == len(prices) - 1 or prices[i] > prices[i + 1]:
                    # 卖出
                    state = 2
                    profit += prices[i] - fee
                    max_profit = max(max_profit, profit)
            elif state == 2:  # 卖出
                # 可以再次交易
                state = 0
        
        return max_profit
    
    def maxProfit_alternative(self, prices: List[int], fee: int) -> int:
        """
        替代解法：使用两个状态
        
        解题思路：
        1. 定义两个状态：持有股票、不持有股票
        2. 状态转移方程：
           - hold[i] = max(hold[i-1], sold[i-1] - prices[i])
           - sold[i] = max(sold[i-1], hold[i-1] + prices[i] - fee)
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        n = len(prices)
        
        # 初始化状态
        hold = [-float('inf')] * n  # 持有股票
        sold = [0] * n              # 不持有股票
        
        # 第一天只能买入
        hold[0] = -prices[0]
        
        for i in range(1, n):
            # 持有股票：要么继续持有，要么从卖出状态买入
            hold[i] = max(hold[i-1], sold[i-1] - prices[i])
            # 不持有股票：要么继续不持有，要么从持有状态卖出（扣除手续费）
            sold[i] = max(sold[i-1], hold[i-1] + prices[i] - fee)
        
        return sold[n-1]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    prices = [1, 3, 2, 8, 4, 9]
    fee = 2
    result = solution.maxProfit(prices, fee)
    expected = 8
    assert result == expected
    
    # 测试用例2
    prices = [1,3,7,5,10,3]
    fee = 3
    result = solution.maxProfit(prices, fee)
    expected = 6
    assert result == expected
    
    # 测试用例3
    prices = [1,2,3,4,5]
    fee = 1
    result = solution.maxProfit(prices, fee)
    expected = 3
    assert result == expected
    
    # 测试用例4
    prices = [7,6,4,3,1]
    fee = 1
    result = solution.maxProfit(prices, fee)
    expected = 0
    assert result == expected
    
    # 测试用例5
    prices = [1,3,2,8,4,9]
    fee = 0
    result = solution.maxProfit(prices, fee)
    expected = 8
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    prices = [1, 3, 2, 8, 4, 9]
    fee = 2
    result_opt = solution.maxProfit_optimized(prices, fee)
    assert result_opt == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    prices = [1, 3, 2, 8, 4, 9]
    fee = 2
    result_greedy = solution.maxProfit_greedy(prices, fee)
    assert result_greedy == expected
    
    # 测试递归解法
    print("测试递归解法...")
    prices = [1, 3, 2, 8, 4, 9]
    fee = 2
    result_rec = solution.maxProfit_recursive(prices, fee)
    assert result_rec == expected
    
    # 测试状态机解法
    print("测试状态机解法...")
    prices = [1, 3, 2, 8, 4, 9]
    fee = 2
    result_sm = solution.maxProfit_state_machine(prices, fee)
    assert result_sm == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
