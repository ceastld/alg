"""
123. 买卖股票的最佳时机III - 标准答案
"""
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. 定义5个状态：未交易、第一次买入、第一次卖出、第二次买入、第二次卖出
        2. 状态转移方程：
           - buy1 = max(buy1, -prices[i])
           - sell1 = max(sell1, buy1 + prices[i])
           - buy2 = max(buy2, sell1 - prices[i])
           - sell2 = max(sell2, buy2 + prices[i])
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # 初始化状态
        buy1 = -prices[0]   # 第一次买入
        sell1 = 0          # 第一次卖出
        buy2 = -prices[0]  # 第二次买入
        sell2 = 0          # 第二次卖出
        
        for i in range(1, len(prices)):
            # 更新状态
            buy1 = max(buy1, -prices[i])
            sell1 = max(sell1, buy1 + prices[i])
            buy2 = max(buy2, sell1 - prices[i])
            sell2 = max(sell2, buy2 + prices[i])
        
        return sell2
    
    def maxProfit_dp(self, prices: List[int]) -> int:
        """
        动态规划解法（二维数组）
        
        解题思路：
        1. dp[i][j] 表示第i天第j个状态的最大利润
        2. j=0: 未交易, j=1: 第一次买入, j=2: 第一次卖出, j=3: 第二次买入, j=4: 第二次卖出
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        n = len(prices)
        dp = [[0] * 5 for _ in range(n)]
        
        # 初始化
        dp[0][0] = 0           # 未交易
        dp[0][1] = -prices[0]  # 第一次买入
        dp[0][2] = 0           # 第一次卖出
        dp[0][3] = -prices[0]  # 第二次买入
        dp[0][4] = 0           # 第二次卖出
        
        for i in range(1, n):
            # 未交易
            dp[i][0] = dp[i-1][0]
            # 第一次买入
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
            # 第一次卖出
            dp[i][2] = max(dp[i-1][2], dp[i-1][1] + prices[i])
            # 第二次买入
            dp[i][3] = max(dp[i-1][3], dp[i-1][2] - prices[i])
            # 第二次卖出
            dp[i][4] = max(dp[i-1][4], dp[i-1][3] + prices[i])
        
        return dp[n-1][4]
    
    def maxProfit_recursive(self, prices: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最大利润
        2. 使用记忆化避免重复计算
        3. 考虑交易次数限制
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        memo = {}
        
        def dfs(i, holding, transactions):
            if (i, holding, transactions) in memo:
                return memo[(i, holding, transactions)]
            
            if i == len(prices) or transactions >= 2:
                return 0
            
            if holding:
                # 当前持有股票，可以选择卖出或继续持有
                result = max(dfs(i + 1, False, transactions + 1) + prices[i], 
                           dfs(i + 1, True, transactions))
            else:
                # 当前不持有股票，可以选择买入或继续不持有
                result = max(dfs(i + 1, True, transactions) - prices[i], 
                           dfs(i + 1, False, transactions))
            
            memo[(i, holding, transactions)] = result
            return result
        
        return dfs(0, False, 0)
    
    def maxProfit_brute_force(self, prices: List[int]) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的交易方案
        2. 计算每种方案的利润
        3. 返回最大利润
        
        时间复杂度：O(n^4)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        max_profit = 0
        n = len(prices)
        
        # 枚举所有可能的交易方案
        for i in range(n):
            for j in range(i + 1, n):
                # 第一次交易
                profit1 = prices[j] - prices[i]
                if profit1 <= 0:
                    continue
                
                # 第二次交易
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        profit2 = prices[l] - prices[k]
                        max_profit = max(max_profit, profit1 + profit2)
        
        return max_profit
    
    def maxProfit_optimized(self, prices: List[int]) -> int:
        """
        优化解法：状态压缩
        
        解题思路：
        1. 使用滚动数组优化空间复杂度
        2. 只保留必要的状态信息
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # 初始化状态
        buy1 = -prices[0]   # 第一次买入
        sell1 = 0          # 第一次卖出
        buy2 = -prices[0]  # 第二次买入
        sell2 = 0          # 第二次卖出
        
        for i in range(1, len(prices)):
            # 更新状态（注意更新顺序）
            buy1 = max(buy1, -prices[i])
            sell1 = max(sell1, buy1 + prices[i])
            buy2 = max(buy2, sell1 - prices[i])
            sell2 = max(sell2, buy2 + prices[i])
        
        return sell2
    
    def maxProfit_state_machine(self, prices: List[int]) -> int:
        """
        状态机解法
        
        解题思路：
        1. 定义状态机：未交易 -> 第一次买入 -> 第一次卖出 -> 第二次买入 -> 第二次卖出
        2. 状态转移：买入、卖出、持有
        3. 计算最终状态的最大利润
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # 状态：0-未交易, 1-第一次买入, 2-第一次卖出, 3-第二次买入, 4-第二次卖出
        state = 0
        profit = 0
        max_profit = 0
        
        for i in range(len(prices)):
            if state == 0:  # 未交易
                if i < len(prices) - 1 and prices[i] < prices[i + 1]:
                    # 第一次买入
                    state = 1
                    profit -= prices[i]
            elif state == 1:  # 第一次买入
                if i == len(prices) - 1 or prices[i] > prices[i + 1]:
                    # 第一次卖出
                    state = 2
                    profit += prices[i]
            elif state == 2:  # 第一次卖出
                if i < len(prices) - 1 and prices[i] < prices[i + 1]:
                    # 第二次买入
                    state = 3
                    profit -= prices[i]
            elif state == 3:  # 第二次买入
                if i == len(prices) - 1 or prices[i] > prices[i + 1]:
                    # 第二次卖出
                    state = 4
                    profit += prices[i]
                    max_profit = max(max_profit, profit)
        
        return max_profit


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    prices = [3,3,5,0,0,3,1,4]
    result = solution.maxProfit(prices)
    expected = 6
    assert result == expected
    
    # 测试用例2
    prices = [1,2,3,4,5]
    result = solution.maxProfit(prices)
    expected = 4
    assert result == expected
    
    # 测试用例3
    prices = [7,6,4,3,1]
    result = solution.maxProfit(prices)
    expected = 0
    assert result == expected
    
    # 测试用例4
    prices = [1]
    result = solution.maxProfit(prices)
    expected = 0
    assert result == expected
    
    # 测试用例5
    prices = [1,2,4,2,5,7,2,4,9,0]
    result = solution.maxProfit(prices)
    expected = 13
    assert result == expected
    
    # 测试DP解法
    print("测试DP解法...")
    prices = [3,3,5,0,0,3,1,4]
    result_dp = solution.maxProfit_dp(prices)
    assert result_dp == expected
    
    # 测试递归解法
    print("测试递归解法...")
    prices = [3,3,5,0,0,3,1,4]
    result_rec = solution.maxProfit_recursive(prices)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    prices = [3,3,5,0,0,3,1,4]
    result_opt = solution.maxProfit_optimized(prices)
    assert result_opt == expected
    
    # 测试状态机解法
    print("测试状态机解法...")
    prices = [3,3,5,0,0,3,1,4]
    result_sm = solution.maxProfit_state_machine(prices)
    assert result_sm == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
