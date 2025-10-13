"""
122. 买卖股票的最佳时机II - 标准答案
"""
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 只要后一天价格比前一天高，就进行交易
        2. 累加所有正收益
        3. 贪心地选择最优解
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        max_profit = 0
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                max_profit += prices[i] - prices[i-1]
        
        return max_profit
    
    def maxProfit_dp(self, prices: List[int]) -> int:
        """
        动态规划解法
        
        解题思路：
        1. dp[i][0] 表示第i天不持有股票的最大利润
        2. dp[i][1] 表示第i天持有股票的最大利润
        3. 状态转移方程：
           - dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
           - dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        
        # 初始化
        dp[0][0] = 0           # 第0天不持有股票
        dp[0][1] = -prices[0]  # 第0天持有股票
        
        for i in range(1, n):
            # 不持有股票：要么继续不持有，要么今天卖出
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            # 持有股票：要么继续持有，要么今天买入
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        
        return dp[n-1][0]
    
    def maxProfit_dp_optimized(self, prices: List[int]) -> int:
        """
        动态规划空间优化
        
        解题思路：
        1. 使用两个变量代替二维数组
        2. hold表示持有股票的最大利润
        3. sold表示不持有股票的最大利润
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        hold = -prices[0]  # 持有股票
        sold = 0           # 不持有股票
        
        for i in range(1, len(prices)):
            # 更新状态
            new_hold = max(hold, sold - prices[i])
            new_sold = max(sold, hold + prices[i])
            
            hold = new_hold
            sold = new_sold
        
        return sold
    
    def maxProfit_recursive(self, prices: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最大利润
        2. 使用记忆化避免重复计算
        
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
                result = max(dfs(i + 1, False) + prices[i], dfs(i + 1, True))
            else:
                # 当前不持有股票，可以选择买入或继续不持有
                result = max(dfs(i + 1, True) - prices[i], dfs(i + 1, False))
            
            memo[(i, holding)] = result
            return result
        
        return dfs(0, False)
    
    def maxProfit_peak_valley(self, prices: List[int]) -> int:
        """
        峰谷法：寻找所有上升趋势
        
        解题思路：
        1. 找到所有价格上升的区间
        2. 计算每个区间的利润
        3. 累加所有利润
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        max_profit = 0
        valley = prices[0]  # 谷底
        peak = prices[0]    # 峰顶
        
        i = 0
        while i < len(prices) - 1:
            # 寻找谷底
            while i < len(prices) - 1 and prices[i] >= prices[i + 1]:
                i += 1
            valley = prices[i]
            
            # 寻找峰顶
            while i < len(prices) - 1 and prices[i] <= prices[i + 1]:
                i += 1
            peak = prices[i]
            
            max_profit += peak - valley
        
        return max_profit
    
    def maxProfit_brute_force(self, prices: List[int]) -> int:
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
                return max(dfs(i + 1, False, profit + prices[i]), 
                          dfs(i + 1, True, profit))
            else:
                # 当前不持有股票，可以选择买入或继续不持有
                return max(dfs(i + 1, True, profit - prices[i]), 
                          dfs(i + 1, False, profit))
        
        return dfs(0, False, 0)
    
    def maxProfit_state_machine(self, prices: List[int]) -> int:
        """
        状态机解法
        
        解题思路：
        1. 定义两个状态：持有股票、不持有股票
        2. 状态转移：买入、卖出、持有
        3. 计算最终状态的最大利润
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # 状态：0-不持有股票，1-持有股票
        state = 0  # 初始状态：不持有股票
        profit = 0
        
        for i in range(len(prices)):
            if state == 0:  # 不持有股票
                if i < len(prices) - 1 and prices[i] < prices[i + 1]:
                    # 买入
                    state = 1
                    profit -= prices[i]
            else:  # 持有股票
                if i == len(prices) - 1 or prices[i] > prices[i + 1]:
                    # 卖出
                    state = 0
                    profit += prices[i]
        
        return profit


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    prices = [7,1,5,3,6,4]
    result = solution.maxProfit(prices)
    expected = 7
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
    prices = [1,2,3,4,5,6]
    result = solution.maxProfit(prices)
    expected = 5
    assert result == expected
    
    # 测试用例5
    prices = [1]
    result = solution.maxProfit(prices)
    expected = 0
    assert result == expected
    
    # 测试DP解法
    print("测试DP解法...")
    prices = [7,1,5,3,6,4]
    result_dp = solution.maxProfit_dp(prices)
    assert result_dp == expected
    
    # 测试DP优化解法
    print("测试DP优化解法...")
    prices = [7,1,5,3,6,4]
    result_opt = solution.maxProfit_dp_optimized(prices)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    prices = [7,1,5,3,6,4]
    result_rec = solution.maxProfit_recursive(prices)
    assert result_rec == expected
    
    # 测试峰谷法
    print("测试峰谷法...")
    prices = [7,1,5,3,6,4]
    result_pv = solution.maxProfit_peak_valley(prices)
    assert result_pv == expected
    
    # 测试状态机解法
    print("测试状态机解法...")
    prices = [7,1,5,3,6,4]
    result_sm = solution.maxProfit_state_machine(prices)
    assert result_sm == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
