"""
121. 买卖股票的最佳时机 - 标准答案
"""
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        标准解法：一次遍历
        
        解题思路：
        1. 遍历每一天的价格
        2. 维护最低买入价格
        3. 计算当前价格与最低价格的差值
        4. 更新最大利润
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        min_price = prices[0]
        max_profit = 0
        
        for price in prices:
            # 更新最低价格
            min_price = min(min_price, price)
            # 计算当前利润
            current_profit = price - min_price
            # 更新最大利润
            max_profit = max(max_profit, current_profit)
        
        return max_profit
    
    def maxProfit_dp(self, prices: List[int]) -> int:
        """
        动态规划解法
        
        解题思路：
        1. dp[i][0] 表示第i天持有股票的最大利润
        2. dp[i][1] 表示第i天不持有股票的最大利润
        3. 状态转移方程：
           - dp[i][0] = max(dp[i-1][0], -prices[i])
           - dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        
        # 初始化
        dp[0][0] = -prices[0]  # 第0天持有股票
        dp[0][1] = 0           # 第0天不持有股票
        
        for i in range(1, n):
            # 持有股票：要么继续持有，要么今天买入
            dp[i][0] = max(dp[i-1][0], -prices[i])
            # 不持有股票：要么继续不持有，要么今天卖出
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        
        return dp[n-1][1]
    
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
            new_hold = max(hold, -prices[i])
            new_sold = max(sold, hold + prices[i])
            
            hold = new_hold
            sold = new_sold
        
        return sold
    
    def maxProfit_brute_force(self, prices: List[int]) -> int:
        """
        暴力解法：双重循环
        
        解题思路：
        1. 枚举所有可能的买入和卖出组合
        2. 计算每种组合的利润
        3. 返回最大利润
        
        时间复杂度：O(n²)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        max_profit = 0
        
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                profit = prices[j] - prices[i]
                max_profit = max(max_profit, profit)
        
        return max_profit
    
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
    
    def maxProfit_greedy(self, prices: List[int]) -> int:
        """
        贪心解法：局部最优
        
        解题思路：
        1. 每次遇到更低价格就更新买入点
        2. 每次遇到更高价格就计算利润
        3. 贪心地选择最优解
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        buy_price = prices[0]
        max_profit = 0
        
        for price in prices[1:]:
            if price < buy_price:
                buy_price = price
            else:
                max_profit = max(max_profit, price - buy_price)
        
        return max_profit


def main():
    """测试标准答案"""
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
    
    # 测试暴力解法
    print("测试暴力解法...")
    prices = [7,1,5,3,6,4]
    result_bf = solution.maxProfit_brute_force(prices)
    assert result_bf == expected
    
    # 测试递归解法
    print("测试递归解法...")
    prices = [7,1,5,3,6,4]
    result_rec = solution.maxProfit_recursive(prices)
    assert result_rec == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    prices = [7,1,5,3,6,4]
    result_greedy = solution.maxProfit_greedy(prices)
    assert result_greedy == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
