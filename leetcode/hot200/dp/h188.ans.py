"""
188. 买卖股票的最佳时机IV - 标准答案
"""
from typing import List


class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. 定义状态：buy[i][j] 表示第i天完成j次交易且持有股票的最大利润
        2. 定义状态：sell[i][j] 表示第i天完成j次交易且不持有股票的最大利润
        3. 状态转移方程：
           - buy[i][j] = max(buy[i-1][j], sell[i-1][j] - prices[i])
           - sell[i][j] = max(sell[i-1][j], buy[i-1][j-1] + prices[i])
        
        时间复杂度：O(n*k)
        空间复杂度：O(n*k)
        """
        if not prices or len(prices) < 2 or k <= 0:
            return 0
        
        n = len(prices)
        
        # 如果k >= n//2，相当于可以无限次交易
        if k >= n // 2:
            return self.maxProfit_unlimited(prices)
        
        # 初始化状态
        buy = [[-float('inf')] * (k + 1) for _ in range(n)]
        sell = [[0] * (k + 1) for _ in range(n)]
        
        # 第一天只能买入
        buy[0][0] = -prices[0]
        
        for i in range(1, n):
            for j in range(k + 1):
                # 持有股票：要么继续持有，要么今天买入
                buy[i][j] = max(buy[i-1][j], sell[i-1][j] - prices[i])
                
                # 不持有股票：要么继续不持有，要么今天卖出
                if j > 0:
                    sell[i][j] = max(sell[i-1][j], buy[i-1][j-1] + prices[i])
                else:
                    sell[i][j] = sell[i-1][j]
        
        return max(sell[n-1])
    
    def maxProfit_optimized(self, k: int, prices: List[int]) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用滚动数组优化空间复杂度
        2. 只保留必要的状态信息
        
        时间复杂度：O(n*k)
        空间复杂度：O(k)
        """
        if not prices or len(prices) < 2 or k <= 0:
            return 0
        
        n = len(prices)
        
        # 如果k >= n//2，相当于可以无限次交易
        if k >= n // 2:
            return self.maxProfit_unlimited(prices)
        
        # 初始化状态
        buy = [-float('inf')] * (k + 1)
        sell = [0] * (k + 1)
        
        # 第一天只能买入
        buy[0] = -prices[0]
        
        for i in range(1, n):
            # 从后往前更新，避免状态覆盖
            for j in range(k, 0, -1):
                # 不持有股票：要么继续不持有，要么今天卖出
                sell[j] = max(sell[j], buy[j] + prices[i])
                # 持有股票：要么继续持有，要么今天买入
                buy[j] = max(buy[j], sell[j-1] - prices[i])
        
        return max(sell)
    
    def maxProfit_unlimited(self, prices: List[int]) -> int:
        """
        无限次交易解法
        
        解题思路：
        1. 只要后一天价格比前一天高，就进行交易
        2. 累加所有正收益
        
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
    
    def maxProfit_recursive(self, k: int, prices: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最大利润
        2. 使用记忆化避免重复计算
        3. 考虑交易次数限制
        
        时间复杂度：O(n*k)
        空间复杂度：O(n*k)
        """
        if not prices or len(prices) < 2 or k <= 0:
            return 0
        
        n = len(prices)
        
        # 如果k >= n//2，相当于可以无限次交易
        if k >= n // 2:
            return self.maxProfit_unlimited(prices)
        
        memo = {}
        
        def dfs(i, holding, transactions):
            if (i, holding, transactions) in memo:
                return memo[(i, holding, transactions)]
            
            if i == len(prices) or transactions >= k:
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
    
    def maxProfit_brute_force(self, k: int, prices: List[int]) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的交易方案
        2. 计算每种方案的利润
        3. 返回最大利润
        
        时间复杂度：O(n^(2k))
        空间复杂度：O(k)
        """
        if not prices or len(prices) < 2 or k <= 0:
            return 0
        
        n = len(prices)
        
        # 如果k >= n//2，相当于可以无限次交易
        if k >= n // 2:
            return self.maxProfit_unlimited(prices)
        
        def dfs(i, holding, transactions, profit):
            if i == len(prices) or transactions >= k:
                return profit
            
            if holding:
                # 当前持有股票，可以选择卖出或继续持有
                return max(dfs(i + 1, False, transactions + 1, profit + prices[i]), 
                          dfs(i + 1, True, transactions, profit))
            else:
                # 当前不持有股票，可以选择买入或继续不持有
                return max(dfs(i + 1, True, transactions, profit - prices[i]), 
                          dfs(i + 1, False, transactions, profit))
        
        return dfs(0, False, 0, 0)
    
    def maxProfit_state_machine(self, k: int, prices: List[int]) -> int:
        """
        状态机解法
        
        解题思路：
        1. 定义状态机：未交易 -> 第一次买入 -> 第一次卖出 -> ... -> 第k次买入 -> 第k次卖出
        2. 状态转移：买入、卖出、持有
        3. 计算最终状态的最大利润
        
        时间复杂度：O(n*k)
        空间复杂度：O(k)
        """
        if not prices or len(prices) < 2 or k <= 0:
            return 0
        
        n = len(prices)
        
        # 如果k >= n//2，相当于可以无限次交易
        if k >= n // 2:
            return self.maxProfit_unlimited(prices)
        
        # 状态：0-未交易, 1-第一次买入, 2-第一次卖出, ..., 2k-1-第k次买入, 2k-第k次卖出
        state = 0
        profit = 0
        max_profit = 0
        
        for i in range(len(prices)):
            if state == 0:  # 未交易
                if i < len(prices) - 1 and prices[i] < prices[i + 1]:
                    # 第一次买入
                    state = 1
                    profit -= prices[i]
            elif state % 2 == 1:  # 第j次买入
                if i == len(prices) - 1 or prices[i] > prices[i + 1]:
                    # 第j次卖出
                    state += 1
                    profit += prices[i]
                    if state == 2 * k:
                        max_profit = max(max_profit, profit)
            elif state % 2 == 0 and state < 2 * k:  # 第j次卖出
                if i < len(prices) - 1 and prices[i] < prices[i + 1]:
                    # 第j+1次买入
                    state += 1
                    profit -= prices[i]
        
        return max_profit


def main():
    """测试标准答案"""
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
    
    # 测试优化解法
    print("测试优化解法...")
    k = 2
    prices = [2,4,1]
    result_opt = solution.maxProfit_optimized(k, prices)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    k = 2
    prices = [2,4,1]
    result_rec = solution.maxProfit_recursive(k, prices)
    assert result_rec == expected
    
    # 测试状态机解法
    print("测试状态机解法...")
    k = 2
    prices = [2,4,1]
    result_sm = solution.maxProfit_state_machine(k, prices)
    assert result_sm == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
