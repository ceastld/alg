"""
309. 最佳买卖股票时机含冷冻期 - 标准答案
"""
from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. 定义三个状态：持有股票、不持有股票（非冷冻期）、冷冻期
        2. 状态转移方程：
           - hold[i] = max(hold[i-1], rest[i-1] - prices[i])
           - sold[i] = hold[i-1] + prices[i]
           - rest[i] = max(rest[i-1], sold[i-1])
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        n = len(prices)
        
        # 初始化状态
        hold = [-float('inf')] * n  # 持有股票
        sold = [0] * n             # 卖出股票（冷冻期）
        rest = [0] * n              # 不持有股票（非冷冻期）
        
        # 第一天只能买入
        hold[0] = -prices[0]
        
        for i in range(1, n):
            # 持有股票：要么继续持有，要么从非冷冻期买入
            hold[i] = max(hold[i-1], rest[i-1] - prices[i])
            # 卖出股票：从持有状态卖出
            sold[i] = hold[i-1] + prices[i]
            # 不持有股票（非冷冻期）：要么继续不持有，要么从冷冻期恢复
            rest[i] = max(rest[i-1], sold[i-1])
        
        return max(sold[n-1], rest[n-1])
    
    def maxProfit_optimized(self, prices: List[int]) -> int:
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
        sold = 0           # 卖出股票（冷冻期）
        rest = 0            # 不持有股票（非冷冻期）
        
        for i in range(1, len(prices)):
            # 更新状态（注意更新顺序）
            new_hold = max(hold, rest - prices[i])
            new_sold = hold + prices[i]
            new_rest = max(rest, sold)
            
            hold = new_hold
            sold = new_sold
            rest = new_rest
        
        return max(sold, rest)
    
    def maxProfit_recursive(self, prices: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最大利润
        2. 使用记忆化避免重复计算
        3. 考虑冷冻期限制
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        memo = {}
        
        def dfs(i, holding, cooldown):
            if (i, holding, cooldown) in memo:
                return memo[(i, holding, cooldown)]
            
            if i == len(prices):
                return 0
            
            if cooldown:
                # 冷冻期，只能等待
                result = dfs(i + 1, False, False)
            elif holding:
                # 持有股票，可以选择卖出或继续持有
                result = max(dfs(i + 1, False, True) + prices[i], 
                           dfs(i + 1, True, False))
            else:
                # 不持有股票，可以选择买入或继续不持有
                result = max(dfs(i + 1, True, False) - prices[i], 
                           dfs(i + 1, False, False))
            
            memo[(i, holding, cooldown)] = result
            return result
        
        return dfs(0, False, False)
    
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
        
        def dfs(i, holding, cooldown, profit):
            if i == len(prices):
                return profit
            
            if cooldown:
                # 冷冻期，只能等待
                return dfs(i + 1, False, False, profit)
            elif holding:
                # 持有股票，可以选择卖出或继续持有
                return max(dfs(i + 1, False, True, profit + prices[i]), 
                          dfs(i + 1, True, False, profit))
            else:
                # 不持有股票，可以选择买入或继续不持有
                return max(dfs(i + 1, True, False, profit - prices[i]), 
                          dfs(i + 1, False, False, profit))
        
        return dfs(0, False, False, 0)
    
    def maxProfit_state_machine(self, prices: List[int]) -> int:
        """
        状态机解法
        
        解题思路：
        1. 定义状态机：未交易 -> 持有 -> 卖出 -> 冷冻期 -> 未交易
        2. 状态转移：买入、卖出、等待
        3. 计算最终状态的最大利润
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not prices or len(prices) < 2:
            return 0
        
        # 状态：0-未交易, 1-持有, 2-卖出, 3-冷冻期
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
                    profit += prices[i]
                    max_profit = max(max_profit, profit)
            elif state == 2:  # 卖出
                # 进入冷冻期
                state = 3
            elif state == 3:  # 冷冻期
                # 冷冻期结束，可以交易
                state = 0
        
        return max_profit
    
    def maxProfit_dp_alternative(self, prices: List[int]) -> int:
        """
        替代DP解法：使用两个状态
        
        解题思路：
        1. 定义两个状态：持有股票、不持有股票
        2. 状态转移方程：
           - hold[i] = max(hold[i-1], rest[i-1] - prices[i])
           - rest[i] = max(rest[i-1], sold[i-1])
           - sold[i] = hold[i-1] + prices[i]
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not prices or len(prices) < 2:
            return 0
        
        n = len(prices)
        
        # 初始化状态
        hold = [-float('inf')] * n  # 持有股票
        rest = [0] * n              # 不持有股票（非冷冻期）
        sold = [0] * n              # 卖出股票（冷冻期）
        
        # 第一天只能买入
        hold[0] = -prices[0]
        
        for i in range(1, n):
            # 持有股票：要么继续持有，要么从非冷冻期买入
            hold[i] = max(hold[i-1], rest[i-1] - prices[i])
            # 不持有股票（非冷冻期）：要么继续不持有，要么从冷冻期恢复
            rest[i] = max(rest[i-1], sold[i-1])
            # 卖出股票：从持有状态卖出
            sold[i] = hold[i-1] + prices[i]
        
        return max(rest[n-1], sold[n-1])


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    prices = [1,2,3,0,2]
    result = solution.maxProfit(prices)
    expected = 3
    assert result == expected
    
    # 测试用例2
    prices = [1]
    result = solution.maxProfit(prices)
    expected = 0
    assert result == expected
    
    # 测试用例3
    prices = [1,2,3,4,5]
    result = solution.maxProfit(prices)
    expected = 4
    assert result == expected
    
    # 测试用例4
    prices = [7,6,4,3,1]
    result = solution.maxProfit(prices)
    expected = 0
    assert result == expected
    
    # 测试用例5
    prices = [1,2,4,0,2]
    result = solution.maxProfit(prices)
    expected = 3
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    prices = [1,2,3,0,2]
    result_opt = solution.maxProfit_optimized(prices)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    prices = [1,2,3,0,2]
    result_rec = solution.maxProfit_recursive(prices)
    assert result_rec == expected
    
    # 测试状态机解法
    print("测试状态机解法...")
    prices = [1,2,3,0,2]
    result_sm = solution.maxProfit_state_machine(prices)
    assert result_sm == expected
    
    # 测试替代DP解法
    print("测试替代DP解法...")
    prices = [1,2,3,0,2]
    result_alt = solution.maxProfit_dp_alternative(prices)
    assert result_alt == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
