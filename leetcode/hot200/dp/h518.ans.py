"""
518. 零钱兑换II - 标准答案
"""
from typing import List


class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        """
        标准解法：动态规划（完全背包）
        
        解题思路：
        1. 问题转化为：在限制总金额的情况下，选择硬币的组合数
        2. 使用一维DP：dp[i] 表示金额为i时的组合数
        3. 状态转移方程：
           - dp[i] += dp[i - coin]
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount)
        """
        if amount == 0:
            return 1
        
        if not coins:
            return 0
        
        # 使用一维数组优化空间
        dp = [0] * (amount + 1)
        dp[0] = 1  # 金额为0时有一种组合（不选择任何硬币）
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        
        return dp[amount]
    
    def change_2d(self, amount: int, coins: List[int]) -> int:
        """
        二维DP解法
        
        解题思路：
        1. dp[i][j] 表示前i种硬币组成金额j的组合数
        2. 状态转移方程：
           - dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]]
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount * len(coins))
        """
        if amount == 0:
            return 1
        
        if not coins:
            return 0
        
        n = len(coins)
        dp = [[0] * (amount + 1) for _ in range(n + 1)]
        
        # 初始化：金额为0时有一种组合
        for i in range(n + 1):
            dp[i][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, amount + 1):
                if j >= coins[i-1]:
                    # 可以选择当前硬币
                    dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]]
                else:
                    # 不能选择当前硬币
                    dp[i][j] = dp[i-1][j]
        
        return dp[n][amount]
    
    def change_recursive(self, amount: int, coins: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个金额的组合数
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount * len(coins))
        """
        if amount == 0:
            return 1
        
        if not coins:
            return 0
        
        memo = {}
        
        def dfs(remaining, coin_index):
            if (remaining, coin_index) in memo:
                return memo[(remaining, coin_index)]
            
            if remaining == 0:
                return 1
            
            if coin_index >= len(coins):
                return 0
            
            # 选择当前硬币或不选择
            result = 0
            if remaining >= coins[coin_index]:
                result += dfs(remaining - coins[coin_index], coin_index)
            result += dfs(remaining, coin_index + 1)
            
            memo[(remaining, coin_index)] = result
            return result
        
        return dfs(amount, 0)
    
    def change_brute_force(self, amount: int, coins: List[int]) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的硬币组合
        2. 计算每种组合的总金额
        3. 统计等于amount的组合数
        
        时间复杂度：O(amount^len(coins))
        空间复杂度：O(len(coins))
        """
        if amount == 0:
            return 1
        
        if not coins:
            return 0
        
        def dfs(remaining, coin_index, current_combination):
            if remaining == 0:
                return 1
            
            if coin_index >= len(coins):
                return 0
            
            result = 0
            # 选择当前硬币
            if remaining >= coins[coin_index]:
                result += dfs(remaining - coins[coin_index], coin_index, current_combination + [coins[coin_index]])
            # 不选择当前硬币
            result += dfs(remaining, coin_index + 1, current_combination)
            
            return result
        
        return dfs(amount, 0, [])
    
    def change_optimized(self, amount: int, coins: List[int]) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用一维数组优化空间复杂度
        2. 从前往后遍历，避免重复使用
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount)
        """
        if amount == 0:
            return 1
        
        if not coins:
            return 0
        
        # 使用一维数组优化空间
        dp = [0] * (amount + 1)
        dp[0] = 1  # 金额为0时有一种组合
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        
        return dp[amount]
    
    def change_alternative(self, amount: int, coins: List[int]) -> int:
        """
        替代解法：使用集合
        
        解题思路：
        1. 使用集合存储所有可能的金额
        2. 逐步更新集合
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount)
        """
        if amount == 0:
            return 1
        
        if not coins:
            return 0
        
        # 使用集合存储所有可能的金额
        possible_amounts = {0: 1}  # 初始状态：金额为0有1种组合
        
        for coin in coins:
            # 创建新的集合，避免重复使用
            new_amounts = {}
            for amt, count in possible_amounts.items():
                new_amt = amt + coin
                if new_amt <= amount:
                    new_amounts[new_amt] = new_amounts.get(new_amt, 0) + count
            
            # 更新集合
            for amt, count in new_amounts.items():
                possible_amounts[amt] = possible_amounts.get(amt, 0) + count
        
        return possible_amounts.get(amount, 0)
    
    def change_dfs(self, amount: int, coins: List[int]) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的硬币组合
        2. 统计等于amount的组合数
        
        时间复杂度：O(amount^len(coins))
        空间复杂度：O(len(coins))
        """
        if amount == 0:
            return 1
        
        if not coins:
            return 0
        
        def dfs(remaining, coin_index):
            if remaining == 0:
                return 1
            
            if coin_index >= len(coins):
                return 0
            
            result = 0
            # 选择当前硬币
            if remaining >= coins[coin_index]:
                result += dfs(remaining - coins[coin_index], coin_index)
            # 不选择当前硬币
            result += dfs(remaining, coin_index + 1)
            
            return result
        
        return dfs(amount, 0)
    
    def change_memo(self, amount: int, coins: List[int]) -> int:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount * len(coins))
        """
        if amount == 0:
            return 1
        
        if not coins:
            return 0
        
        memo = {}
        
        def dfs(remaining, coin_index):
            if (remaining, coin_index) in memo:
                return memo[(remaining, coin_index)]
            
            if remaining == 0:
                return 1
            
            if coin_index >= len(coins):
                return 0
            
            result = 0
            # 选择当前硬币
            if remaining >= coins[coin_index]:
                result += dfs(remaining - coins[coin_index], coin_index)
            # 不选择当前硬币
            result += dfs(remaining, coin_index + 1)
            
            memo[(remaining, coin_index)] = result
            return result
        
        return dfs(amount, 0)
    
    def change_greedy(self, amount: int, coins: List[int]) -> int:
        """
        贪心解法：优先选择大面额硬币
        
        解题思路：
        1. 优先选择面额较大的硬币
        2. 贪心地选择最优解
        
        时间复杂度：O(amount * len(coins))
        空间复杂度：O(amount)
        """
        if amount == 0:
            return 1
        
        if not coins:
            return 0
        
        # 按面额从大到小排序
        sorted_coins = sorted(coins, reverse=True)
        
        def dfs(remaining, coin_index):
            if remaining == 0:
                return 1
            
            if coin_index >= len(sorted_coins):
                return 0
            
            result = 0
            # 选择当前硬币
            if remaining >= sorted_coins[coin_index]:
                result += dfs(remaining - sorted_coins[coin_index], coin_index)
            # 不选择当前硬币
            result += dfs(remaining, coin_index + 1)
            
            return result
        
        return dfs(amount, 0)


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    amount = 5
    coins = [1, 2, 5]
    result = solution.change(amount, coins)
    expected = 4
    assert result == expected
    
    # 测试用例2
    amount = 3
    coins = [2]
    result = solution.change(amount, coins)
    expected = 0
    assert result == expected
    
    # 测试用例3
    amount = 10
    coins = [10]
    result = solution.change(amount, coins)
    expected = 1
    assert result == expected
    
    # 测试用例4
    amount = 0
    coins = [1, 2, 5]
    result = solution.change(amount, coins)
    expected = 1
    assert result == expected
    
    # 测试用例5
    amount = 5
    coins = [1, 2, 3]
    result = solution.change(amount, coins)
    expected = 5
    assert result == expected
    
    # 测试二维DP解法
    print("测试二维DP解法...")
    amount = 5
    coins = [1, 2, 5]
    result_2d = solution.change_2d(amount, coins)
    assert result_2d == expected
    
    # 测试递归解法
    print("测试递归解法...")
    amount = 5
    coins = [1, 2, 5]
    result_rec = solution.change_recursive(amount, coins)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    amount = 5
    coins = [1, 2, 5]
    result_opt = solution.change_optimized(amount, coins)
    assert result_opt == expected
    
    # 测试替代解法
    print("测试替代解法...")
    amount = 5
    coins = [1, 2, 5]
    result_alt = solution.change_alternative(amount, coins)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    amount = 5
    coins = [1, 2, 5]
    result_dfs = solution.change_dfs(amount, coins)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    amount = 5
    coins = [1, 2, 5]
    result_memo = solution.change_memo(amount, coins)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    amount = 5
    coins = [1, 2, 5]
    result_greedy = solution.change_greedy(amount, coins)
    assert result_greedy == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
