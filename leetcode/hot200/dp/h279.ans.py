"""
279. 完全平方数 - 标准答案
"""
from typing import List


class Solution:
    def numSquares(self, n: int) -> int:
        """
        标准解法：动态规划（完全背包）
        
        解题思路：
        1. 问题转化为：在限制总金额的情况下，选择完全平方数的最少数量
        2. 使用一维DP：dp[i] 表示金额为i时的最少完全平方数数量
        3. 状态转移方程：
           - dp[i] = min(dp[i], dp[i - j*j] + 1)
        
        时间复杂度：O(n * sqrt(n))
        空间复杂度：O(n)
        """
        if n <= 0:
            return 0
        
        # 使用一维数组优化空间
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # 金额为0时需要0个完全平方数
        
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        
        return dp[n]
    
    def numSquares_2d(self, n: int) -> int:
        """
        二维DP解法
        
        解题思路：
        1. dp[i][j] 表示前i个完全平方数组成金额j的最少数量
        2. 状态转移方程：
           - dp[i][j] = min(dp[i-1][j], dp[i][j-i*i] + 1)
        
        时间复杂度：O(n * sqrt(n))
        空间复杂度：O(n * sqrt(n))
        """
        if n <= 0:
            return 0
        
        # 生成所有可能的完全平方数
        squares = []
        i = 1
        while i * i <= n:
            squares.append(i * i)
            i += 1
        
        m = len(squares)
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        
        # 初始化：金额为0时需要0个完全平方数
        for i in range(m + 1):
            dp[i][0] = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if j >= squares[i-1]:
                    # 可以选择当前完全平方数
                    dp[i][j] = min(dp[i-1][j], dp[i][j-squares[i-1]] + 1)
                else:
                    # 不能选择当前完全平方数
                    dp[i][j] = dp[i-1][j]
        
        return dp[m][n]
    
    def numSquares_recursive(self, n: int) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个金额的最少完全平方数数量
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n * sqrt(n))
        空间复杂度：O(n)
        """
        if n <= 0:
            return 0
        
        memo = {}
        
        def dfs(remaining):
            if remaining in memo:
                return memo[remaining]
            
            if remaining == 0:
                return 0
            
            if remaining < 0:
                return float('inf')
            
            result = float('inf')
            j = 1
            while j * j <= remaining:
                result = min(result, dfs(remaining - j * j) + 1)
                j += 1
            
            memo[remaining] = result
            return result
        
        return dfs(n)
    
    def numSquares_brute_force(self, n: int) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的完全平方数组合
        2. 计算每种组合的总和
        3. 返回等于n的最少数量
        
        时间复杂度：O(n^n)
        空间复杂度：O(n)
        """
        if n <= 0:
            return 0
        
        def dfs(remaining, count):
            if remaining == 0:
                return count
            
            if remaining < 0:
                return float('inf')
            
            result = float('inf')
            j = 1
            while j * j <= remaining:
                result = min(result, dfs(remaining - j * j, count + 1))
                j += 1
            
            return result
        
        return dfs(n, 0)
    
    def numSquares_optimized(self, n: int) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用一维数组优化空间复杂度
        2. 从前往后遍历，避免重复使用
        
        时间复杂度：O(n * sqrt(n))
        空间复杂度：O(n)
        """
        if n <= 0:
            return 0
        
        # 使用一维数组优化空间
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # 金额为0时需要0个完全平方数
        
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        
        return dp[n]
    
    def numSquares_alternative(self, n: int) -> int:
        """
        替代解法：使用集合
        
        解题思路：
        1. 使用集合存储所有可能的金额
        2. 逐步更新集合
        
        时间复杂度：O(n * sqrt(n))
        空间复杂度：O(n)
        """
        if n <= 0:
            return 0
        
        # 使用集合存储所有可能的金额
        possible_amounts = {0: 0}  # 初始状态：金额为0需要0个完全平方数
        
        for i in range(1, n + 1):
            j = 1
            min_count = float('inf')
            while j * j <= i:
                if i - j * j in possible_amounts:
                    min_count = min(min_count, possible_amounts[i - j * j] + 1)
                j += 1
            
            if min_count != float('inf'):
                possible_amounts[i] = min_count
        
        return possible_amounts.get(n, float('inf'))
    
    def numSquares_dfs(self, n: int) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的完全平方数组合
        2. 统计等于n的最少数量
        
        时间复杂度：O(n^n)
        空间复杂度：O(n)
        """
        if n <= 0:
            return 0
        
        def dfs(remaining, count):
            if remaining == 0:
                return count
            
            if remaining < 0:
                return float('inf')
            
            result = float('inf')
            j = 1
            while j * j <= remaining:
                result = min(result, dfs(remaining - j * j, count + 1))
                j += 1
            
            return result
        
        return dfs(n, 0)
    
    def numSquares_memo(self, n: int) -> int:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(n * sqrt(n))
        空间复杂度：O(n)
        """
        if n <= 0:
            return 0
        
        memo = {}
        
        def dfs(remaining):
            if remaining in memo:
                return memo[remaining]
            
            if remaining == 0:
                return 0
            
            if remaining < 0:
                return float('inf')
            
            result = float('inf')
            j = 1
            while j * j <= remaining:
                result = min(result, dfs(remaining - j * j) + 1)
                j += 1
            
            memo[remaining] = result
            return result
        
        return dfs(n)
    
    def numSquares_greedy(self, n: int) -> int:
        """
        贪心解法：优先选择大完全平方数
        
        解题思路：
        1. 优先选择较大的完全平方数
        2. 贪心地选择最优解
        
        时间复杂度：O(n * sqrt(n))
        空间复杂度：O(n)
        """
        if n <= 0:
            return 0
        
        # 按完全平方数从大到小排序
        squares = []
        i = 1
        while i * i <= n:
            squares.append(i * i)
            i += 1
        
        squares.sort(reverse=True)
        
        def dfs(remaining, count):
            if remaining == 0:
                return count
            
            if remaining < 0:
                return float('inf')
            
            result = float('inf')
            for square in squares:
                if square <= remaining:
                    result = min(result, dfs(remaining - square, count + 1))
                    if result != float('inf'):
                        break  # 由于排序，找到第一个解就是最优解
            
            return result
        
        return dfs(n, 0)
    
    def numSquares_iterative(self, n: int) -> int:
        """
        迭代解法：使用栈
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 避免递归调用栈
        
        时间复杂度：O(n * sqrt(n))
        空间复杂度：O(n)
        """
        if n <= 0:
            return 0
        
        # 使用栈模拟递归
        stack = [(n, 0)]  # (remaining, count)
        result = float('inf')
        
        while stack:
            remaining, count = stack.pop()
            
            if remaining == 0:
                result = min(result, count)
                continue
            
            if remaining < 0:
                continue
            
            j = 1
            while j * j <= remaining:
                stack.append((remaining - j * j, count + 1))
                j += 1
        
        return result if result != float('inf') else 0


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    n = 12
    result = solution.numSquares(n)
    expected = 3
    assert result == expected
    
    # 测试用例2
    n = 13
    result = solution.numSquares(n)
    expected = 2
    assert result == expected
    
    # 测试用例3
    n = 1
    result = solution.numSquares(n)
    expected = 1
    assert result == expected
    
    # 测试用例4
    n = 4
    result = solution.numSquares(n)
    expected = 1
    assert result == expected
    
    # 测试用例5
    n = 9
    result = solution.numSquares(n)
    expected = 1
    assert result == expected
    
    # 测试二维DP解法
    print("测试二维DP解法...")
    n = 12
    result_2d = solution.numSquares_2d(n)
    assert result_2d == expected
    
    # 测试递归解法
    print("测试递归解法...")
    n = 12
    result_rec = solution.numSquares_recursive(n)
    assert result_rec == expected
    
    # 测试优化解法
    print("测试优化解法...")
    n = 12
    result_opt = solution.numSquares_optimized(n)
    assert result_opt == expected
    
    # 测试替代解法
    print("测试替代解法...")
    n = 12
    result_alt = solution.numSquares_alternative(n)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    n = 12
    result_dfs = solution.numSquares_dfs(n)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    n = 12
    result_memo = solution.numSquares_memo(n)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    n = 12
    result_greedy = solution.numSquares_greedy(n)
    assert result_greedy == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    n = 12
    result_iter = solution.numSquares_iterative(n)
    assert result_iter == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
