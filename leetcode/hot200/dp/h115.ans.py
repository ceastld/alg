"""
115. 不同的子序列 - 标准答案
"""
from typing import List


class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. dp[i][j] 表示s的前i个字符中t的前j个字符出现的个数
        2. 状态转移方程：
           - 如果s[i-1] == t[j-1]，则dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
           - 否则，dp[i][j] = dp[i-1][j]
        3. 初始化：dp[i][0] = 1（空字符串是任何字符串的子序列）
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not t:
            return 1
        
        if not s:
            return 0
        
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化：空字符串是任何字符串的子序列
        for i in range(m + 1):
            dp[i][0] = 1
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i-1] == t[j-1]:
                    # 可以选择匹配或不匹配
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                else:
                    # 只能不匹配
                    dp[i][j] = dp[i-1][j]
        
        return dp[m][n]
    
    def numDistinct_optimized(self, s: str, t: str) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用一维数组优化空间复杂度
        2. 从后往前遍历，避免重复使用
        
        时间复杂度：O(m * n)
        空间复杂度：O(n)
        """
        if not t:
            return 1
        
        if not s:
            return 0
        
        m, n = len(s), len(t)
        dp = [0] * (n + 1)
        dp[0] = 1  # 空字符串是任何字符串的子序列
        
        for i in range(1, m + 1):
            # 从后往前遍历，避免重复使用
            for j in range(n, 0, -1):
                if s[i-1] == t[j-1]:
                    dp[j] += dp[j-1]
        
        return dp[n]
    
    def numDistinct_recursive(self, s: str, t: str) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的可能个数
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not t:
            return 1
        
        if not s:
            return 0
        
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if j == len(t):
                return 1
            
            if i == len(s):
                return 0
            
            if s[i] == t[j]:
                # 可以选择匹配或不匹配
                result = dfs(i + 1, j + 1) + dfs(i + 1, j)
            else:
                # 只能不匹配
                result = dfs(i + 1, j)
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, 0)
    
    def numDistinct_brute_force(self, s: str, t: str) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的子序列
        2. 检查每个子序列是否等于t
        3. 统计个数
        
        时间复杂度：O(2^m)
        空间复杂度：O(m)
        """
        if not t:
            return 1
        
        if not s:
            return 0
        
        def dfs(i, j):
            if j == len(t):
                return 1
            
            if i == len(s):
                return 0
            
            if s[i] == t[j]:
                # 可以选择匹配或不匹配
                return dfs(i + 1, j + 1) + dfs(i + 1, j)
            else:
                # 只能不匹配
                return dfs(i + 1, j)
        
        return dfs(0, 0)
    
    def numDistinct_alternative(self, s: str, t: str) -> int:
        """
        替代解法：使用字典
        
        解题思路：
        1. 使用字典存储所有可能的状态
        2. 逐步更新状态
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not t:
            return 1
        
        if not s:
            return 0
        
        m, n = len(s), len(t)
        
        # 使用字典存储所有可能的状态
        dp = {(0, 0): 1}  # 初始状态：空字符串是任何字符串的子序列
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i-1] == t[j-1]:
                    # 可以选择匹配或不匹配
                    dp[(i, j)] = dp.get((i-1, j-1), 0) + dp.get((i-1, j), 0)
                else:
                    # 只能不匹配
                    dp[(i, j)] = dp.get((i-1, j), 0)
        
        return dp.get((m, n), 0)
    
    def numDistinct_dfs(self, s: str, t: str) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的子序列
        2. 检查每个子序列是否等于t
        
        时间复杂度：O(2^m)
        空间复杂度：O(m)
        """
        if not t:
            return 1
        
        if not s:
            return 0
        
        def dfs(i, j):
            if j == len(t):
                return 1
            
            if i == len(s):
                return 0
            
            if s[i] == t[j]:
                # 可以选择匹配或不匹配
                return dfs(i + 1, j + 1) + dfs(i + 1, j)
            else:
                # 只能不匹配
                return dfs(i + 1, j)
        
        return dfs(0, 0)
    
    def numDistinct_memo(self, s: str, t: str) -> int:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not t:
            return 1
        
        if not s:
            return 0
        
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if j == len(t):
                return 1
            
            if i == len(s):
                return 0
            
            if s[i] == t[j]:
                # 可以选择匹配或不匹配
                result = dfs(i + 1, j + 1) + dfs(i + 1, j)
            else:
                # 只能不匹配
                result = dfs(i + 1, j)
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, 0)
    
    def numDistinct_greedy(self, s: str, t: str) -> int:
        """
        贪心解法：优先选择匹配字符
        
        解题思路：
        1. 优先选择匹配的字符
        2. 贪心地选择最优解
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not t:
            return 1
        
        if not s:
            return 0
        
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if j == len(t):
                return 1
            
            if i == len(s):
                return 0
            
            # 优先选择匹配的字符
            if s[i] == t[j]:
                result = dfs(i + 1, j + 1) + dfs(i + 1, j)
            else:
                result = dfs(i + 1, j)
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, 0)
    
    def numDistinct_iterative(self, s: str, t: str) -> int:
        """
        迭代解法：使用栈
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 避免递归调用栈
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not t:
            return 1
        
        if not s:
            return 0
        
        m, n = len(s), len(t)
        memo = {}
        stack = [(0, 0)]
        
        while stack:
            i, j = stack.pop()
            
            if (i, j) in memo:
                continue
            
            if j == len(t):
                memo[(i, j)] = 1
                continue
            
            if i == len(s):
                memo[(i, j)] = 0
                continue
            
            # 检查依赖项是否已计算
            if s[i] == t[j]:
                if (i + 1, j + 1) not in memo:
                    stack.append((i, j))
                    stack.append((i + 1, j + 1))
                    continue
                
                if (i + 1, j) not in memo:
                    stack.append((i, j))
                    stack.append((i + 1, j))
                    continue
                
                # 计算当前值
                memo[(i, j)] = memo[(i + 1, j + 1)] + memo[(i + 1, j)]
            else:
                if (i + 1, j) not in memo:
                    stack.append((i, j))
                    stack.append((i + 1, j))
                    continue
                
                # 计算当前值
                memo[(i, j)] = memo[(i + 1, j)]
        
        return memo[(0, 0)]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "rabbbit"
    t = "rabbit"
    result = solution.numDistinct(s, t)
    expected = 3
    assert result == expected
    
    # 测试用例2
    s = "babgbag"
    t = "bag"
    result = solution.numDistinct(s, t)
    expected = 5
    assert result == expected
    
    # 测试用例3
    s = "a"
    t = "a"
    result = solution.numDistinct(s, t)
    expected = 1
    assert result == expected
    
    # 测试用例4
    s = "a"
    t = "b"
    result = solution.numDistinct(s, t)
    expected = 0
    assert result == expected
    
    # 测试用例5
    s = "a"
    t = ""
    result = solution.numDistinct(s, t)
    expected = 1
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    s = "rabbbit"
    t = "rabbit"
    result_opt = solution.numDistinct_optimized(s, t)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    s = "rabbbit"
    t = "rabbit"
    result_rec = solution.numDistinct_recursive(s, t)
    assert result_rec == expected
    
    # 测试替代解法
    print("测试替代解法...")
    s = "rabbbit"
    t = "rabbit"
    result_alt = solution.numDistinct_alternative(s, t)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    s = "rabbbit"
    t = "rabbit"
    result_dfs = solution.numDistinct_dfs(s, t)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    s = "rabbbit"
    t = "rabbit"
    result_memo = solution.numDistinct_memo(s, t)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    s = "rabbbit"
    t = "rabbit"
    result_greedy = solution.numDistinct_greedy(s, t)
    assert result_greedy == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    s = "rabbbit"
    t = "rabbit"
    result_iter = solution.numDistinct_iterative(s, t)
    assert result_iter == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
