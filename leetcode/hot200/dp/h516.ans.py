"""
516. 最长回文子序列 - 标准答案
"""
from typing import List


class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. dp[i][j] 表示字符串s从位置i到位置j的最长回文子序列长度
        2. 状态转移方程：
           - 如果s[i] == s[j]，则dp[i][j] = dp[i+1][j-1] + 2
           - 否则，dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        3. 从右下角开始填充DP表
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s:
            return 0
        
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        # 单个字符都是回文，长度为1
        for i in range(n):
            dp[i][i] = 1
        
        # 填充DP表
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        
        return dp[0][n - 1]
    
    def longestPalindromeSubseq_optimized(self, s: str) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用一维数组优化空间复杂度
        2. 从右下角开始填充DP表
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not s:
            return 0
        
        n = len(s)
        dp = [0] * n
        
        # 单个字符都是回文，长度为1
        for i in range(n):
            dp[i] = 1
        
        # 填充DP表
        for i in range(n - 1, -1, -1):
            prev = 0
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[j] = prev + 2
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = dp[j]
        
        return dp[n - 1]
    
    def longestPalindromeSubseq_recursive(self, s: str) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最长回文子序列长度
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s:
            return 0
        
        n = len(s)
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i > j:
                return 0
            
            if i == j:
                return 1
            
            if s[i] == s[j]:
                result = dfs(i + 1, j - 1) + 2
            else:
                result = max(dfs(i + 1, j), dfs(i, j - 1))
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, n - 1)
    
    def longestPalindromeSubseq_brute_force(self, s: str) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的子序列
        2. 检查每个子序列是否为回文
        3. 返回最长回文子序列的长度
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not s:
            return 0
        
        def is_palindrome(subseq):
            left, right = 0, len(subseq) - 1
            while left < right:
                if subseq[left] != subseq[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        def dfs(i, current_subseq):
            if i == len(s):
                return len(current_subseq) if is_palindrome(current_subseq) else 0
            
            # 选择当前字符或不选择
            return max(dfs(i + 1, current_subseq + s[i]), dfs(i + 1, current_subseq))
        
        return dfs(0, "")
    
    def longestPalindromeSubseq_alternative(self, s: str) -> int:
        """
        替代解法：使用LCS
        
        解题思路：
        1. 最长回文子序列 = 原字符串与反转字符串的LCS
        2. 使用LCS算法求解
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s:
            return 0
        
        # 反转字符串
        reversed_s = s[::-1]
        
        # 计算LCS
        n = len(s)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if s[i - 1] == reversed_s[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[n][n]
    
    def longestPalindromeSubseq_dfs(self, s: str) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的子序列
        2. 检查每个子序列是否为回文
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not s:
            return 0
        
        def is_palindrome(subseq):
            left, right = 0, len(subseq) - 1
            while left < right:
                if subseq[left] != subseq[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        def dfs(i, current_subseq):
            if i == len(s):
                return len(current_subseq) if is_palindrome(current_subseq) else 0
            
            # 选择当前字符或不选择
            return max(dfs(i + 1, current_subseq + s[i]), dfs(i + 1, current_subseq))
        
        return dfs(0, "")
    
    def longestPalindromeSubseq_memo(self, s: str) -> int:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s:
            return 0
        
        n = len(s)
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i > j:
                return 0
            
            if i == j:
                return 1
            
            if s[i] == s[j]:
                result = dfs(i + 1, j - 1) + 2
            else:
                result = max(dfs(i + 1, j), dfs(i, j - 1))
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, n - 1)
    
    def longestPalindromeSubseq_greedy(self, s: str) -> int:
        """
        贪心解法：优先选择相同字符
        
        解题思路：
        1. 优先选择相同的字符
        2. 贪心地选择最优解
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s:
            return 0
        
        n = len(s)
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i > j:
                return 0
            
            if i == j:
                return 1
            
            # 优先选择相同的字符
            if s[i] == s[j]:
                result = dfs(i + 1, j - 1) + 2
            else:
                result = max(dfs(i + 1, j), dfs(i, j - 1))
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, n - 1)
    
    def longestPalindromeSubseq_iterative(self, s: str) -> int:
        """
        迭代解法：使用栈
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 避免递归调用栈
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s:
            return 0
        
        n = len(s)
        memo = {}
        stack = [(0, n - 1)]
        
        while stack:
            i, j = stack.pop()
            
            if (i, j) in memo:
                continue
            
            if i > j:
                memo[(i, j)] = 0
                continue
            
            if i == j:
                memo[(i, j)] = 1
                continue
            
            # 检查依赖项是否已计算
            if (i + 1, j - 1) not in memo:
                stack.append((i, j))
                stack.append((i + 1, j - 1))
                continue
            
            if (i + 1, j) not in memo:
                stack.append((i, j))
                stack.append((i + 1, j))
                continue
            
            if (i, j - 1) not in memo:
                stack.append((i, j))
                stack.append((i, j - 1))
                continue
            
            # 计算当前值
            if s[i] == s[j]:
                memo[(i, j)] = memo[(i + 1, j - 1)] + 2
            else:
                memo[(i, j)] = max(memo[(i + 1, j)], memo[(i, j - 1)])
        
        return memo[(0, n - 1)]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "bbbab"
    result = solution.longestPalindromeSubseq(s)
    expected = 4
    assert result == expected
    
    # 测试用例2
    s = "cbbd"
    result = solution.longestPalindromeSubseq(s)
    expected = 2
    assert result == expected
    
    # 测试用例3
    s = "a"
    result = solution.longestPalindromeSubseq(s)
    expected = 1
    assert result == expected
    
    # 测试用例4
    s = "ab"
    result = solution.longestPalindromeSubseq(s)
    expected = 1
    assert result == expected
    
    # 测试用例5
    s = "racecar"
    result = solution.longestPalindromeSubseq(s)
    expected = 7
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    s = "bbbab"
    result_opt = solution.longestPalindromeSubseq_optimized(s)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    s = "bbbab"
    result_rec = solution.longestPalindromeSubseq_recursive(s)
    assert result_rec == expected
    
    # 测试替代解法
    print("测试替代解法...")
    s = "bbbab"
    result_alt = solution.longestPalindromeSubseq_alternative(s)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    s = "bbbab"
    result_dfs = solution.longestPalindromeSubseq_dfs(s)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    s = "bbbab"
    result_memo = solution.longestPalindromeSubseq_memo(s)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    s = "bbbab"
    result_greedy = solution.longestPalindromeSubseq_greedy(s)
    assert result_greedy == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    s = "bbbab"
    result_iter = solution.longestPalindromeSubseq_iterative(s)
    assert result_iter == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
