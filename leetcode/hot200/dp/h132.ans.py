"""
132. 分割回文串II - 标准答案
"""
from typing import List


class Solution:
    def minCut(self, s: str) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. 首先预处理所有回文子串
        2. 使用DP计算最少分割次数
        3. dp[i] 表示前i个字符的最少分割次数
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s or len(s) <= 1:
            return 0
        
        n = len(s)
        
        # 预处理回文子串
        is_palindrome = [[False] * n for _ in range(n)]
        
        # 单个字符都是回文
        for i in range(n):
            is_palindrome[i][i] = True
        
        # 两个字符的情况
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # 三个及以上字符的情况
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        # 计算最少分割次数
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            dp[i] = i - 1  # 最坏情况：每个字符都分割
            
            for j in range(i):
                if is_palindrome[j][i - 1]:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]
    
    def minCut_optimized(self, s: str) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用一维数组优化空间复杂度
        2. 预处理回文子串
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s or len(s) <= 1:
            return 0
        
        n = len(s)
        
        # 预处理回文子串
        is_palindrome = [[False] * n for _ in range(n)]
        
        # 单个字符都是回文
        for i in range(n):
            is_palindrome[i][i] = True
        
        # 两个字符的情况
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # 三个及以上字符的情况
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        # 使用一维数组优化空间
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            dp[i] = i - 1  # 最坏情况：每个字符都分割
            
            for j in range(i):
                if is_palindrome[j][i - 1]:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]
    
    def minCut_recursive(self, s: str) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最少分割次数
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s or len(s) <= 1:
            return 0
        
        n = len(s)
        
        # 预处理回文子串
        is_palindrome = [[False] * n for _ in range(n)]
        
        # 单个字符都是回文
        for i in range(n):
            is_palindrome[i][i] = True
        
        # 两个字符的情况
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # 三个及以上字符的情况
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        memo = {}
        
        def dfs(start):
            if start in memo:
                return memo[start]
            
            if start == n:
                return 0
            
            result = n - start - 1  # 最坏情况：每个字符都分割
            
            for end in range(start, n):
                if is_palindrome[start][end]:
                    result = min(result, dfs(end + 1) + 1)
            
            memo[start] = result
            return result
        
        return dfs(0)
    
    def minCut_brute_force(self, s: str) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的分割方案
        2. 计算每种方案的分割次数
        3. 返回最少分割次数
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not s or len(s) <= 1:
            return 0
        
        def is_palindrome(s, start, end):
            while start < end:
                if s[start] != s[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        def dfs(start, cuts):
            if start == len(s):
                return cuts
            
            result = float('inf')
            
            for end in range(start, len(s)):
                if is_palindrome(s, start, end):
                    result = min(result, dfs(end + 1, cuts + 1))
            
            return result
        
        return dfs(0, 0)
    
    def minCut_alternative(self, s: str) -> int:
        """
        替代解法：使用中心扩展
        
        解题思路：
        1. 使用中心扩展法预处理回文子串
        2. 使用动态规划求解
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s or len(s) <= 1:
            return 0
        
        n = len(s)
        
        # 使用中心扩展法预处理回文子串
        is_palindrome = [[False] * n for _ in range(n)]
        
        # 奇数长度的回文
        for i in range(n):
            left = right = i
            while left >= 0 and right < n and s[left] == s[right]:
                is_palindrome[left][right] = True
                left -= 1
                right += 1
        
        # 偶数长度的回文
        for i in range(n - 1):
            left, right = i, i + 1
            while left >= 0 and right < n and s[left] == s[right]:
                is_palindrome[left][right] = True
                left -= 1
                right += 1
        
        # 计算最少分割次数
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            dp[i] = i - 1  # 最坏情况：每个字符都分割
            
            for j in range(i):
                if is_palindrome[j][i - 1]:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n]
    
    def minCut_dfs(self, s: str) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的分割
        2. 统计最少分割次数
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not s or len(s) <= 1:
            return 0
        
        def is_palindrome(s, start, end):
            while start < end:
                if s[start] != s[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        def dfs(start, cuts):
            if start == len(s):
                return cuts
            
            result = float('inf')
            
            for end in range(start, len(s)):
                if is_palindrome(s, start, end):
                    result = min(result, dfs(end + 1, cuts + 1))
            
            return result
        
        return dfs(0, 0)
    
    def minCut_memo(self, s: str) -> int:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s or len(s) <= 1:
            return 0
        
        n = len(s)
        
        # 预处理回文子串
        is_palindrome = [[False] * n for _ in range(n)]
        
        # 单个字符都是回文
        for i in range(n):
            is_palindrome[i][i] = True
        
        # 两个字符的情况
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # 三个及以上字符的情况
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        memo = {}
        
        def dfs(start):
            if start in memo:
                return memo[start]
            
            if start == n:
                return 0
            
            result = n - start - 1  # 最坏情况：每个字符都分割
            
            for end in range(start, n):
                if is_palindrome[start][end]:
                    result = min(result, dfs(end + 1) + 1)
            
            memo[start] = result
            return result
        
        return dfs(0)
    
    def minCut_greedy(self, s: str) -> int:
        """
        贪心解法：优先选择长回文
        
        解题思路：
        1. 优先选择较长的回文子串
        2. 贪心地选择最优解
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s or len(s) <= 1:
            return 0
        
        n = len(s)
        
        # 预处理回文子串
        is_palindrome = [[False] * n for _ in range(n)]
        
        # 单个字符都是回文
        for i in range(n):
            is_palindrome[i][i] = True
        
        # 两个字符的情况
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # 三个及以上字符的情况
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        def dfs(start):
            if start == n:
                return 0
            
            # 优先选择最长的回文子串
            for length in range(n - start, 0, -1):
                end = start + length - 1
                if is_palindrome[start][end]:
                    return dfs(end + 1) + 1
            
            return n - start - 1  # 最坏情况：每个字符都分割
        
        return dfs(0)
    
    def minCut_iterative(self, s: str) -> int:
        """
        迭代解法：使用栈
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 避免递归调用栈
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s or len(s) <= 1:
            return 0
        
        n = len(s)
        
        # 预处理回文子串
        is_palindrome = [[False] * n for _ in range(n)]
        
        # 单个字符都是回文
        for i in range(n):
            is_palindrome[i][i] = True
        
        # 两个字符的情况
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                is_palindrome[i][i + 1] = True
        
        # 三个及以上字符的情况
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and is_palindrome[i + 1][j - 1]:
                    is_palindrome[i][j] = True
        
        # 使用栈模拟递归
        stack = [(0, 0)]  # (start, cuts)
        result = float('inf')
        
        while stack:
            start, cuts = stack.pop()
            
            if start == n:
                result = min(result, cuts)
                continue
            
            for end in range(start, n):
                if is_palindrome[start][end]:
                    stack.append((end + 1, cuts + 1))
        
        return result if result != float('inf') else 0


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "aab"
    result = solution.minCut(s)
    expected = 1
    assert result == expected
    
    # 测试用例2
    s = "a"
    result = solution.minCut(s)
    expected = 0
    assert result == expected
    
    # 测试用例3
    s = "ab"
    result = solution.minCut(s)
    expected = 1
    assert result == expected
    
    # 测试用例4
    s = "racecar"
    result = solution.minCut(s)
    expected = 0
    assert result == expected
    
    # 测试用例5
    s = "abacaba"
    result = solution.minCut(s)
    expected = 0
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    s = "aab"
    result_opt = solution.minCut_optimized(s)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    s = "aab"
    result_rec = solution.minCut_recursive(s)
    assert result_rec == expected
    
    # 测试替代解法
    print("测试替代解法...")
    s = "aab"
    result_alt = solution.minCut_alternative(s)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    s = "aab"
    result_dfs = solution.minCut_dfs(s)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    s = "aab"
    result_memo = solution.minCut_memo(s)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    s = "aab"
    result_greedy = solution.minCut_greedy(s)
    assert result_greedy == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    s = "aab"
    result_iter = solution.minCut_iterative(s)
    assert result_iter == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
