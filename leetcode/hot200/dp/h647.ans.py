"""
647. 回文子串 - 标准答案
"""
from typing import List


class Solution:
    def countSubstrings(self, s: str) -> int:
        """
        标准解法：中心扩展法
        
        解题思路：
        1. 遍历所有可能的回文中心
        2. 从中心向两边扩展，检查是否为回文
        3. 考虑奇数长度和偶数长度的回文
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        if not s:
            return 0
        
        n = len(s)
        count = 0
        
        for i in range(n):
            # 奇数长度的回文
            count += self.expand_around_center(s, i, i)
            # 偶数长度的回文
            count += self.expand_around_center(s, i, i + 1)
        
        return count
    
    def expand_around_center(self, s: str, left: int, right: int) -> int:
        """从中心向两边扩展，返回回文子串的数量"""
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    def countSubstrings_dp(self, s: str) -> int:
        """
        动态规划解法
        
        解题思路：
        1. dp[i][j] 表示字符串s从位置i到位置j是否为回文
        2. 状态转移方程：
           - 如果s[i] == s[j]且dp[i+1][j-1]为真，则dp[i][j]为真
           - 否则dp[i][j]为假
        3. 从右下角开始填充DP表
        
        时间复杂度：O(n^2)
        空间复杂度：O(n^2)
        """
        if not s:
            return 0
        
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        count = 0
        
        # 单个字符都是回文
        for i in range(n):
            dp[i][i] = True
            count += 1
        
        # 两个字符的情况
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                count += 1
        
        # 三个及以上字符的情况
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    count += 1
        
        return count
    
    def countSubstrings_optimized(self, s: str) -> int:
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
        dp = [False] * n
        count = 0
        
        # 单个字符都是回文
        for i in range(n):
            dp[i] = True
            count += 1
        
        # 两个字符的情况
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i] = True
                count += 1
        
        # 三个及以上字符的情况
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j] and dp[i + 1]:
                    dp[i] = True
                    count += 1
        
        return count
    
    def countSubstrings_recursive(self, s: str) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的回文子串数量
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
                if i + 1 > j - 1:
                    result = 1
                else:
                    result = dfs(i + 1, j - 1) + 1
            else:
                result = dfs(i + 1, j) + dfs(i, j - 1) - dfs(i + 1, j - 1)
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, n - 1)
    
    def countSubstrings_brute_force(self, s: str) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的子串
        2. 检查每个子串是否为回文
        3. 统计回文子串的数量
        
        时间复杂度：O(n^3)
        空间复杂度：O(1)
        """
        if not s:
            return 0
        
        n = len(s)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                if self.is_palindrome(s, i, j):
                    count += 1
        
        return count
    
    def is_palindrome(self, s: str, start: int, end: int) -> bool:
        """检查子串是否为回文"""
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True
    
    def countSubstrings_alternative(self, s: str) -> int:
        """
        替代解法：使用中心扩展法
        
        解题思路：
        1. 遍历所有可能的回文中心
        2. 从中心向两边扩展，检查是否为回文
        3. 考虑奇数长度和偶数长度的回文
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        if not s:
            return 0
        
        n = len(s)
        count = 0
        
        for i in range(n):
            # 奇数长度的回文
            left = right = i
            while left >= 0 and right < n and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
            
            # 偶数长度的回文
            left, right = i, i + 1
            while left >= 0 and right < n and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
        
        return count
    
    def countSubstrings_dfs(self, s: str) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的子串
        2. 检查每个子串是否为回文
        
        时间复杂度：O(n^3)
        空间复杂度：O(n)
        """
        if not s:
            return 0
        
        n = len(s)
        count = 0
        
        def dfs(i, j):
            nonlocal count
            
            if i > j:
                return
            
            if self.is_palindrome(s, i, j):
                count += 1
            
            dfs(i + 1, j)
            dfs(i, j - 1)
        
        dfs(0, n - 1)
        return count
    
    def countSubstrings_memo(self, s: str) -> int:
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
                if i + 1 > j - 1:
                    result = 1
                else:
                    result = dfs(i + 1, j - 1) + 1
            else:
                result = dfs(i + 1, j) + dfs(i, j - 1) - dfs(i + 1, j - 1)
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, n - 1)
    
    def countSubstrings_greedy(self, s: str) -> int:
        """
        贪心解法：优先选择长回文
        
        解题思路：
        1. 优先选择较长的回文子串
        2. 贪心地选择最优解
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        if not s:
            return 0
        
        n = len(s)
        count = 0
        
        for i in range(n):
            # 奇数长度的回文
            left = right = i
            while left >= 0 and right < n and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
            
            # 偶数长度的回文
            left, right = i, i + 1
            while left >= 0 and right < n and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
        
        return count
    
    def countSubstrings_iterative(self, s: str) -> int:
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
                memo[(i, j)] = memo[(i + 1, j - 1)] + 1
            else:
                memo[(i, j)] = memo[(i + 1, j)] + memo[(i, j - 1)] - memo[(i + 1, j - 1)]
        
        return memo[(0, n - 1)]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "abc"
    result = solution.countSubstrings(s)
    expected = 3
    assert result == expected
    
    # 测试用例2
    s = "aaa"
    result = solution.countSubstrings(s)
    expected = 6
    assert result == expected
    
    # 测试用例3
    s = "a"
    result = solution.countSubstrings(s)
    expected = 1
    assert result == expected
    
    # 测试用例4
    s = "ab"
    result = solution.countSubstrings(s)
    expected = 2
    assert result == expected
    
    # 测试用例5
    s = "racecar"
    result = solution.countSubstrings(s)
    expected = 10
    assert result == expected
    
    # 测试DP解法
    print("测试DP解法...")
    s = "abc"
    result_dp = solution.countSubstrings_dp(s)
    assert result_dp == expected
    
    # 测试优化解法
    print("测试优化解法...")
    s = "abc"
    result_opt = solution.countSubstrings_optimized(s)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    s = "abc"
    result_rec = solution.countSubstrings_recursive(s)
    assert result_rec == expected
    
    # 测试替代解法
    print("测试替代解法...")
    s = "abc"
    result_alt = solution.countSubstrings_alternative(s)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    s = "abc"
    result_dfs = solution.countSubstrings_dfs(s)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    s = "abc"
    result_memo = solution.countSubstrings_memo(s)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    s = "abc"
    result_greedy = solution.countSubstrings_greedy(s)
    assert result_greedy == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    s = "abc"
    result_iter = solution.countSubstrings_iterative(s)
    assert result_iter == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
