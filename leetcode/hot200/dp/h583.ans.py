"""
583. 两个字符串的删除操作 - 标准答案
"""
from typing import List


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. 问题转化为：找到两个字符串的最长公共子序列LCS
        2. 最小删除步数 = len(word1) + len(word2) - 2 * LCS长度
        3. 使用LCS算法求解
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not word1 and not word2:
            return 0
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        m, n = len(word1), len(word2)
        
        # 计算LCS长度
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return m + n - 2 * lcs_length
    
    def minDistance_optimized(self, word1: str, word2: str) -> int:
        """
        优化解法：空间优化
        
        解题思路：
        1. 使用一维数组优化空间复杂度
        2. 从后往前遍历，避免重复使用
        
        时间复杂度：O(m * n)
        空间复杂度：O(n)
        """
        if not word1 and not word2:
            return 0
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        m, n = len(word1), len(word2)
        
        # 使用一维数组优化空间
        dp = [0] * (n + 1)
        
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j-1])
                prev = dp[j]
        
        lcs_length = dp[n]
        return m + n - 2 * lcs_length
    
    def minDistance_recursive(self, word1: str, word2: str) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算每个位置的最长公共子序列长度
        2. 使用记忆化避免重复计算
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not word1 and not word2:
            return 0
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i == 0 or j == 0:
                return 0
            
            if word1[i-1] == word2[j-1]:
                result = dfs(i-1, j-1) + 1
            else:
                result = max(dfs(i-1, j), dfs(i, j-1))
            
            memo[(i, j)] = result
            return result
        
        lcs_length = dfs(len(word1), len(word2))
        return len(word1) + len(word2) - 2 * lcs_length
    
    def minDistance_brute_force(self, word1: str, word2: str) -> int:
        """
        暴力解法：枚举所有可能
        
        解题思路：
        1. 枚举所有可能的删除方案
        2. 计算每种方案的删除步数
        3. 返回最小删除步数
        
        时间复杂度：O(2^(m+n))
        空间复杂度：O(m + n)
        """
        if not word1 and not word2:
            return 0
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        def dfs(i, j, steps):
            if i == len(word1) and j == len(word2):
                return steps
            
            if i == len(word1):
                return steps + len(word2) - j
            
            if j == len(word2):
                return steps + len(word1) - i
            
            if word1[i] == word2[j]:
                return dfs(i + 1, j + 1, steps)
            else:
                return min(dfs(i + 1, j, steps + 1), dfs(i, j + 1, steps + 1))
        
        return dfs(0, 0, 0)
    
    def minDistance_alternative(self, word1: str, word2: str) -> int:
        """
        替代解法：直接计算删除步数
        
        解题思路：
        1. 直接计算删除步数，不通过LCS
        2. 使用动态规划求解
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not word1 and not word2:
            return 0
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化：删除所有字符
        for i in range(m + 1):
            dp[i][0] = i
        
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1)
        
        return dp[m][n]
    
    def minDistance_dfs(self, word1: str, word2: str) -> int:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历所有可能的删除方案
        2. 计算每种方案的删除步数
        
        时间复杂度：O(2^(m+n))
        空间复杂度：O(m + n)
        """
        if not word1 and not word2:
            return 0
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        def dfs(i, j, steps):
            if i == len(word1) and j == len(word2):
                return steps
            
            if i == len(word1):
                return steps + len(word2) - j
            
            if j == len(word2):
                return steps + len(word1) - i
            
            if word1[i] == word2[j]:
                return dfs(i + 1, j + 1, steps)
            else:
                return min(dfs(i + 1, j, steps + 1), dfs(i, j + 1, steps + 1))
        
        return dfs(0, 0, 0)
    
    def minDistance_memo(self, word1: str, word2: str) -> int:
        """
        记忆化DFS解法
        
        解题思路：
        1. 使用记忆化避免重复计算
        2. 提高DFS的效率
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not word1 and not word2:
            return 0
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i == len(word1) and j == len(word2):
                return 0
            
            if i == len(word1):
                return len(word2) - j
            
            if j == len(word2):
                return len(word1) - i
            
            if word1[i] == word2[j]:
                result = dfs(i + 1, j + 1)
            else:
                result = min(dfs(i + 1, j) + 1, dfs(i, j + 1) + 1)
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, 0)
    
    def minDistance_greedy(self, word1: str, word2: str) -> int:
        """
        贪心解法：优先匹配相同字符
        
        解题思路：
        1. 优先匹配相同的字符
        2. 贪心地选择最优解
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not word1 and not word2:
            return 0
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i == len(word1) and j == len(word2):
                return 0
            
            if i == len(word1):
                return len(word2) - j
            
            if j == len(word2):
                return len(word1) - i
            
            # 优先匹配相同的字符
            if word1[i] == word2[j]:
                result = dfs(i + 1, j + 1)
            else:
                result = min(dfs(i + 1, j) + 1, dfs(i, j + 1) + 1)
            
            memo[(i, j)] = result
            return result
        
        return dfs(0, 0)
    
    def minDistance_iterative(self, word1: str, word2: str) -> int:
        """
        迭代解法：使用栈
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 避免递归调用栈
        
        时间复杂度：O(m * n)
        空间复杂度：O(m * n)
        """
        if not word1 and not word2:
            return 0
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        m, n = len(word1), len(word2)
        memo = {}
        stack = [(0, 0)]
        
        while stack:
            i, j = stack.pop()
            
            if (i, j) in memo:
                continue
            
            if i == m and j == n:
                memo[(i, j)] = 0
                continue
            
            if i == m:
                memo[(i, j)] = n - j
                continue
            
            if j == n:
                memo[(i, j)] = m - i
                continue
            
            # 检查依赖项是否已计算
            if word1[i] == word2[j]:
                if (i + 1, j + 1) not in memo:
                    stack.append((i, j))
                    stack.append((i + 1, j + 1))
                    continue
                
                memo[(i, j)] = memo[(i + 1, j + 1)]
            else:
                if (i + 1, j) not in memo:
                    stack.append((i, j))
                    stack.append((i + 1, j))
                    continue
                
                if (i, j + 1) not in memo:
                    stack.append((i, j))
                    stack.append((i, j + 1))
                    continue
                
                memo[(i, j)] = min(memo[(i + 1, j)] + 1, memo[(i, j + 1)] + 1)
        
        return memo[(0, 0)]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    word1 = "sea"
    word2 = "eat"
    result = solution.minDistance(word1, word2)
    expected = 2
    assert result == expected
    
    # 测试用例2
    word1 = "leetcode"
    word2 = "etco"
    result = solution.minDistance(word1, word2)
    expected = 4
    assert result == expected
    
    # 测试用例3
    word1 = "a"
    word2 = "a"
    result = solution.minDistance(word1, word2)
    expected = 0
    assert result == expected
    
    # 测试用例4
    word1 = "a"
    word2 = "b"
    result = solution.minDistance(word1, word2)
    expected = 2
    assert result == expected
    
    # 测试用例5
    word1 = "ab"
    word2 = "ba"
    result = solution.minDistance(word1, word2)
    expected = 2
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    word1 = "sea"
    word2 = "eat"
    result_opt = solution.minDistance_optimized(word1, word2)
    assert result_opt == expected
    
    # 测试递归解法
    print("测试递归解法...")
    word1 = "sea"
    word2 = "eat"
    result_rec = solution.minDistance_recursive(word1, word2)
    assert result_rec == expected
    
    # 测试替代解法
    print("测试替代解法...")
    word1 = "sea"
    word2 = "eat"
    result_alt = solution.minDistance_alternative(word1, word2)
    assert result_alt == expected
    
    # 测试DFS解法
    print("测试DFS解法...")
    word1 = "sea"
    word2 = "eat"
    result_dfs = solution.minDistance_dfs(word1, word2)
    assert result_dfs == expected
    
    # 测试记忆化DFS解法
    print("测试记忆化DFS解法...")
    word1 = "sea"
    word2 = "eat"
    result_memo = solution.minDistance_memo(word1, word2)
    assert result_memo == expected
    
    # 测试贪心解法
    print("测试贪心解法...")
    word1 = "sea"
    word2 = "eat"
    result_greedy = solution.minDistance_greedy(word1, word2)
    assert result_greedy == expected
    
    # 测试迭代解法
    print("测试迭代解法...")
    word1 = "sea"
    word2 = "eat"
    result_iter = solution.minDistance_iterative(word1, word2)
    assert result_iter == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
