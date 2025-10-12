"""
1143. 最长公共子序列 - 标准答案
"""
from typing import List


class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """
        标准解法：二维动态规划
        
        解题思路：
        1. dp[i][j] 表示text1[0:i]和text2[0:j]的最长公共子序列长度
        2. 状态转移方程：
           - 如果text1[i-1] == text2[j-1]，则dp[i][j] = dp[i-1][j-1] + 1
           - 否则dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        3. 边界条件：dp[0][j] = 0, dp[i][0] = 0
        
        时间复杂度：O(m×n)
        空间复杂度：O(m×n)
        """
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def longestCommonSubsequence_optimized(self, text1: str, text2: str) -> int:
        """
        空间优化解法：滚动数组
        
        解题思路：
        1. 使用一维数组代替二维数组
        2. 每次只保存当前行的状态
        3. 空间复杂度从O(m×n)优化到O(min(m,n))
        
        时间复杂度：O(m×n)
        空间复杂度：O(min(m,n))
        """
        # 选择较短的字符串作为列，优化空间
        if len(text1) < len(text2):
            text1, text2 = text2, text1
        
        m, n = len(text1), len(text2)
        dp = [0] * (n + 1)
        
        for i in range(1, m + 1):
            prev = 0  # 保存dp[i-1][j-1]的值
            for j in range(1, n + 1):
                temp = dp[j]  # 保存dp[i-1][j]的值
                if text1[i-1] == text2[j-1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j-1])
                prev = temp  # 更新prev为dp[i-1][j]
        
        return dp[n]
    
    def longestCommonSubsequence_recursive(self, text1: str, text2: str) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算，使用记忆化避免重复计算
        2. 自顶向下的动态规划
        
        时间复杂度：O(m×n)
        空间复杂度：O(m×n)
        """
        memo = {}
        
        def dfs(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i < 0 or j < 0:
                return 0
            
            if text1[i] == text2[j]:
                memo[(i, j)] = dfs(i-1, j-1) + 1
            else:
                memo[(i, j)] = max(dfs(i-1, j), dfs(i, j-1))
            
            return memo[(i, j)]
        
        return dfs(len(text1)-1, len(text2)-1)
    
    def longestCommonSubsequence_brute_force(self, text1: str, text2: str) -> int:
        """
        暴力解法：枚举所有子序列
        
        解题思路：
        1. 枚举text1的所有子序列
        2. 检查每个子序列是否在text2中
        3. 返回最长公共子序列的长度
        
        时间复杂度：O(2^m × n)
        空间复杂度：O(m)
        """
        def get_subsequences(text):
            """获取字符串的所有子序列"""
            subsequences = []
            n = len(text)
            
            for mask in range(1, 1 << n):
                subsequence = []
                for i in range(n):
                    if mask & (1 << i):
                        subsequence.append(text[i])
                subsequences.append(''.join(subsequence))
            
            return subsequences
        
        # 获取text1的所有子序列
        subsequences1 = get_subsequences(text1)
        
        max_length = 0
        for subseq in subsequences1:
            if self.is_subsequence(subseq, text2):
                max_length = max(max_length, len(subseq))
        
        return max_length
    
    def is_subsequence(self, s: str, t: str) -> bool:
        """判断s是否是t的子序列"""
        i = j = 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == len(s)
    
    def longestCommonSubsequence_optimized_v2(self, text1: str, text2: str) -> int:
        """
        进一步优化：使用更少的内存
        
        解题思路：
        1. 使用两个一维数组交替更新
        2. 进一步减少内存使用
        
        时间复杂度：O(m×n)
        空间复杂度：O(min(m,n))
        """
        if len(text1) < len(text2):
            text1, text2 = text2, text1
        
        m, n = len(text1), len(text2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev
        
        return prev[n]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    text1 = "abcde"
    text2 = "ace"
    result = solution.longestCommonSubsequence(text1, text2)
    expected = 3
    assert result == expected
    
    # 测试用例2
    text1 = "abc"
    text2 = "abc"
    result = solution.longestCommonSubsequence(text1, text2)
    expected = 3
    assert result == expected
    
    # 测试用例3
    text1 = "abc"
    text2 = "def"
    result = solution.longestCommonSubsequence(text1, text2)
    expected = 0
    assert result == expected
    
    # 测试用例4
    text1 = "bsbininm"
    text2 = "jmjkbkjkv"
    result = solution.longestCommonSubsequence(text1, text2)
    expected = 1
    assert result == expected
    
    # 测试用例5
    text1 = "oxcpqrsvwf"
    text2 = "shmtulqrypy"
    result = solution.longestCommonSubsequence(text1, text2)
    expected = 2
    assert result == expected
    
    # 测试空间优化解法
    print("测试空间优化解法...")
    text1 = "abcde"
    text2 = "ace"
    result_opt = solution.longestCommonSubsequence_optimized(text1, text2)
    expected_opt = 3
    assert result_opt == expected_opt
    
    # 测试递归解法
    print("测试递归解法...")
    text1 = "abcde"
    text2 = "ace"
    result_rec = solution.longestCommonSubsequence_recursive(text1, text2)
    expected_rec = 3
    assert result_rec == expected_rec
    
    # 测试暴力解法
    print("测试暴力解法...")
    text1 = "abc"
    text2 = "abc"
    result_bf = solution.longestCommonSubsequence_brute_force(text1, text2)
    expected_bf = 3
    assert result_bf == expected_bf
    
    # 测试进一步优化解法
    print("测试进一步优化解法...")
    text1 = "abcde"
    text2 = "ace"
    result_opt2 = solution.longestCommonSubsequence_optimized_v2(text1, text2)
    expected_opt2 = 3
    assert result_opt2 == expected_opt2
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
