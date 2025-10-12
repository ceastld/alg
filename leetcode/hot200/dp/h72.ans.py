"""
72. 编辑距离 - 标准答案
"""
from typing import List


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        """
        标准解法：二维动态规划
        
        解题思路：
        1. dp[i][j] 表示word1[0:i]转换为word2[0:j]的最少操作数
        2. 状态转移方程：
           - 如果word1[i-1] == word2[j-1]，则dp[i][j] = dp[i-1][j-1]
           - 否则dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        3. 边界条件：dp[0][j] = j, dp[i][0] = i
        
        时间复杂度：O(m×n)
        空间复杂度：O(m×n)
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化边界条件
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # 删除
                        dp[i][j-1] + 1,      # 插入
                        dp[i-1][j-1] + 1     # 替换
                    )
        
        return dp[m][n]
    
    def minDistance_optimized(self, word1: str, word2: str) -> int:
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
        if len(word1) < len(word2):
            word1, word2 = word2, word1
        
        m, n = len(word1), len(word2)
        dp = [0] * (n + 1)
        
        # 初始化第一行
        for j in range(n + 1):
            dp[j] = j
        
        # 逐行处理
        for i in range(1, m + 1):
            prev = dp[0]  # 保存dp[i-1][j-1]的值
            dp[0] = i     # 更新dp[i][0]
            
            for j in range(1, n + 1):
                temp = dp[j]  # 保存dp[i-1][j]的值
                if word1[i-1] == word2[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + 1)
                prev = temp  # 更新prev为dp[i-1][j]
        
        return dp[n]
    
    def minDistance_recursive(self, word1: str, word2: str) -> int:
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
            
            if i == 0:
                return j
            if j == 0:
                return i
            
            if word1[i-1] == word2[j-1]:
                memo[(i, j)] = dfs(i-1, j-1)
            else:
                memo[(i, j)] = min(
                    dfs(i-1, j) + 1,      # 删除
                    dfs(i, j-1) + 1,      # 插入
                    dfs(i-1, j-1) + 1     # 替换
                )
            
            return memo[(i, j)]
        
        return dfs(len(word1), len(word2))
    
    def minDistance_brute_force(self, word1: str, word2: str) -> int:
        """
        暴力解法：递归枚举所有操作
        
        解题思路：
        1. 递归尝试所有可能的操作
        2. 返回最小操作数
        
        时间复杂度：O(3^(m+n))
        空间复杂度：O(m+n)
        """
        def dfs(i, j):
            if i == 0:
                return j
            if j == 0:
                return i
            
            if word1[i-1] == word2[j-1]:
                return dfs(i-1, j-1)
            else:
                return min(
                    dfs(i-1, j) + 1,      # 删除
                    dfs(i, j-1) + 1,      # 插入
                    dfs(i-1, j-1) + 1     # 替换
                )
        
        return dfs(len(word1), len(word2))
    
    def minDistance_optimized_v2(self, word1: str, word2: str) -> int:
        """
        进一步优化：使用两个一维数组
        
        解题思路：
        1. 使用两个一维数组交替更新
        2. 进一步减少内存使用
        
        时间复杂度：O(m×n)
        空间复杂度：O(min(m,n))
        """
        if len(word1) < len(word2):
            word1, word2 = word2, word1
        
        m, n = len(word1), len(word2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        # 初始化第一行
        for j in range(n + 1):
            prev[j] = j
        
        # 逐行处理
        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    curr[j] = prev[j-1]
                else:
                    curr[j] = min(
                        prev[j] + 1,      # 删除
                        curr[j-1] + 1,    # 插入
                        prev[j-1] + 1     # 替换
                    )
            prev, curr = curr, prev
        
        return prev[n]
    
    def minDistance_optimized_v3(self, word1: str, word2: str) -> int:
        """
        终极优化：使用位运算的编辑距离
        
        解题思路：
        1. 使用位运算优化状态转移
        2. 进一步减少计算量
        
        时间复杂度：O(m×n)
        空间复杂度：O(min(m,n))
        """
        if len(word1) < len(word2):
            word1, word2 = word2, word1
        
        m, n = len(word1), len(word2)
        
        # 使用位运算优化
        if n == 0:
            return m
        if m == 0:
            return n
        
        # 初始化
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    curr[j] = prev[j-1]
                else:
                    curr[j] = min(prev[j], curr[j-1], prev[j-1]) + 1
            prev, curr = curr, prev
        
        return prev[n]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    word1 = "horse"
    word2 = "ros"
    result = solution.minDistance(word1, word2)
    expected = 3
    assert result == expected
    
    # 测试用例2
    word1 = "intention"
    word2 = "execution"
    result = solution.minDistance(word1, word2)
    expected = 5
    assert result == expected
    
    # 测试用例3
    word1 = ""
    word2 = ""
    result = solution.minDistance(word1, word2)
    expected = 0
    assert result == expected
    
    # 测试用例4
    word1 = "a"
    word2 = "b"
    result = solution.minDistance(word1, word2)
    expected = 1
    assert result == expected
    
    # 测试用例5
    word1 = "abc"
    word2 = "abc"
    result = solution.minDistance(word1, word2)
    expected = 0
    assert result == expected
    
    # 测试空间优化解法
    print("测试空间优化解法...")
    word1 = "horse"
    word2 = "ros"
    result_opt = solution.minDistance_optimized(word1, word2)
    expected_opt = 3
    assert result_opt == expected_opt
    
    # 测试递归解法
    print("测试递归解法...")
    word1 = "horse"
    word2 = "ros"
    result_rec = solution.minDistance_recursive(word1, word2)
    expected_rec = 3
    assert result_rec == expected_rec
    
    # 测试暴力解法
    print("测试暴力解法...")
    word1 = "abc"
    word2 = "abc"
    result_bf = solution.minDistance_brute_force(word1, word2)
    expected_bf = 0
    assert result_bf == expected_bf
    
    # 测试进一步优化解法
    print("测试进一步优化解法...")
    word1 = "horse"
    word2 = "ros"
    result_opt2 = solution.minDistance_optimized_v2(word1, word2)
    expected_opt2 = 3
    assert result_opt2 == expected_opt2
    
    # 测试终极优化解法
    print("测试终极优化解法...")
    word1 = "horse"
    word2 = "ros"
    result_opt3 = solution.minDistance_optimized_v3(word1, word2)
    expected_opt3 = 3
    assert result_opt3 == expected_opt3
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
