"""
LeetCode 72. Edit Distance

题目描述：
给你两个单词word1和word2，请返回将word1转换成word2所使用的最少操作数。
你可以对一个单词进行如下三种操作：
- 插入一个字符
- 删除一个字符
- 替换一个字符

示例：
word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将'h'替换为'r')
rorse -> rose (删除'r')
rose -> ros (删除'e')

数据范围：
- 0 <= word1.length, word2.length <= 500
- word1和word2由小写英文字母组成
"""

class Solution:
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word1), len(word2)
        
        # dp[i][j] 表示word1前i个字符转换为word2前j个字符的最少操作数
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化边界条件
        for i in range(m + 1):
            dp[i][0] = i  # word1前i个字符转换为空字符串需要i次删除操作
        for j in range(n + 1):
            dp[0][j] = j  # 空字符串转换为word2前j个字符需要j次插入操作
        
        # 填充dp表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    # 字符相同，不需要操作
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # 字符不同，取三种操作的最小值
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # 删除word1[i-1]
                        dp[i][j-1],    # 插入word2[j-1]
                        dp[i-1][j-1]   # 替换word1[i-1]为word2[j-1]
                    )
        
        return dp[m][n]
