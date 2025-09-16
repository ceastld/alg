"""
LeetCode 221. Maximal Square

题目描述：
在一个由'0'和'1'组成的二维矩阵内，找到只包含'1'的最大正方形，并返回其面积。

示例：
matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：4

数据范围：
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 300
- matrix[i][j]为'0'或'1'
"""

class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        dp = [[0] * len(matrix[0]) for _ in range(len(matrix))]
        for i in range(len(matrix)):
            dp[i][0] = int(matrix[i][0])
        for j in range(len(matrix[0])):
            dp[0][j] = int(matrix[0][j])
        for i in range(1, len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[i][j] == "1":
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return max(max(row) for row in dp) ** 2
