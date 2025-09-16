"""
LeetCode 48. Rotate Image

题目描述：
给定一个n×n的二维矩阵matrix表示一个图像。请你将图像顺时针旋转90度。
你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

示例：
matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]

数据范围：
- n == matrix.length == matrix[i].length
- 1 <= n <= 20
- -1000 <= matrix[i][j] <= 1000
"""

class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        n = len(matrix)
        
        # 转置矩阵
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # 翻转每一行
        for i in range(n):
            matrix[i].reverse()
