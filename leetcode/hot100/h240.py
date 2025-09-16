"""
LeetCode 240. Search a 2D Matrix II

题目描述：
编写一个高效的算法来搜索m×n矩阵matrix中的一个目标值target。该矩阵具有以下特性：
- 每行的元素从左到右升序排列
- 每列的元素从上到下升序排列

示例：
matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true

数据范围：
- m == matrix.length
- n == matrix[i].length
- 1 <= n, m <= 300
- -10^9 <= matrix[i][j] <= 10^9
- 每行的所有元素从左到右升序排列
- 每列的所有元素从上到下升序排列
- -10^9 <= target <= 10^9
"""

class Solution:
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        
        row, col = 0, len(matrix[0]) - 1
        
        while row < len(matrix) and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        
        return False
