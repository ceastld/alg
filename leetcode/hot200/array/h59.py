"""
59. 螺旋矩阵 II
给你一个正整数 n ，生成一个包含 1 到 n² 所有元素，
且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。

题目链接：https://leetcode.cn/problems/spiral-matrix-ii/

示例 1:
输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]

示例 2:
输入：n = 1
输出：[[1]]

提示：
- 1 <= n <= 20
"""
from typing import List


class Solution:
    """
    59. 螺旋矩阵 II
    模拟题，按层填充
    """
    
    def generateMatrix(self, n: int) -> List[List[int]]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        matrix = [[0] * n for _ in range(n)]
        num = 1
        top, bottom = 0, n - 1
        left, right = 0, n - 1
        while top <= bottom and left <= right:
            for col in range(left, right + 1):
                matrix[top][col] = num
                num += 1
            top += 1
            for row in range(top, bottom + 1):
                matrix[row][right] = num
                num += 1
            right -= 1
            for col in range(right, left - 1, -1):
                matrix[bottom][col] = num
                num += 1
            bottom -= 1
            for row in range(bottom, top - 1, -1):
                matrix[row][left] = num
                num += 1
            left += 1
        return matrix


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.generateMatrix(3) == [[1, 2, 3], [8, 9, 4], [7, 6, 5]]
    
    # 测试用例2
    assert solution.generateMatrix(1) == [[1]]
    
    # 测试用例3
    assert solution.generateMatrix(2) == [[1, 2], [4, 3]]
    
    # 测试用例4
    assert solution.generateMatrix(4) == [[1, 2, 3, 4], [12, 13, 14, 5], [11, 16, 15, 6], [10, 9, 8, 7]]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
