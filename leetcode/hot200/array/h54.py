"""
54. 螺旋矩阵
给你一个 m 行 n 列的矩阵 matrix ，请按照顺时针螺旋顺序，返回矩阵中的所有元素。

题目链接：https://leetcode.cn/problems/spiral-matrix/

示例 1:
输入: matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出: [1,2,3,6,9,8,7,4,5]

示例 2:
输入: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]

提示：
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 10
- -100 <= matrix[i][j] <= 100
"""
from typing import List

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        标准解法：模拟 + 边界控制
        
        解题思路：
        1. 定义四个边界：top, bottom, left, right
        2. 按照右->下->左->上的顺序遍历
        3. 每遍历完一行或一列后，更新对应的边界
        4. 当边界相遇时停止遍历
        
        时间复杂度：O(m*n)
        空间复杂度：O(1)
        """
        if not matrix or not matrix[0]:
            return []
        
        result = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        
        while top <= bottom and left <= right:
            # 从左到右
            for j in range(left, right + 1):
                result.append(matrix[top][j])
            top += 1
            
            # 从上到下
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1
            
            # 从右到左
            if top <= bottom:
                for j in range(right, left - 1, -1):
                    result.append(matrix[bottom][j])
                bottom -= 1
            
            # 从下到上
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1
        
        return result

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected = [1, 2, 3, 6, 9, 8, 7, 4, 5]
    assert solution.spiralOrder(matrix) == expected
    
    # 测试用例2
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    expected = [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
    assert solution.spiralOrder(matrix) == expected
    
    # 测试用例3
    matrix = [[1]]
    assert solution.spiralOrder(matrix) == [1]
    
    # 测试用例4
    matrix = [[1, 2], [3, 4]]
    assert solution.spiralOrder(matrix) == [1, 2, 4, 3]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
