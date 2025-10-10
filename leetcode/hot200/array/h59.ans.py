"""
59. 螺旋矩阵 II - 标准答案
"""
from typing import List


class Solution:
    """
    59. 螺旋矩阵 II - 标准解法
    """
    
    def generateMatrix(self, n: int) -> List[List[int]]:
        """
        标准解法：按层填充
        
        解题思路：
        1. 按层从外到内填充
        2. 每层按上、右、下、左的顺序填充
        3. 维护四个边界：top, bottom, left, right
        4. 使用num变量记录当前要填充的数字
        
        时间复杂度：O(n²)
        空间复杂度：O(1) 不考虑输出矩阵
        """
        matrix = [[0] * n for _ in range(n)]
        num = 1
        top, bottom = 0, n - 1
        left, right = 0, n - 1
        
        while top <= bottom and left <= right:
            # 上边：从左到右
            for col in range(left, right + 1):
                matrix[top][col] = num
                num += 1
            top += 1
            
            # 右边：从上到下
            for row in range(top, bottom + 1):
                matrix[row][right] = num
                num += 1
            right -= 1
            
            # 下边：从右到左
            for col in range(right, left - 1, -1):
                matrix[bottom][col] = num
                num += 1
            bottom -= 1
            
            # 左边：从下到上
            for row in range(bottom, top - 1, -1):
                matrix[row][left] = num
                num += 1
            left += 1
        
        return matrix


def main():
    """测试标准答案"""
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
