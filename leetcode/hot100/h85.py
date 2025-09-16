"""
LeetCode 85. Maximal Rectangle

题目描述：
给定一个仅包含0和1的二维二进制矩阵，找出只包含1的最大矩形，并返回其面积。

示例：
matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。

数据范围：
- rows == matrix.length
- cols == matrix[0].length
- 1 <= row, cols <= 200
- matrix[i][j]为'0'或'1'
"""

class Solution:
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        heights = [0] * cols
        max_area = 0
        
        for i in range(rows):
            # 更新高度数组
            for j in range(cols):
                if matrix[i][j] == '1':
                    heights[j] += 1
                else:
                    heights[j] = 0
            
            # 计算当前行的最大矩形面积
            max_area = max(max_area, self.largestRectangleArea(heights))
        
        return max_area
    
    def largestRectangleArea(self, heights):
        """LeetCode 84的解法"""
        stack = []
        max_area = 0
        
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area
