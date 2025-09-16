"""
LeetCode 84. Largest Rectangle in Histogram

题目描述：
给定n个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为1。
求在该柱状图中，能够勾勒出来的矩形的最大面积。

示例：
heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中阴影区域，面积为10。

数据范围：
- 1 <= heights.length <= 10^5
- 0 <= heights[i] <= 10^4
"""

class Solution:
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack = []
        max_area = 0
        
        for i, h in enumerate(heights):
            # 如果当前高度小于栈顶高度，计算以栈顶为高的矩形面积
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                # 宽度 = 当前索引 - 栈顶索引 - 1
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        # 处理栈中剩余的元素
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area
