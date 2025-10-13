"""
84. 柱状图中最大的矩形
给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

题目链接：https://leetcode.cn/problems/largest-rectangle-in-histogram/

示例 1:
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10

示例 2:
输入：heights = [2,4]
输出：4

提示：
- 1 <= heights.length <= 10^5
- 0 <= heights[i] <= 10^4
"""

from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.largestRectangleArea([2,1,5,6,2,3]) == 10
    
    # 测试用例2
    assert solution.largestRectangleArea([2,4]) == 4
    
    # 测试用例3
    assert solution.largestRectangleArea([1]) == 1
    
    # 测试用例4
    assert solution.largestRectangleArea([1,1]) == 2
    
    # 测试用例5
    assert solution.largestRectangleArea([2,1,2]) == 3
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
