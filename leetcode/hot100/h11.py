"""
LeetCode 11. Container With Most Water

题目描述：
给定一个长度为n的整数数组height。有n条垂线，第i条线的两个端点是(i, 0)和(i, height[i])。
找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。

示例：
height = [1,8,6,2,5,4,8,3,7]
输出：49
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为49。

数据范围：
- n == height.length
- 2 <= n <= 10^5
- 0 <= height[i] <= 10^4
"""

class Solution:
    def maxArea(self, height: list[int]) -> int:
        left, right = 0, len(height) - 1
        max_water = 0
        
        while left < right:
            # 计算当前面积
            width = right - left
            current_height = min(height[left], height[right])
            current_area = width * current_height
            max_water = max(max_water, current_area)
            
            # 移动较短的边
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_water
