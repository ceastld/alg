"""
11. 盛最多水的容器
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

说明：你不能倾斜容器。

题目链接：https://leetcode.cn/problems/container-with-most-water/

示例 1:
输入：height = [1,8,6,2,5,4,8,3,7]
输出：49
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

示例 2:
输入：height = [1,1]
输出：1

提示：
- n == height.length
- 2 <= n <= 105
- 0 <= height[i] <= 104
"""
from typing import List


class Solution:
    """
    11. 盛最多水的容器
    双指针经典题目
    """
    
    def maxArea(self, height: List[int]) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        left = 0
        right = len(height) - 1
        max_area = 0
        
        # 缓存当前左右指针的高度，用于剪枝优化
        left_height = height[left]
        right_height = height[right]
        
        while left < right:
            # 计算当前面积：面积 = 较短边的高度 × 宽度
            area = min(height[left], height[right]) * (right - left)
            max_area = max(max_area, area)
            
            # 关键优化：移动较短的指针
            if height[left] < height[right]:
                left += 1
                # 剪枝优化：跳过所有高度小于等于当前左指针高度的位置
                # 因为这些位置不可能产生更大的面积
                while left < right and height[left] <= left_height:
                    left += 1
                # 更新左指针高度缓存
                left_height = height[left]
            else:
                right -= 1
                # 剪枝优化：跳过所有高度小于等于当前右指针高度的位置
                # 因为这些位置不可能产生更大的面积
                while left < right and height[right] <= right_height:
                    right -= 1
                # 更新右指针高度缓存
                right_height = height[right]
                
        return max_area


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    
    # 测试用例2
    assert solution.maxArea([1, 1]) == 1
    
    # 测试用例3
    assert solution.maxArea([4, 3, 2, 1, 4]) == 16
    
    # 测试用例4
    assert solution.maxArea([1, 2, 1]) == 2
    
    # 测试用例5
    assert solution.maxArea([2, 3, 4, 5, 18, 17, 6]) == 17
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
