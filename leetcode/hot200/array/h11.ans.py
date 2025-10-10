"""
11. 盛最多水的容器 - 标准答案
"""
from typing import List


class Solution:
    """
    11. 盛最多水的容器 - 标准解法
    """
    
    def maxArea(self, height: List[int]) -> int:
        """
        标准解法：双指针法
        
        解题思路：
        1. 使用双指针从两端向中间移动
        2. 每次移动较短的指针，因为移动较长的指针不会增加面积
        3. 记录过程中的最大面积
        4. 面积 = min(height[left], height[right]) * (right - left)
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        left, right = 0, len(height) - 1
        max_area = 0
        
        while left < right:
            # 计算当前面积
            current_area = min(height[left], height[right]) * (right - left)
            max_area = max(max_area, current_area)
            
            # 移动较短的指针
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area


def main():
    """测试标准答案"""
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
