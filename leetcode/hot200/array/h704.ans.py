"""
704. 二分查找 - 标准答案
"""
from typing import List


class Solution:
    """
    704. 二分查找 - 标准解法
    """
    
    def search(self, nums: List[int], target: int) -> int:
        """
        标准解法：经典二分查找
        
        解题思路：
        1. 使用双指针维护搜索区间 [left, right]
        2. 每次取中间位置 mid = (left + right) // 2
        3. 比较 nums[mid] 与 target：
           - 相等：找到目标，返回 mid
           - nums[mid] < target：目标在右半部分，left = mid + 1
           - nums[mid] > target：目标在左半部分，right = mid - 1
        4. 当 left > right 时，搜索结束，返回 -1
        
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.search([-1, 0, 3, 5, 9, 12], 9) == 4
    
    # 测试用例2
    assert solution.search([-1, 0, 3, 5, 9, 12], 2) == -1
    
    # 测试用例3
    assert solution.search([5], 5) == 0
    
    # 测试用例4
    assert solution.search([5], -5) == -1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
