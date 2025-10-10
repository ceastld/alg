"""
LeetCode 15. 3Sum - 优秀解答

题目描述：
给你一个包含n个整数的数组nums，判断nums中是否存在三个元素a，b，c，使得a + b + c = 0？
请你找出所有和为0且不重复的三元组。

示例：
nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]

数据范围：
- 3 <= nums.length <= 3000
- -10^5 <= nums[i] <= 10^5
"""

from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        优秀解法：排序 + 双指针
        时间复杂度：O(n²)
        空间复杂度：O(1) 不考虑输出空间
        """
        # 1. 排序数组，便于去重和双指针操作
        nums.sort()
        result = []
        n = len(nums)
        
        # 2. 固定第一个数，用双指针找后两个数
        for i in range(n - 2):
            # 跳过重复的第一个数
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            # 3. 双指针找后两个数
            left, right = i + 1, n - 1
            
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                
                if total == 0:
                    # 找到有效三元组
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # 跳过重复的left和right
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    # 移动指针
                    left += 1
                    right -= 1
                    
                elif total < 0:
                    # 和太小，left右移
                    left += 1
                else:
                    # 和太大，right左移
                    right -= 1
        
        return result


# 测试用例
def test_threeSum():
    solution = Solution()
    
    # 测试用例1
    nums1 = [-1, 0, 1, 2, -1, -4]
    result1 = solution.threeSum(nums1)
    print(f"输入: {nums1}")
    print(f"输出: {result1}")
    print(f"期望: [[-1, -1, 2], [-1, 0, 1]]")
    print()
    
    # 测试用例2
    nums2 = [0, 1, 1]
    result2 = solution.threeSum(nums2)
    print(f"输入: {nums2}")
    print(f"输出: {result2}")
    print(f"期望: []")
    print()
    
    # 测试用例3
    nums3 = [0, 0, 0]
    result3 = solution.threeSum(nums3)
    print(f"输入: {nums3}")
    print(f"输出: {result3}")
    print(f"期望: [[0, 0, 0]]")


if __name__ == "__main__":
    test_threeSum()
