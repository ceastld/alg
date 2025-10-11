"""
15. 三数之和 - 标准答案
"""
from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        标准解法：排序 + 双指针
        
        解题思路：
        1. 先对数组排序
        2. 固定第一个数，用双指针找另外两个数
        3. 去重：跳过重复的元素
        4. 剪枝：如果第一个数大于0，直接返回
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        nums.sort()
        result = []
        n = len(nums)
        
        for i in range(n - 2):
            # 剪枝：如果第一个数大于0，直接返回
            if nums[i] > 0:
                break
            
            # 去重：跳过重复的第一个数
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            left, right = i + 1, n - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    # 去重：跳过重复的left和right
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [-1, 0, 1, 2, -1, -4]
    result = solution.threeSum(nums)
    expected = [[-1, -1, 2], [-1, 0, 1]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = []
    result = solution.threeSum(nums)
    assert result == []
    
    # 测试用例3
    nums = [0]
    result = solution.threeSum(nums)
    assert result == []
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
