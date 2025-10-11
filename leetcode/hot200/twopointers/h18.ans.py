"""
18. 四数之和 - 标准答案
"""
from typing import List


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        标准解法：排序 + 双指针
        
        解题思路：
        1. 先对数组排序
        2. 固定前两个数，用双指针找后两个数
        3. 去重：跳过重复的元素
        4. 剪枝：如果前两个数之和已经大于target，直接跳过
        
        时间复杂度：O(n^3)
        空间复杂度：O(1)
        """
        nums.sort()
        result = []
        n = len(nums)
        
        for i in range(n - 3):
            # 剪枝：如果第一个数已经大于target，直接跳过
            if nums[i] > target and nums[i] > 0:
                break
            
            # 去重：跳过重复的第一个数
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            for j in range(i + 1, n - 2):
                # 剪枝：如果前两个数之和已经大于target，直接跳过
                if nums[i] + nums[j] > target and nums[i] + nums[j] > 0:
                    break
                
                # 去重：跳过重复的第二个数
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                
                left, right = j + 1, n - 1
                while left < right:
                    total = nums[i] + nums[j] + nums[left] + nums[right]
                    if total == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        # 去重：跳过重复的left和right
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif total < target:
                        left += 1
                    else:
                        right -= 1
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    result = solution.fourSum(nums, target)
    expected = [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = [2, 2, 2, 2, 2]
    target = 8
    result = solution.fourSum(nums, target)
    expected = [[2, 2, 2, 2]]
    assert len(result) == len(expected)
    
    # 测试用例3
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    result = solution.fourSum(nums, target)
    assert len(result) == 3
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
