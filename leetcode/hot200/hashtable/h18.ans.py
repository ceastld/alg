"""
18. 四数之和 - 标准答案
"""
from typing import List


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        标准解法：排序 + 双指针
        
        解题思路：
        1. 先对数组进行排序
        2. 固定前两个数，用双指针在剩余数组中寻找两数之和等于目标值
        3. 使用双指针技巧，左指针从第二个数后开始，右指针从末尾开始
        4. 根据当前四数之和调整指针位置
        5. 跳过重复元素避免重复结果
        
        时间复杂度：O(n³)
        空间复杂度：O(1)
        """
        nums.sort()
        result = []
        
        for i in range(len(nums) - 3):
            # 跳过重复的第一个数
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            for j in range(i + 1, len(nums) - 2):
                # 跳过重复的第二个数
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                
                left, right = j + 1, len(nums) - 1
                
                while left < right:
                    current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    
                    if current_sum == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        
                        # 跳过重复的左指针
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        # 跳过重复的右指针
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        
                        left += 1
                        right -= 1
                    elif current_sum < target:
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
    for quad in expected:
        assert quad in result
    
    # 测试用例2
    nums = [2, 2, 2, 2, 2]
    target = 8
    result = solution.fourSum(nums, target)
    assert [2, 2, 2, 2] in result
    
    # 测试用例3
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    result = solution.fourSum(nums, target)
    assert len(result) == 3
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
