"""
15. 三数之和 - 标准答案
"""
from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        标准解法：排序 + 双指针
        
        解题思路：
        1. 先对数组进行排序
        2. 固定第一个数，用双指针在剩余数组中寻找两数之和等于负的第一个数
        3. 使用双指针技巧，左指针从第一个数后开始，右指针从末尾开始
        4. 根据当前三数之和调整指针位置
        5. 跳过重复元素避免重复结果
        
        时间复杂度：O(n²)
        空间复杂度：O(1)
        """
        nums.sort()
        result = []
        
        for i in range(len(nums) - 2):
            # 跳过重复的第一个数
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            # 如果第一个数大于0，后面不可能有解
            if nums[i] > 0:
                break
            
            left, right = i + 1, len(nums) - 1
            
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                if current_sum == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # 跳过重复的左指针
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    # 跳过重复的右指针
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif current_sum < 0:
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
    for triplet in expected:
        assert triplet in result
    
    # 测试用例2
    nums = []
    assert solution.threeSum(nums) == []
    
    # 测试用例3
    nums = [0]
    assert solution.threeSum(nums) == []
    
    # 测试用例4
    nums = [0, 0, 0]
    result = solution.threeSum(nums)
    assert [0, 0, 0] in result
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
