"""
15. 三数之和 - 标准答案
"""
from typing import List


class Solution:
    """
    15. 三数之和 - 标准解法
    """
    
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        标准解法：双指针 + 去重
        
        解题思路：
        1. 先排序数组
        2. 固定第一个数，用双指针找另外两个数
        3. 跳过重复元素避免重复解
        4. 当和等于0时记录解，并跳过重复元素
        
        时间复杂度：O(n²)
        空间复杂度：O(1) 不考虑输出数组
        """
        nums.sort()
        result = []
        n = len(nums)
        
        for i in range(n - 2):
            # 跳过重复的第一个数
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            # 如果第一个数大于0，后面不可能有解
            if nums[i] > 0:
                break
            
            left, right = i + 1, n - 1
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                if current_sum == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # 跳过重复元素
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
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
    result_1 = solution.threeSum([-1, 0, 1, 2, -1, -4])
    assert len(result_1) == 2
    assert [-1, -1, 2] in result_1
    assert [-1, 0, 1] in result_1
    
    # 测试用例2
    assert solution.threeSum([0, 1, 1]) == []
    
    # 测试用例3
    assert solution.threeSum([0, 0, 0]) == [[0, 0, 0]]
    
    # 测试用例4
    result_4 = solution.threeSum([-2, 0, 1, 1, 2])
    assert len(result_4) == 2
    assert [-2, 0, 2] in result_4
    assert [-2, 1, 1] in result_4
    
    # 测试用例5
    assert solution.threeSum([1, 2, -2, -1]) == []
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
