"""
1. 两数之和 - 标准答案
"""
from typing import List


class Solution:
    """
    1. 两数之和 - 标准解法
    """
    
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        标准解法：哈希表法
        
        解题思路：
        1. 使用哈希表存储已遍历的元素及其索引
        2. 遍历数组，对每个元素计算complement = target - nums[i]
        3. 如果complement在哈希表中，返回两个索引
        4. 否则将当前元素加入哈希表
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        num_to_index = {}
        
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_to_index:
                return [num_to_index[complement], i]
            num_to_index[num] = i
        
        return []


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.twoSum([2, 7, 11, 15], 9) == [0, 1]
    
    # 测试用例2
    assert solution.twoSum([3, 2, 4], 6) == [1, 2]
    
    # 测试用例3
    assert solution.twoSum([3, 3], 6) == [0, 1]
    
    # 测试用例4
    assert solution.twoSum([-1, -2, -3, -4, -5], -8) == [2, 4]
    
    # 测试用例5
    assert solution.twoSum([0, 4, 3, 0], 0) == [0, 3]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
