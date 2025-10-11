"""
1365. 有多少小于当前数字的数字 - 标准答案
"""
from typing import List


class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        """
        标准解法：计数排序
        
        解题思路：
        1. 由于nums[i]的范围是0-100，可以使用计数排序
        2. 统计每个数字出现的次数
        3. 计算前缀和，得到小于当前数字的个数
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 统计每个数字出现的次数
        count = [0] * 101
        for num in nums:
            count[num] += 1
        
        # 计算前缀和
        for i in range(1, 101):
            count[i] += count[i - 1]
        
        # 构建结果
        result = []
        for num in nums:
            if num == 0:
                result.append(0)
            else:
                result.append(count[num - 1])
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [8, 1, 2, 2, 3]
    expected = [4, 0, 1, 1, 3]
    assert solution.smallerNumbersThanCurrent(nums) == expected
    
    # 测试用例2
    nums = [6, 5, 4, 8]
    expected = [2, 1, 0, 3]
    assert solution.smallerNumbersThanCurrent(nums) == expected
    
    # 测试用例3
    nums = [7, 7, 7, 7]
    expected = [0, 0, 0, 0]
    assert solution.smallerNumbersThanCurrent(nums) == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
