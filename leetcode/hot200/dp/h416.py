"""
416. 分割等和子集
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

题目链接：https://leetcode.cn/problems/partition-equal-subset-sum/

示例 1:
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。

示例 2:
输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。

提示：
- 1 <= nums.length <= 200
- 1 <= nums[i] <= 100
"""

from typing import List


class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        if s % 2 == 1:
            return False
        target = s // 2
        dp = 1 << target
        for num in nums:
            dp |= dp >> num
            if dp & 1:
                return True
        return False


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,5,11,5]
    result = solution.canPartition(nums)
    expected = True
    assert result == expected
    
    # 测试用例2
    nums = [1,2,3,5]
    result = solution.canPartition(nums)
    expected = False
    assert result == expected
    
    # 测试用例3
    nums = [1,1,1,1]
    result = solution.canPartition(nums)
    expected = True
    assert result == expected
    
    # 测试用例4
    nums = [1,2,5]
    result = solution.canPartition(nums)
    expected = False
    assert result == expected
    
    # 测试用例5
    nums = [1,1]
    result = solution.canPartition(nums)
    expected = True
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
