"""
209. 长度最小的子数组
给定一个含有 n 个正整数的数组和一个正整数 target ，
找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，
并返回其长度。如果不存在符合条件的子数组，返回 0。

题目链接：https://leetcode.cn/problems/minimum-size-subarray-sum/

示例 1:
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。

示例 2:
输入：target = 4, nums = [1,4,4]
输出：1

示例 3:
输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0

提示：
- 1 <= target <= 109
- 1 <= nums.length <= 105
- 1 <= nums[i] <= 104
"""
from typing import List


class Solution:
    """
    209. 长度最小的子数组
    滑动窗口经典题目
    """
    
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        slow = 0
        min_length = float('inf')
        sum = 0
        for fast in range(len(nums)):
            sum += nums[fast]
            while sum >= target:
                min_length = min(min_length, fast - slow + 1)
                sum -= nums[slow]
                slow += 1
        return min_length if min_length != float('inf') else 0
            
            


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.minSubArrayLen(7, [2, 3, 1, 2, 4, 3]) == 2
    
    # 测试用例2
    assert solution.minSubArrayLen(4, [1, 4, 4]) == 1
    
    # 测试用例3
    assert solution.minSubArrayLen(11, [1, 1, 1, 1, 1, 1, 1, 1]) == 0
    
    # 测试用例4
    assert solution.minSubArrayLen(15, [1, 2, 3, 4, 5]) == 5
    
    # 测试用例5
    assert solution.minSubArrayLen(6, [10, 2, 3]) == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
