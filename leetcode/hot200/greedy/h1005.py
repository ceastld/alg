"""
1005. K次取反后最大化的数组和
给你一个整数数组 nums 和一个整数 k ，按以下方法修改该数组：

选择某个下标 i 并将 nums[i] 替换为 -nums[i] 。
重复这个过程恰好 k 次。可以多次选择同一个下标 i 。

以这种方式修改数组后，返回数组 可能的最大和 。

题目链接：https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/

示例 1:
输入：nums = [4,2,3], k = 1
输出：5
解释：选择下标 1 ，nums 变为 [4,-2,3] 。

示例 2:
输入：nums = [3,-1,0,2], k = 3
输出：6
解释：选择下标 (1, 2, 2) ，nums 变为 [3,1,0,2] 。

示例 3:
输入：nums = [2,-3,-1,5,-4], k = 2
输出：13
解释：选择下标 (1, 4) ，nums 变为 [2,3,-1,5,4] 。

提示：
- 1 <= nums.length <= 10^4
- -100 <= nums[i] <= 100
- 1 <= k <= 10^4
"""

import bisect
from typing import List


class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        nums.sort()
        zero_index = bisect.bisect_left(nums, 0)
        if k < zero_index:
            for i in range(k):
                nums[i] = -nums[i]
            return sum(nums)
        else:
            for i in range(zero_index):
                if nums[i] < 0:
                    nums[i] = -nums[i]
                else:
                    break
            if (k - zero_index) % 2 == 1:
                return sum(nums) - 2 * min(nums)
            else:
                return sum(nums)
    

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.largestSumAfterKNegations([4,2,3], 1) == 5
    
    # 测试用例2
    assert solution.largestSumAfterKNegations([3,-1,0,2], 3) == 6
    
    # 测试用例3
    assert solution.largestSumAfterKNegations([2,-3,-1,5,-4], 2) == 13
            
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
