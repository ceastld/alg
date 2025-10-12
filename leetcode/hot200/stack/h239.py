"""
239. 滑动窗口最大值
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。

题目链接：https://leetcode.cn/problems/sliding-window-maximum/

示例 1:
输入: nums = [1,3,-1,-3,5,3,6,7], k = 3
输出: [3,3,5,5,6,7]
解释:
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

示例 2:
输入: nums = [1], k = 1
输出: [1]

提示：
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
- 1 <= k <= nums.length
"""

from collections import deque
from typing import List


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        请在这里实现你的解法
        """
        q = deque()
        result = []
        for i in range(len(nums)):
            while q and nums[q[-1]] <= nums[i]:
                q.pop()
            q.append(i)
            if q[0] <= i - k:
                q.popleft()
            if i >= k - 1:
                result.append(nums[q[0]])
        return result


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    result = solution.maxSlidingWindow(nums, k)
    expected = [3, 3, 5, 5, 6, 7]
    assert result == expected

    # 测试用例2
    nums = [1]
    k = 1
    result = solution.maxSlidingWindow(nums, k)
    expected = [1]
    assert result == expected

    # 测试用例3
    nums = [1, -1]
    k = 1
    result = solution.maxSlidingWindow(nums, k)
    expected = [1, -1]
    assert result == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
