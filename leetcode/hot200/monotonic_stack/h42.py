"""
42. 接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

题目链接：https://leetcode.cn/problems/trapping-rain-water/

示例 1:
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。

示例 2:
输入：height = [4,2,0,3,2,5]
输出：9

提示：
- n == height.length
- 1 <= n <= 2 * 10^4
- 0 <= height[i] <= 10^5
"""

from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
    
    # 测试用例2
    assert solution.trap([4,2,0,3,2,5]) == 9
    
    # 测试用例3
    assert solution.trap([3,0,2,0,4]) == 7
    
    # 测试用例4
    assert solution.trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
    
    # 测试用例5
    assert solution.trap([2,0,2]) == 2
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
