"""
435. 无重叠区间
给定一个区间的集合 intervals ，其中 intervals[i] = [starti, endi] 。返回 需要移除区间的最小数量，使剩余区间互不重叠 。

题目链接：https://leetcode.cn/problems/non-overlapping-intervals/

示例 1:
输入: intervals = [[1,2],[2,3],[3,4],[1,3]]
输出: 1
解释: 移除 [1,3] 后，剩下的区间没有重叠。

示例 2:
输入: intervals = [ [1,2], [1,2], [1,2] ]
输出: 2
解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。

示例 3:
输入: intervals = [ [1,2], [2,3] ]
输出: 0
解释: 你不需要移除任何区间，因为它们已经是无重叠的了。

提示：
- 1 <= intervals.length <= 10^5
- intervals[i].length == 2
- -5 * 10^4 <= starti < endi <= 5 * 10^4
"""

from typing import List


class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.eraseOverlapIntervals([[1,2],[2,3],[3,4],[1,3]]) == 1
    
    # 测试用例2
    assert solution.eraseOverlapIntervals([[1,2],[1,2],[1,2]]) == 2
    
    # 测试用例3
    assert solution.eraseOverlapIntervals([[1,2],[2,3]]) == 0
    
    # 测试用例4
    assert solution.eraseOverlapIntervals([[1,2],[1,3],[2,3],[3,4]]) == 1
    
    # 测试用例5
    assert solution.eraseOverlapIntervals([[1,100],[11,22],[1,11],[2,12]]) == 2
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
