"""
56. 合并区间
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

题目链接：https://leetcode.cn/problems/merge-intervals/

示例 1:
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

示例 2:
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。

提示：
- 1 <= intervals.length <= 10^4
- intervals[i].length == 2
- 0 <= starti <= endi <= 10^4
"""

from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        result = []
        for i in intervals:
            if not result or result[-1][1] < i[0]:
                result.append(i)
            else:
                result[-1][1] = max(result[-1][1], i[1])
        return result


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    
    # 测试用例2
    assert solution.merge([[1,4],[4,5]]) == [[1,5]]
    
    # 测试用例3
    assert solution.merge([[1,4],[0,4]]) == [[0,4]]
    
    # 测试用例4
    assert solution.merge([[1,4],[2,3]]) == [[1,4]]
    
    # 测试用例5
    assert solution.merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
