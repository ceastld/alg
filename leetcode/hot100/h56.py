"""
LeetCode 56. Merge Intervals

题目描述：
以数组intervals表示若干个区间的集合，其中单个区间为intervals[i] = [starti, endi]。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

示例：
intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间[1,3]和[2,6]重叠，将它们合并为[1,6]。

数据范围：
- 1 <= intervals.length <= 10^4
- intervals[i].length == 2
- 0 <= starti <= endi <= 10^4
"""

class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if not intervals:
            return []
        
        # 按起始位置排序
        intervals.sort(key=lambda x: x[0])
        
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            
            # 如果当前区间与上一个区间重叠，则合并
            if current[0] <= last[1]:
                last[1] = max(last[1], current[1])
            else:
                # 不重叠，直接添加
                merged.append(current)
        
        return merged
