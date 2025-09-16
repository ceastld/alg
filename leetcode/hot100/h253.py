"""
LeetCode 253. Meeting Rooms II

题目描述：
给你一个会议时间安排的数组intervals，每个会议时间都会包括开始和结束的时间intervals[i] = [starti, endi]，返回所需会议室的最小数量。

示例：
intervals = [[0,30],[5,10],[15,20]]
输出：2

数据范围：
- 1 <= intervals.length <= 10^4
- 0 <= starti < endi <= 10^6
"""

class Solution:
    def minMeetingRooms(self, intervals: list[list[int]]) -> int:
        import heapq
        
        if not intervals:
            return 0
        
        # 按开始时间排序
        intervals.sort()
        
        # 最小堆存储结束时间
        heap = []
        
        for start, end in intervals:
            # 如果当前会议开始时，有会议室已结束，则复用
            if heap and heap[0] <= start:
                heapq.heappop(heap)
            
            # 将当前会议结束时间加入堆
            heapq.heappush(heap, end)
        
        return len(heap)
