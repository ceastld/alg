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
