class Solution:
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        from collections import Counter
        import heapq
        
        # 统计频次
        count = Counter(nums)
        
        # 使用堆找前k个
        return heapq.nlargest(k, count.keys(), key=count.get)
