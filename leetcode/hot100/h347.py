"""
LeetCode 347. Top K Frequent Elements

题目描述：
给你一个整数数组nums和一个整数k，请你返回其中出现频率前k高的元素。你可以按任意顺序返回答案。

示例：
nums = [1,1,1,2,2,3], k = 2
输出：[1,2]

数据范围：
- 1 <= nums.length <= 10^5
- k的取值范围是[1, 数组中不相同的元素的个数]
- 题目数据保证答案唯一，换句话说，数组中前k个高频元素的集合是唯一的
"""

class Solution:
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        from collections import Counter
        import heapq
        
        # 统计频次
        count = Counter(nums)
        
        # 使用堆找前k个
        return heapq.nlargest(k, count.keys(), key=count.get)
