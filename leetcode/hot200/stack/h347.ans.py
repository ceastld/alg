"""
347. 前 K 个高频元素 - 标准答案
"""
from typing import List
from collections import Counter
import heapq


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        标准解法：堆排序
        
        解题思路：
        1. 使用Counter统计每个元素的出现频率
        2. 使用最小堆维护前k个高频元素
        3. 遍历所有元素，维护堆的大小为k
        4. 最后返回堆中的所有元素
        
        时间复杂度：O(n log k)
        空间复杂度：O(n)
        """
        # 统计频率
        count = Counter(nums)
        
        # 使用最小堆
        heap = []
        
        for num, freq in count.items():
            heapq.heappush(heap, (freq, num))
            # 保持堆的大小为k
            if len(heap) > k:
                heapq.heappop(heap)
        
        # 返回结果
        return [num for freq, num in heap]
    
    def topKFrequent_bucket(self, nums: List[int], k: int) -> List[int]:
        """
        优化解法：桶排序
        
        解题思路：
        1. 使用Counter统计频率
        2. 使用桶排序，按频率分组
        3. 从高频到低频遍历，取前k个元素
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        # 统计频率
        count = Counter(nums)
        
        # 创建桶，索引为频率，值为元素列表
        buckets = [[] for _ in range(len(nums) + 1)]
        
        for num, freq in count.items():
            buckets[freq].append(num)
        
        # 从高频到低频遍历
        result = []
        for i in range(len(buckets) - 1, -1, -1):
            if buckets[i]:
                result.extend(buckets[i])
                if len(result) >= k:
                    break
        
        return result[:k]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    result = solution.topKFrequent(nums, k)
    expected = [1, 2]
    assert set(result) == set(expected)
    
    # 测试用例2
    nums = [1]
    k = 1
    result = solution.topKFrequent(nums, k)
    expected = [1]
    assert result == expected
    
    # 测试用例3
    nums = [1, 2]
    k = 2
    result = solution.topKFrequent(nums, k)
    expected = [1, 2]
    assert set(result) == set(expected)
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
