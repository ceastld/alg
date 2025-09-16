"""
LeetCode 215. Kth Largest Element in an Array

题目描述：
给定整数数组nums和整数k，请返回数组中第k个最大的元素。
请注意，你需要找的是数组排序后的第k个最大的元素，而不是第k个不同的元素。
你必须设计并实现时间复杂度为O(n)的算法解决此问题。

示例：
nums = [3,2,1,5,6,4], k = 2
输出：5

数据范围：
- 1 <= k <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
"""

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # Method: Bucket/Counting Sort (O(n) guaranteed)
        # Since nums[i] is in range [-10^4, 10^4], we can use counting sort
        
        # Create bucket array with size 20001 (from -10000 to 10000)
        # Index mapping: num -> index = num + 10000
        bucket_size = 20001
        buckets = [0] * bucket_size
        
        # Count frequency of each number
        for num in nums:
            index = num + 10000  # Map [-10000, 10000] to [0, 20000]
            buckets[index] += 1
        
        # Traverse from largest to smallest to find k-th largest
        count = 0
        for i in range(bucket_size - 1, -1, -1):
            count += buckets[i]
            if count >= k:
                # Convert index back to original number
                return i - 10000
