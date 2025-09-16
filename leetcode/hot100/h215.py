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
