"""
1005. K次取反后最大化的数组和 - 标准答案
"""
from typing import List


class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 优先对负数取反，因为负数取反后能增加和
        2. 如果k还有剩余，对最小的正数反复取反
        3. 贪心策略：优先处理负数，然后处理最小正数
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        # 排序，优先处理负数
        nums.sort()
        
        # 对负数取反
        for i in range(len(nums)):
            if nums[i] < 0 and k > 0:
                nums[i] = -nums[i]
                k -= 1
            else:
                break
        
        # 如果k还有剩余，对最小的正数反复取反
        if k > 0:
            # 找到最小的正数
            min_positive = min(nums)
            # 如果k是奇数，对最小正数取反一次
            if k % 2 == 1:
                for i in range(len(nums)):
                    if nums[i] == min_positive:
                        nums[i] = -nums[i]
                        break
        
        return sum(nums)
    
    def largestSumAfterKNegations_alternative(self, nums: List[int], k: int) -> int:
        """
        替代解法：贪心算法（使用堆）
        
        解题思路：
        1. 使用最小堆维护数组
        2. 每次取出最小值进行取反
        3. 重复k次
        
        时间复杂度：O(n log n + k log n)
        空间复杂度：O(n)
        """
        import heapq
        
        # 创建最小堆
        heap = nums[:]
        heapq.heapify(heap)
        
        # 进行k次取反
        for _ in range(k):
            min_val = heapq.heappop(heap)
            heapq.heappush(heap, -min_val)
        
        return sum(heap)
    
    def largestSumAfterKNegations_optimized(self, nums: List[int], k: int) -> int:
        """
        优化解法：贪心算法（一次遍历）
        
        解题思路：
        1. 统计负数的个数
        2. 如果负数个数 >= k，直接对最小的k个负数取反
        3. 否则，对所有负数取反，然后对最小正数反复取反
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        # 排序
        nums.sort()
        
        # 统计负数个数
        negative_count = 0
        for num in nums:
            if num < 0:
                negative_count += 1
            else:
                break
        
        # 如果负数个数 >= k，直接对最小的k个负数取反
        if negative_count >= k:
            for i in range(k):
                nums[i] = -nums[i]
        else:
            # 对所有负数取反
            for i in range(negative_count):
                nums[i] = -nums[i]
            
            # 对最小正数反复取反
            remaining_k = k - negative_count
            if remaining_k % 2 == 1:
                # 找到最小的正数
                min_positive = min(nums)
                for i in range(len(nums)):
                    if nums[i] == min_positive:
                        nums[i] = -nums[i]
                        break
        
        return sum(nums)
    
    def largestSumAfterKNegations_detailed(self, nums: List[int], k: int) -> int:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 优先对负数取反，因为负数取反后能增加和
        2. 如果k还有剩余，对最小的正数反复取反
        3. 贪心策略：优先处理负数，然后处理最小正数
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        # 排序，优先处理负数
        nums.sort()
        
        # 对负数取反
        for i in range(len(nums)):
            if nums[i] < 0 and k > 0:
                nums[i] = -nums[i]
                k -= 1
            else:
                break
        
        # 如果k还有剩余，对最小的正数反复取反
        if k > 0:
            # 找到最小的正数
            min_positive = min(nums)
            # 如果k是奇数，对最小正数取反一次
            if k % 2 == 1:
                for i in range(len(nums)):
                    if nums[i] == min_positive:
                        nums[i] = -nums[i]
                        break
        
        return sum(nums)
    
    def largestSumAfterKNegations_brute_force(self, nums: List[int], k: int) -> int:
        """
        暴力解法：递归
        
        解题思路：
        1. 使用递归尝试所有可能的取反组合
        2. 返回能获得的最大和
        
        时间复杂度：O(2^k)
        空间复杂度：O(k)
        """
        def backtrack(index, remaining_k):
            if remaining_k == 0:
                return sum(nums)
            
            if index == len(nums):
                return sum(nums)
            
            # 不取反当前元素
            result = backtrack(index + 1, remaining_k)
            
            # 取反当前元素
            nums[index] = -nums[index]
            result = max(result, backtrack(index + 1, remaining_k - 1))
            nums[index] = -nums[index]  # 回溯
            
            return result
        
        return backtrack(0, k)
    
    def largestSumAfterKNegations_step_by_step(self, nums: List[int], k: int) -> int:
        """
        分步解法：贪心算法（分步处理）
        
        解题思路：
        1. 第一步：对负数取反
        2. 第二步：对最小正数反复取反
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        # 排序
        nums.sort()
        
        # 第一步：对负数取反
        for i in range(len(nums)):
            if nums[i] < 0 and k > 0:
                nums[i] = -nums[i]
                k -= 1
            else:
                break
        
        # 第二步：对最小正数反复取反
        if k > 0:
            min_positive = min(nums)
            if k % 2 == 1:
                for i in range(len(nums)):
                    if nums[i] == min_positive:
                        nums[i] = -nums[i]
                        break
        
        return sum(nums)


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.largestSumAfterKNegations([4,2,3], 1) == 5
    
    # 测试用例2
    assert solution.largestSumAfterKNegations([3,-1,0,2], 3) == 6
    
    # 测试用例3
    assert solution.largestSumAfterKNegations([2,-3,-1,5,-4], 2) == 13
    
    # 测试用例4
    assert solution.largestSumAfterKNegations([1,3,2,6,7,9], 3) == 24
    
    # 测试用例5
    assert solution.largestSumAfterKNegations([-1,2,3,4,5], 2) == 15
    
    # 测试用例6：边界情况
    assert solution.largestSumAfterKNegations([1], 1) == -1
    assert solution.largestSumAfterKNegations([-1], 1) == 1
    
    # 测试用例7：全负数
    assert solution.largestSumAfterKNegations([-1,-2,-3], 2) == 4
    
    # 测试用例8：全正数
    assert solution.largestSumAfterKNegations([1,2,3], 2) == 4
    
    # 测试用例9：k为0
    assert solution.largestSumAfterKNegations([1,2,3], 0) == 6
    
    # 测试用例10：k很大
    assert solution.largestSumAfterKNegations([1,2,3], 100) == 4
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
