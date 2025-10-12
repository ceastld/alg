"""
918. 环形子数组的最大和 - 标准答案
"""
from typing import List


class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        """
        标准解法：Kadane算法 + 环形处理
        
        解题思路：
        1. 环形数组的最大子数组和有两种情况：
           - 情况1：最大子数组不跨越数组边界（普通Kadane算法）
           - 情况2：最大子数组跨越数组边界（总和 - 最小子数组和）
        2. 取两种情况的最大值
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 情况1：最大子数组不跨越边界（普通Kadane算法）
        max_sum = nums[0]
        current_sum = nums[0]
        
        for i in range(1, len(nums)):
            current_sum = max(nums[i], current_sum + nums[i])
            max_sum = max(max_sum, current_sum)
        
        # 情况2：最大子数组跨越边界
        # 计算最小子数组和
        min_sum = nums[0]
        current_sum = nums[0]
        total_sum = nums[0]
        
        for i in range(1, len(nums)):
            current_sum = min(nums[i], current_sum + nums[i])
            min_sum = min(min_sum, current_sum)
            total_sum += nums[i]
        
        # 如果所有元素都是负数，返回最大元素
        if total_sum == min_sum:
            return max_sum
        
        # 返回两种情况的最大值
        return max(max_sum, total_sum - min_sum)
    
    def maxSubarraySumCircular_optimized(self, nums: List[int]) -> int:
        """
        优化解法：一次遍历
        
        解题思路：
        1. 在一次遍历中同时计算最大子数组和、最小子数组和、总和
        2. 避免重复遍历数组
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        max_sum = min_sum = current_max = current_min = total_sum = nums[0]
        
        for i in range(1, len(nums)):
            # 更新最大子数组和
            current_max = max(nums[i], current_max + nums[i])
            max_sum = max(max_sum, current_max)
            
            # 更新最小子数组和
            current_min = min(nums[i], current_min + nums[i])
            min_sum = min(min_sum, current_min)
            
            # 更新总和
            total_sum += nums[i]
        
        # 如果所有元素都是负数，返回最大元素
        if total_sum == min_sum:
            return max_sum
        
        # 返回两种情况的最大值
        return max(max_sum, total_sum - min_sum)
    
    def maxSubarraySumCircular_brute_force(self, nums: List[int]) -> int:
        """
        暴力解法：枚举所有可能的子数组
        
        解题思路：
        1. 将数组扩展为两倍长度，模拟环形
        2. 枚举所有可能的子数组
        3. 返回最大和
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        n = len(nums)
        # 扩展数组为两倍长度，模拟环形
        extended = nums + nums
        max_sum = float('-inf')
        
        # 枚举所有可能的子数组（长度不超过n）
        for i in range(n):
            current_sum = 0
            for j in range(i, i + n):
                current_sum += extended[j]
                max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def maxSubarraySumCircular_dp(self, nums: List[int]) -> int:
        """
        动态规划解法：显式DP
        
        解题思路：
        1. 使用两个DP数组分别计算最大子数组和和最小子数组和
        2. 计算总和
        3. 返回两种情况的最大值
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(nums)
        
        # 计算最大子数组和
        max_dp = [0] * n
        max_dp[0] = nums[0]
        for i in range(1, n):
            max_dp[i] = max(nums[i], max_dp[i-1] + nums[i])
        
        # 计算最小子数组和
        min_dp = [0] * n
        min_dp[0] = nums[0]
        for i in range(1, n):
            min_dp[i] = min(nums[i], min_dp[i-1] + nums[i])
        
        # 计算总和
        total_sum = sum(nums)
        
        # 如果所有元素都是负数，返回最大元素
        if total_sum == min(min_dp):
            return max(max_dp)
        
        # 返回两种情况的最大值
        return max(max(max_dp), total_sum - min(min_dp))


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,-2,3,-2]
    result = solution.maxSubarraySumCircular(nums)
    expected = 3
    assert result == expected
    
    # 测试用例2
    nums = [5,-3,5]
    result = solution.maxSubarraySumCircular(nums)
    expected = 10
    assert result == expected
    
    # 测试用例3
    nums = [-3,-2,-3]
    result = solution.maxSubarraySumCircular(nums)
    expected = -2
    assert result == expected
    
    # 测试用例4
    nums = [1]
    result = solution.maxSubarraySumCircular(nums)
    expected = 1
    assert result == expected
    
    # 测试用例5
    nums = [1,2,3,4,5]
    result = solution.maxSubarraySumCircular(nums)
    expected = 15
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [1,-2,3,-2]
    result_opt = solution.maxSubarraySumCircular_optimized(nums)
    expected_opt = 3
    assert result_opt == expected_opt
    
    # 测试暴力解法
    print("测试暴力解法...")
    nums = [1,-2,3,-2]
    result_bf = solution.maxSubarraySumCircular_brute_force(nums)
    expected_bf = 3
    assert result_bf == expected_bf
    
    # 测试DP解法
    print("测试DP解法...")
    nums = [1,-2,3,-2]
    result_dp = solution.maxSubarraySumCircular_dp(nums)
    expected_dp = 3
    assert result_dp == expected_dp
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
