"""
53. 最大子数组和 - 标准答案
"""
from typing import List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        """
        标准解法：Kadane算法
        
        解题思路：
        1. 使用Kadane算法（动态规划）
        2. 维护当前子数组和和最大子数组和
        3. 如果当前子数组和为负，则重新开始
        4. 每次更新最大子数组和
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        max_sum = nums[0]
        current_sum = nums[0]
        
        for i in range(1, len(nums)):
            # 如果当前子数组和为负，则重新开始
            current_sum = max(nums[i], current_sum + nums[i])
            # 更新最大子数组和
            max_sum = max(max_sum, current_sum)
        
        return max_sum
    
    def maxSubArray_dp(self, nums: List[int]) -> int:
        """
        动态规划解法：显式DP数组
        
        解题思路：
        1. dp[i]表示以nums[i]结尾的最大子数组和
        2. dp[i] = max(nums[i], dp[i-1] + nums[i])
        3. 返回dp数组中的最大值
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i-1] + nums[i])
        
        return max(dp)
    
    def maxSubArray_divide_conquer(self, nums: List[int]) -> int:
        """
        分治法：递归求解
        
        解题思路：
        1. 将数组分为左右两部分
        2. 最大子数组要么在左半部分，要么在右半部分，要么跨越中点
        3. 递归求解三种情况的最大值
        
        时间复杂度：O(n log n)
        空间复杂度：O(log n)
        """
        def divide_conquer(left: int, right: int) -> int:
            if left == right:
                return nums[left]
            
            mid = (left + right) // 2
            
            # 左半部分的最大子数组和
            left_max = divide_conquer(left, mid)
            
            # 右半部分的最大子数组和
            right_max = divide_conquer(mid + 1, right)
            
            # 跨越中点的最大子数组和
            left_sum = float('-inf')
            current_sum = 0
            for i in range(mid, left - 1, -1):
                current_sum += nums[i]
                left_sum = max(left_sum, current_sum)
            
            right_sum = float('-inf')
            current_sum = 0
            for i in range(mid + 1, right + 1):
                current_sum += nums[i]
                right_sum = max(right_sum, current_sum)
            
            cross_max = left_sum + right_sum
            
            return max(left_max, right_max, cross_max)
        
        return divide_conquer(0, len(nums) - 1)
    
    def maxSubArray_brute_force(self, nums: List[int]) -> int:
        """
        暴力解法：枚举所有子数组
        
        解题思路：
        1. 枚举所有可能的子数组
        2. 计算每个子数组的和
        3. 返回最大和
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        max_sum = float('-inf')
        
        for i in range(len(nums)):
            current_sum = 0
            for j in range(i, len(nums)):
                current_sum += nums[j]
                max_sum = max(max_sum, current_sum)
        
        return max_sum


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    result = solution.maxSubArray(nums)
    expected = 6
    assert result == expected
    
    # 测试用例2
    nums = [1]
    result = solution.maxSubArray(nums)
    expected = 1
    assert result == expected
    
    # 测试用例3
    nums = [5,4,-1,7,8]
    result = solution.maxSubArray(nums)
    expected = 23
    assert result == expected
    
    # 测试用例4
    nums = [-1]
    result = solution.maxSubArray(nums)
    expected = -1
    assert result == expected
    
    # 测试用例5
    nums = [-2,-1]
    result = solution.maxSubArray(nums)
    expected = -1
    assert result == expected
    
    # 测试DP解法
    print("测试DP解法...")
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    result_dp = solution.maxSubArray_dp(nums)
    expected_dp = 6
    assert result_dp == expected_dp
    
    # 测试分治法
    print("测试分治法...")
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    result_dc = solution.maxSubArray_divide_conquer(nums)
    expected_dc = 6
    assert result_dc == expected_dc
    
    # 测试暴力解法
    print("测试暴力解法...")
    nums = [-2,1,-3,4,-1,2,1,-5,4]
    result_bf = solution.maxSubArray_brute_force(nums)
    expected_bf = 6
    assert result_bf == expected_bf
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
