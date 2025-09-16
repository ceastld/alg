"""
LeetCode 152. Maximum Product Subarray

题目描述：
给你一个整数数组nums，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
测试用例的答案是一个32位整数。子数组是数组的连续子序列。

示例：
nums = [2,3,-2,4]
输出：6
解释：子数组[2,3]有最大乘积6。

数据范围：
- 1 <= nums.length <= 2 * 10^4
- -10 <= nums[i] <= 10
- nums的任何前缀或后缀的乘积都保证是一个32位整数
"""

from typing import List

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        """
        最大乘积子数组 - 动态规划解法
        
        Args:
            nums: 整数数组
            
        Returns:
            最大乘积
        """
        if not nums:
            return 0
        
        # 初始化：以第一个元素结尾的最大和最小乘积
        max_prod = min_prod = result = nums[0]
        
        for i in range(1, len(nums)):
            # 保存之前的值，避免在计算过程中被覆盖
            prev_max = max_prod
            prev_min = min_prod
            
            # 当前元素可以：
            # 1. 自己单独成子数组
            # 2. 与之前的最大乘积相乘
            # 3. 与之前的最小乘积相乘（负数情况）
            max_prod = max(nums[i], prev_max * nums[i], prev_min * nums[i])
            min_prod = min(nums[i], prev_max * nums[i], prev_min * nums[i])
            
            # 更新全局最大值
            result = max(result, max_prod)
        
        return result
    
    def maxProductBruteForce(self, nums: List[int]) -> int:
        """
        暴力解法 - 枚举所有子数组
        
        Args:
            nums: 整数数组
            
        Returns:
            最大乘积
        """
        if not nums:
            return 0
        
        max_product = float('-inf')
        
        for i in range(len(nums)):
            product = 1
            for j in range(i, len(nums)):
                product *= nums[j]
                max_product = max(max_product, product)
        
        return max_product
    
    def maxProductOptimized(self, nums: List[int]) -> int:
        """
        优化版本 - 处理零和负数的情况
        
        Args:
            nums: 整数数组
            
        Returns:
            最大乘积
        """
        if not nums:
            return 0
        
        # 从左到右和从右到左分别计算
        max_product = nums[0]
        
        # 从左到右
        product = 1
        for num in nums:
            product *= num
            max_product = max(max_product, product)
            if product == 0:
                product = 1  # 遇到0重新开始
        
        # 从右到左
        product = 1
        for num in reversed(nums):
            product *= num
            max_product = max(max_product, product)
            if product == 0:
                product = 1  # 遇到0重新开始
        
        return max_product


# 测试用例
def test_max_product():
    """测试最大乘积子数组功能"""
    solution = Solution()
    
    # 测试用例1：包含负数
    nums1 = [2, 3, -2, 4]
    result1 = solution.maxProduct(nums1)
    print(f"测试1 - 数组: {nums1}")
    print(f"最大乘积: {result1}")  # 6 (子数组 [2, 3])
    print()
    
    # 测试用例2：包含负数
    nums2 = [-2, 0, -1]
    result2 = solution.maxProduct(nums2)
    print(f"测试2 - 数组: {nums2}")
    print(f"最大乘积: {result2}")  # 0 (子数组 [0])
    print()
    
    # 测试用例3：全负数
    nums3 = [-2, -3, -4]
    result3 = solution.maxProduct(nums3)
    print(f"测试3 - 数组: {nums3}")
    print(f"最大乘积: {result3}")  # 24 (子数组 [-2, -3, -4])
    print()
    
    # 测试用例4：包含0
    nums4 = [2, 0, 3, 4]
    result4 = solution.maxProduct(nums4)
    print(f"测试4 - 数组: {nums4}")
    print(f"最大乘积: {result4}")  # 12 (子数组 [3, 4])
    print()
    
    # 测试用例5：单个元素
    nums5 = [-1]
    result5 = solution.maxProduct(nums5)
    print(f"测试5 - 数组: {nums5}")
    print(f"最大乘积: {result5}")  # -1
    print()
    
    # 测试用例6：复杂情况
    nums6 = [2, -5, -2, -4, 3]
    result6 = solution.maxProduct(nums6)
    print(f"测试6 - 数组: {nums6}")
    print(f"最大乘积: {result6}")  # 24 (子数组 [-2, -4, 3])
    print()
    
    # 对比不同解法
    print("=== 解法对比 ===")
    test_nums = [2, 3, -2, 4]
    dp_result = solution.maxProduct(test_nums)
    brute_result = solution.maxProductBruteForce(test_nums)
    opt_result = solution.maxProductOptimized(test_nums)
    
    print(f"动态规划结果: {dp_result}")
    print(f"暴力解法结果: {brute_result}")
    print(f"优化解法结果: {opt_result}")
    print(f"结果一致: {dp_result == brute_result == opt_result}")


if __name__ == "__main__":
    test_max_product()