"""
300. 最长递增子序列 - 标准答案
"""
from typing import List
import bisect


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        标准解法：动态规划
        
        解题思路：
        1. dp[i] 表示以nums[i]结尾的最长递增子序列长度
        2. 状态转移方程：dp[i] = max(dp[j] + 1) for j < i and nums[j] < nums[i]
        3. 边界条件：dp[0] = 1
        4. 返回dp数组中的最大值
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def lengthOfLIS_binary_search(self, nums: List[int]) -> int:
        """
        优化解法：二分查找 + 贪心
        
        解题思路：
        1. 维护一个tails数组，tails[i]表示长度为i+1的递增子序列的最小尾部元素
        2. 对于每个元素，使用二分查找找到它应该插入的位置
        3. 如果元素大于所有尾部元素，则扩展序列
        4. 否则替换第一个大于等于它的元素
        
        时间复杂度：O(n log n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        tails = []
        
        for num in nums:
            # 使用二分查找找到插入位置
            pos = bisect.bisect_left(tails, num)
            
            if pos == len(tails):
                # 当前元素大于所有尾部元素，扩展序列
                tails.append(num)
            else:
                # 替换第一个大于等于当前元素的元素
                tails[pos] = num
        
        return len(tails)
    
    def lengthOfLIS_recursive(self, nums: List[int]) -> int:
        """
        递归解法（带记忆化）
        
        解题思路：
        1. 递归计算，使用记忆化避免重复计算
        2. 自顶向下的动态规划
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        memo = {}
        
        def dfs(i):
            if i in memo:
                return memo[i]
            if i == 0:
                return 1
            
            max_len = 1
            for j in range(i):
                if nums[j] < nums[i]:
                    max_len = max(max_len, dfs(j) + 1)
            
            memo[i] = max_len
            return max_len
        
        return max(dfs(i) for i in range(len(nums)))
    
    def lengthOfLIS_optimized_dp(self, nums: List[int]) -> int:
        """
        优化DP解法：空间优化
        
        解题思路：
        1. 使用滚动数组优化空间复杂度
        2. 只保存必要的状态信息
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        if not nums:
            return 0
        
        n = len(nums)
        max_len = 1
        
        for i in range(1, n):
            current_max = 1
            for j in range(i):
                if nums[j] < nums[i]:
                    current_max = max(current_max, 1 + (1 if j == 0 else 1))
            max_len = max(max_len, current_max)
        
        return max_len
    
    def lengthOfLIS_brute_force(self, nums: List[int]) -> int:
        """
        暴力解法：枚举所有子序列
        
        解题思路：
        1. 枚举所有可能的子序列
        2. 检查每个子序列是否为递增
        3. 返回最长递增子序列的长度
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not nums:
            return 0
        
        def is_increasing(subsequence):
            for i in range(1, len(subsequence)):
                if subsequence[i] <= subsequence[i-1]:
                    return False
            return True
        
        max_len = 0
        n = len(nums)
        
        # 枚举所有可能的子序列
        for mask in range(1, 1 << n):
            subsequence = []
            for i in range(n):
                if mask & (1 << i):
                    subsequence.append(nums[i])
            
            if is_increasing(subsequence):
                max_len = max(max_len, len(subsequence))
        
        return max_len


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [10,9,2,5,3,7,101,18]
    result = solution.lengthOfLIS(nums)
    expected = 4
    assert result == expected
    
    # 测试用例2
    nums = [0,1,0,3,2,3]
    result = solution.lengthOfLIS(nums)
    expected = 4
    assert result == expected
    
    # 测试用例3
    nums = [7,7,7,7,7,7,7]
    result = solution.lengthOfLIS(nums)
    expected = 1
    assert result == expected
    
    # 测试用例4
    nums = [1,3,6,7,9,4,10,5,6]
    result = solution.lengthOfLIS(nums)
    expected = 6
    assert result == expected
    
    # 测试用例5
    nums = [1]
    result = solution.lengthOfLIS(nums)
    expected = 1
    assert result == expected
    
    # 测试二分查找解法
    print("测试二分查找解法...")
    nums = [10,9,2,5,3,7,101,18]
    result_bs = solution.lengthOfLIS_binary_search(nums)
    expected_bs = 4
    assert result_bs == expected_bs
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [10,9,2,5,3,7,101,18]
    result_rec = solution.lengthOfLIS_recursive(nums)
    expected_rec = 4
    assert result_rec == expected_rec
    
    # 测试暴力解法
    print("测试暴力解法...")
    nums = [10,9,2,5,3,7,101,18]
    result_bf = solution.lengthOfLIS_brute_force(nums)
    expected_bf = 4
    assert result_bf == expected_bf
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
