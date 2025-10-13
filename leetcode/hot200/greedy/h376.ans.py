"""
376. 摆动序列 - 标准答案
"""
from typing import List


class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 维护两个状态：上升和下降
        2. 遍历数组，根据当前趋势更新状态
        3. 如果趋势改变，长度+1
        4. 贪心策略：只保留趋势变化的点
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if len(nums) < 2:
            return len(nums)
        
        up = 1  # 上升趋势的长度
        down = 1  # 下降趋势的长度
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                # 当前是上升趋势
                up = down + 1
            elif nums[i] < nums[i-1]:
                # 当前是下降趋势
                down = up + 1
        
        return max(up, down)
    
    def wiggleMaxLength_alternative(self, nums: List[int]) -> int:
        """
        替代解法：贪心算法（状态机）
        
        解题思路：
        1. 使用状态机：上升、下降、相等
        2. 根据当前状态和下一个元素决定状态转换
        3. 只在状态改变时增加长度
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if len(nums) < 2:
            return len(nums)
        
        length = 1
        prev_diff = 0
        
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i-1]
            
            # 如果当前差值与上一个差值符号相反，或者上一个差值为0
            if (diff > 0 and prev_diff <= 0) or (diff < 0 and prev_diff >= 0):
                length += 1
                prev_diff = diff
        
        return length
    
    def wiggleMaxLength_dp(self, nums: List[int]) -> int:
        """
        动态规划解法
        
        解题思路：
        1. dp[i][0] 表示以i结尾且最后是下降趋势的最长摆动序列长度
        2. dp[i][1] 表示以i结尾且最后是上升趋势的最长摆动序列长度
        3. 状态转移：根据当前元素与前面元素的关系更新状态
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if len(nums) < 2:
            return len(nums)
        
        n = len(nums)
        # dp[i][0] 表示以i结尾且最后是下降趋势的最长摆动序列长度
        # dp[i][1] 表示以i结尾且最后是上升趋势的最长摆动序列长度
        dp = [[1, 1] for _ in range(n)]
        
        for i in range(1, n):
            if nums[i] > nums[i-1]:
                # 当前是上升趋势
                dp[i][1] = dp[i-1][0] + 1
                dp[i][0] = dp[i-1][0]
            elif nums[i] < nums[i-1]:
                # 当前是下降趋势
                dp[i][0] = dp[i-1][1] + 1
                dp[i][1] = dp[i-1][1]
            else:
                # 当前相等，保持之前的状态
                dp[i][0] = dp[i-1][0]
                dp[i][1] = dp[i-1][1]
        
        return max(dp[n-1][0], dp[n-1][1])
    
    def wiggleMaxLength_optimized(self, nums: List[int]) -> int:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 只维护两个变量：上升和下降趋势的长度
        2. 根据当前趋势更新对应的长度
        3. 返回两个长度的最大值
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if len(nums) < 2:
            return len(nums)
        
        up = 1
        down = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                up = down + 1
            elif nums[i] < nums[i-1]:
                down = up + 1
        
        return max(up, down)
    
    def wiggleMaxLength_detailed(self, nums: List[int]) -> int:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 维护两个状态：上升趋势和下降趋势的长度
        2. 遍历数组，根据当前元素与前一元素的关系更新状态
        3. 只在趋势改变时增加长度
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if len(nums) < 2:
            return len(nums)
        
        # 维护两个状态
        up = 1    # 上升趋势的长度
        down = 1  # 下降趋势的长度
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                # 当前是上升趋势，更新上升长度
                up = down + 1
            elif nums[i] < nums[i-1]:
                # 当前是下降趋势，更新下降长度
                down = up + 1
            # 如果相等，不更新任何状态
        
        return max(up, down)
    
    def wiggleMaxLength_brute_force(self, nums: List[int]) -> int:
        """
        暴力解法：回溯
        
        解题思路：
        1. 使用回溯算法尝试所有可能的子序列
        2. 检查每个子序列是否为摆动序列
        3. 返回最长的摆动序列长度
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if len(nums) < 2:
            return len(nums)
        
        def is_wiggle(sequence):
            if len(sequence) < 2:
                return True
            
            for i in range(1, len(sequence)):
                if i == 1:
                    continue
                if (sequence[i] - sequence[i-1]) * (sequence[i-1] - sequence[i-2]) >= 0:
                    return False
            return True
        
        def backtrack(index, current):
            if index == len(nums):
                if is_wiggle(current):
                    return len(current)
                return 0
            
            # 不选择当前元素
            result = backtrack(index + 1, current)
            
            # 选择当前元素
            current.append(nums[index])
            result = max(result, backtrack(index + 1, current))
            current.pop()
            
            return result
        
        return backtrack(0, [])


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.wiggleMaxLength([1,7,4,9,2,5]) == 6
    
    # 测试用例2
    assert solution.wiggleMaxLength([1,17,5,10,13,15,10,5,16,8]) == 7
    
    # 测试用例3
    assert solution.wiggleMaxLength([1,2,3,4,5,6,7,8,9]) == 2
    
    # 测试用例4
    assert solution.wiggleMaxLength([1,1,7,4,9,2,5]) == 6
    
    # 测试用例5
    assert solution.wiggleMaxLength([1,2,2,2,3,4]) == 2
    
    # 测试用例6：边界情况
    assert solution.wiggleMaxLength([1]) == 1
    assert solution.wiggleMaxLength([1,2]) == 2
    assert solution.wiggleMaxLength([2,1]) == 2
    
    # 测试用例7：单调递增
    assert solution.wiggleMaxLength([1,2,3,4,5]) == 2
    
    # 测试用例8：单调递减
    assert solution.wiggleMaxLength([5,4,3,2,1]) == 2
    
    # 测试用例9：相等元素
    assert solution.wiggleMaxLength([1,1,1,1,1]) == 1
    
    # 测试用例10：复杂情况
    assert solution.wiggleMaxLength([1,2,3,4,5,6,7,8,9,10]) == 2
    assert solution.wiggleMaxLength([1,17,5,10,13,15,10,5,16,8]) == 7
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
