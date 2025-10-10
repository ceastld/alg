"""
209. 长度最小的子数组 - 标准答案
"""
from typing import List


class Solution:
    """
    209. 长度最小的子数组 - 标准解法
    """
    
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        """
        标准解法：滑动窗口
        
        解题思路：
        1. 使用滑动窗口，维护窗口和
        2. 右指针扩展窗口，当窗口和>=target时
        3. 左指针收缩窗口，记录最小长度
        4. 返回最小长度，如果不存在返回0
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        n = len(nums)
        left = 0
        window_sum = 0
        min_length = float('inf')
        
        for right in range(n):
            window_sum += nums[right]
            
            while window_sum >= target:
                min_length = min(min_length, right - left + 1)
                window_sum -= nums[left]
                left += 1
        
        return min_length if min_length != float('inf') else 0


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.minSubArrayLen(7, [2, 3, 1, 2, 4, 3]) == 2
    
    # 测试用例2
    assert solution.minSubArrayLen(4, [1, 4, 4]) == 1
    
    # 测试用例3
    assert solution.minSubArrayLen(11, [1, 1, 1, 1, 1, 1, 1, 1]) == 0
    
    # 测试用例4
    assert solution.minSubArrayLen(15, [1, 2, 3, 4, 5]) == 5
    
    # 测试用例5
    assert solution.minSubArrayLen(6, [10, 2, 3]) == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
