"""
239. 滑动窗口最大值 - 标准答案
"""
from typing import List
from collections import deque


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        标准解法：单调双端队列
        
        解题思路：
        1. 使用双端队列存储数组元素的索引
        2. 队列中保持递减顺序，队首元素是当前窗口的最大值
        3. 当窗口滑动时，移除过期元素，添加新元素
        4. 维护队列的单调性，确保队首始终是最大值
        
        时间复杂度：O(n)
        空间复杂度：O(k)
        """
        if not nums or k == 0:
            return []
        
        # 使用双端队列存储索引
        dq = deque()
        result = []
        
        for i in range(len(nums)):
            # 移除队列中超出窗口范围的元素
            while dq and dq[0] <= i - k:
                dq.popleft()
            
            # 移除队列中比当前元素小的元素
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            
            # 添加当前元素索引
            dq.append(i)
            
            # 当窗口大小达到k时，开始记录结果
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    result = solution.maxSlidingWindow(nums, k)
    expected = [3, 3, 5, 5, 6, 7]
    assert result == expected
    
    # 测试用例2
    nums = [1]
    k = 1
    result = solution.maxSlidingWindow(nums, k)
    expected = [1]
    assert result == expected
    
    # 测试用例3
    nums = [1, -1]
    k = 1
    result = solution.maxSlidingWindow(nums, k)
    expected = [1, -1]
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
