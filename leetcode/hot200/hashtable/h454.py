"""
454. 四数相加 II
给你四个整数数组 nums1、nums2、nums3 和 nums4，数组长度都是 n，请你计算有多少个元组 (i, j, k, l) 能满足：

0 <= i, j, k, l < n
nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0

题目链接：https://leetcode.cn/problems/4sum-ii/

示例 1:
输入: nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
输出: 2
解释:
两个元组如下：
1. (0, 0, 0, 1) -> nums1[0] + nums2[0] + nums3[0] + nums4[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> nums1[1] + nums2[1] + nums3[0] + nums4[0] = 2 + (-1) + (-1) + 0 = 0

示例 2:
输入: nums1 = [0], nums2 = [0], nums3 = [0], nums4 = [0]
输出: 1

提示：
- n == nums1.length
- n == nums2.length  
- n == nums3.length
- n == nums4.length
- 1 <= n <= 200
- -2^28 <= nums1[i], nums2[i], nums3[i], nums4[i] <= 2^28
"""
from collections import Counter
from typing import List


class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        """
        请在这里实现你的解法
        """
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums1 = [1, 2]
    nums2 = [-2, -1]
    nums3 = [-1, 2]
    nums4 = [0, 2]
    assert solution.fourSumCount(nums1, nums2, nums3, nums4) == 2
    
    # 测试用例2
    nums1 = [0]
    nums2 = [0]
    nums3 = [0]
    nums4 = [0]
    assert solution.fourSumCount(nums1, nums2, nums3, nums4) == 1
    
    # 测试用例3
    nums1 = [-1, -1]
    nums2 = [-1, 1]
    nums3 = [-1, 1]
    nums4 = [1, -1]
    assert solution.fourSumCount(nums1, nums2, nums3, nums4) == 6
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
