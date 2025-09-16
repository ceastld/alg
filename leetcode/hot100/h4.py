"""
LeetCode 4. Median of Two Sorted Arrays

题目描述：
给定两个大小分别为m和n的正序（从小到大）数组nums1和nums2。
请你找出并返回这两个正序数组的中位数。

示例：
nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3]，中位数是2

数据范围：
- nums1.length == m
- nums2.length == n
- 0 <= m <= 1000
- 0 <= n <= 1000
- 1 <= m + n <= 2000
- -10^6 <= nums1[i], nums2[i] <= 10^6
"""

class Solution:
    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        # 确保nums1是较短的数组
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        
        while left <= right:
            # 在nums1中切分
            cut1 = (left + right) // 2
            # 在nums2中切分
            cut2 = (m + n + 1) // 2 - cut1
            
            # 获取切分点左右的元素
            left1 = float('-inf') if cut1 == 0 else nums1[cut1 - 1]
            right1 = float('inf') if cut1 == m else nums1[cut1]
            left2 = float('-inf') if cut2 == 0 else nums2[cut2 - 1]
            right2 = float('inf') if cut2 == n else nums2[cut2]
            
            # 检查切分是否正确
            if left1 <= right2 and left2 <= right1:
                # 找到正确的切分
                if (m + n) % 2 == 0:
                    return (max(left1, left2) + min(right1, right2)) / 2
                else:
                    return max(left1, left2)
            elif left1 > right2:
                # 切分点太靠右，向左调整
                right = cut1 - 1
            else:
                # 切分点太靠左，向右调整
                left = cut1 + 1
