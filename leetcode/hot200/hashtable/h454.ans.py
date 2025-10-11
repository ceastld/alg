"""
454. 四数相加 II - 标准答案
"""
from typing import List
from collections import Counter


class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        """
        标准解法：哈希表 + 分组
        
        解题思路：
        1. 将四个数组分成两组：(nums1, nums2) 和 (nums3, nums4)
        2. 计算第一组所有可能的和，用哈希表记录每个和出现的次数
        3. 计算第二组所有可能的和，对于每个和，查找哈希表中是否存在相反数
        4. 累加所有匹配的组合数
        
        时间复杂度：O(n²)
        空间复杂度：O(n²)
        """
        countAB = Counter(a+b for a in nums1 for b in nums2)
        countCD = Counter(c+d for c in nums3 for d in nums4)
        rlt = 0
        for i in countAB:
            rlt += countAB[i] * countCD[-i]
        return rlt


def main():
    """测试标准答案"""
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
