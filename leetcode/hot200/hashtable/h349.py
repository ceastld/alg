"""
349. 两个数组的交集
给定两个数组 nums1 和 nums2 ，返回 它们的交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。

题目链接：https://leetcode.cn/problems/intersection-of-two-arrays/

示例 1:
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]

示例 2:
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]
解释：[4,9] 也是可通过的

提示：
- 1 <= nums1.length, nums2.length <= 1000
- 0 <= nums1[i], nums2[i] <= 1000
"""
from typing import List


class Solution:
    """
    349. 两个数组的交集
    哈希表经典题目
    """
    
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        set1 = set(nums1)
        set2 = set(nums2)
        if len(set1) < len(set2):
            set1, set2 = set2, set1
        result = []
        for num in set1:
            if num in set2:
                result.append(num)
        return result
        


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    result1 = solution.intersection([1, 2, 2, 1], [2, 2])
    assert set(result1) == {2}
    
    # 测试用例2
    result2 = solution.intersection([4, 9, 5], [9, 4, 9, 8, 4])
    assert set(result2) == {9, 4}
    
    # 测试用例3
    result3 = solution.intersection([1, 2, 3], [4, 5, 6])
    assert result3 == []
    
    # 测试用例4
    result4 = solution.intersection([1, 2, 3], [1, 2, 3])
    assert set(result4) == {1, 2, 3}
    
    # 测试用例5
    result5 = solution.intersection([1], [1])
    assert result5 == [1]
    
    # 测试用例6
    result6 = solution.intersection([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
    assert set(result6) == {3, 4, 5}
    
    # 测试用例7
    result7 = solution.intersection([], [1, 2, 3])
    assert result7 == []
    
    # 测试用例8
    result8 = solution.intersection([1, 2, 3], [])
    assert result8 == []
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
