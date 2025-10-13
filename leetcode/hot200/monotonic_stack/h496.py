"""
496. 下一个更大元素I
nums1 中数字 x 的 下一个更大元素 是指 x 在 nums2 中对应位置 右侧 的 第一个 比 x 大的元素。给你两个 没有重复元素 的数组 nums1 和 nums2 ，下标从 0 开始计数，其中nums1 是 nums2 的子集。

对于每个 0 <= i < nums1.length ，找出满足 nums1[i] == nums2[j] 的下标 j ，并且在 nums2 确定 nums2[j] 的 下一个更大元素 。如果不存在下一个更大元素，那么本次查询的答案是 -1 。

返回一个长度为 nums1.length 的数组 ans 作为答案，满足 ans[i] 是如上所述的 下一个更大元素 。

题目链接：https://leetcode.cn/problems/next-greater-element-i/

示例 1:
输入：nums1 = [4,1,2], nums2 = [1,3,4,2]
输出：[-1,3,-1]
解释：nums1 中每个值的下一个更大元素如下所述：
- 4 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 。
- 1 ，用加粗斜体标识，nums2 = [1,3,4,2]。下一个更大元素是 3 。
- 2 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 。

示例 2:
输入：nums1 = [2,4], nums2 = [1,2,3,4]
输出：[3,-1]
解释：nums1 中每个值的下一个更大元素如下所述：
- 2 ，用加粗斜体标识，nums2 = [1,2,3,4]。下一个更大元素是 3 。
- 4 ，用加粗斜体标识，nums2 = [1,2,3,4]。不存在下一个更大元素，所以答案是 -1 。

提示：
- 1 <= nums1.length <= nums2.length <= 1000
- 0 <= nums1[i], nums2[i] <= 10^4
- nums1 和 nums2 中所有整数 互不相同
- nums1 中的所有整数同样出现在 nums2 中
"""

from typing import List


class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        res = {}
        for num in nums2:
            while stack and num > stack[-1]:
                res[stack.pop()] = num
            stack.append(num)
        return [res.get(num, -1) for num in nums1]


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.nextGreaterElement([4,1,2], [1,3,4,2]) == [-1,3,-1]
    
    # 测试用例2
    assert solution.nextGreaterElement([2,4], [1,2,3,4]) == [3,-1]
    
    # 测试用例3
    assert solution.nextGreaterElement([1,3,5,2,4], [6,5,4,3,2,1,7]) == [7,7,7,7,7]
    
    # 测试用例4
    assert solution.nextGreaterElement([4,1,2], [1,2,3,4]) == [-1,2,3]
    
    # 测试用例5
    assert solution.nextGreaterElement([1], [1]) == [-1]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
