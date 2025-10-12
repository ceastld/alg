"""
347. 前 K 个高频元素
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。

题目链接：https://leetcode.cn/problems/top-k-frequent-elements/

示例 1:
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]

示例 2:
输入: nums = [1], k = 1
输出: [1]

提示：
- 1 <= nums.length <= 10^5
- k 的取值范围是 [1, 数组中不相同的元素的个数]
- 题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的

进阶：你所设计算法的时间复杂度 必须 优于 O(n log n) ，其中 n 是数组大小。
"""
from typing import List


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    result = solution.topKFrequent(nums, k)
    expected = [1, 2]
    assert set(result) == set(expected)
    
    # 测试用例2
    nums = [1]
    k = 1
    result = solution.topKFrequent(nums, k)
    expected = [1]
    assert result == expected
    
    # 测试用例3
    nums = [1, 2]
    k = 2
    result = solution.topKFrequent(nums, k)
    expected = [1, 2]
    assert set(result) == set(expected)
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
