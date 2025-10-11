"""
31. 下一个排列
实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

题目链接：https://leetcode.cn/problems/next-permutation/

示例 1:
输入: nums = [1,2,3]
输出: [1,3,2]

示例 2:
输入: nums = [3,2,1]
输出: [1,2,3]

示例 3:
输入: nums = [1,1,5]
输出: [1,5,1]

提示：
- 1 <= nums.length <= 100
- 0 <= nums[i] <= 100
"""
from typing import List


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 2, 3]
    solution.nextPermutation(nums)
    assert nums == [1, 3, 2]
    
    # 测试用例2
    nums = [3, 2, 1]
    solution.nextPermutation(nums)
    assert nums == [1, 2, 3]
    
    # 测试用例3
    nums = [1, 1, 5]
    solution.nextPermutation(nums)
    assert nums == [1, 5, 1]
    
    # 测试用例4
    nums = [1, 3, 2]
    solution.nextPermutation(nums)
    assert nums == [2, 1, 3]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
