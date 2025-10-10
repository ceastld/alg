"""
27. 移除元素
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，
并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

题目链接：https://leetcode.cn/problems/remove-element/

示例 1:
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。

示例 2:
输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
解释：函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。注意这五个元素可为任意顺序。你不需要考虑数组中超出新长度后面的元素。

提示：
- 0 <= nums.length <= 100
- 0 <= nums[i] <= 50
- 0 <= val <= 100
"""
from typing import List


class Solution:
    """
    27. 移除元素
    双指针经典题目
    """
    
    def removeElement(self, nums: List[int], val: int) -> int:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums_1 = [3, 2, 2, 3]
    assert solution.removeElement(nums_1, 3) == 2
    
    # 测试用例2
    nums_2 = [0, 1, 2, 2, 3, 0, 4, 2]
    assert solution.removeElement(nums_2, 2) == 5
    
    # 测试用例3
    nums_3 = [1]
    assert solution.removeElement(nums_3, 1) == 0
    
    # 测试用例4
    nums_4 = [4, 5]
    assert solution.removeElement(nums_4, 5) == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
