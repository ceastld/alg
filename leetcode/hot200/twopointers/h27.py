"""
27. 移除元素
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

题目链接：https://leetcode.cn/problems/remove-element/

示例 1:
输入: nums = [3,2,2,3], val = 3
输出: 2, nums = [2,2]
解释: 函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。

示例 2:
输入: nums = [0,1,2,2,3,0,4,2], val = 2
输出: 5, nums = [0,1,4,0,3]
解释: 函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。

提示：
- 0 <= nums.length <= 100
- 0 <= nums[i] <= 50
- 0 <= val <= 100
"""
from typing import List


class Solution:
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
    nums = [3, 2, 2, 3]
    val = 3
    result = solution.removeElement(nums, val)
    assert result == 2
    assert nums[:result] == [2, 2]
    
    # 测试用例2
    nums = [0, 1, 2, 2, 3, 0, 4, 2]
    val = 2
    result = solution.removeElement(nums, val)
    assert result == 5
    # 注意：顺序可能不同，这里只检查长度
    
    # 测试用例3
    nums = []
    val = 0
    result = solution.removeElement(nums, val)
    assert result == 0
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
