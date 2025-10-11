"""
922. 按奇偶排序数组II
给定一个非负整数数组 nums，nums 中一半整数是奇数，一半整数是偶数。

对数组进行排序，以便当 nums[i] 为奇数时，i 也是奇数；当 nums[i] 为偶数时，i 也是偶数。

你可以返回任何满足上述条件的答案数组。

题目链接：https://leetcode.cn/problems/sort-array-by-parity-ii/

示例 1:
输入: nums = [4,2,5,7]
输出: [4,7,2,5]
解释: [4,2,5,7], [4,7,2,5], [2,4,7,5], [2,7,4,5] 也会被接受。

示例 2:
输入: nums = [2,3]
输出: [2,3]

提示：
- 2 <= nums.length <= 2 * 10^4
- nums.length 是偶数
- nums 中一半整数是奇数，一半整数是偶数
"""
from typing import List


class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        j = 1
        for i in range(0, len(nums), 2):
            if nums[i] % 2 != 0:
                while j < len(nums) and nums[j] % 2 != 0:
                    j += 2
                nums[i], nums[j] = nums[j], nums[i]
        return nums


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [4, 2, 5, 7]
    result = solution.sortArrayByParityII(nums)
    # 验证结果：偶数索引位置是偶数，奇数索引位置是奇数
    for i, num in enumerate(result):
        if i % 2 == 0:
            assert num % 2 == 0
        else:
            assert num % 2 == 1
    
    # 测试用例2
    nums = [2, 3]
    result = solution.sortArrayByParityII(nums)
    for i, num in enumerate(result):
        if i % 2 == 0:
            assert num % 2 == 0
        else:
            assert num % 2 == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
