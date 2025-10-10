"""
189. 旋转数组
给你一个数组，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

题目链接：https://leetcode.cn/problems/rotate-array/

示例 1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]

示例 2:
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释: 
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]

提示：
- 1 <= nums.length <= 105
- -231 <= nums[i] <= 231 - 1
- 0 <= k <= 105
"""
from typing import List


class Solution:
    """
    189. 旋转数组
    数组操作经典题目
    """
    
    def rotate(self, nums: List[int], k: int) -> None:
        """
        请在这里实现你的解法
        注意：必须原地修改数组，不能返回任何值
        """
        # TODO: 在这里实现你的解法
        k = k % len(nums)
        nums[:] = nums[-k:] + nums[:-k]


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums_1 = [1, 2, 3, 4, 5, 6, 7]
    solution.rotate(nums_1, 3)
    assert nums_1 == [5, 6, 7, 1, 2, 3, 4]
    
    # 测试用例2
    nums_2 = [-1, -100, 3, 99]
    solution.rotate(nums_2, 2)
    assert nums_2 == [3, 99, -1, -100]
    
    # 测试用例3
    nums_3 = [1, 2]
    solution.rotate(nums_3, 1)
    assert nums_3 == [2, 1]
    
    # 测试用例4
    nums_4 = [1, 2, 3, 4, 5]
    solution.rotate(nums_4, 0)
    assert nums_4 == [1, 2, 3, 4, 5]
    
    # 测试用例5
    nums_5 = [1, 2, 3, 4, 5, 6]
    solution.rotate(nums_5, 2)
    assert nums_5 == [5, 6, 1, 2, 3, 4]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
