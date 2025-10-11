"""
18. 四数之和
给你一个由 n 个整数组成的数组 nums，和一个目标值 target。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]]（若两个四元组元素一一对应，则认为两个四元组重复）：

0 <= a, b, c, d < n
a、b、c 和 d 互不相同
nums[a] + nums[b] + nums[c] + nums[d] == target

你可以按任意顺序返回答案。

题目链接：https://leetcode.cn/problems/4sum/

示例 1:
输入: nums = [1,0,-1,0,-2,2], target = 0
输出: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

示例 2:
输入: nums = [2,2,2,2,2], target = 8
输出: [[2,2,2,2]]

提示：
- 1 <= nums.length <= 200
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
"""
from typing import List


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        请在这里实现你的解法
        """
        nums.sort()
        result = []
        for i in range(len(nums) - 3):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            if nums[i] > target:
                break
            for j in range(i+1, len(nums) - 2):
                if j > i+1 and nums[j] == nums[j-1]:
                    continue
                if nums[i] + nums[j] > target:
                    break
                k,l = j+1,len(nums)-1
                while k < l:
                    sum = nums[i] + nums[j] + nums[k] + nums[l]
                    if sum == target:
                        result.append([nums[i], nums[j], nums[k], nums[l]])
                        while k < l and nums[k] == nums[k+1]:
                            k += 1
                            while k < l and nums[l] == nums[l-1]:
                                l -= 1
                    k += 1
                    l -= 1
        return result


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    result = solution.fourSum(nums, target)
    expected = [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
    assert len(result) == len(expected)
    for quad in expected:
        assert quad in result
    
    # 测试用例2
    nums = [2, 2, 2, 2, 2]
    target = 8
    result = solution.fourSum(nums, target)
    assert [2, 2, 2, 2] in result
    
    # 测试用例3
    nums = [1, 0, -1, 0, -2, 2]
    target = 0
    result = solution.fourSum(nums, target)
    assert len(result) == 3
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
