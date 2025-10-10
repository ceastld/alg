"""
15. 三数之和
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，
使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

题目链接：https://leetcode.cn/problems/3sum/

示例 1:
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。

示例 2:
输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。

示例 3:
输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。

提示：
- 3 <= nums.length <= 3000
- -105 <= nums[i] <= 105
"""
from typing import List


class Solution:
    """
    15. 三数之和
    双指针 + 去重经典题目
    """
    
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        nums.sort()
        result = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            if nums[i] > 0:
                break
            j,k = i+1,len(nums)-1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                if sum == 0:
                    result.append([nums[i], nums[j], nums[k]])
                    while j < k and nums[j] == nums[j+1]:
                        j += 1
                    while j < k and nums[k] == nums[k-1]:
                        k -= 1
                    j += 1
                    k -= 1
                elif sum < 0:
                    j += 1
                else:
                    k -= 1
        return result


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    result_1 = solution.threeSum([-1, 0, 1, 2, -1, -4])
    assert len(result_1) == 2
    assert [-1, -1, 2] in result_1
    assert [-1, 0, 1] in result_1
    
    # 测试用例2
    assert solution.threeSum([0, 1, 1]) == []
    
    # 测试用例3
    assert solution.threeSum([0, 0, 0]) == [[0, 0, 0]]
    
    # 测试用例4
    result_4 = solution.threeSum([-2, 0, 1, 1, 2])
    assert len(result_4) == 2
    assert [-2, 0, 2] in result_4
    assert [-2, 1, 1] in result_4
    
    # 测试用例5
    assert solution.threeSum([1, 2, -2, -1]) == []
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
