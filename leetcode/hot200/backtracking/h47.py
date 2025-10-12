"""
47. 全排列II
给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。

题目链接：https://leetcode.cn/problems/permutations-ii/

示例 1:
输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]

示例 2:
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

提示：
- 1 <= nums.length <= 8
- -10 <= nums[i] <= 10
"""

from typing import List


class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(start, nums):
            # 递归终止条件
            if start == len(nums):
                result.append(nums)  # 添加当前排列的副本
                return
            
            # 用set记录当前位置已经使用过的数字，避免重复
            used = set()
            for i in range(start, len(nums)):
                # 如果当前数字在这个位置已经用过，跳过
                if nums[i] in used:
                    continue
                used.add(nums[i])
                
                # 交换
                temp = nums.copy()
                temp[start], temp[i] = temp[i], temp[start]
                # 递归处理下一个位置
                dfs(start + 1, temp)
                
        
        result = []
        dfs(0, nums)
        return result


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    nums = [1, 1, 2]
    result = solution.permuteUnique(nums)
    expected = [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
    assert len(result) == len(expected)

    # 测试用例2
    nums = [1, 2, 3]
    result = solution.permuteUnique(nums)
    expected = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    assert len(result) == len(expected)

    # 测试用例3
    nums = [1, 1, 1]
    result = solution.permuteUnique(nums)
    expected = [[1, 1, 1]]
    assert result == expected

    # 测试用例4
    nums = [1, 2]
    result = solution.permuteUnique(nums)
    expected = [[1, 2], [2, 1]]
    assert len(result) == len(expected)

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
