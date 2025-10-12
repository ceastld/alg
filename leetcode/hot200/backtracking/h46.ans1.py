"""
46. 全排列 - 优化答案
"""

from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(x, arr):
            if x == len(arr) - 1:
                res.append(arr)  # 添加排列方案
                return
            for i in range(x, len(arr)):
                temp = arr.copy()
                temp[i], temp[x] = temp[x], temp[i]  # 交换，将 nums[i] 固定在第 x 位
                dfs(x + 1, temp)  # 开启固定第 x + 1 位元素

        res = []
        dfs(0, nums)
        return res


def main():
    """测试优化答案"""
    solution = Solution()

    # 测试用例1
    nums = [1, 2, 3]
    result = solution.permute(nums)
    expected = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    assert len(result) == len(expected)

    # 测试用例2
    nums = [0, 1]
    result = solution.permute(nums)
    expected = [[0, 1], [1, 0]]
    assert len(result) == len(expected)

    # 测试用例3
    nums = [1]
    result = solution.permute(nums)
    expected = [[1]]
    assert result == expected

    nums = [1, 2, 3]
    result_opt = solution.permute(nums)
    assert len(result_opt) == 6
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
