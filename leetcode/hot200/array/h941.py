"""
941. 有效的山脉数组
给定一个整数数组 arr，如果它是有效的山脉数组就返回 true，否则返回 false。

让我们回顾一下，如果 arr 满足下述条件，那么它是一个山脉数组：

arr.length >= 3
存在 i（0 < i < arr.length - 1）使得：
arr[0] < arr[1] < ... < arr[i-1] < arr[i]
arr[i] > arr[i+1] > ... > arr[arr.length - 1]

题目链接：https://leetcode.cn/problems/valid-mountain-array/

示例 1:
输入: arr = [2,1]
输出: false

示例 2:
输入: arr = [3,5,5]
输出: false

示例 3:
输入: arr = [0,3,2,1]
输出: true

提示：
- 1 <= arr.length <= 10^4
- 0 <= arr[i] <= 10^4
"""

from typing import List


class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        i = 0
        while i < len(arr)-1 and arr[i] < arr[i + 1]:
            i += 1
        if i == 0 or i == len(arr) - 1:
            return False
        while i < len(arr)-1 and arr[i] > arr[i + 1]:
            i += 1
        return i == len(arr) - 1


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    arr = [2, 1]
    assert solution.validMountainArray(arr) == False

    # 测试用例2
    arr = [3, 5, 5]
    assert solution.validMountainArray(arr) == False

    # 测试用例3
    arr = [0, 3, 2, 1]
    assert solution.validMountainArray(arr) == True

    # 测试用例4
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert solution.validMountainArray(arr) == False

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
