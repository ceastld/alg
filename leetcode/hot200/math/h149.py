"""
149. 直线上最多的点数
给你一个数组 points ，其中 points[i] = [xi, yi] 表示 X-Y 平面上的一个点。求最多有多少个点在同一条直线上。

题目链接：https://leetcode.cn/problems/max-points-on-a-line/

示例 1:
输入：points = [[1,1],[2,2],[3,3]]
输出：3

示例 2:
输入：points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
输出：4

提示：
- 1 <= points.length <= 300
- points[i].length == 2
- -10^4 <= xi, yi <= 10^4
- points 中的所有点 互不相同
"""

from typing import List


class Slope:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x * other.y == self.y * other.x

    @staticmethod
    def from_points(p1: List[int], p2: List[int]) -> "Slope":
        return Slope(p2[0] - p1[0], p2[1] - p1[1])


class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x: (x[0], x[1]))
        slopes = [[Slope.from_points(points[i], points[j]) for j in range(i+1,len(points))] for i in range(len(points))]
        def get_slope(i,j):
            return slopes[i][j-i-1]
        
        max_points = 1
        def dfs():
            for i in range(len(points)):
                for j in range(i+1,len(points)):
                    if get_slope(i,j) == Slope(0,1):
                        dfs(i,j)
        dfs()
        return max_points


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    points = [[1, 1], [2, 2], [3, 3]]
    result = solution.maxPoints(points)
    expected = 3
    assert result == expected

    # 测试用例2
    points = [[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]]
    result = solution.maxPoints(points)
    expected = 4
    assert result == expected

    # 测试用例3
    points = [[1, 1], [2, 2], [3, 3], [4, 4]]
    result = solution.maxPoints(points)
    expected = 4
    assert result == expected

    # 测试用例4
    points = [[0, 0]]
    result = solution.maxPoints(points)
    expected = 1
    assert result == expected

    # 测试用例5
    points = [[0, 0], [1, 1], [0, 0]]
    result = solution.maxPoints(points)
    expected = 3
    assert result == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
