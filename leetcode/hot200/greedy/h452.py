"""
452. 用最少数量的箭引爆气球
有一些球形气球贴在一堵用 XY 平面表示的墙上。墙上的气球用一个二维数组 points 表示，其中 points[i] = [xstart, xend] 表示水平直径在 xstart 和 xend 之间的气球。你不知道气球的确切 y 坐标。

一支弓箭可以沿着 x 轴从不同点 完全垂直 地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend，且满足  xstart ≤ x ≤ xend，则该气球会被 引爆 。可以射出的弓箭的数量 没有限制 。 弓箭一旦被射出之后，可以无限地前进。

给你一个数组 points ，返回引爆所有气球所必须射出的 最小 弓箭数 。

题目链接：https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/

示例 1:
输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
解释：气球可以用2支箭来爆破:
-在x = 6处射出箭，击破气球[2,8]和[1,6]。
-在x = 11处射出箭，击破气球[10,16]和[7,12]。

示例 2:
输入：points = [[1,2],[3,4],[5,6],[7,8]]
输出：4
解释：每个气球需要射出一支箭，总共需要4支箭。

示例 3:
输入：points = [[1,2],[2,3],[3,4],[4,5]]
输出：2
解释：气球可以用2支箭来爆破:
- 在x = 2处射出箭，击破气球[1,2]和[2,3]。
- 在x = 4处射出箭，击破气球[3,4]和[4,5]。

提示：
- 1 <= points.length <= 10^5
- points[i].length == 2
- -2^31 <= xstart < xend <= 2^31 - 1
"""

from typing import List


class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x: x[1])
        end = points[0][1]
        count = 1
        for p in points[1:]:
            if p[0] > end:
                count += 1
                end = p[1]
        return count

def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.findMinArrowShots([[3,9],[7,12],[3,8],[6,8],[9,10],[2,9],[0,9],[3,9],[0,6],[2,8]]) == 2
    
    # 测试用例2
    assert solution.findMinArrowShots([[9,12],[1,10],[4,11],[8,12],[3,9],[6,9],[6,7]]) == 2
    
    # 测试用例3
    assert solution.findMinArrowShots([[1,2],[2,3],[3,4],[4,5]]) == 2
    
    # 测试用例4
    assert solution.findMinArrowShots([[1,2],[2,3],[3,4],[4,5],[5,6]]) == 3
    
    # 测试用例5
    assert solution.findMinArrowShots([[1,2]]) == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
