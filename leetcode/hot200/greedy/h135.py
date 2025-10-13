"""
135. 分发糖果
n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。

你需要按照以下要求，给这些孩子分发糖果：

每个孩子至少分发一个糖果。
相邻的孩子中，评分高的孩子必须获得更多的糖果。
请你计算并返回满足条件的分发糖果的最少数量。

题目链接：https://leetcode.cn/problems/candy/

示例 1:
输入：ratings = [1,0,2]
输出：5
解释：你可以分别给这三个孩子分发 2、1、2 颗糖果。

示例 2:
输入：ratings = [1,2,2]
输出：4
解释：你可以分别给这三个孩子分发 1、2、1 颗糖果。
     第三个孩子只得到 1 颗糖果，这满足题目的两个条件。

提示：
- n == ratings.length
- 1 <= n <= 2 * 10^4
- 0 <= ratings[i] <= 2 * 10^4
"""

from typing import List


class Solution:
    def candy(self, ratings: List[int]) -> int:
        candy = [1] * len(ratings)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candy[i] = candy[i - 1] + 1
        for i in range(len(ratings) - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candy[i] = max(candy[i], candy[i + 1] + 1)
        return sum(candy)


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.candy([1,0,2]) == 5
    
    # 测试用例2
    assert solution.candy([1,2,2]) == 4
    
    # 测试用例3
    assert solution.candy([1,2,3,4,5]) == 15
    
    # 测试用例4
    assert solution.candy([5,4,3,2,1]) == 15
    
    # 测试用例5
    assert solution.candy([1,3,2,2,1]) == 7
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
