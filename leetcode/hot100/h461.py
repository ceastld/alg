"""
LeetCode 461. Hamming Distance

题目描述：
两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目。
给你两个整数x和y，计算并返回它们之间的汉明距离。

示例：
x = 1, y = 4
输出：2
解释：
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
上面的箭头指出了对应二进制位不同的位置。

数据范围：
- 0 <= x, y <= 2^31 - 1
"""

class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')
