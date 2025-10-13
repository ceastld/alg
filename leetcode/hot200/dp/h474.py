"""
474. 一和零
给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。

如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。

题目链接：https://leetcode.cn/problems/ones-and-zeroes/

示例 1:
输入：strs = ["10","0001","111001","1","0"], m = 5, n = 3
输出：4
解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
其他满足题意但较小的子集包括 {"0001","1"} 以及 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。

示例 2:
输入：strs = ["10","0","1"], m = 1, n = 1
输出：2
解释：最大的子集是 {"0","1"} ，所以答案是 2 。

提示：
- 1 <= strs.length <= 600
- 1 <= strs[i].length <= 100
- strs[i] 仅由字符 '0' 和 '1' 组成
- 1 <= m, n <= 100
"""

from typing import List


class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:
            zeros = s.count("0")
            ones = len(s) - zeros
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    v = dp[i - zeros][j - ones] + 1
                    if v > dp[i][j]:
                        dp[i][j] = v
        return dp[m][n]


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    strs = ["10", "0001", "111001", "1", "0"]
    m = 5
    n = 3
    result = solution.findMaxForm(strs, m, n)
    expected = 4
    assert result == expected

    # 测试用例2
    strs = ["10", "0", "1"]
    m = 1
    n = 1
    result = solution.findMaxForm(strs, m, n)
    expected = 2
    assert result == expected

    # 测试用例3
    strs = ["10", "0001", "111001", "1", "0"]
    m = 3
    n = 2
    result = solution.findMaxForm(strs, m, n)
    expected = 3
    assert result == expected

    # 测试用例4
    strs = ["10", "0001", "111001", "1", "0"]
    m = 1
    n = 1
    result = solution.findMaxForm(strs, m, n)
    expected = 2
    assert result == expected

    # 测试用例5
    strs = ["10", "0001", "111001", "1", "0"]
    m = 0
    n = 0
    result = solution.findMaxForm(strs, m, n)
    expected = 0
    assert result == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
