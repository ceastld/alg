"""
115. 不同的子序列
给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数。

题目数据保证答案符合 32 位带符号整数范围。

题目链接：https://leetcode.cn/problems/distinct-subsequences/

示例 1:
输入：s = "rabbbit", t = "rabbit"
输出：3
解释：
如下所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
rabbbit
rabbbit
rabbbit

示例 2:
输入：s = "babgbag", t = "bag"
输出：5
解释：
如下所示, 有 5 种可以从 s 中得到 "bag" 的方案。
babgbag
babgbag
babgbag
babgbag
babgbag

提示：
- 1 <= s.length, t.length <= 1000
- s 和 t 仅由英文字母组成
"""

from bisect import bisect_left
from collections import defaultdict
from typing import List


def convert_to_LIS(s2, s1):
    m = len(s1)
    hashmap = defaultdict(list)
    for i in range(m - 1, -1, -1):
        hashmap[s1[i]].append(i)
    candidates = []
    for c in s2:
        if c in hashmap:
            candidates.extend(hashmap[c])
    return candidates

def LIS(candidates):
    stack = []
    for num in candidates:
        idx = bisect_left(stack, num)
        if idx < len(stack):
            stack[idx] = num
        else:
            stack.append(num)
    return len(stack)

def numsOfLIS(nums: List[int]) -> int:
    if not nums:
        return 0
    tails = []
    dp = []
    for num in nums:
        count = 1
        index = bisect_left(tails, num)

        if index > 0:
            count = 0
            for kk, vv in dp[index - 1].items():
                if kk < num:
                    count += vv
        if index == len(tails):
            tails.append(num)
            dp.append({num: count})
        else:
            tails[index] = num
            if num in dp[index]:
                dp[index][num] += count
            else:
                dp[index][num] = count
    return sum(dp[-1].values()) if dp else 0


class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        if not t:
            return 1
        con = convert_to_LIS(s, t)
        if LIS(con) != len(t):
            return 0
        return numsOfLIS(con)


def main():
    """测试用例"""
    solution = Solution()

    # 测试用例1
    s = "rabbbit"
    t = "rabbit"
    result = solution.numDistinct(s, t)
    expected = 3
    assert result == expected

    # 测试用例2
    s = "babgbag"
    t = "bag"
    result = solution.numDistinct(s, t)
    expected = 5
    assert result == expected

    # 测试用例3
    s = "a"
    t = "a"
    result = solution.numDistinct(s, t)
    expected = 1
    assert result == expected

    # 测试用例4
    s = "a"
    t = "b"
    result = solution.numDistinct(s, t)
    expected = 0
    assert result == expected

    # 测试用例5
    s = "a"
    t = ""
    result = solution.numDistinct(s, t)
    expected = 1
    assert result == expected

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
