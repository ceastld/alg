"""
1049. 最后一块石头的重量II
有一堆石头，用整数数组 stones 表示。其中 stones[i] 表示第 i 块石头的重量。

每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：

如果 x == y，那么两块石头都会被完全粉碎；
如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。

最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。

题目链接：https://leetcode.cn/problems/last-stone-weight-ii/

示例 1:
输入：stones = [2,7,4,1,8,1]
输出：1
解释：
组合 2 和 4，得到 2，所以数组转化为 [2,7,1,8,1]，
组合 7 和 8，得到 1，所以数组转化为 [2,1,1,1]，
组合 2 和 1，得到 1，所以数组转化为 [1,1,1]，
组合 1 和 1，得到 0，所以数组转化为 [1]，这就是最优值。

示例 2:
输入：stones = [31,26,33,21,40]
输出：5

提示：
- 1 <= stones.length <= 30
- 1 <= stones[i] <= 100
"""

from typing import List

class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        s = sum(stones)
        target = s // 2
        dp = [False] * (target + 1)
        dp[0] = True
        for stone in stones:
            for i in range(target, stone - 1, -1):
                dp[i] = dp[i] or dp[i - stone]
        for i in range(target, -1, -1):
            if dp[i]:
                print(s,s-2*i)
                return s - 2 * i
        return s


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    stones = [2,7,4,1,8,1]
    result = solution.lastStoneWeightII(stones)
    expected = 1
    assert result == expected
    
    # 测试用例2
    stones = [31,26,33,21,40]
    result = solution.lastStoneWeightII(stones)
    expected = 5
    assert result == expected
    
    # 测试用例3
    stones = [1,2]
    result = solution.lastStoneWeightII(stones)
    expected = 1
    assert result == expected
        
    stones = [1]
    result = solution.lastStoneWeightII(stones)
    expected = 1
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
