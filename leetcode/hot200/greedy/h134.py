"""
134. 加油站
在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站开始出发，开始时油箱为空。

给定两个整数数组 gas 和 cost，如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。如果题目有解，该答案即为唯一答案。

题目链接：https://leetcode.cn/problems/gas-station/

示例 1:
输入: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
输出: 3
解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 号加油站可以作为起始点。

示例 2:
输入: gas = [2,3,4], cost = [3,4,3]
输出: -1
解释:
你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
开往 1 号加油站，此时油箱有 3 - 4 + 3 = 2 升汽油
开往 2 号加油站，你需要消耗 4 升汽油，但是你的油箱只有 2 升汽油。
因此，无论怎样，你都不可能绕环路行驶一周。

提示：
- gas.length == n
- cost.length == n
- 1 <= n <= 10^5
- 0 <= gas[i], cost[i] <= 10^4
"""

from typing import List


class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        plus = [gas[i] - cost[i] for i in range(len(gas))]
        if sum(plus) < 0:
            return -1
        s = 0
        start = 0
        for i in range(len(plus)):
            s += plus[i]
            if s < 0:
                start = i + 1
                s = 0
        return start
        
def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    assert solution.canCompleteCircuit([1,2,3,4,5], [3,4,5,1,2]) == 3
    
    # 测试用例2
    assert solution.canCompleteCircuit([2,3,4], [3,4,3]) == -1
    
    # 测试用例3
    assert solution.canCompleteCircuit([1,2,3,4,5], [3,4,5,1,2]) == 3
    
    # 测试用例4
    assert solution.canCompleteCircuit([5,1,2,3,4], [4,4,1,5,1]) == 4
    
    # 测试用例5
    assert solution.canCompleteCircuit([2,3,4], [3,4,3]) == -1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
