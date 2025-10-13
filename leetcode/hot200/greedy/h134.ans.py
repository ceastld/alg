"""
134. 加油站 - 标准答案
"""
from typing import List


class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 如果总油量小于总消耗，则无解
        2. 从0开始遍历，维护当前油量和起始位置
        3. 如果当前油量小于0，说明从当前起始位置无法到达当前位置
        4. 更新起始位置为下一个位置，重置当前油量
        5. 贪心策略：一旦发现从某个位置开始无法到达某个位置，就尝试从下一个位置开始
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 检查是否有解
        if sum(gas) < sum(cost):
            return -1
        
        start = 0
        current_gas = 0
        
        for i in range(len(gas)):
            current_gas += gas[i] - cost[i]
            
            # 如果当前油量小于0，说明从start开始无法到达位置i
            if current_gas < 0:
                start = i + 1
                current_gas = 0
        
        return start
    
    def canCompleteCircuit_alternative(self, gas: List[int], cost: List[int]) -> int:
        """
        替代解法：贪心算法（一次遍历）
        
        解题思路：
        1. 维护两个变量：total_tank和current_tank
        2. total_tank记录总的油量差值
        3. current_tank记录从当前起始位置开始的油量
        4. 如果current_tank < 0，更新起始位置
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        total_tank = 0
        current_tank = 0
        start = 0
        
        for i in range(len(gas)):
            total_tank += gas[i] - cost[i]
            current_tank += gas[i] - cost[i]
            
            # 如果当前油量小于0，说明从start开始无法到达位置i
            if current_tank < 0:
                start = i + 1
                current_tank = 0
        
        return start if total_tank >= 0 else -1
    
    def canCompleteCircuit_brute_force(self, gas: List[int], cost: List[int]) -> int:
        """
        暴力解法：尝试每个起始位置
        
        解题思路：
        1. 尝试从每个位置开始
        2. 模拟绕环路行驶一周的过程
        3. 如果从某个位置开始能成功，返回该位置
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        n = len(gas)
        
        for start in range(n):
            current_gas = 0
            can_complete = True
            
            for i in range(n):
                current_pos = (start + i) % n
                current_gas += gas[current_pos] - cost[current_pos]
                
                if current_gas < 0:
                    can_complete = False
                    break
            
            if can_complete:
                return start
        
        return -1
    
    def canCompleteCircuit_optimized(self, gas: List[int], cost: List[int]) -> int:
        """
        优化解法：贪心算法（详细版）
        
        解题思路：
        1. 首先检查是否有解（总油量 >= 总消耗）
        2. 使用贪心策略找到起始位置
        3. 维护当前油量和起始位置
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        n = len(gas)
        
        # 检查是否有解
        total_gas = sum(gas)
        total_cost = sum(cost)
        if total_gas < total_cost:
            return -1
        
        start = 0
        current_gas = 0
        
        for i in range(n):
            current_gas += gas[i] - cost[i]
            
            # 如果当前油量小于0，说明从start开始无法到达位置i
            if current_gas < 0:
                # 尝试从下一个位置开始
                start = i + 1
                current_gas = 0
        
        return start
    
    def canCompleteCircuit_detailed(self, gas: List[int], cost: List[int]) -> int:
        """
        详细解法：贪心算法（带注释）
        
        解题思路：
        1. 维护当前油量和起始位置
        2. 遍历每个加油站
        3. 更新当前油量
        4. 如果油量不足，更新起始位置
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not gas or not cost:
            return -1
        
        n = len(gas)
        start = 0
        current_gas = 0
        
        for i in range(n):
            # 在位置i加油
            current_gas += gas[i]
            # 从位置i到位置i+1消耗油量
            current_gas -= cost[i]
            
            # 如果油量不足，说明从start开始无法到达位置i+1
            if current_gas < 0:
                # 尝试从下一个位置开始
                start = i + 1
                current_gas = 0
        
        return start


def main():
    """测试标准答案"""
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
    
    # 测试用例6：边界情况
    assert solution.canCompleteCircuit([1], [1]) == 0
    assert solution.canCompleteCircuit([1], [2]) == -1
    
    # 测试用例7：单站
    assert solution.canCompleteCircuit([2], [2]) == 0
    assert solution.canCompleteCircuit([1], [2]) == -1
    
    # 测试用例8：两站
    assert solution.canCompleteCircuit([1,2], [2,1]) == 1
    assert solution.canCompleteCircuit([2,1], [1,2]) == 0
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
