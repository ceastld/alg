"""
649. Dota2参议院 - 标准答案
"""
from typing import List
from collections import deque

class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        """
        标准解法：贪心算法 + 队列
        
        解题思路：
        1. 使用两个队列分别存储R和D的位置
        2. 每次从两个队列头部取出位置，位置小的先行动
        3. 先行动的参议员会禁止后行动的参议员
        4. 被禁止的参议员从队列中移除，未被禁止的参议员重新加入队列
        5. 重复直到某一方队列为空
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        radiant_queue = deque()
        dire_queue = deque()
        
        # 初始化队列
        for i, senator in enumerate(senate):
            if senator == 'R':
                radiant_queue.append(i)
            else:
                dire_queue.append(i)
        
        # 模拟投票过程
        while radiant_queue and dire_queue:
            r_pos = radiant_queue.popleft()
            d_pos = dire_queue.popleft()
            
            if r_pos < d_pos:
                # R先行动，禁止D，R重新加入队列
                radiant_queue.append(r_pos + len(senate))
            else:
                # D先行动，禁止R，D重新加入队列
                dire_queue.append(d_pos + len(senate))
        
        return "Radiant" if radiant_queue else "Dire"

def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.predictPartyVictory("RD") == "Radiant"
    
    # 测试用例2
    assert solution.predictPartyVictory("RDD") == "Dire"
    
    # 测试用例3
    assert solution.predictPartyVictory("RRDDD") == "Radiant"
    
    print("所有测试用例通过！")
