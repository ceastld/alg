"""
455. 分发饼干 - 标准答案
"""
from typing import List


class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 将孩子和饼干都按从小到大排序
        2. 使用双指针，分别指向孩子和饼干
        3. 如果当前饼干能满足当前孩子，则分配并移动两个指针
        4. 否则只移动饼干指针，尝试更大的饼干
        5. 贪心策略：优先满足胃口小的孩子
        
        时间复杂度：O(n log n + m log m)
        空间复杂度：O(1)
        """
        # 排序
        g.sort()
        s.sort()
        
        child_idx = 0
        cookie_idx = 0
        satisfied = 0
        
        # 双指针遍历
        while child_idx < len(g) and cookie_idx < len(s):
            if s[cookie_idx] >= g[child_idx]:
                # 当前饼干能满足当前孩子
                satisfied += 1
                child_idx += 1
                cookie_idx += 1
            else:
                # 当前饼干太小，尝试下一个饼干
                cookie_idx += 1
        
        return satisfied
    
    def findContentChildren_alternative(self, g: List[int], s: List[int]) -> int:
        """
        替代解法：贪心算法（从大饼干开始分配）
        
        解题思路：
        1. 将孩子和饼干都按从大到小排序
        2. 优先用大饼干满足大胃口的孩子
        3. 如果当前饼干能满足当前孩子，则分配
        4. 否则尝试下一个孩子
        
        时间复杂度：O(n log n + m log m)
        空间复杂度：O(1)
        """
        # 从大到小排序
        g.sort(reverse=True)
        s.sort(reverse=True)
        
        child_idx = 0
        cookie_idx = 0
        satisfied = 0
        
        while child_idx < len(g) and cookie_idx < len(s):
            if s[cookie_idx] >= g[child_idx]:
                # 当前饼干能满足当前孩子
                satisfied += 1
                child_idx += 1
                cookie_idx += 1
            else:
                # 当前孩子胃口太大，尝试下一个孩子
                child_idx += 1
        
        return satisfied
    
    def findContentChildren_optimized(self, g: List[int], s: List[int]) -> int:
        """
        优化解法：贪心算法（使用内置函数）
        
        解题思路：
        1. 排序后使用双指针
        2. 使用enumerate和zip优化代码
        3. 提前终止条件
        
        时间复杂度：O(n log n + m log m)
        空间复杂度：O(1)
        """
        if not g or not s:
            return 0
        
        g.sort()
        s.sort()
        
        satisfied = 0
        cookie_idx = 0
        
        for child_greed in g:
            # 找到第一个能满足当前孩子的最小饼干
            while cookie_idx < len(s) and s[cookie_idx] < child_greed:
                cookie_idx += 1
            
            if cookie_idx < len(s):
                satisfied += 1
                cookie_idx += 1
            else:
                # 没有更多饼干了
                break
        
        return satisfied


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.findContentChildren([1,2,3], [1,1]) == 1
    
    # 测试用例2
    assert solution.findContentChildren([1,2], [1,2,3]) == 2
    
    # 测试用例3
    assert solution.findContentChildren([1,2,3], []) == 0
    
    # 测试用例4
    assert solution.findContentChildren([10,9,8,7], [5,6,7,8]) == 2
    
    # 测试用例5：边界情况
    assert solution.findContentChildren([], [1,2,3]) == 0
    assert solution.findContentChildren([1,2,3], [1,2,3]) == 3
    
    # 测试用例6：大胃口孩子
    assert solution.findContentChildren([5,6,7,8], [1,2,3,4]) == 0
    
    # 测试用例7：大饼干
    assert solution.findContentChildren([1,2,3], [10,11,12]) == 3
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
