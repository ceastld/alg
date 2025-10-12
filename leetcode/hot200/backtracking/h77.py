"""
77. 组合
给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。

你可以按 任何顺序 返回答案。

题目链接：https://leetcode.cn/problems/combinations/

示例 1:
输入: n = 4, k = 2
输出: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]

示例 2:
输入: n = 1, k = 1
输出: [[1]]

提示：
- 1 <= n <= 20
- 1 <= k <= n
"""
from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        请在这里实现你的解法
        """
        path = []
        def backtrack(start):
            if len(path) == k:
                result.append(path.copy())
                return
            for i in range(start, n+1):
                path.append(i)
                backtrack(i+1)
                path.pop()
        result = []
        backtrack(1)
        return result


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    n, k = 4, 2
    result = solution.combine(n, k)
    expected = [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    assert len(result) == len(expected)
    
    # 测试用例2
    n, k = 1, 1
    result = solution.combine(n, k)
    expected = [[1]]
    assert result == expected
    
    # 测试用例3
    n, k = 3, 3
    result = solution.combine(n, k)
    expected = [[1,2,3]]
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
