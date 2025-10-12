"""
216. 组合总和III
找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：

- 只使用数字1到9
- 每个数字 最多使用一次

返回 所有可能的有效组合的列表 。列表中的每个组合都是唯一的。

题目链接：https://leetcode.cn/problems/combination-sum-iii/

示例 1:
输入: k = 3, n = 7
输出: [[1,2,4]]
解释:
1 + 2 + 4 = 7
没有其他符合的组合了。

示例 2:
输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]
解释:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
没有其他符合的组合了。

示例 3:
输入: k = 4, n = 1
输出: []
解释: 不存在有效的组合。

提示:
- 2 <= k <= 9
- 1 <= n <= 60
"""

from typing import List


class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        def dfs(start, path, remaining):
            if remaining == 0 and len(path) == k:
                result.append(path.copy())
                return
            if start > remaining:
                return
            for i in range(start, min(10, remaining+1)):
                dfs(i+1, path+[i], remaining - i)
                
        result = []
        dfs(1, [], n)
        return result


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    k, n = 3, 7
    result = solution.combinationSum3(k, n)
    expected = [[1,2,4]]
    assert result == expected
    
    # 测试用例2
    k, n = 3, 9
    result = solution.combinationSum3(k, n)
    expected = [[1,2,6], [1,3,5], [2,3,4]]
    assert result == expected
    
    # 测试用例3
    k, n = 4, 1
    result = solution.combinationSum3(k, n)
    expected = []
    assert result == expected
    
    # 测试用例4
    k, n = 2, 18
    result = solution.combinationSum3(k, n)
    expected = []
    assert result == expected
    
    # 测试用例5
    k, n = 9, 45
    result = solution.combinationSum3(k, n)
    expected = [[1,2,3,4,5,6,7,8,9]]
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
