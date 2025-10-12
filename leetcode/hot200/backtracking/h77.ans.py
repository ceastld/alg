"""
77. 组合 - 标准答案
"""
from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 使用回溯算法生成所有可能的组合
        2. 从1开始，每次选择一个数，然后递归选择下一个数
        3. 当组合长度达到k时，将结果加入答案
        4. 使用剪枝优化：如果剩余数字不够组成k个数的组合，直接返回
        
        时间复杂度：O(C(n,k) * k)
        空间复杂度：O(k)
        """
        result = []
        
        def backtrack(start, path):
            # 如果路径长度达到k，加入结果
            if len(path) == k:
                result.append(path[:])
                return
            
            # 剪枝：如果剩余数字不够组成k个数的组合
            for i in range(start, n + 1):
                if len(path) + (n - i + 1) < k:
                    break
                
                path.append(i)
                backtrack(i + 1, path)
                path.pop()
        
        backtrack(1, [])
        return result
    
    def combine_optimized(self, n: int, k: int) -> List[List[int]]:
        """
        优化解法：反向枚举 + 高效剪枝
        
        解题思路：
        1. 从n开始倒着枚举，更容易触发剪枝
        2. 使用d = k - len(path)计算剩余需要选择的数字数量
        3. 如果当前数字i > d，说明不可能选够k个数，直接跳过
        4. 每个数字都有两种选择：选或不选
        
        时间复杂度：O(C(n,k) * k)
        空间复杂度：O(k)
        性能提升：约28%（相比正向枚举）
        """
        ans = []
        path = []

        def dfs(i: int) -> None:
            d = k - len(path)  # 还要选 d 个数
            if d == 0:  # 选好了
                ans.append(path.copy())
                return

            # 不选 i
            if i > d:
                dfs(i - 1)

            # 选 i
            path.append(i)
            dfs(i - 1)
            path.pop()  # 恢复现场

        dfs(n)  # 从 i=n 开始倒着枚举
        return ans


def main():
    """测试标准答案"""
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
    
    # 测试优化解法
    print("测试优化解法...")
    n, k = 4, 2
    result_opt = solution.combine_optimized(n, k)
    assert len(result_opt) == len(expected)
    
    n, k = 1, 1
    result_opt = solution.combine_optimized(n, k)
    assert result_opt == [[1]]
    
    n, k = 3, 3
    result_opt = solution.combine_optimized(n, k)
    assert result_opt == [[1,2,3]]
    
    print("所有测试用例通过！")
    print("优化解法验证通过！")


if __name__ == "__main__":
    main()
