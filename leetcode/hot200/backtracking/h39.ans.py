"""
39. 组合总和 - 标准答案
"""
from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 使用回溯算法生成所有可能的组合
        2. 对数组排序，便于剪枝
        3. 每次选择一个数，递归搜索剩余目标
        4. 使用剪枝优化：如果当前数大于剩余目标，直接跳过
        5. 允许重复选择同一个数，所以递归时从当前位置开始
        
        时间复杂度：O(2^target)
        空间复杂度：O(target)
        """
        result = []
        candidates.sort()  # 排序便于剪枝
        
        def backtrack(start, path, remaining):
            # 如果剩余目标为0，找到解
            if remaining == 0:
                result.append(path[:])
                return
            
            # 从start开始遍历候选数
            for i in range(start, len(candidates)):
                num = candidates[i]
                
                # 剪枝：如果当前数大于剩余目标，直接跳过
                if num > remaining:
                    break
                
                path.append(num)
                backtrack(i, path, remaining - num)  # 允许重复选择
                path.pop()
        
        backtrack(0, [], target)
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    candidates = [2, 3, 6, 7]
    target = 7
    result = solution.combinationSum(candidates, target)
    expected = [[2,2,3],[7]]
    assert len(result) == len(expected)
    
    # 测试用例2
    candidates = [2, 3, 5]
    target = 8
    result = solution.combinationSum(candidates, target)
    expected = [[2,2,2,2],[2,3,3],[3,5]]
    assert len(result) == len(expected)
    
    # 测试用例3
    candidates = [2]
    target = 1
    result = solution.combinationSum(candidates, target)
    expected = []
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
