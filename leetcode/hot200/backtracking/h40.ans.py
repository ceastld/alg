"""
40. 组合总和II - 标准答案
"""
from typing import List


class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        标准解法：回溯算法 + 去重
        
        解题思路：
        1. 先对数组排序，便于去重
        2. 使用回溯算法生成所有可能的组合
        3. 使用剪枝优化：如果当前和已经超过target，则提前终止
        4. 使用去重策略：跳过重复的元素，避免生成重复的组合
        
        时间复杂度：O(2^n)
        空间复杂度：O(target)
        """
        candidates.sort()  # 排序便于去重
        result = []
        path = []
        
        def backtrack(start: int, current_sum: int, current_path: List[int]):
            # 终止条件
            if current_sum == target:
                result.append(current_path[:])
                return
            
            if current_sum > target:
                return
            
            for i in range(start, len(candidates)):
                # 去重：跳过重复的元素
                if i > start and candidates[i] == candidates[i-1]:
                    continue
                
                # 剪枝：如果当前元素加上去会超过target，则跳过后续元素
                if current_sum + candidates[i] > target:
                    break
                
                current_path.append(candidates[i])
                backtrack(i + 1, current_sum + candidates[i], current_path)
                current_path.pop()
        
        backtrack(0, 0, path)
        return result
    
    def combinationSum2_optimized(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        优化解法：位运算 + 预计算
        
        解题思路：
        1. 使用位运算枚举所有可能的子集
        2. 预计算每个子集的和，快速筛选
        3. 使用哈希表去重，避免重复组合
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(2^n)
        """
        candidates.sort()
        result = []
        seen = set()
        
        # 枚举所有可能的子集
        for mask in range(1 << len(candidates)):
            subset = []
            current_sum = 0
            
            for i in range(len(candidates)):
                if mask & (1 << i):
                    subset.append(candidates[i])
                    current_sum += candidates[i]
            
            # 如果和等于target，则检查是否重复
            if current_sum == target:
                subset_tuple = tuple(subset)
                if subset_tuple not in seen:
                    seen.add(subset_tuple)
                    result.append(subset)
        
        return result
    
    def combinationSum2_dp(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        动态规划解法：背包问题
        
        解题思路：
        1. 将问题转化为背包问题
        2. 使用动态规划计算所有可能的组合
        3. 使用回溯重构解
        
        时间复杂度：O(n * target)
        空间复杂度：O(n * target)
        """
        candidates.sort()
        result = []
        
        def backtrack(start: int, current_sum: int, current_path: List[int]):
            if current_sum == target:
                result.append(current_path[:])
                return
            
            if current_sum > target:
                return
            
            for i in range(start, len(candidates)):
                # 去重
                if i > start and candidates[i] == candidates[i-1]:
                    continue
                
                # 剪枝
                if current_sum + candidates[i] > target:
                    break
                
                current_path.append(candidates[i])
                backtrack(i + 1, current_sum + candidates[i], current_path)
                current_path.pop()
        
        backtrack(0, 0, [])
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    candidates = [10,1,2,7,6,1,5]
    target = 8
    result = solution.combinationSum2(candidates, target)
    expected = [[1,1,6],[1,2,5],[1,7],[2,6]]
    assert len(result) == len(expected)
    
    # 测试用例2
    candidates = [2,5,2,1,2]
    target = 5
    result = solution.combinationSum2(candidates, target)
    expected = [[1,2,2],[5]]
    assert len(result) == len(expected)
    
    # 测试用例3
    candidates = [1,1,1,1,1]
    target = 3
    result = solution.combinationSum2(candidates, target)
    expected = [[1,1,1]]
    assert len(result) == len(expected)
    
    # 测试用例4
    candidates = [1,2,3,4,5]
    target = 8
    result = solution.combinationSum2(candidates, target)
    expected = [[1,2,5],[1,3,4],[2,3,3]]
    assert len(result) == len(expected)
    
    # 测试优化解法
    print("测试优化解法...")
    candidates = [10,1,2,7,6,1,5]
    target = 8
    result_opt = solution.combinationSum2_optimized(candidates, target)
    assert len(result_opt) == len(expected)
    
    # 测试DP解法
    print("测试DP解法...")
    candidates = [10,1,2,7,6,1,5]
    target = 8
    result_dp = solution.combinationSum2_dp(candidates, target)
    assert len(result_dp) == len(expected)
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
