"""
491. 递增子序列 - 标准答案
"""
from typing import List


class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        """
        标准解法：回溯算法 + 去重
        
        解题思路：
        1. 使用回溯算法生成所有可能的递增子序列
        2. 使用去重策略：跳过重复的元素，避免生成重复的子序列
        3. 使用剪枝优化：如果当前元素小于序列最后一个元素，则跳过
        4. 使用哈希表去重，避免重复子序列
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(n)
        """
        result = []
        path = []
        
        def backtrack(start: int, current_path: List[int]):
            # 如果当前序列长度大于等于2，则加入结果
            if len(current_path) >= 2:
                result.append(current_path[:])
            
            # 使用集合记录当前层已使用的元素，避免重复
            used = set()
            
            for i in range(start, len(nums)):
                # 去重：跳过重复的元素
                if nums[i] in used:
                    continue
                
                # 剪枝：如果当前元素小于序列最后一个元素，则跳过
                if current_path and nums[i] < current_path[-1]:
                    continue
                
                used.add(nums[i])
                current_path.append(nums[i])
                backtrack(i + 1, current_path)
                current_path.pop()
        
        backtrack(0, path)
        return result
    
    def findSubsequences_optimized(self, nums: List[int]) -> List[List[int]]:
        """
        优化解法：位运算 + 去重
        
        解题思路：
        1. 使用位运算枚举所有可能的子序列
        2. 使用哈希表去重，避免重复子序列
        3. 使用排序优化去重效率
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(2^n)
        """
        result = []
        seen = set()
        
        # 枚举所有可能的子序列
        for mask in range(1 << len(nums)):
            subsequence = []
            for i in range(len(nums)):
                if mask & (1 << i):
                    subsequence.append(nums[i])
            
            # 检查是否为递增序列
            if len(subsequence) >= 2 and self.is_increasing(subsequence):
                # 使用元组作为哈希键去重
                subsequence_tuple = tuple(subsequence)
                if subsequence_tuple not in seen:
                    seen.add(subsequence_tuple)
                    result.append(subsequence)
        
        return result
    
    def is_increasing(self, arr: List[int]) -> bool:
        """检查数组是否为递增序列"""
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                return False
        return True
    
    def findSubsequences_iterative(self, nums: List[int]) -> List[List[int]]:
        """
        迭代解法：使用队列 + BFS
        
        解题思路：
        1. 使用队列存储部分子序列
        2. 每次从队列中取出一个部分子序列
        3. 添加下一个元素，形成新的子序列
        4. 使用去重策略避免重复子序列
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(2^n)
        """
        from collections import deque
        
        result = []
        queue = deque([([], 0)])  # (当前子序列, 当前位置)
        seen = set()
        
        while queue:
            current_subsequence, start = queue.popleft()
            
            # 如果当前序列长度大于等于2，则加入结果
            if len(current_subsequence) >= 2:
                subsequence_tuple = tuple(current_subsequence)
                if subsequence_tuple not in seen:
                    seen.add(subsequence_tuple)
                    result.append(current_subsequence)
            
            # 添加下一个元素
            used = set()
            for i in range(start, len(nums)):
                # 去重：跳过重复的元素
                if nums[i] in used:
                    continue
                
                # 剪枝：如果当前元素小于序列最后一个元素，则跳过
                if current_subsequence and nums[i] < current_subsequence[-1]:
                    continue
                
                used.add(nums[i])
                new_subsequence = current_subsequence + [nums[i]]
                queue.append((new_subsequence, i + 1))
        
        return result
    
    def findSubsequences_recursive(self, nums: List[int]) -> List[List[int]]:
        """
        递归解法：分治思想
        
        解题思路：
        1. 将问题分解为包含当前元素和不包含当前元素两种情况
        2. 递归处理剩余元素
        3. 合并结果
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(n)
        """
        result = []
        
        def backtrack(index: int, current_path: List[int]):
            if len(current_path) >= 2:
                result.append(current_path[:])
            
            if index == len(nums):
                return
            
            # 不包含当前元素
            backtrack(index + 1, current_path)
            
            # 包含当前元素（如果满足递增条件）
            if not current_path or nums[index] >= current_path[-1]:
                current_path.append(nums[index])
                backtrack(index + 1, current_path)
                current_path.pop()
        
        backtrack(0, [])
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [4,6,7,7]
    result = solution.findSubsequences(nums)
    expected = [[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = [4,4,3,2,1]
    result = solution.findSubsequences(nums)
    expected = [[4,4]]
    assert result == expected
    
    # 测试用例3
    nums = [1,2,3,4]
    result = solution.findSubsequences(nums)
    expected = [[1,2],[1,3],[1,4],[1,2,3],[1,2,4],[1,3,4],[1,2,3,4],[2,3],[2,4],[2,3,4],[3,4]]
    assert len(result) == len(expected)
    
    # 测试用例4
    nums = [1,1,1]
    result = solution.findSubsequences(nums)
    expected = [[1,1],[1,1,1]]
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [4,6,7,7]
    result_opt = solution.findSubsequences_optimized(nums)
    assert len(result_opt) == len(expected)
    
    # 测试迭代解法
    print("测试迭代解法...")
    nums = [4,6,7,7]
    result_iter = solution.findSubsequences_iterative(nums)
    assert len(result_iter) == len(expected)
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [4,6,7,7]
    result_rec = solution.findSubsequences_recursive(nums)
    assert len(result_rec) == len(expected)
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
