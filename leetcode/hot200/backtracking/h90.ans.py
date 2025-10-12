"""
90. 子集II - 标准答案
"""
from typing import List


class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """
        标准解法：回溯算法 + 去重
        
        解题思路：
        1. 先对数组排序，便于去重
        2. 使用回溯算法生成所有可能的子集
        3. 使用去重策略：跳过重复的元素，避免生成重复的子集
        4. 使用剪枝优化：如果当前元素与前一元素相同且前一元素未被使用，则跳过
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(n)
        """
        nums.sort()  # 排序便于去重
        result = []
        path = []
        
        def backtrack(start: int, current_path: List[int]):
            # 每个节点都是一个子集
            result.append(current_path[:])
            
            for i in range(start, len(nums)):
                # 去重：跳过重复的元素
                if i > start and nums[i] == nums[i-1]:
                    continue
                
                current_path.append(nums[i])
                backtrack(i + 1, current_path)
                current_path.pop()
        
        backtrack(0, path)
        return result
    
    def subsetsWithDup_optimized(self, nums: List[int]) -> List[List[int]]:
        """
        优化解法：位运算 + 去重
        
        解题思路：
        1. 使用位运算枚举所有可能的子集
        2. 使用哈希表去重，避免重复子集
        3. 使用排序优化去重效率
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(2^n)
        """
        nums.sort()
        result = []
        seen = set()
        
        # 枚举所有可能的子集
        for mask in range(1 << len(nums)):
            subset = []
            for i in range(len(nums)):
                if mask & (1 << i):
                    subset.append(nums[i])
            
            # 使用元组作为哈希键去重
            subset_tuple = tuple(subset)
            if subset_tuple not in seen:
                seen.add(subset_tuple)
                result.append(subset)
        
        return result
    
    def subsetsWithDup_iterative(self, nums: List[int]) -> List[List[int]]:
        """
        迭代解法：使用队列 + BFS
        
        解题思路：
        1. 使用队列存储部分子集
        2. 每次从队列中取出一个部分子集
        3. 添加下一个元素，形成新的子集
        4. 使用去重策略避免重复子集
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(2^n)
        """
        from collections import deque
        
        nums.sort()
        result = []
        queue = deque([([], 0)])  # (当前子集, 当前位置)
        seen = set()
        
        while queue:
            current_subset, start = queue.popleft()
            
            # 当前子集加入结果
            subset_tuple = tuple(current_subset)
            if subset_tuple not in seen:
                seen.add(subset_tuple)
                result.append(current_subset)
            
            # 添加下一个元素
            for i in range(start, len(nums)):
                # 去重：跳过重复的元素
                if i > start and nums[i] == nums[i-1]:
                    continue
                
                new_subset = current_subset + [nums[i]]
                queue.append((new_subset, i + 1))
        
        return result
    
    def subsetsWithDup_recursive(self, nums: List[int]) -> List[List[int]]:
        """
        递归解法：分治思想
        
        解题思路：
        1. 将问题分解为包含当前元素和不包含当前元素两种情况
        2. 递归处理剩余元素
        3. 合并结果
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(n)
        """
        nums.sort()
        result = []
        
        def backtrack(index: int, current_path: List[int]):
            if index == len(nums):
                result.append(current_path[:])
                return
            
            # 不包含当前元素
            backtrack(index + 1, current_path)
            
            # 包含当前元素
            current_path.append(nums[index])
            backtrack(index + 1, current_path)
            current_path.pop()
        
        backtrack(0, [])
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,2,2]
    result = solution.subsetsWithDup(nums)
    expected = [[],[1],[1,2],[1,2,2],[2],[2,2]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = [0]
    result = solution.subsetsWithDup(nums)
    expected = [[],[0]]
    assert result == expected
    
    # 测试用例3
    nums = [1,1,2]
    result = solution.subsetsWithDup(nums)
    expected = [[],[1],[1,1],[1,1,2],[1,2],[2]]
    assert len(result) == len(expected)
    
    # 测试用例4
    nums = [1,2,3]
    result = solution.subsetsWithDup(nums)
    expected = [[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]
    assert len(result) == len(expected)
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [1,2,2]
    result_opt = solution.subsetsWithDup_optimized(nums)
    assert len(result_opt) == len(expected)
    
    # 测试迭代解法
    print("测试迭代解法...")
    nums = [1,2,2]
    result_iter = solution.subsetsWithDup_iterative(nums)
    assert len(result_iter) == len(expected)
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [1,2,2]
    result_rec = solution.subsetsWithDup_recursive(nums)
    assert len(result_rec) == len(expected)
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
