"""
47. 全排列II - 标准答案
"""
from typing import List


class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """
        标准解法：回溯算法 + 去重
        
        解题思路：
        1. 先对数组排序，便于去重
        2. 使用回溯算法生成所有可能的排列
        3. 使用去重策略：跳过重复的元素，避免生成重复的排列
        4. 使用剪枝优化：如果当前元素与前一元素相同且前一元素未被使用，则跳过
        
        时间复杂度：O(n! * n)
        空间复杂度：O(n)
        """
        nums.sort()  # 排序便于去重
        result = []
        path = []
        used = [False] * len(nums)
        
        def backtrack():
            # 终止条件
            if len(path) == len(nums):
                result.append(path[:])
                return
            
            for i in range(len(nums)):
                # 跳过已使用的元素
                if used[i]:
                    continue
                
                # 去重：跳过重复的元素
                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                
                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()
                used[i] = False
        
        backtrack()
        return result
    
    def permuteUnique_optimized(self, nums: List[int]) -> List[List[int]]:
        """
        优化解法：原地交换 + 去重
        
        解题思路：
        1. 使用原地交换避免visited数组的开销
        2. 通过交换元素位置来生成不同的排列
        3. 递归时交换当前元素与后续元素
        4. 回溯时恢复交换，保持数组原状
        5. 使用去重策略避免重复排列
        
        时间复杂度：O(n! * n)
        空间复杂度：O(1) - 除了结果数组
        """
        nums.sort()
        result = []
        
        def backtrack(start):
            if start == len(nums):
                result.append(nums[:])
                return
            
            used = set()
            for i in range(start, len(nums)):
                # 去重：跳过重复的元素
                if nums[i] in used:
                    continue
                
                used.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]
        
        backtrack(0)
        return result
    
    def permuteUnique_iterative(self, nums: List[int]) -> List[List[int]]:
        """
        迭代解法：使用队列 + BFS
        
        解题思路：
        1. 使用队列存储部分排列
        2. 每次从队列中取出一个部分排列
        3. 添加一个未使用的元素，形成新的部分排列
        4. 当部分排列长度等于数组长度时，加入结果
        5. 使用去重策略避免重复排列
        
        时间复杂度：O(n! * n)
        空间复杂度：O(n! * n)
        """
        from collections import deque
        
        nums.sort()
        result = []
        queue = deque([([], [False] * len(nums))])  # (当前排列, 已使用标记)
        seen = set()
        
        while queue:
            path, used = queue.popleft()
            
            if len(path) == len(nums):
                path_tuple = tuple(path)
                if path_tuple not in seen:
                    seen.add(path_tuple)
                    result.append(path)
                continue
            
            used_set = set()
            for i in range(len(nums)):
                if used[i] or nums[i] in used_set:
                    continue
                
                used_set.add(nums[i])
                new_path = path + [nums[i]]
                new_used = used[:]
                new_used[i] = True
                queue.append((new_path, new_used))
        
        return result
    
    def permuteUnique_recursive(self, nums: List[int]) -> List[List[int]]:
        """
        递归解法：分治思想
        
        解题思路：
        1. 将问题分解为包含当前元素和不包含当前元素两种情况
        2. 递归处理剩余元素
        3. 合并结果
        
        时间复杂度：O(n! * n)
        空间复杂度：O(n)
        """
        nums.sort()
        result = []
        
        def backtrack(remaining: List[int], current_path: List[int]):
            if not remaining:
                result.append(current_path[:])
                return
            
            used = set()
            for i in range(len(remaining)):
                if remaining[i] in used:
                    continue
                
                used.add(remaining[i])
                current_path.append(remaining[i])
                backtrack(remaining[:i] + remaining[i+1:], current_path)
                current_path.pop()
        
        backtrack(nums, [])
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1,1,2]
    result = solution.permuteUnique(nums)
    expected = [[1,1,2],[1,2,1],[2,1,1]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = [1,2,3]
    result = solution.permuteUnique(nums)
    expected = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    assert len(result) == len(expected)
    
    # 测试用例3
    nums = [1,1,1]
    result = solution.permuteUnique(nums)
    expected = [[1,1,1]]
    assert result == expected
    
    # 测试用例4
    nums = [1,2]
    result = solution.permuteUnique(nums)
    expected = [[1,2],[2,1]]
    assert len(result) == len(expected)
    
    # 测试优化解法
    print("测试优化解法...")
    nums = [1,1,2]
    result_opt = solution.permuteUnique_optimized(nums)
    assert len(result_opt) == len(expected)
    
    # 测试迭代解法
    print("测试迭代解法...")
    nums = [1,1,2]
    result_iter = solution.permuteUnique_iterative(nums)
    assert len(result_iter) == len(expected)
    
    # 测试递归解法
    print("测试递归解法...")
    nums = [1,1,2]
    result_rec = solution.permuteUnique_recursive(nums)
    assert len(result_rec) == len(expected)
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
