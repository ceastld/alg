"""
46. 全排列 - 标准答案
"""
from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 使用回溯算法生成所有可能的排列
        2. 使用visited数组标记已使用的元素
        3. 每次选择一个未使用的元素，递归生成剩余元素的排列
        4. 当路径长度达到数组长度时，将结果加入答案
        
        时间复杂度：O(n! * n)
        空间复杂度：O(n)
        """
        result = []
        visited = [False] * len(nums)
        
        def backtrack(path):
            # 如果路径长度达到数组长度，加入结果
            if len(path) == len(nums):
                result.append(path[:])
                return
            
            # 遍历所有未使用的元素
            for i in range(len(nums)):
                if not visited[i]:
                    visited[i] = True
                    path.append(nums[i])
                    backtrack(path)
                    path.pop()
                    visited[i] = False
        
        backtrack([])
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 2, 3]
    result = solution.permute(nums)
    expected = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = [0, 1]
    result = solution.permute(nums)
    expected = [[0,1],[1,0]]
    assert len(result) == len(expected)
    
    # 测试用例3
    nums = [1]
    result = solution.permute(nums)
    expected = [[1]]
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
