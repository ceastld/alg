"""
78. 子集 - 标准答案
"""
from typing import List


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 使用回溯算法生成所有可能的子集
        2. 每个元素都有两种选择：包含或不包含
        3. 递归处理每个元素，生成所有可能的组合
        4. 当处理完所有元素时，将当前路径加入结果
        
        时间复杂度：O(2^n * n)
        空间复杂度：O(n)
        """
        result = []
        
        def backtrack(start, path):
            # 将当前路径加入结果（包括空集）
            result.append(path[:])
            
            # 从start开始遍历剩余元素
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(i + 1, path)
                path.pop()
        
        backtrack(0, [])
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [1, 2, 3]
    result = solution.subsets(nums)
    expected = [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    assert len(result) == len(expected)
    
    # 测试用例2
    nums = [0]
    result = solution.subsets(nums)
    expected = [[],[0]]
    assert len(result) == len(expected)
    
    # 测试用例3
    nums = [1, 2]
    result = solution.subsets(nums)
    expected = [[],[1],[2],[1,2]]
    assert len(result) == len(expected)
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
