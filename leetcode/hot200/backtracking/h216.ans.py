"""
216. 组合总和III - 标准答案
"""
from typing import List


class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 使用回溯算法生成所有可能的组合
        2. 从1到9中选择k个数字，使它们的和等于n
        3. 使用剪枝优化：如果当前和已经超过n，则提前终止
        4. 使用剪枝优化：如果剩余数字不足以填满k个，则提前终止
        
        时间复杂度：O(C(9,k) * k)
        空间复杂度：O(k)
        """
        result = []
        path = []
        
        def backtrack(start: int, current_sum: int, current_path: List[int]):
            # 终止条件
            if len(current_path) == k:
                if current_sum == n:
                    result.append(current_path[:])
                return
            
            # 剪枝优化
            if current_sum > n:
                return
            
            # 剪枝优化：剩余数字不足以填满k个
            remaining = k - len(current_path)
            if 9 - start + 1 < remaining:
                return
            
            for i in range(start, 10):
                # 剪枝优化：如果当前数字加上去会超过n，则跳过
                if current_sum + i > n:
                    break
                
                current_path.append(i)
                backtrack(i + 1, current_sum + i, current_path)
                current_path.pop()
        
        backtrack(1, 0, path)
        return result
    
    def combinationSum3_optimized(self, k: int, n: int) -> List[List[int]]:
        """
        优化解法：位运算 + 预计算
        
        解题思路：
        1. 使用位运算枚举所有可能的组合
        2. 预计算每个组合的和，快速筛选
        3. 使用位运算快速计算组合中数字的个数
        
        时间复杂度：O(2^9)
        空间复杂度：O(1)
        """
        result = []
        
        # 枚举所有可能的组合
        for mask in range(1 << 9):
            # 计算当前组合中数字的个数
            if bin(mask).count('1') != k:
                continue
            
            # 计算当前组合的和
            current_sum = 0
            combination = []
            for i in range(9):
                if mask & (1 << i):
                    num = i + 1
                    current_sum += num
                    combination.append(num)
            
            # 如果和等于n，则加入结果
            if current_sum == n:
                result.append(combination)
        
        return result


def main():
    """测试标准答案"""
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
    
    # 测试优化解法
    print("测试优化解法...")
    k, n = 3, 7
    result_opt = solution.combinationSum3_optimized(k, n)
    assert result_opt == expected
    
    k, n = 3, 9
    result_opt = solution.combinationSum3_optimized(k, n)
    assert result_opt == expected
    
    print("所有测试用例通过！")
    print("优化解法验证通过！")


if __name__ == "__main__":
    main()
