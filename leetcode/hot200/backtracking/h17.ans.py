"""
17. 电话号码的字母组合 - 标准答案
"""
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 建立数字到字母的映射表
        2. 使用回溯算法生成所有可能的字母组合
        3. 递归处理每个数字对应的字母
        4. 当处理完所有数字时，将当前组合加入结果
        
        时间复杂度：O(3^m * 4^n)，其中m是3个字母的数字个数，n是4个字母的数字个数
        空间复杂度：O(m+n)
        """
        if not digits:
            return []
        
        # 数字到字母的映射
        digit_to_letters = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        
        result = []
        
        def backtrack(index: int, current_combination: str):
            # 终止条件：处理完所有数字
            if index == len(digits):
                result.append(current_combination)
                return
            
            # 获取当前数字对应的字母
            current_digit = digits[index]
            letters = digit_to_letters[current_digit]
            
            # 遍历当前数字对应的所有字母
            for letter in letters:
                backtrack(index + 1, current_combination + letter)
        
        backtrack(0, "")
        return result
    
    def letterCombinations_iterative(self, digits: str) -> List[str]:
        """
        迭代解法：队列 + BFS
        
        解题思路：
        1. 使用队列存储部分组合
        2. 每次从队列中取出一个部分组合
        3. 添加下一个数字对应的字母，形成新的组合
        4. 当组合长度等于数字长度时，加入结果
        
        时间复杂度：O(3^m * 4^n)
        空间复杂度：O(3^m * 4^n)
        """
        if not digits:
            return []
        
        digit_to_letters = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        
        from collections import deque
        queue = deque([''])
        
        for digit in digits:
            letters = digit_to_letters[digit]
            size = len(queue)
            
            for _ in range(size):
                current = queue.popleft()
                for letter in letters:
                    queue.append(current + letter)
        
        return list(queue)
    
    def letterCombinations_optimized(self, digits: str) -> List[str]:
        """
        优化解法：预计算 + 位运算
        
        解题思路：
        1. 预计算所有可能的组合数量
        2. 使用位运算快速生成组合
        3. 避免递归调用栈的开销
        
        时间复杂度：O(3^m * 4^n)
        空间复杂度：O(1)
        """
        if not digits:
            return []
        
        digit_to_letters = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        
        # 计算总组合数
        total_combinations = 1
        for digit in digits:
            total_combinations *= len(digit_to_letters[digit])
        
        result = []
        
        # 生成所有组合
        for i in range(total_combinations):
            combination = ""
            temp = i
            
            for digit in digits:
                letters = digit_to_letters[digit]
                combination += letters[temp % len(letters)]
                temp //= len(letters)
            
            result.append(combination)
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    digits = "23"
    result = solution.letterCombinations(digits)
    expected = ["ad","ae","af","bd","be","bf","cd","ce","cf"]
    assert set(result) == set(expected)
    
    # 测试用例2
    digits = ""
    result = solution.letterCombinations(digits)
    expected = []
    assert result == expected
    
    # 测试用例3
    digits = "2"
    result = solution.letterCombinations(digits)
    expected = ["a","b","c"]
    assert set(result) == set(expected)
    
    # 测试用例4
    digits = "234"
    result = solution.letterCombinations(digits)
    expected = ["adg","adh","adi","aeg","aeh","aei","afg","afh","afi",
                "bdg","bdh","bdi","beg","beh","bei","bfg","bfh","bfi",
                "cdg","cdh","cdi","ceg","ceh","cei","cfg","cfh","cfi"]
    assert set(result) == set(expected)
    
    # 测试迭代解法
    print("测试迭代解法...")
    digits = "23"
    result_iter = solution.letterCombinations_iterative(digits)
    assert set(result_iter) == set(expected)
    
    # 测试优化解法
    print("测试优化解法...")
    digits = "23"
    result_opt = solution.letterCombinations_optimized(digits)
    assert set(result_opt) == set(expected)
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
