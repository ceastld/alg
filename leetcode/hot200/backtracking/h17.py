"""
17. 电话号码的字母组合
给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

题目链接：https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

示例 1:
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]

示例 2:
输入：digits = ""
输出：[]

示例 3:
输入：digits = "2"
输出：["a","b","c"]

提示：
- 0 <= digits.length <= 4
- digits[i] 是范围 ['2', '9'] 的一个数字。
"""

from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        """
        请在这里实现你的解法
        """
        # TODO: 在这里实现你的解法
        pass


def main():
    """测试用例"""
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
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
