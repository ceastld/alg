"""
1002. 查找常用字符
给你一个字符串数组 words，请你找出所有在 words 的每个字符串中都出现的共用字符（包括重复字符），并以数组形式返回。你可以按任意顺序返回答案。

题目链接：https://leetcode.cn/problems/find-common-characters/

示例 1:
输入: words = ["bella","label","roller"]
输出: ["e","l","l"]

示例 2:
输入: words = ["cool","lock","cook"]
输出: ["c","o"]

提示：
- 1 <= words.length <= 100
- 1 <= words[i].length <= 100
- words[i] 由小写英文字母组成
"""
from typing import List
from collections import Counter

class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        """
        请在这里实现你的解法
        """
        count = Counter(words[0])
        for word in words[1:]:
            count &= Counter(word)
        return list(count.elements())


def main():
    """测试用例"""
    solution = Solution()
    
    # 测试用例1
    words = ["bella", "label", "roller"]
    result = solution.commonChars(words)
    expected = ["e", "l", "l"]
    assert sorted(result) == sorted(expected)
    
    # 测试用例2
    words = ["cool", "lock", "cook"]
    result = solution.commonChars(words)
    expected = ["c", "o"]
    assert sorted(result) == sorted(expected)
    
    # 测试用例3
    words = ["a"]
    result = solution.commonChars(words)
    expected = ["a"]
    assert sorted(result) == sorted(expected)
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
