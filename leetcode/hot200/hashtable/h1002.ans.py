"""
1002. 查找常用字符 - 标准答案
"""
from typing import List
from collections import Counter


class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        """
        标准解法：字符计数 + 取最小值
        
        解题思路：
        1. 统计每个字符串中每个字符的出现次数
        2. 对每个字符，取在所有字符串中出现次数的最小值
        3. 根据最小值构造结果数组
        
        时间复杂度：O(n*m)，其中n是字符串数量，m是平均字符串长度
        空间复杂度：O(1) - 最多26个字符
        """
        if not words:
            return []
        
        # 统计第一个字符串的字符频率
        min_count = Counter(words[0])
        
        # 对每个字符串，更新最小计数
        for word in words[1:]:
            word_count = Counter(word)
            # 只保留在两个字符串中都出现的字符
            min_count &= word_count
        
        # 构造结果
        result = []
        for char, count in min_count.items():
            result.extend([char] * count)
        
        return result


def main():
    """测试标准答案"""
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
