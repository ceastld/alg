"""
1207. 独一无二的出现次数 - 标准答案
"""
from typing import List
from collections import Counter


class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        """
        标准解法：哈希表计数
        
        解题思路：
        1. 统计每个数字的出现次数
        2. 检查所有出现次数是否唯一
        3. 使用集合来判断是否有重复的出现次数
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        # 统计每个数字的出现次数
        count = Counter(arr)
        
        # 获取所有出现次数
        occurrences = list(count.values())
        
        # 检查出现次数是否唯一
        return len(occurrences) == len(set(occurrences))


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    arr = [1, 2, 2, 3, 3, 3]
    assert solution.uniqueOccurrences(arr) == True
    
    # 测试用例2
    arr = [1, 2]
    assert solution.uniqueOccurrences(arr) == False
    
    # 测试用例3
    arr = [-3, 0, 1, -3, 1, 1, 1, -3, 10, 0]
    assert solution.uniqueOccurrences(arr) == True
    
    # 测试用例4
    arr = [1, 1, 2, 2, 2, 3]
    assert solution.uniqueOccurrences(arr) == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
