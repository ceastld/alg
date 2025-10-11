"""
1207. 独一无二的出现次数
给你一个整数数组 arr，请你帮忙统计数组中每个数的出现次数。

如果每个数的出现次数都是独一无二的，就返回 true；否则返回 false。

题目链接：https://leetcode.cn/problems/unique-number-of-occurrences/

示例 1:
输入: arr = [1,2,2,3,3,3]
输出: true
解释: 1 出现 1 次，2 出现 2 次，3 出现 3 次。没有两个数的出现次数相同。

示例 2:
输入: arr = [1,2]
输出: false
解释: 1 出现 1 次，2 出现 1 次。1 和 2 的出现次数相同。

示例 3:
输入: arr = [-3,0,1,-3,1,1,1,-3,10,0]
输出: true

提示：
- 1 <= arr.length <= 1000
- -1000 <= arr[i] <= 1000
"""
from typing import List
from collections import Counter

class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        """
        请在这里实现你的解法
        """
        count = Counter(arr)
        values = set()
        for v in count.values():
            if v in values:
                return False
            values.add(v)
        return True

def main():
    """测试用例"""
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
    assert solution.uniqueOccurrences(arr) == True
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
