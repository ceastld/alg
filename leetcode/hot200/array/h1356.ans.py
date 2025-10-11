"""
1356. 根据数字二进制下1的数目排序 - 标准答案
"""
from typing import List


class Solution:
    def sortByBits(self, arr: List[int]) -> List[int]:
        """
        标准解法：自定义排序
        
        解题思路：
        1. 计算每个数字的二进制表示中1的个数
        2. 按照1的个数进行排序，如果1的个数相同，按数值大小排序
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        def count_bits(num):
            count = 0
            while num:
                count += num & 1
                num >>= 1
            return count
        
        return sorted(arr, key=lambda x: (count_bits(x), x))


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    expected = [0, 1, 2, 4, 8, 3, 5, 6, 7]
    assert solution.sortByBits(arr) == expected
    
    # 测试用例2
    arr = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    expected = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    assert solution.sortByBits(arr) == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
