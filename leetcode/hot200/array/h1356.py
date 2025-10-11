"""
1356. 根据数字二进制下1的数目排序
给你一个整数数组 arr。请你将数组中的元素按照其二进制表示中数字 1 的数目升序排序。

如果存在多个数字二进制中 1 的数目相同，则必须按照数值大小升序排序。

请你返回排序后的数组。

题目链接：https://leetcode.cn/problems/sort-integers-by-the-number-of-1-bits/

示例 1:
输入: arr = [0,1,2,3,4,5,6,7,8]
输出: [0,1,2,4,8,3,5,6,7]
解释: [0] 是唯一一个有 0 个 1 的数。
[1,2,4,8] 都有 1 个 1 。
[3,5,6] 有 2 个 1 。
[7] 有 3 个 1 。
按照 1 的个数排序后得到 [0,1,2,4,8,3,5,6,7]

示例 2:
输入: arr = [1024,512,256,128,64,32,16,8,4,2,1]
输出: [1,2,4,8,16,32,64,128,256,512,1024]

提示：
- 1 <= arr.length <= 500
- 0 <= arr[i] <= 10^4
"""
from typing import List


class Solution:
    def sortByBits(self, arr: List[int]) -> List[int]:
        arr.sort(key=lambda x: (x.bit_count(), x))
        return arr


def main():
    """测试用例"""
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
