"""
922. 按奇偶排序数组II - 标准答案
"""
from typing import List


class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        """
        标准解法：双指针
        
        解题思路：
        1. 使用两个指针，一个指向偶数位置，一个指向奇数位置
        2. 如果偶数位置是奇数，奇数位置是偶数，则交换
        3. 否则移动对应的指针
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        n = len(nums)
        even, odd = 0, 1
        
        while even < n and odd < n:
            if nums[even] % 2 == 0:
                even += 2
            elif nums[odd] % 2 == 1:
                odd += 2
            else:
                nums[even], nums[odd] = nums[odd], nums[even]
                even += 2
                odd += 2
        
        return nums


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums = [4, 2, 5, 7]
    result = solution.sortArrayByParityII(nums)
    # 验证结果：偶数索引位置是偶数，奇数索引位置是奇数
    for i, num in enumerate(result):
        if i % 2 == 0:
            assert num % 2 == 0
        else:
            assert num % 2 == 1
    
    # 测试用例2
    nums = [2, 3]
    result = solution.sortArrayByParityII(nums)
    for i, num in enumerate(result):
        if i % 2 == 0:
            assert num % 2 == 0
        else:
            assert num % 2 == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
