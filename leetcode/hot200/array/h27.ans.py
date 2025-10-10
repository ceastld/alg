"""
27. 移除元素 - 标准答案
"""
from typing import List


class Solution:
    """
    27. 移除元素 - 标准解法
    """
    
    def removeElement(self, nums: List[int], val: int) -> int:
        """
        标准解法：双指针法
        
        解题思路：
        1. 使用双指针，slow指向下一个要填充的位置
        2. fast遍历数组，遇到不等于val的元素就放到slow位置
        3. slow最终指向新数组的长度
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums_1 = [3, 2, 2, 3]
    assert solution.removeElement(nums_1, 3) == 2
    
    # 测试用例2
    nums_2 = [0, 1, 2, 2, 3, 0, 4, 2]
    assert solution.removeElement(nums_2, 2) == 5
    
    # 测试用例3
    nums_3 = [1]
    assert solution.removeElement(nums_3, 1) == 0
    
    # 测试用例4
    nums_4 = [4, 5]
    assert solution.removeElement(nums_4, 5) == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
