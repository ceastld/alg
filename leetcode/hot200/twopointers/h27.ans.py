"""
27. 移除元素 - 标准答案
"""
from typing import List


class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        """
        标准解法：双指针法
        
        解题思路：
        1. 使用快慢指针，快指针遍历数组
        2. 慢指针指向下一个要填充的位置
        3. 当快指针指向的元素不等于val时，将其复制到慢指针位置
        4. 慢指针的位置就是新数组的长度
        
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
    nums = [3, 2, 2, 3]
    val = 3
    result = solution.removeElement(nums, val)
    assert result == 2
    assert nums[:result] == [2, 2]
    
    # 测试用例2
    nums = [0, 1, 2, 2, 3, 0, 4, 2]
    val = 2
    result = solution.removeElement(nums, val)
    assert result == 5
    # 注意：顺序可能不同，这里只检查长度
    
    # 测试用例3
    nums = []
    val = 0
    result = solution.removeElement(nums, val)
    assert result == 0
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
