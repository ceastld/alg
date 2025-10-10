"""
283. 移动零 - 标准答案
"""
from typing import List


class Solution:
    """
    283. 移动零 - 标准解法
    """
    
    def moveZeroes(self, nums: List[int]) -> None:
        """
        标准解法：双指针法
        
        解题思路：
        1. 使用双指针，slow指向下一个非零元素应该放置的位置
        2. fast遍历数组，将非零元素移到slow位置
        3. 最后将slow之后的位置都设为0
        4. 这样保持了非零元素的相对顺序
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        slow = 0
        
        # 将所有非零元素移到前面
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow] = nums[fast]
                slow += 1
        
        # 将剩余位置设为0
        for i in range(slow, len(nums)):
            nums[i] = 0


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    nums_1 = [0, 1, 0, 3, 12]
    solution.moveZeroes(nums_1)
    assert nums_1 == [1, 3, 12, 0, 0]
    
    # 测试用例2
    nums_2 = [0]
    solution.moveZeroes(nums_2)
    assert nums_2 == [0]
    
    # 测试用例3
    nums_3 = [1, 2, 3, 4, 5]
    solution.moveZeroes(nums_3)
    assert nums_3 == [1, 2, 3, 4, 5]
    
    # 测试用例4
    nums_4 = [0, 0, 0, 1, 2, 3]
    solution.moveZeroes(nums_4)
    assert nums_4 == [1, 2, 3, 0, 0, 0]
    
    # 测试用例5
    nums_5 = [1, 0, 1, 0, 1]
    solution.moveZeroes(nums_5)
    assert nums_5 == [1, 1, 1, 0, 0]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
