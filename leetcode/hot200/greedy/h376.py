"""
376. 摆动序列
如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 摆动序列 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。

相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。

子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。

给你一个整数数组 nums ，返回 nums 中作为 摆动序列 的 最长子序列的长度 。

题目链接：https://leetcode.cn/problems/wiggle-subsequence/

示例 1:
输入：nums = [1,7,4,9,2,5]
输出：6
解释：整个序列均为摆动序列，各元素之间的差值为 (6, -3, 5, -7, 3) 。

示例 2:
输入：nums = [1,17,5,10,13,15,10,5,16,8]
输出：7
解释：这个序列包含几个长度为 7 摆动序列。
其中一个是 [1, 17, 10, 13, 10, 16, 8] ，各元素之间的差值为 (16, -7, 3, -3, 6, -8) 。

示例 3:
输入：nums = [1,2,3,4,5,6,7,8,9]
输出：2

提示：
- 1 <= nums.length <= 1000
- 0 <= nums[i] <= 1000
"""

from typing import List


class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 维护两个状态：上升和下降
        2. 遍历数组，根据当前趋势更新状态
        3. 如果趋势改变，长度+1
        4. 贪心策略：只保留趋势变化的点
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if len(nums) < 2:
            return len(nums)
        
        up = 1  # 上升趋势的长度
        down = 1  # 下降趋势的长度
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                # 当前是上升趋势
                up = down + 1
            elif nums[i] < nums[i-1]:
                # 当前是下降趋势
                down = up + 1
        
        return max(up, down)

def main():
    """测试用例"""
    solution = Solution()

    # # 测试用例1
    # assert solution.wiggleMaxLength([1, 7, 4, 9, 2, 5]) == 6

    # # 测试用例2
    # assert solution.wiggleMaxLength([1, 17, 5, 10, 13, 15, 10, 5, 16, 8]) == 7

    # # 测试用例3
    # assert solution.wiggleMaxLength([1, 2, 3, 4, 5, 6, 7, 8, 9]) == 2

    # 测试用例4
    assert solution.wiggleMaxLength([1, 1, 7, 4, 9, 2, 5]) == 6

    # 测试用例5
    assert solution.wiggleMaxLength([1, 2, 2, 2, 3, 4]) == 2

    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
