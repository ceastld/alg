"""
84. 柱状图中最大的矩形 - 标准答案
"""
from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        """
        标准解法：单调栈
        
        解题思路：
        1. 使用单调栈维护递增序列
        2. 对于每个柱子，找到其左边和右边第一个比它小的柱子
        3. 计算以当前柱子为高的最大矩形面积
        4. 返回所有矩形面积中的最大值
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not heights:
            return 0
        
        # 添加哨兵，简化边界处理
        heights = [0] + heights + [0]
        stack = []
        max_area = 0
        
        for i in range(len(heights)):
            # 当前高度小于栈顶高度时，计算面积
            while stack and heights[i] < heights[stack[-1]]:
                # 弹出栈顶元素作为高度
                h = heights[stack.pop()]
                # 计算宽度
                w = i - stack[-1] - 1
                # 更新最大面积
                max_area = max(max_area, h * w)
            
            stack.append(i)
        
        return max_area
    
    def largestRectangleArea_brute_force(self, heights: List[int]) -> int:
        """
        暴力解法：双重循环
        
        解题思路：
        1. 对于每个柱子，找到其左边和右边第一个比它小的柱子
        2. 计算以当前柱子为高的最大矩形面积
        3. 返回所有矩形面积中的最大值
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        if not heights:
            return 0
        
        max_area = 0
        n = len(heights)
        
        for i in range(n):
            # 找到左边第一个比当前柱子小的位置
            left = i
            while left >= 0 and heights[left] >= heights[i]:
                left -= 1
            
            # 找到右边第一个比当前柱子小的位置
            right = i
            while right < n and heights[right] >= heights[i]:
                right += 1
            
            # 计算面积
            area = heights[i] * (right - left - 1)
            max_area = max(max_area, area)
        
        return max_area
    
    def largestRectangleArea_dp(self, heights: List[int]) -> int:
        """
        动态规划解法：预处理左右边界
        
        解题思路：
        1. 预处理每个柱子左边和右边第一个比它小的位置
        2. 计算以每个柱子为高的最大矩形面积
        3. 返回所有矩形面积中的最大值
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not heights:
            return 0
        
        n = len(heights)
        left = [-1] * n
        right = [n] * n
        
        # 计算每个柱子左边第一个比它小的位置
        for i in range(1, n):
            j = i - 1
            while j >= 0 and heights[j] >= heights[i]:
                j = left[j]
            left[i] = j
        
        # 计算每个柱子右边第一个比它小的位置
        for i in range(n - 2, -1, -1):
            j = i + 1
            while j < n and heights[j] >= heights[i]:
                j = right[j]
            right[i] = j
        
        # 计算最大面积
        max_area = 0
        for i in range(n):
            area = heights[i] * (right[i] - left[i] - 1)
            max_area = max(max_area, area)
        
        return max_area
    
    def largestRectangleArea_optimized(self, heights: List[int]) -> int:
        """
        优化解法：单调栈（空间优化）
        
        解题思路：
        1. 使用单调栈维护递增序列
        2. 当遇到更小的柱子时，计算面积
        3. 使用哨兵简化边界处理
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not heights:
            return 0
        
        # 添加哨兵
        heights = [0] + heights + [0]
        stack = []
        max_area = 0
        
        for i, h in enumerate(heights):
            while stack and h < heights[stack[-1]]:
                height = heights[stack.pop()]
                width = i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        return max_area
    
    def largestRectangleArea_detailed(self, heights: List[int]) -> int:
        """
        详细解法：单调栈（带详细注释）
        
        解题思路：
        1. 使用单调栈维护递增序列
        2. 对于每个柱子，找到其左边和右边第一个比它小的柱子
        3. 计算以当前柱子为高的最大矩形面积
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not heights:
            return 0
        
        # 添加哨兵，简化边界处理
        heights = [0] + heights + [0]
        stack = []
        max_area = 0
        
        for i in range(len(heights)):
            current_height = heights[i]
            
            # 当前高度小于栈顶高度时，需要计算面积
            while stack and current_height < heights[stack[-1]]:
                # 弹出栈顶元素作为高度
                height = heights[stack.pop()]
                # 计算宽度：当前索引 - 新的栈顶索引 - 1
                width = i - stack[-1] - 1
                # 更新最大面积
                max_area = max(max_area, height * width)
            
            # 将当前索引压入栈中
            stack.append(i)
        
        return max_area
    
    def largestRectangleArea_alternative(self, heights: List[int]) -> int:
        """
        替代解法：单调栈（不使用哨兵）
        
        解题思路：
        1. 使用单调栈维护递增序列
        2. 手动处理边界情况
        3. 计算以每个柱子为高的最大矩形面积
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not heights:
            return 0
        
        stack = []
        max_area = 0
        
        for i in range(len(heights)):
            # 当前高度小于栈顶高度时，计算面积
            while stack and heights[i] < heights[stack[-1]]:
                height = heights[stack.pop()]
                # 计算宽度
                if stack:
                    width = i - stack[-1] - 1
                else:
                    width = i
                max_area = max(max_area, height * width)
            
            stack.append(i)
        
        # 处理栈中剩余元素
        while stack:
            height = heights[stack.pop()]
            if stack:
                width = len(heights) - stack[-1] - 1
            else:
                width = len(heights)
            max_area = max(max_area, height * width)
        
        return max_area


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.largestRectangleArea([2,1,5,6,2,3]) == 10
    
    # 测试用例2
    assert solution.largestRectangleArea([2,4]) == 4
    
    # 测试用例3
    assert solution.largestRectangleArea([1]) == 1
    
    # 测试用例4
    assert solution.largestRectangleArea([1,1]) == 2
    
    # 测试用例5
    assert solution.largestRectangleArea([2,1,2]) == 3
    
    # 测试用例6：边界情况
    assert solution.largestRectangleArea([]) == 0
    assert solution.largestRectangleArea([0]) == 0
    
    # 测试用例7：单调递增
    assert solution.largestRectangleArea([1,2,3,4,5]) == 9
    
    # 测试用例8：单调递减
    assert solution.largestRectangleArea([5,4,3,2,1]) == 9
    
    # 测试用例9：复杂情况
    assert solution.largestRectangleArea([3,1,3,2,2]) == 6
    assert solution.largestRectangleArea([2,1,5,6,2,3]) == 10
    
    # 测试用例10：全等高度
    assert solution.largestRectangleArea([2,2,2,2]) == 8
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
