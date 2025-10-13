"""
42. 接雨水 - 标准答案
"""
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        """
        标准解法：单调栈
        
        解题思路：
        1. 使用单调栈维护递减序列
        2. 当遇到比栈顶元素高的柱子时，计算雨水
        3. 雨水 = (当前高度 - 栈顶高度) * (当前索引 - 栈顶索引 - 1)
        4. 累加所有雨水量
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not height:
            return 0
        
        stack = []
        water = 0
        
        for i in range(len(height)):
            # 当前高度大于栈顶高度时，计算雨水
            while stack and height[i] > height[stack[-1]]:
                # 弹出栈顶元素作为底部
                bottom = stack.pop()
                
                if not stack:
                    break
                
                # 计算雨水
                left = stack[-1]
                right = i
                h = min(height[left], height[right]) - height[bottom]
                w = right - left - 1    
                water += h * w
            
            stack.append(i)
        
        return water
    
    def trap_two_pointers(self, height: List[int]) -> int:
        """
        双指针解法：左右指针
        
        解题思路：
        1. 使用左右指针从两端向中间移动
        2. 维护左右最大值
        3. 每次移动较小的一边，计算该位置的积水量
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        if not height:
            return 0
        
        left, right = 0, len(height) - 1
        left_max, right_max = 0, 0
        water = 0
        
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
        
        return water
    
    def trap_dp(self, height: List[int]) -> int:
        """
        动态规划解法：预处理左右最大值
        
        解题思路：
        1. 预处理每个位置左边和右边的最大值
        2. 对于每个位置，积水量 = min(左边最大值, 右边最大值) - 当前高度
        3. 累加所有积水量
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not height:
            return 0
        
        n = len(height)
        left_max = [0] * n
        right_max = [0] * n
        
        # 计算每个位置左边的最大值
        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i-1], height[i])
        
        # 计算每个位置右边的最大值
        right_max[n-1] = height[n-1]
        for i in range(n-2, -1, -1):
            right_max[i] = max(right_max[i+1], height[i])
        
        # 计算积水量
        water = 0
        for i in range(n):
            water += min(left_max[i], right_max[i]) - height[i]
        
        return water
    
    def trap_brute_force(self, height: List[int]) -> int:
        """
        暴力解法：双重循环
        
        解题思路：
        1. 对于每个位置，找到左边和右边的最大值
        2. 积水量 = min(左边最大值, 右边最大值) - 当前高度
        3. 累加所有积水量
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        if not height:
            return 0
        
        water = 0
        n = len(height)
        
        for i in range(1, n-1):
            # 找到左边最大值
            left_max = max(height[:i])
            # 找到右边最大值
            right_max = max(height[i+1:])
            
            # 计算积水量
            h = min(left_max, right_max) - height[i]
            if h > 0:
                water += h
        
        return water
    
    def trap_optimized(self, height: List[int]) -> int:
        """
        优化解法：单调栈（空间优化）
        
        解题思路：
        1. 使用单调栈维护递减序列
        2. 当遇到更高柱子时，计算雨水
        3. 使用变量优化空间使用
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not height:
            return 0
        
        stack = []
        water = 0
        
        for i, h in enumerate(height):
            while stack and h > height[stack[-1]]:
                bottom = stack.pop()
                if stack:
                    left = stack[-1]
                    water += (min(height[left], h) - height[bottom]) * (i - left - 1)
            stack.append(i)
        
        return water
    
    def trap_detailed(self, height: List[int]) -> int:
        """
        详细解法：单调栈（带详细注释）
        
        解题思路：
        1. 使用单调栈维护递减序列
        2. 当遇到比栈顶元素高的柱子时，形成凹槽
        3. 计算凹槽中的积水量
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        if not height:
            return 0
        
        stack = []
        water = 0
        
        for i in range(len(height)):
            current_height = height[i]
            
            # 当前高度大于栈顶高度时，需要计算雨水
            while stack and current_height > height[stack[-1]]:
                # 弹出栈顶元素作为底部
                bottom_index = stack.pop()
                bottom_height = height[bottom_index]
                
                # 如果栈为空，说明没有左边界，无法积水
                if not stack:
                    break
                
                # 获取左边界
                left_index = stack[-1]
                left_height = height[left_index]
                
                # 计算积水量
                # 高度 = min(左边界高度, 右边界高度) - 底部高度
                water_height = min(left_height, current_height) - bottom_height
                # 宽度 = 右边界索引 - 左边界索引 - 1
                water_width = i - left_index - 1
                # 积水量 = 高度 × 宽度
                water += water_height * water_width
            
            # 将当前索引压入栈中
            stack.append(i)
        
        return water


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
    
    # 测试用例2
    assert solution.trap([4,2,0,3,2,5]) == 9
    
    # 测试用例3
    assert solution.trap([3,0,2,0,4]) == 7
    
    # 测试用例4
    assert solution.trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6
    
    # 测试用例5
    assert solution.trap([2,0,2]) == 2
    
    # 测试用例6：边界情况
    assert solution.trap([]) == 0
    assert solution.trap([1]) == 0
    assert solution.trap([1,2]) == 0
    
    # 测试用例7：单调递增
    assert solution.trap([1,2,3,4,5]) == 0
    
    # 测试用例8：单调递减
    assert solution.trap([5,4,3,2,1]) == 0
    
    # 测试用例9：复杂情况
    assert solution.trap([3,2,1,2,3]) == 4
    assert solution.trap([1,0,2,1,0,1,3,2,1,2,1]) == 6
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
