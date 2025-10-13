"""
503. 下一个更大元素II - 标准答案
"""
from typing import List


class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        """
        标准解法：单调栈（循环数组）
        
        解题思路：
        1. 将数组扩展为两倍长度，模拟循环数组
        2. 使用单调栈维护递减序列
        3. 对于每个元素，找到其下一个更大元素
        4. 只处理前n个元素的结果
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # 遍历两倍长度的数组，模拟循环
        for i in range(2 * n):
            # 使用模运算获取实际索引
            actual_index = i % n
            current_num = nums[actual_index]
            
            # 当前元素大于栈顶元素时，弹出并更新结果
            while stack and current_num > nums[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = current_num
            
            # 只处理前n个元素
            if i < n:
                stack.append(actual_index)
        
        return result
    
    def nextGreaterElements_alternative(self, nums: List[int]) -> List[int]:
        """
        替代解法：单调栈（两次遍历）
        
        解题思路：
        1. 第一次遍历：处理正常的下一个更大元素
        2. 第二次遍历：处理循环情况下的下一个更大元素
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # 第一次遍历：处理正常情况
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = nums[i]
            stack.append(i)
        
        # 第二次遍历：处理循环情况
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = nums[i]
            # 第二次遍历不需要再压入栈中
        
        return result
    
    def nextGreaterElements_brute_force(self, nums: List[int]) -> List[int]:
        """
        暴力解法：双重循环
        
        解题思路：
        1. 对于每个元素，从下一个位置开始循环查找
        2. 如果找到更大的元素，记录并跳出
        3. 否则设为-1
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        n = len(nums)
        result = [-1] * n
        
        for i in range(n):
            # 从下一个位置开始循环查找
            for j in range(1, n):
                next_index = (i + j) % n
                if nums[next_index] > nums[i]:
                    result[i] = nums[next_index]
                    break
        
        return result
    
    def nextGreaterElements_optimized(self, nums: List[int]) -> List[int]:
        """
        优化解法：单调栈（空间优化）
        
        解题思路：
        1. 使用单调栈维护递减序列
        2. 遍历两倍长度数组，但只处理前n个元素
        3. 使用模运算处理循环
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # 遍历两倍长度，模拟循环
        for i in range(2 * n - 1):
            # 获取实际索引
            actual_index = i % n
            
            # 当前元素大于栈顶元素时，弹出并更新
            while stack and nums[actual_index] > nums[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = nums[actual_index]
            
            # 只在前n次遍历中压入栈
            if i < n:
                stack.append(actual_index)
        
        return result
    
    def nextGreaterElements_detailed(self, nums: List[int]) -> List[int]:
        """
        详细解法：单调栈（带详细注释）
        
        解题思路：
        1. 将循环数组问题转化为线性数组问题
        2. 使用单调栈维护递减序列
        3. 遍历两倍长度数组，处理循环情况
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # 遍历两倍长度数组，模拟循环
        for i in range(2 * n):
            # 使用模运算获取实际索引
            actual_index = i % n
            current_num = nums[actual_index]
            
            # 当前元素大于栈顶元素时，需要处理栈中所有较小元素
            while stack and current_num > nums[stack[-1]]:
                # 弹出栈顶元素并更新其下一个更大元素
                prev_index = stack.pop()
                result[prev_index] = current_num
            
            # 只在前n次遍历中将元素压入栈
            if i < n:
                stack.append(actual_index)
        
        return result
    
    def nextGreaterElements_step_by_step(self, nums: List[int]) -> List[int]:
        """
        分步解法：单调栈（分步处理）
        
        解题思路：
        1. 第一步：处理正常的下一个更大元素
        2. 第二步：处理循环情况下的下一个更大元素
        
        时间复杂度：O(n)
        空间复杂度：O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # 第一步：处理正常情况
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = nums[i]
            stack.append(i)
        
        # 第二步：处理循环情况
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = nums[i]
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.nextGreaterElements([1,2,1]) == [2,-1,2]
    
    # 测试用例2
    assert solution.nextGreaterElements([1,2,3,4,3]) == [2,3,4,-1,4]
    
    # 测试用例3
    assert solution.nextGreaterElements([1,1,1,1,1]) == [-1,-1,-1,-1,-1]
    
    # 测试用例4
    assert solution.nextGreaterElements([5,4,3,2,1]) == [-1,5,5,5,5]
    
    # 测试用例5
    assert solution.nextGreaterElements([1,2,3,4,5]) == [2,3,4,5,-1]
    
    # 测试用例6：边界情况
    assert solution.nextGreaterElements([1]) == [-1]
    assert solution.nextGreaterElements([1,2]) == [2,-1]
    
    # 测试用例7：单调递增
    assert solution.nextGreaterElements([1,2,3,4]) == [2,3,4,-1]
    
    # 测试用例8：单调递减
    assert solution.nextGreaterElements([4,3,2,1]) == [-1,4,4,4]
    
    # 测试用例9：复杂情况
    assert solution.nextGreaterElements([1,3,4,2]) == [3,4,-1,3]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
