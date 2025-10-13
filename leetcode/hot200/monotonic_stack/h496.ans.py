"""
496. 下一个更大元素I - 标准答案
"""
from typing import List


class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        标准解法：单调栈 + 哈希表
        
        解题思路：
        1. 使用单调栈找到nums2中每个元素的下一个更大元素
        2. 使用哈希表存储每个元素的下一个更大元素
        3. 遍历nums1，从哈希表中获取对应的下一个更大元素
        
        时间复杂度：O(m + n)
        空间复杂度：O(n)
        """
        # 使用单调栈找到nums2中每个元素的下一个更大元素
        next_greater = {}
        stack = []
        
        for num in nums2:
            # 当前元素大于栈顶元素时，弹出并记录
            while stack and num > stack[-1]:
                next_greater[stack.pop()] = num
            stack.append(num)
        
        # 栈中剩余元素没有下一个更大元素，设为-1
        while stack:
            next_greater[stack.pop()] = -1
        
        # 根据nums1获取结果
        result = []
        for num in nums1:
            result.append(next_greater[num])
        
        return result
    
    def nextGreaterElement_brute_force(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        暴力解法：双重循环
        
        解题思路：
        1. 对于nums1中的每个元素，在nums2中找到其位置
        2. 从该位置开始向右查找第一个更大的元素
        
        时间复杂度：O(m * n)
        空间复杂度：O(1)
        """
        result = []
        
        for num1 in nums1:
            # 在nums2中找到num1的位置
            index = nums2.index(num1)
            
            # 从该位置开始向右查找第一个更大的元素
            found = False
            for i in range(index + 1, len(nums2)):
                if nums2[i] > num1:
                    result.append(nums2[i])
                    found = True
                    break
            
            if not found:
                result.append(-1)
        
        return result
    
    def nextGreaterElement_optimized(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        优化解法：单调栈（从右到左）
        
        解题思路：
        1. 从右到左遍历nums2
        2. 维护一个单调栈，存储元素
        3. 对于每个元素，找到栈中第一个大于它的元素
        
        时间复杂度：O(m + n)
        空间复杂度：O(n)
        """
        next_greater = {}
        stack = []
        
        # 从右到左遍历nums2
        for i in range(len(nums2) - 1, -1, -1):
            num = nums2[i]
            
            # 弹出所有小于等于当前元素的元素
            while stack and stack[-1] <= num:
                stack.pop()
            
            # 如果栈不为空，说明有更大的元素
            if stack:
                next_greater[num] = stack[-1]
            else:
                next_greater[num] = -1
            
            # 将当前元素压入栈中
            stack.append(num)
        
        # 根据nums1获取结果
        return [next_greater[num] for num in nums1]
    
    def nextGreaterElement_detailed(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        详细解法：单调栈（带详细注释）
        
        解题思路：
        1. 使用单调栈维护递减序列
        2. 当遇到更大元素时，更新栈中所有较小元素的下一个更大元素
        3. 使用哈希表存储结果
        
        时间复杂度：O(m + n)
        空间复杂度：O(n)
        """
        # 存储每个元素的下一个更大元素
        next_greater = {}
        # 单调栈，存储递减序列
        stack = []
        
        for num in nums2:
            # 当前元素大于栈顶元素时，需要处理栈中所有较小元素
            while stack and num > stack[-1]:
                # 弹出栈顶元素并记录其下一个更大元素
                smaller_num = stack.pop()
                next_greater[smaller_num] = num
            
            # 将当前元素压入栈中
            stack.append(num)
        
        # 栈中剩余元素没有下一个更大元素
        while stack:
            next_greater[stack.pop()] = -1
        
        # 根据nums1获取结果
        result = []
        for num in nums1:
            result.append(next_greater[num])
        
        return result
    
    def nextGreaterElement_alternative(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        替代解法：使用字典存储索引
        
        解题思路：
        1. 先建立nums2中每个元素的索引映射
        2. 对于nums1中的每个元素，找到其在nums2中的位置
        3. 从该位置开始向右查找第一个更大的元素
        
        时间复杂度：O(m * n)
        空间复杂度：O(n)
        """
        # 建立nums2中每个元素的索引映射
        index_map = {num: i for i, num in enumerate(nums2)}
        
        result = []
        for num1 in nums1:
            # 找到num1在nums2中的位置
            start_index = index_map[num1]
            
            # 从该位置开始向右查找第一个更大的元素
            found = False
            for i in range(start_index + 1, len(nums2)):
                if nums2[i] > num1:
                    result.append(nums2[i])
                    found = True
                    break
            
            if not found:
                result.append(-1)
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.nextGreaterElement([4,1,2], [1,3,4,2]) == [-1,3,-1]
    
    # 测试用例2
    assert solution.nextGreaterElement([2,4], [1,2,3,4]) == [3,-1]
    
    # 测试用例3
    assert solution.nextGreaterElement([1,3,5,2,4], [6,5,4,3,2,1,7]) == [7,7,7,7,7]
    
    # 测试用例4
    assert solution.nextGreaterElement([4,1,2], [1,2,3,4]) == [-1,2,3]
    
    # 测试用例5
    assert solution.nextGreaterElement([1], [1]) == [-1]
    
    # 测试用例6：边界情况
    assert solution.nextGreaterElement([1], [1,2]) == [2]
    assert solution.nextGreaterElement([2], [1,2]) == [-1]
    
    # 测试用例7：单调递增
    assert solution.nextGreaterElement([1,2,3], [1,2,3,4]) == [2,3,4]
    
    # 测试用例8：单调递减
    assert solution.nextGreaterElement([3,2,1], [3,2,1]) == [-1,-1,-1]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
