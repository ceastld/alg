"""
69. x 的平方根 - 标准答案
"""
from typing import List


class Solution:
    def mySqrt(self, x: int) -> int:
        """
        标准解法：二分查找
        
        解题思路：
        1. 在[0, x]范围内二分查找
        2. 找到最大的mid使得mid*mid <= x
        3. 使用二分查找优化时间复杂度
        
        时间复杂度：O(log x)
        空间复杂度：O(1)
        """
        if x < 2:
            return x
        
        left, right = 2, x // 2
        
        while left <= right:
            mid = left + (right - left) // 2
            square = mid * mid
            
            if square == x:
                return mid
            elif square < x:
                left = mid + 1
            else:
                right = mid - 1
        
        return right
    
    def mySqrt_newton(self, x: int) -> int:
        """
        牛顿迭代法
        
        解题思路：
        1. 使用牛顿迭代法求解方程y^2 = x
        2. 迭代公式：y = (y + x/y) / 2
        3. 收敛速度很快
        
        时间复杂度：O(log x)
        空间复杂度：O(1)
        """
        if x < 2:
            return x
        
        # 初始猜测值
        y = x
        while y * y > x:
            y = (y + x // y) // 2
        
        return y
    
    def mySqrt_linear(self, x: int) -> int:
        """
        线性搜索解法
        
        解题思路：
        1. 从1开始线性搜索
        2. 找到最大的i使得i*i <= x
        3. 简单但效率较低
        
        时间复杂度：O(√x)
        空间复杂度：O(1)
        """
        if x < 2:
            return x
        
        for i in range(1, x + 1):
            if i * i > x:
                return i - 1
        
        return x
    
    def mySqrt_bit_manipulation(self, x: int) -> int:
        """
        位运算解法
        
        解题思路：
        1. 使用位运算优化二分查找
        2. 从最高位开始逐位确定结果
        3. 利用位运算的性质
        
        时间复杂度：O(log x)
        空间复杂度：O(1)
        """
        if x < 2:
            return x
        
        # 找到最高位
        bit = 0
        while (1 << bit) <= x:
            bit += 1
        bit -= 1
        
        result = 0
        for i in range(bit, -1, -1):
            candidate = result | (1 << i)
            if candidate * candidate <= x:
                result = candidate
        
        return result
    
    def mySqrt_math(self, x: int) -> int:
        """
        数学公式解法
        
        解题思路：
        1. 使用数学公式近似计算
        2. 利用对数性质：sqrt(x) = x^(1/2) = e^(ln(x)/2)
        3. 需要处理精度问题
        
        时间复杂度：O(1)
        空间复杂度：O(1)
        """
        if x < 2:
            return x
        
        import math
        
        # 使用数学公式计算
        result = int(math.exp(0.5 * math.log(x)))
        
        # 调整结果
        if (result + 1) * (result + 1) <= x:
            return result + 1
        return result
    
    def mySqrt_optimized_binary(self, x: int) -> int:
        """
        优化二分查找解法
        
        解题思路：
        1. 优化搜索范围
        2. 使用更精确的边界条件
        3. 减少不必要的计算
        
        时间复杂度：O(log x)
        空间复杂度：O(1)
        """
        if x < 2:
            return x
        
        # 优化搜索范围
        left, right = 1, x
        
        while left < right:
            mid = left + (right - left + 1) // 2
            if mid > x // mid:
                right = mid - 1
            else:
                left = mid
        
        return left
    
    def mySqrt_recursive(self, x: int) -> int:
        """
        递归解法
        
        解题思路：
        1. 使用递归实现二分查找
        2. 分治思想
        
        时间复杂度：O(log x)
        空间复杂度：O(log x)
        """
        if x < 2:
            return x
        
        def binary_search(left, right):
            if left > right:
                return right
            
            mid = left + (right - left) // 2
            square = mid * mid
            
            if square == x:
                return mid
            elif square < x:
                return binary_search(mid + 1, right)
            else:
                return binary_search(left, mid - 1)
        
        return binary_search(1, x // 2)
    
    def mySqrt_brute_force(self, x: int) -> int:
        """
        暴力解法
        
        解题思路：
        1. 从0开始逐个尝试
        2. 找到最大的i使得i*i <= x
        
        时间复杂度：O(√x)
        空间复杂度：O(1)
        """
        if x < 2:
            return x
        
        for i in range(x + 1):
            if i * i > x:
                return i - 1
        
        return x


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    x = 4
    result = solution.mySqrt(x)
    expected = 2
    assert result == expected
    
    # 测试用例2
    x = 8
    result = solution.mySqrt(x)
    expected = 2
    assert result == expected
    
    # 测试用例3
    x = 0
    result = solution.mySqrt(x)
    expected = 0
    assert result == expected
    
    # 测试用例4
    x = 1
    result = solution.mySqrt(x)
    expected = 1
    assert result == expected
    
    # 测试用例5
    x = 9
    result = solution.mySqrt(x)
    expected = 3
    assert result == expected
    
    # 测试用例6
    x = 15
    result = solution.mySqrt(x)
    expected = 3
    assert result == expected
    
    # 测试牛顿迭代法
    print("测试牛顿迭代法...")
    x = 8
    result_newton = solution.mySqrt_newton(x)
    expected_newton = 2
    assert result_newton == expected_newton
    
    # 测试线性搜索
    print("测试线性搜索...")
    x = 8
    result_linear = solution.mySqrt_linear(x)
    expected_linear = 2
    assert result_linear == expected_linear
    
    # 测试位运算解法
    print("测试位运算解法...")
    x = 8
    result_bit = solution.mySqrt_bit_manipulation(x)
    expected_bit = 2
    assert result_bit == expected_bit
    
    # 测试数学公式解法
    print("测试数学公式解法...")
    x = 8
    result_math = solution.mySqrt_math(x)
    expected_math = 2
    assert result_math == expected_math
    
    # 测试优化二分查找
    print("测试优化二分查找...")
    x = 8
    result_opt = solution.mySqrt_optimized_binary(x)
    expected_opt = 2
    assert result_opt == expected_opt
    
    # 测试递归解法
    print("测试递归解法...")
    x = 8
    result_rec = solution.mySqrt_recursive(x)
    expected_rec = 2
    assert result_rec == expected_rec
    
    # 测试暴力解法
    print("测试暴力解法...")
    x = 8
    result_bf = solution.mySqrt_brute_force(x)
    expected_bf = 2
    assert result_bf == expected_bf
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
