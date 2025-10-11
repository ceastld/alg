"""
202. 快乐数 - 预计算优化答案
使用预计算的方法，先生成1-198的快乐数结果，然后硬编码到代码中
"""
happy_data = bytearray.fromhex("82482213001080001402442492080000c82000000201001580")

def f(n: int) -> int:
    total = 0
    while n > 0:
        digit = n % 10
        total += digit * digit
        n //= 10
    return total

class Solution:
    """
    202. 快乐数 - 预计算优化解法
    """    
    def isHappy(self, n: int) -> bool:
        """
        预计算优化解法
        
        解题思路：
        1. 对任意数字n，计算两次f(n)后，结果必然 <= 198
        2. 使用预计算的1-198的快乐数结果直接查表
        3. 时间复杂度O(1)，空间复杂度O(1)
        
        时间复杂度：O(1)
        空间复杂度：O(1)
        """
        # 计算两次f(n)，确保结果 <= 198

        n = f(f(n))
        # 直接查表返回结果（从hex编码的bytearray中提取bit）
        byte_index = (n - 1) // 8
        bit_index = (n - 1) % 8
        return bool((happy_data[byte_index] >> (7 - bit_index)) & 1)


def main():
    """测试预计算优化答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.isHappy(19) == True
    
    # 测试用例2
    assert solution.isHappy(2) == False
    
    # 测试用例3
    assert solution.isHappy(1) == True
    
    # 测试用例4
    assert solution.isHappy(7) == True
    
    # 测试用例5
    assert solution.isHappy(4) == False
    
    # 测试用例6
    assert solution.isHappy(10) == True
    
    # 测试用例7
    assert solution.isHappy(13) == True
    
    # 测试用例8
    assert solution.isHappy(16) == False
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
