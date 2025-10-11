"""
202. 快乐数 - 标准答案
"""


class Solution:
    """
    202. 快乐数 - 标准解法
    """
    
    def isHappy(self, n: int) -> bool:
        """
        标准解法：哈希表法
        
        解题思路：
        1. 使用哈希表记录已经计算过的数字
        2. 重复计算数字的平方和
        3. 如果结果为1，返回True
        4. 如果结果重复出现，说明进入循环，返回False
        
        时间复杂度：O(log n)
        空间复杂度：O(log n)
        """
        seen = set()
        
        while n != 1 and n not in seen:
            seen.add(n)
            n = self.get_next(n)
        
        return n == 1
    
    def isHappy_floyd(self, n: int) -> bool:
        """
        快慢指针法（Floyd判圈算法）
        
        解题思路：
        1. 使用快慢指针检测循环
        2. 快指针每次计算两次平方和
        3. 慢指针每次计算一次平方和
        4. 如果快慢指针相遇，说明有循环
        
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        def get_next(num):
            total = 0
            while num > 0:
                digit = num % 10
                total += digit * digit
                num //= 10
            return total
        
        slow = n
        fast = get_next(n)
        
        while fast != 1 and slow != fast:
            slow = get_next(slow)
            fast = get_next(get_next(fast))
        
        return fast == 1
    
    def isHappy_math(self, n: int) -> bool:
        """
        数学法（利用数学规律）
        
        解题思路：
        1. 根据数学研究，只有1和7是快乐数
        2. 其他数字最终都会进入循环
        3. 直接判断是否为1或7
        
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        while n > 9:
            n = self.get_next(n)
        
        return n == 1 or n == 7
    
    def get_next(self, n: int) -> int:
        """计算下一个数字"""
        total = 0
        while n > 0:
            digit = n % 10
            total += digit * digit
            n //= 10
        return total


def main():
    """测试标准答案"""
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
