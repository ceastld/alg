"""
738. 单调递增的数字 - 标准答案
"""


class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 将数字转换为字符串
        2. 从右到左遍历，找到第一个不满足单调递增的位置
        3. 将该位置减1，并将后面的所有位置设为9
        4. 贪心策略：尽可能保持高位数字不变
        
        时间复杂度：O(log n)
        空间复杂度：O(log n)
        """
        # 将数字转换为字符串
        s = list(str(n))
        
        # 从右到左遍历，找到第一个不满足单调递增的位置
        for i in range(len(s) - 1, 0, -1):
            if s[i] < s[i - 1]:
                # 将前一位减1，并将后面的所有位置设为9
                s[i - 1] = str(int(s[i - 1]) - 1)
                for j in range(i, len(s)):
                    s[j] = '9'
        
        return int(''.join(s))
    
    def monotoneIncreasingDigits_alternative(self, n: int) -> int:
        """
        替代解法：贪心算法（使用数字操作）
        
        解题思路：
        1. 从右到左遍历数字的每一位
        2. 找到第一个不满足单调递增的位置
        3. 将该位置减1，并将后面的所有位置设为9
        
        时间复杂度：O(log n)
        空间复杂度：O(1)
        """
        # 将数字转换为字符串
        s = list(str(n))
        
        # 从右到左遍历，找到第一个不满足单调递增的位置
        for i in range(len(s) - 1, 0, -1):
            if s[i] < s[i - 1]:
                # 将前一位减1，并将后面的所有位置设为9
                s[i - 1] = str(int(s[i - 1]) - 1)
                for j in range(i, len(s)):
                    s[j] = '9'
        
        return int(''.join(s))
    
    def monotoneIncreasingDigits_optimized(self, n: int) -> int:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 将数字转换为字符串
        2. 从右到左遍历，找到第一个不满足单调递增的位置
        3. 将该位置减1，并将后面的所有位置设为9
        
        时间复杂度：O(log n)
        空间复杂度：O(log n)
        """
        # 将数字转换为字符串
        s = list(str(n))
        
        # 从右到左遍历，找到第一个不满足单调递增的位置
        for i in range(len(s) - 1, 0, -1):
            if s[i] < s[i - 1]:
                # 将前一位减1，并将后面的所有位置设为9
                s[i - 1] = str(int(s[i - 1]) - 1)
                for j in range(i, len(s)):
                    s[j] = '9'
        
        return int(''.join(s))
    
    def monotoneIncreasingDigits_detailed(self, n: int) -> int:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 将数字转换为字符串
        2. 从右到左遍历，找到第一个不满足单调递增的位置
        3. 将该位置减1，并将后面的所有位置设为9
        4. 贪心策略：尽可能保持高位数字不变
        
        时间复杂度：O(log n)
        空间复杂度：O(log n)
        """
        # 将数字转换为字符串
        s = list(str(n))
        
        # 从右到左遍历，找到第一个不满足单调递增的位置
        for i in range(len(s) - 1, 0, -1):
            if s[i] < s[i - 1]:
                # 将前一位减1，并将后面的所有位置设为9
                s[i - 1] = str(int(s[i - 1]) - 1)
                for j in range(i, len(s)):
                    s[j] = '9'
        
        return int(''.join(s))
    
    def monotoneIncreasingDigits_brute_force(self, n: int) -> int:
        """
        暴力解法：枚举
        
        解题思路：
        1. 从n开始向下枚举
        2. 检查每个数字是否满足单调递增
        3. 返回第一个满足条件的数字
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        def is_monotone_increasing(num):
            s = str(num)
            for i in range(len(s) - 1):
                if s[i] > s[i + 1]:
                    return False
            return True
        
        for i in range(n, -1, -1):
            if is_monotone_increasing(i):
                return i
        
        return 0
    
    def monotoneIncreasingDigits_step_by_step(self, n: int) -> int:
        """
        分步解法：贪心算法（分步处理）
        
        解题思路：
        1. 第一步：将数字转换为字符串
        2. 第二步：从右到左遍历，找到第一个不满足单调递增的位置
        3. 第三步：将该位置减1，并将后面的所有位置设为9
        
        时间复杂度：O(log n)
        空间复杂度：O(log n)
        """
        # 第一步：将数字转换为字符串
        s = list(str(n))
        
        # 第二步：从右到左遍历，找到第一个不满足单调递增的位置
        for i in range(len(s) - 1, 0, -1):
            if s[i] < s[i - 1]:
                # 第三步：将该位置减1，并将后面的所有位置设为9
                s[i - 1] = str(int(s[i - 1]) - 1)
                for j in range(i, len(s)):
                    s[j] = '9'
        
        return int(''.join(s))
    
    def monotoneIncreasingDigits_recursive(self, n: int) -> int:
        """
        递归解法：贪心算法
        
        解题思路：
        1. 将数字转换为字符串
        2. 递归处理每个位置
        3. 如果当前位置不满足单调递增，则调整
        
        时间复杂度：O(log n)
        空间复杂度：O(log n)
        """
        def helper(s, index):
            if index == 0:
                return s
            
            if s[index] < s[index - 1]:
                # 将前一位减1，并将后面的所有位置设为9
                s[index - 1] = str(int(s[index - 1]) - 1)
                for j in range(index, len(s)):
                    s[j] = '9'
            
            return helper(s, index - 1)
        
        s = list(str(n))
        return int(''.join(helper(s, len(s) - 1)))


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.monotoneIncreasingDigits(10) == 9
    
    # 测试用例2
    assert solution.monotoneIncreasingDigits(1234) == 1234
    
    # 测试用例3
    assert solution.monotoneIncreasingDigits(332) == 299
    
    # 测试用例4
    assert solution.monotoneIncreasingDigits(120) == 119
    
    # 测试用例5
    assert solution.monotoneIncreasingDigits(100) == 99
    
    # 测试用例6：边界情况
    assert solution.monotoneIncreasingDigits(0) == 0
    assert solution.monotoneIncreasingDigits(1) == 1
    
    # 测试用例7：单调递增
    assert solution.monotoneIncreasingDigits(1234) == 1234
    assert solution.monotoneIncreasingDigits(1111) == 1111
    
    # 测试用例8：单调递减
    assert solution.monotoneIncreasingDigits(4321) == 3999
    
    # 测试用例9：复杂情况
    assert solution.monotoneIncreasingDigits(332) == 299
    assert solution.monotoneIncreasingDigits(120) == 119
    assert solution.monotoneIncreasingDigits(100) == 99
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
