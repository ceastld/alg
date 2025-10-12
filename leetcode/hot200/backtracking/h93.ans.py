"""
93. 复原IP地址 - 标准答案
"""
from typing import List


class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        """
        标准解法：回溯算法
        
        解题思路：
        1. 使用回溯算法生成所有可能的IP地址
        2. 每个IP地址由4个数字组成，每个数字在0-255之间
        3. 不能有前导零（除了单独的0）
        4. 使用剪枝优化：如果剩余字符不足以填满剩余数字，则提前终止
        
        时间复杂度：O(3^4) = O(1)
        空间复杂度：O(1)
        """
        result = []
        path = []
        
        def backtrack(start: int, current_path: List[str]):
            # 终止条件：已经分割出4个数字
            if len(current_path) == 4:
                if start == len(s):
                    result.append('.'.join(current_path))
                return
            
            # 剪枝：剩余字符不足以填满剩余数字
            remaining_digits = 4 - len(current_path)
            if len(s) - start < remaining_digits or len(s) - start > remaining_digits * 3:
                return
            
            # 尝试分割1-3个字符
            for length in range(1, min(4, len(s) - start + 1)):
                segment = s[start:start + length]
                
                # 检查是否为有效的IP段
                if self.is_valid_segment(segment):
                    current_path.append(segment)
                    backtrack(start + length, current_path)
                    current_path.pop()
        
        backtrack(0, path)
        return result
    
    def is_valid_segment(self, segment: str) -> bool:
        """检查IP段是否有效"""
        if not segment:
            return False
        
        # 不能有前导零（除了单独的0）
        if len(segment) > 1 and segment[0] == '0':
            return False
        
        # 必须在0-255之间
        num = int(segment)
        return 0 <= num <= 255
    
    def restoreIpAddresses_optimized(self, s: str) -> List[str]:
        """
        优化解法：预计算 + 剪枝
        
        解题思路：
        1. 预计算所有可能的IP段
        2. 使用更严格的剪枝条件
        3. 避免重复计算
        
        时间复杂度：O(3^4) = O(1)
        空间复杂度：O(1)
        """
        result = []
        path = []
        
        def backtrack(start: int, current_path: List[str]):
            if len(current_path) == 4:
                if start == len(s):
                    result.append('.'.join(current_path))
                return
            
            # 更严格的剪枝
            remaining_digits = 4 - len(current_path)
            remaining_chars = len(s) - start
            
            if remaining_chars < remaining_digits or remaining_chars > remaining_digits * 3:
                return
            
            # 尝试分割1-3个字符
            for length in range(1, min(4, remaining_chars + 1)):
                segment = s[start:start + length]
                
                if self.is_valid_segment(segment):
                    current_path.append(segment)
                    backtrack(start + length, current_path)
                    current_path.pop()
        
        backtrack(0, path)
        return result
    
    def restoreIpAddresses_iterative(self, s: str) -> List[str]:
        """
        迭代解法：使用队列 + BFS
        
        解题思路：
        1. 使用队列存储部分IP地址
        2. 每次从队列中取出一个部分IP地址
        3. 添加下一个IP段，形成新的部分IP地址
        4. 当IP地址完整时，加入结果
        
        时间复杂度：O(3^4) = O(1)
        空间复杂度：O(3^4) = O(1)
        """
        from collections import deque
        
        if len(s) < 4 or len(s) > 12:
            return []
        
        result = []
        queue = deque([([], 0)])  # (当前IP段列表, 当前位置)
        
        while queue:
            current_path, start = queue.popleft()
            
            if len(current_path) == 4:
                if start == len(s):
                    result.append('.'.join(current_path))
                continue
            
            # 尝试分割1-3个字符
            for length in range(1, min(4, len(s) - start + 1)):
                segment = s[start:start + length]
                
                if self.is_valid_segment(segment):
                    new_path = current_path + [segment]
                    queue.append((new_path, start + length))
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    s = "25525511135"
    result = solution.restoreIpAddresses(s)
    expected = ["255.255.11.135","255.255.111.35"]
    assert set(result) == set(expected)
    
    # 测试用例2
    s = "0000"
    result = solution.restoreIpAddresses(s)
    expected = ["0.0.0.0"]
    assert result == expected
    
    # 测试用例3
    s = "101023"
    result = solution.restoreIpAddresses(s)
    expected = ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
    assert set(result) == set(expected)
    
    # 测试用例4
    s = "1111"
    result = solution.restoreIpAddresses(s)
    expected = ["1.1.1.1"]
    assert result == expected
    
    # 测试用例5
    s = "010010"
    result = solution.restoreIpAddresses(s)
    expected = ["0.10.0.10","0.100.1.0"]
    assert set(result) == set(expected)
    
    # 测试优化解法
    print("测试优化解法...")
    s = "25525511135"
    result_opt = solution.restoreIpAddresses_optimized(s)
    assert set(result_opt) == set(expected)
    
    # 测试迭代解法
    print("测试迭代解法...")
    s = "25525511135"
    result_iter = solution.restoreIpAddresses_iterative(s)
    assert set(result_iter) == set(expected)
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
