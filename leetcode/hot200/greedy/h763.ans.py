"""
763. 划分字母区间 - 标准答案
"""
from typing import List


class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 记录每个字符最后出现的位置
        2. 遍历字符串，维护当前片段的右边界
        3. 当遍历到右边界时，开始新的片段
        4. 贪心策略：尽可能延长当前片段
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 记录每个字符最后出现的位置
        last = {}
        for i, char in enumerate(s):
            last[char] = i
        
        result = []
        start = 0
        end = 0
        
        for i, char in enumerate(s):
            # 更新当前片段的右边界
            end = max(end, last[char])
            
            # 如果遍历到右边界，开始新的片段
            if i == end:
                result.append(end - start + 1)
                start = end + 1
        
        return result
    
    def partitionLabels_alternative(self, s: str) -> List[int]:
        """
        替代解法：贪心算法（使用数组）
        
        解题思路：
        1. 使用数组记录每个字符最后出现的位置
        2. 遍历字符串，维护当前片段的右边界
        3. 当遍历到右边界时，开始新的片段
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 使用数组记录每个字符最后出现的位置
        last = [-1] * 26
        for i, char in enumerate(s):
            last[ord(char) - ord('a')] = i
        
        result = []
        start = 0
        end = 0
        
        for i, char in enumerate(s):
            # 更新当前片段的右边界
            end = max(end, last[ord(char) - ord('a')])
            
            # 如果遍历到右边界，开始新的片段
            if i == end:
                result.append(end - start + 1)
                start = end + 1
        
        return result
    
    def partitionLabels_optimized(self, s: str) -> List[int]:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 记录每个字符最后出现的位置
        2. 遍历字符串，维护当前片段的右边界
        3. 当遍历到右边界时，开始新的片段
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 记录每个字符最后出现的位置
        last = {}
        for i, char in enumerate(s):
            last[char] = i
        
        result = []
        start = 0
        end = 0
        
        for i, char in enumerate(s):
            # 更新当前片段的右边界
            end = max(end, last[char])
            
            # 如果遍历到右边界，开始新的片段
            if i == end:
                result.append(end - start + 1)
                start = end + 1
        
        return result
    
    def partitionLabels_detailed(self, s: str) -> List[int]:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 记录每个字符最后出现的位置
        2. 遍历字符串，维护当前片段的右边界
        3. 当遍历到右边界时，开始新的片段
        4. 贪心策略：尽可能延长当前片段
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 记录每个字符最后出现的位置
        last = {}
        for i, char in enumerate(s):
            last[char] = i
        
        result = []
        start = 0
        end = 0
        
        for i, char in enumerate(s):
            # 更新当前片段的右边界
            end = max(end, last[char])
            
            # 如果遍历到右边界，开始新的片段
            if i == end:
                result.append(end - start + 1)
                start = end + 1
        
        return result
    
    def partitionLabels_brute_force(self, s: str) -> List[int]:
        """
        暴力解法：回溯
        
        解题思路：
        1. 使用回溯算法尝试所有可能的划分
        2. 检查每个划分是否满足条件
        3. 返回最长的划分
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        def is_valid_partition(partition):
            for part in partition:
                # 检查每个片段中的字符是否只出现在该片段中
                chars_in_part = set(part)
                for char in chars_in_part:
                    # 检查该字符是否在其他片段中出现
                    for other_part in partition:
                        if other_part != part and char in other_part:
                            return False
            return True
        
        def backtrack(index, current_partition):
            if index == len(s):
                if is_valid_partition(current_partition):
                    return [len(part) for part in current_partition]
                return []
            
            result = []
            # 尝试将当前字符加入最后一个片段
            if current_partition:
                current_partition[-1] += s[index]
                result = backtrack(index + 1, current_partition)
                current_partition[-1] = current_partition[-1][:-1]
            
            # 尝试开始新的片段
            current_partition.append(s[index])
            new_result = backtrack(index + 1, current_partition)
            current_partition.pop()
            
            # 选择更长的结果
            if len(new_result) > len(result):
                result = new_result
            
            return result
        
        return backtrack(0, [])
    
    def partitionLabels_step_by_step(self, s: str) -> List[int]:
        """
        分步解法：贪心算法（分步处理）
        
        解题思路：
        1. 第一步：记录每个字符最后出现的位置
        2. 第二步：贪心划分片段
        
        时间复杂度：O(n)
        空间复杂度：O(1)
        """
        # 第一步：记录每个字符最后出现的位置
        last = {}
        for i, char in enumerate(s):
            last[char] = i
        
        # 第二步：贪心划分片段
        result = []
        start = 0
        end = 0
        
        for i, char in enumerate(s):
            end = max(end, last[char])
            if i == end:
                result.append(end - start + 1)
                start = end + 1
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.partitionLabels("ababcbacadefegdehijhklij") == [9,7,8]
    
    # 测试用例2
    assert solution.partitionLabels("eccbbbbdec") == [10]
    
    # 测试用例3
    assert solution.partitionLabels("ababcbacadefegdehijhklij") == [9,7,8]
    
    # 测试用例4
    assert solution.partitionLabels("a") == [1]
    
    # 测试用例5
    assert solution.partitionLabels("ab") == [1,1]
    
    # 测试用例6：边界情况
    assert solution.partitionLabels("") == []
    assert solution.partitionLabels("a") == [1]
    
    # 测试用例7：单字符
    assert solution.partitionLabels("aaaa") == [4]
    
    # 测试用例8：无重复字符
    assert solution.partitionLabels("abcdef") == [1,1,1,1,1,1]
    
    # 测试用例9：复杂情况
    assert solution.partitionLabels("ababcbacadefegdehijhklij") == [9,7,8]
    assert solution.partitionLabels("eccbbbbdec") == [10]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
