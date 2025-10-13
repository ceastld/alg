"""
56. 合并区间 - 标准答案
"""
from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 按左端点排序
        2. 遍历区间，如果当前区间与前一个区间重叠，则合并
        3. 否则，将当前区间加入结果
        4. 贪心策略：优先处理左端点小的区间
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not intervals:
            return []
        
        # 按左端点排序
        intervals.sort(key=lambda x: x[0])
        
        result = [intervals[0]]
        
        for i in range(1, len(intervals)):
            current = intervals[i]
            last = result[-1]
            
            # 如果当前区间与前一个区间重叠，则合并
            if current[0] <= last[1]:
                result[-1] = [last[0], max(last[1], current[1])]
            else:
                # 否则，将当前区间加入结果
                result.append(current)
        
        return result
    
    def merge_alternative(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        替代解法：贪心算法（使用栈）
        
        解题思路：
        1. 按左端点排序
        2. 使用栈维护合并后的区间
        3. 如果当前区间与栈顶区间重叠，则合并
        
        时间复杂度：O(n log n)
        空间复杂度：O(n)
        """
        if not intervals:
            return []
        
        # 按左端点排序
        intervals.sort(key=lambda x: x[0])
        
        stack = [intervals[0]]
        
        for i in range(1, len(intervals)):
            current = intervals[i]
            top = stack[-1]
            
            # 如果当前区间与栈顶区间重叠，则合并
            if current[0] <= top[1]:
                stack[-1] = [top[0], max(top[1], current[1])]
            else:
                # 否则，将当前区间压入栈
                stack.append(current)
        
        return stack
    
    def merge_optimized(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 按左端点排序
        2. 使用变量维护当前区间的右端点
        3. 如果当前区间与前一个区间重叠，则合并
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not intervals:
            return []
        
        # 按左端点排序
        intervals.sort(key=lambda x: x[0])
        
        result = [intervals[0]]
        
        for i in range(1, len(intervals)):
            current = intervals[i]
            last = result[-1]
            
            # 如果当前区间与前一个区间重叠，则合并
            if current[0] <= last[1]:
                result[-1] = [last[0], max(last[1], current[1])]
            else:
                # 否则，将当前区间加入结果
                result.append(current)
        
        return result
    
    def merge_detailed(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 按左端点排序
        2. 遍历区间，如果当前区间与前一个区间重叠，则合并
        3. 否则，将当前区间加入结果
        4. 贪心策略：优先处理左端点小的区间
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not intervals:
            return []
        
        # 按左端点排序
        intervals.sort(key=lambda x: x[0])
        
        result = [intervals[0]]
        
        for i in range(1, len(intervals)):
            current = intervals[i]
            last = result[-1]
            
            # 如果当前区间与前一个区间重叠，则合并
            if current[0] <= last[1]:
                # 合并区间：左端点取较小的，右端点取较大的
                result[-1] = [last[0], max(last[1], current[1])]
            else:
                # 否则，将当前区间加入结果
                result.append(current)
        
        return result
    
    def merge_brute_force(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        暴力解法：双重循环
        
        解题思路：
        1. 对于每个区间，检查是否与其他区间重叠
        2. 如果重叠，则合并
        3. 重复直到没有重叠
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        if not intervals:
            return []
        
        # 按左端点排序
        intervals.sort(key=lambda x: x[0])
        
        merged = True
        while merged:
            merged = False
            for i in range(len(intervals) - 1):
                if intervals[i][1] >= intervals[i + 1][0]:
                    # 合并区间
                    intervals[i] = [intervals[i][0], max(intervals[i][1], intervals[i + 1][1])]
                    intervals.pop(i + 1)
                    merged = True
                    break
        
        return intervals
    
    def merge_step_by_step(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        分步解法：贪心算法（分步处理）
        
        解题思路：
        1. 第一步：按左端点排序
        2. 第二步：贪心合并区间
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not intervals:
            return []
        
        # 第一步：按左端点排序
        intervals.sort(key=lambda x: x[0])
        
        # 第二步：贪心合并区间
        result = [intervals[0]]
        
        for i in range(1, len(intervals)):
            current = intervals[i]
            last = result[-1]
            
            if current[0] <= last[1]:
                result[-1] = [last[0], max(last[1], current[1])]
            else:
                result.append(current)
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    
    # 测试用例2
    assert solution.merge([[1,4],[4,5]]) == [[1,5]]
    
    # 测试用例3
    assert solution.merge([[1,4],[0,4]]) == [[0,4]]
    
    # 测试用例4
    assert solution.merge([[1,4],[2,3]]) == [[1,4]]
    
    # 测试用例5
    assert solution.merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    
    # 测试用例6：边界情况
    assert solution.merge([]) == []
    assert solution.merge([[1,2]]) == [[1,2]]
    
    # 测试用例7：全重叠
    assert solution.merge([[1,4],[2,3],[3,5]]) == [[1,5]]
    
    # 测试用例8：无重叠
    assert solution.merge([[1,2],[3,4],[5,6]]) == [[1,2],[3,4],[5,6]]
    
    # 测试用例9：复杂情况
    assert solution.merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    assert solution.merge([[1,4],[4,5]]) == [[1,5]]
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
