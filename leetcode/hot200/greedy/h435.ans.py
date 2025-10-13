"""
435. 无重叠区间 - 标准答案
"""
from typing import List


class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 按右端点排序
        2. 选择右端点最小的区间，这样能留下更多空间给后续区间
        3. 移除所有与当前区间重叠的区间
        4. 贪心策略：优先选择右端点最小的区间
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not intervals:
            return 0
        
        # 按右端点排序
        intervals.sort(key=lambda x: x[1])
        
        # 选择第一个区间
        count = 1
        end = intervals[0][1]
        
        # 遍历剩余区间
        for i in range(1, len(intervals)):
            # 如果当前区间的左端点大于等于前一个区间的右端点，不重叠
            if intervals[i][0] >= end:
                count += 1
                end = intervals[i][1]
        
        # 返回需要移除的区间数量
        return len(intervals) - count
    
    def eraseOverlapIntervals_alternative(self, intervals: List[List[int]]) -> int:
        """
        替代解法：贪心算法（按左端点排序）
        
        解题思路：
        1. 按左端点排序
        2. 维护当前区间的右端点
        3. 如果当前区间与前一区间重叠，选择右端点更小的区间
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not intervals:
            return 0
        
        # 按左端点排序
        intervals.sort(key=lambda x: x[0])
        
        removed = 0
        end = intervals[0][1]
        
        for i in range(1, len(intervals)):
            if intervals[i][0] < end:
                # 重叠，需要移除一个区间
                removed += 1
                # 选择右端点更小的区间
                end = min(end, intervals[i][1])
            else:
                # 不重叠，更新右端点
                end = intervals[i][1]
        
        return removed
    
    def eraseOverlapIntervals_optimized(self, intervals: List[List[int]]) -> int:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 按右端点排序
        2. 使用变量维护当前区间的右端点
        3. 贪心策略：优先选择右端点最小的区间
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not intervals:
            return 0
        
        # 按右端点排序
        intervals.sort(key=lambda x: x[1])
        
        # 选择第一个区间
        count = 1
        end = intervals[0][1]
        
        # 遍历剩余区间
        for i in range(1, len(intervals)):
            if intervals[i][0] >= end:
                count += 1
                end = intervals[i][1]
        
        return len(intervals) - count
    
    def eraseOverlapIntervals_detailed(self, intervals: List[List[int]]) -> int:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 按右端点排序
        2. 选择右端点最小的区间，这样能留下更多空间给后续区间
        3. 移除所有与当前区间重叠的区间
        4. 贪心策略：优先选择右端点最小的区间
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not intervals:
            return 0
        
        # 按右端点排序
        intervals.sort(key=lambda x: x[1])
        
        # 选择第一个区间
        count = 1
        end = intervals[0][1]
        
        # 遍历剩余区间
        for i in range(1, len(intervals)):
            # 如果当前区间的左端点大于等于前一个区间的右端点，不重叠
            if intervals[i][0] >= end:
                count += 1
                end = intervals[i][1]
        
        # 返回需要移除的区间数量
        return len(intervals) - count
    
    def eraseOverlapIntervals_brute_force(self, intervals: List[List[int]]) -> int:
        """
        暴力解法：回溯
        
        解题思路：
        1. 使用回溯算法尝试所有可能的区间组合
        2. 检查每个组合是否无重叠
        3. 返回需要移除的最少区间数量
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        def is_overlapping(interval1, interval2):
            return not (interval1[1] <= interval2[0] or interval2[1] <= interval1[0])
        
        def backtrack(index, current_intervals):
            if index == len(intervals):
                # 检查当前区间组合是否无重叠
                for i in range(len(current_intervals)):
                    for j in range(i + 1, len(current_intervals)):
                        if is_overlapping(current_intervals[i], current_intervals[j]):
                            return float('inf')
                return len(intervals) - len(current_intervals)
            
            # 不选择当前区间
            result = backtrack(index + 1, current_intervals)
            
            # 选择当前区间
            current_intervals.append(intervals[index])
            result = min(result, backtrack(index + 1, current_intervals))
            current_intervals.pop()
            
            return result
        
        return backtrack(0, [])
    
    def eraseOverlapIntervals_step_by_step(self, intervals: List[List[int]]) -> int:
        """
        分步解法：贪心算法（分步处理）
        
        解题思路：
        1. 第一步：按右端点排序
        2. 第二步：贪心选择区间
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not intervals:
            return 0
        
        # 第一步：按右端点排序
        intervals.sort(key=lambda x: x[1])
        
        # 第二步：贪心选择区间
        count = 1
        end = intervals[0][1]
        
        for i in range(1, len(intervals)):
            if intervals[i][0] >= end:
                count += 1
                end = intervals[i][1]
        
        return len(intervals) - count


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.eraseOverlapIntervals([[1,2],[2,3],[3,4],[1,3]]) == 1
    
    # 测试用例2
    assert solution.eraseOverlapIntervals([[1,2],[1,2],[1,2]]) == 2
    
    # 测试用例3
    assert solution.eraseOverlapIntervals([[1,2],[2,3]]) == 0
    
    # 测试用例4
    assert solution.eraseOverlapIntervals([[1,2],[1,3],[2,3],[3,4]]) == 1
    
    # 测试用例5
    assert solution.eraseOverlapIntervals([[1,100],[11,22],[1,11],[2,12]]) == 2
    
    # 测试用例6：边界情况
    assert solution.eraseOverlapIntervals([]) == 0
    assert solution.eraseOverlapIntervals([[1,2]]) == 0
    
    # 测试用例7：全重叠
    assert solution.eraseOverlapIntervals([[1,2],[1,2],[1,2]]) == 2
    
    # 测试用例8：无重叠
    assert solution.eraseOverlapIntervals([[1,2],[3,4],[5,6]]) == 0
    
    # 测试用例9：复杂情况
    assert solution.eraseOverlapIntervals([[1,2],[2,3],[3,4],[1,3]]) == 1
    assert solution.eraseOverlapIntervals([[1,2],[1,3],[2,3],[3,4]]) == 1
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
