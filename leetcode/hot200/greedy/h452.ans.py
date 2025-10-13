"""
452. 用最少数量的箭引爆气球 - 标准答案
"""
from typing import List


class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 按右端点排序
        2. 每次选择右端点最靠左的气球，用一支箭射穿它
        3. 同时射穿所有与它有重叠的气球
        4. 贪心策略：优先处理右端点最靠左的气球
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not points:
            return 0
        
        # 按右端点排序
        points.sort(key=lambda x: x[1])
        
        arrows = 1
        end = points[0][1]
        
        for i in range(1, len(points)):
            # 如果当前气球的左端点大于前一个气球的右端点，需要新的箭
            if points[i][0] > end:
                arrows += 1
                end = points[i][1]
        
        return arrows
    
    def findMinArrowShots_alternative(self, points: List[List[int]]) -> int:
        """
        替代解法：贪心算法（按左端点排序）
        
        解题思路：
        1. 按左端点排序
        2. 维护当前箭能射到的右端点
        3. 如果当前气球的左端点大于当前右端点，需要新的箭
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not points:
            return 0
        
        # 按左端点排序
        points.sort(key=lambda x: x[0])
        
        arrows = 1
        end = points[0][1]
        
        for i in range(1, len(points)):
            if points[i][0] > end:
                arrows += 1
                end = points[i][1]
            else:
                # 更新右端点为更小的值
                end = min(end, points[i][1])
        
        return arrows
    
    def findMinArrowShots_optimized(self, points: List[List[int]]) -> int:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 按右端点排序
        2. 使用变量维护当前箭能射到的右端点
        3. 贪心策略：优先处理右端点最靠左的气球
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not points:
            return 0
        
        # 按右端点排序
        points.sort(key=lambda x: x[1])
        
        arrows = 1
        end = points[0][1]
        
        for i in range(1, len(points)):
            if points[i][0] > end:
                arrows += 1
                end = points[i][1]
        
        return arrows
    
    def findMinArrowShots_detailed(self, points: List[List[int]]) -> int:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 按右端点排序
        2. 每次选择右端点最靠左的气球，用一支箭射穿它
        3. 同时射穿所有与它有重叠的气球
        4. 贪心策略：优先处理右端点最靠左的气球
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not points:
            return 0
        
        # 按右端点排序
        points.sort(key=lambda x: x[1])
        
        arrows = 1
        end = points[0][1]
        
        for i in range(1, len(points)):
            # 如果当前气球的左端点大于前一个气球的右端点，需要新的箭
            if points[i][0] > end:
                arrows += 1
                end = points[i][1]
        
        return arrows
    
    def findMinArrowShots_brute_force(self, points: List[List[int]]) -> int:
        """
        暴力解法：回溯
        
        解题思路：
        1. 使用回溯算法尝试所有可能的箭的位置
        2. 检查每个位置能射穿多少个气球
        
        时间复杂度：O(2^n)
        空间复杂度：O(n)
        """
        if not points:
            return 0
        
        def can_shot(arrow_pos, balloon):
            return balloon[0] <= arrow_pos <= balloon[1]
        
        def backtrack(index, arrows_used):
            if index == len(points):
                return arrows_used
            
            # 尝试不射箭
            result = backtrack(index + 1, arrows_used)
            
            # 尝试射箭
            for arrow_pos in range(points[index][0], points[index][1] + 1):
                # 检查这个位置能射穿多少个气球
                shot_balloons = []
                for i in range(index, len(points)):
                    if can_shot(arrow_pos, points[i]):
                        shot_balloons.append(i)
                
                # 递归处理剩余的气球
                remaining_points = [points[i] for i in range(len(points)) if i not in shot_balloons]
                if remaining_points:
                    result = min(result, backtrack(0, arrows_used + 1))
                else:
                    result = min(result, arrows_used + 1)
            
            return result
        
        return backtrack(0, 0)
    
    def findMinArrowShots_step_by_step(self, points: List[List[int]]) -> int:
        """
        分步解法：贪心算法（分步处理）
        
        解题思路：
        1. 第一步：按右端点排序
        2. 第二步：贪心选择箭的位置
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not points:
            return 0
        
        # 第一步：按右端点排序
        points.sort(key=lambda x: x[1])
        
        # 第二步：贪心选择箭的位置
        arrows = 1
        end = points[0][1]
        
        for i in range(1, len(points)):
            if points[i][0] > end:
                arrows += 1
                end = points[i][1]
        
        return arrows
    
    def findMinArrowShots_alternative_sort(self, points: List[List[int]]) -> int:
        """
        替代排序解法：贪心算法
        
        解题思路：
        1. 按左端点排序
        2. 维护当前箭能射到的右端点
        3. 如果当前气球的左端点大于当前右端点，需要新的箭
        
        时间复杂度：O(n log n)
        空间复杂度：O(1)
        """
        if not points:
            return 0
        
        # 按左端点排序
        points.sort(key=lambda x: x[0])
        
        arrows = 1
        end = points[0][1]
        
        for i in range(1, len(points)):
            if points[i][0] > end:
                arrows += 1
                end = points[i][1]
            else:
                # 更新右端点为更小的值
                end = min(end, points[i][1])
        
        return arrows


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    assert solution.findMinArrowShots([[10,16],[2,8],[1,6],[7,12]]) == 2
    
    # 测试用例2
    assert solution.findMinArrowShots([[1,2],[3,4],[5,6],[7,8]]) == 4
    
    # 测试用例3
    assert solution.findMinArrowShots([[1,2],[2,3],[3,4],[4,5]]) == 2
    
    # 测试用例4
    assert solution.findMinArrowShots([[1,2],[2,3],[3,4],[4,5],[5,6]]) == 3
    
    # 测试用例5
    assert solution.findMinArrowShots([[1,2]]) == 1
    
    # 测试用例6：边界情况
    assert solution.findMinArrowShots([]) == 0
    assert solution.findMinArrowShots([[1,1]]) == 1
    
    # 测试用例7：重叠情况
    assert solution.findMinArrowShots([[1,2],[1,3],[2,3]]) == 1
    
    # 测试用例8：不重叠情况
    assert solution.findMinArrowShots([[1,2],[3,4],[5,6]]) == 3
    
    # 测试用例9：复杂情况
    assert solution.findMinArrowShots([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]) == 3
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
