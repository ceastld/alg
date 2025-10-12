"""
149. 直线上最多的点数 - 标准答案
"""
from typing import List
from collections import defaultdict
import math


class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        """
        标准解法：斜率统计法
        
        解题思路：
        1. 对于每个点，计算它与其他所有点的斜率
        2. 使用哈希表统计相同斜率的点对数量
        3. 斜率用最简分数表示，避免浮点数精度问题
        4. 返回最大点数
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if len(points) <= 2:
            return len(points)
        
        max_points = 0
        
        for i in range(len(points)):
            slope_count = defaultdict(int)
            same_point = 0
            
            for j in range(len(points)):
                if i == j:
                    continue
                
                # 计算斜率
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                
                if dx == 0 and dy == 0:
                    same_point += 1
                elif dx == 0:
                    slope_count['vertical'] += 1
                elif dy == 0:
                    slope_count['horizontal'] += 1
                else:
                    # 计算最简分数形式的斜率
                    gcd = math.gcd(dx, dy)
                    slope = (dx // gcd, dy // gcd)
                    slope_count[slope] += 1
            
            # 找到最大斜率对应的点数
            max_slope_count = max(slope_count.values()) if slope_count else 0
            max_points = max(max_points, max_slope_count + same_point + 1)
        
        return max_points
    
    def maxPoints_optimized(self, points: List[List[int]]) -> int:
        """
        优化解法：使用字符串表示斜率
        
        解题思路：
        1. 将斜率转换为字符串形式，避免元组比较的开销
        2. 使用更简洁的斜率计算方式
        3. 处理垂直和水平线的特殊情况
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if len(points) <= 2:
            return len(points)
        
        max_points = 0
        
        for i in range(len(points)):
            slope_count = defaultdict(int)
            same_point = 0
            
            for j in range(len(points)):
                if i == j:
                    continue
                
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                
                if dx == 0 and dy == 0:
                    same_point += 1
                elif dx == 0:
                    slope_count['vertical'] += 1
                elif dy == 0:
                    slope_count['horizontal'] += 1
                else:
                    # 使用字符串表示斜率
                    gcd = math.gcd(dx, dy)
                    slope = f"{dx//gcd}/{dy//gcd}"
                    slope_count[slope] += 1
            
            max_slope_count = max(slope_count.values()) if slope_count else 0
            max_points = max(max_points, max_slope_count + same_point + 1)
        
        return max_points
    
    def maxPoints_brute_force(self, points: List[List[int]]) -> int:
        """
        暴力解法：三点共线判断
        
        解题思路：
        1. 枚举所有可能的三点组合
        2. 判断三点是否共线
        3. 统计每条直线上的点数
        
        时间复杂度：O(n^3)
        空间复杂度：O(1)
        """
        if len(points) <= 2:
            return len(points)
        
        max_points = 0
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                count = 2  # 当前两点
                
                for k in range(len(points)):
                    if k == i or k == j:
                        continue
                    
                    # 判断三点是否共线
                    if self.isCollinear(points[i], points[j], points[k]):
                        count += 1
                
                max_points = max(max_points, count)
        
        return max_points
    
    def isCollinear(self, p1: List[int], p2: List[int], p3: List[int]) -> bool:
        """
        判断三点是否共线
        
        使用叉积判断：如果三点共线，则向量p1p2和p1p3的叉积为0
        """
        return (p2[1] - p1[1]) * (p3[0] - p1[0]) == (p3[1] - p1[1]) * (p2[0] - p1[0])
    
    def maxPoints_float_slope(self, points: List[List[int]]) -> int:
        """
        浮点数斜率解法（注意精度问题）
        
        解题思路：
        1. 直接计算浮点数斜率
        2. 使用epsilon处理浮点数精度问题
        3. 适用于对精度要求不高的场景
        
        时间复杂度：O(n^2)
        空间复杂度：O(n)
        """
        if len(points) <= 2:
            return len(points)
        
        max_points = 0
        epsilon = 1e-9
        
        for i in range(len(points)):
            slope_count = defaultdict(int)
            same_point = 0
            
            for j in range(len(points)):
                if i == j:
                    continue
                
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                
                if dx == 0 and dy == 0:
                    same_point += 1
                elif dx == 0:
                    slope_count['vertical'] += 1
                elif dy == 0:
                    slope_count['horizontal'] += 1
                else:
                    # 使用浮点数斜率（注意精度问题）
                    slope = dy / dx
                    # 四舍五入到小数点后9位，避免精度问题
                    slope_rounded = round(slope, 9)
                    slope_count[slope_rounded] += 1
            
            max_slope_count = max(slope_count.values()) if slope_count else 0
            max_points = max(max_points, max_slope_count + same_point + 1)
        
        return max_points


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    points = [[1,1],[2,2],[3,3]]
    result = solution.maxPoints(points)
    expected = 3
    assert result == expected
    
    # 测试用例2
    points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
    result = solution.maxPoints(points)
    expected = 4
    assert result == expected
    
    # 测试用例3
    points = [[1,1],[2,2],[3,3],[4,4]]
    result = solution.maxPoints(points)
    expected = 4
    assert result == expected
    
    # 测试用例4
    points = [[0,0]]
    result = solution.maxPoints(points)
    expected = 1
    assert result == expected
    
    # 测试用例5
    points = [[0,0],[1,1],[0,0]]
    result = solution.maxPoints(points)
    expected = 3
    assert result == expected
    
    # 测试优化解法
    print("测试优化解法...")
    points = [[1,1],[2,2],[3,3]]
    result_opt = solution.maxPoints_optimized(points)
    assert result_opt == expected
    
    # 测试暴力解法
    print("测试暴力解法...")
    points = [[1,1],[2,2],[3,3]]
    result_bf = solution.maxPoints_brute_force(points)
    assert result_bf == expected
    
    # 测试浮点数解法
    print("测试浮点数解法...")
    points = [[1,1],[2,2],[3,3]]
    result_float = solution.maxPoints_float_slope(points)
    assert result_float == expected
    
    print("所有测试用例通过！")
    print("所有解法验证通过！")


if __name__ == "__main__":
    main()
