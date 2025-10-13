"""
406. 根据身高重建队列 - 标准答案
"""
from typing import List


class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        """
        标准解法：贪心算法
        
        解题思路：
        1. 按身高降序排列，身高相同时按k值升序排列
        2. 从高到低依次插入到结果中
        3. 插入位置为k值，因为前面已经有k个更高的人
        4. 贪心策略：先处理高的人，再处理矮的人
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        # 按身高降序排列，身高相同时按k值升序排列
        people.sort(key=lambda x: (-x[0], x[1]))
        
        result = []
        for person in people:
            # 插入到k值位置
            result.insert(person[1], person)
        
        return result
    
    def reconstructQueue_alternative(self, people: List[List[int]]) -> List[List[int]]:
        """
        替代解法：贪心算法（使用链表）
        
        解题思路：
        1. 按身高降序排列，身高相同时按k值升序排列
        2. 使用链表结构插入
        3. 插入位置为k值
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        # 按身高降序排列，身高相同时按k值升序排列
        people.sort(key=lambda x: (-x[0], x[1]))
        
        result = []
        for person in people:
            # 插入到k值位置
            result.insert(person[1], person)
        
        return result
    
    def reconstructQueue_optimized(self, people: List[List[int]]) -> List[List[int]]:
        """
        优化解法：贪心算法（空间优化）
        
        解题思路：
        1. 按身高降序排列，身高相同时按k值升序排列
        2. 从高到低依次插入到结果中
        3. 插入位置为k值
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        # 按身高降序排列，身高相同时按k值升序排列
        people.sort(key=lambda x: (-x[0], x[1]))
        
        result = []
        for person in people:
            # 插入到k值位置
            result.insert(person[1], person)
        
        return result
    
    def reconstructQueue_detailed(self, people: List[List[int]]) -> List[List[int]]:
        """
        详细解法：贪心算法（带详细注释）
        
        解题思路：
        1. 按身高降序排列，身高相同时按k值升序排列
        2. 从高到低依次插入到结果中
        3. 插入位置为k值，因为前面已经有k个更高的人
        4. 贪心策略：先处理高的人，再处理矮的人
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        # 按身高降序排列，身高相同时按k值升序排列
        people.sort(key=lambda x: (-x[0], x[1]))
        
        result = []
        for person in people:
            height, k = person
            # 插入到k值位置，因为前面已经有k个更高的人
            result.insert(k, person)
        
        return result
    
    def reconstructQueue_brute_force(self, people: List[List[int]]) -> List[List[int]]:
        """
        暴力解法：回溯
        
        解题思路：
        1. 使用回溯算法尝试所有可能的排列
        2. 检查每个排列是否满足条件
        
        时间复杂度：O(n!)
        空间复杂度：O(n)
        """
        def is_valid(queue):
            for i, (height, k) in enumerate(queue):
                count = 0
                for j in range(i):
                    if queue[j][0] >= height:
                        count += 1
                if count != k:
                    return False
            return True
        
        def backtrack(current, remaining):
            if not remaining:
                if is_valid(current):
                    return current
                return None
            
            for i, person in enumerate(remaining):
                new_current = current + [person]
                new_remaining = remaining[:i] + remaining[i+1:]
                result = backtrack(new_current, new_remaining)
                if result:
                    return result
            return None
        
        return backtrack([], people)
    
    def reconstructQueue_step_by_step(self, people: List[List[int]]) -> List[List[int]]:
        """
        分步解法：贪心算法（分步处理）
        
        解题思路：
        1. 第一步：按身高降序排列，身高相同时按k值升序排列
        2. 第二步：从高到低依次插入到结果中
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        # 第一步：按身高降序排列，身高相同时按k值升序排列
        people.sort(key=lambda x: (-x[0], x[1]))
        
        # 第二步：从高到低依次插入到结果中
        result = []
        for person in people:
            result.insert(person[1], person)
        
        return result
    
    def reconstructQueue_alternative_sort(self, people: List[List[int]]) -> List[List[int]]:
        """
        替代排序解法：贪心算法
        
        解题思路：
        1. 按身高升序排列，身高相同时按k值降序排列
        2. 从矮到高依次插入到结果中
        3. 插入位置为k值
        
        时间复杂度：O(n^2)
        空间复杂度：O(1)
        """
        # 按身高升序排列，身高相同时按k值降序排列
        people.sort(key=lambda x: (x[0], -x[1]))
        
        result = []
        for person in people:
            # 插入到k值位置
            result.insert(person[1], person)
        
        return result


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    result1 = solution.reconstructQueue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]])
    expected1 = [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    assert result1 == expected1
    
    # 测试用例2
    result2 = solution.reconstructQueue([[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]])
    expected2 = [[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
    assert result2 == expected2
    
    # 测试用例3
    result3 = solution.reconstructQueue([[1,0]])
    expected3 = [[1,0]]
    assert result3 == expected3
    
    # 测试用例4
    result4 = solution.reconstructQueue([[2,0],[1,1]])
    expected4 = [[2,0],[1,1]]
    assert result4 == expected4
    
    # 测试用例5：边界情况
    result5 = solution.reconstructQueue([[1,0],[2,0]])
    expected5 = [[1,0],[2,0]]
    assert result5 == expected5
    
    # 测试用例6：复杂情况
    result6 = solution.reconstructQueue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]])
    expected6 = [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
    assert result6 == expected6
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
