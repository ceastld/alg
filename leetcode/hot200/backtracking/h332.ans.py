"""
332. 重新安排行程 - 标准答案
"""
from typing import List
from collections import defaultdict, deque
import heapq


class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        def dfs(curr: str):
            while vec[curr]:
                tmp = heapq.heappop(vec[curr])
                dfs(tmp)
            stack.append(curr)

        vec = defaultdict(list)
        for depart, arrive in tickets:
            vec[depart].append(arrive)
        for key in vec:
            heapq.heapify(vec[key])
        
        stack = list()
        dfs("JFK")
        return stack[::-1]
    
    def findItinerary_optimized(self, tickets: List[List[str]]) -> List[str]:
        """
        优化解法：Hierholzer算法 + heapq
        
        解题思路：
        1. 使用Hierholzer算法找欧拉路径
        2. 使用heapq管理每个机场的目的地，自动按字典序排序
        3. 使用栈存储路径
        4. 当无法继续前进时，将当前节点加入结果
        5. 最后反转结果得到正确路径
        
        时间复杂度：O(E log E)
        空间复杂度：O(E)
        """
        # 构建邻接表，使用heapq管理每个机场的目的地
        graph = defaultdict(list)
        for src, dst in tickets:
            heapq.heappush(graph[src], dst)
        
        stack = ["JFK"]
        result = []
        
        while stack:
            current = stack[-1]
            if graph[current]:
                # 选择字典序最小的目的地
                destination = heapq.heappop(graph[current])
                stack.append(destination)
            else:
                # 无法继续前进，将当前节点加入结果
                result.append(stack.pop())
        
        return result[::-1]
    
    def findItinerary_dfs(self, tickets: List[List[str]]) -> List[str]:
        """
        DFS解法：深度优先搜索
        
        解题思路：
        1. 使用DFS遍历图
        2. 使用贪心策略选择字典序最小的路径
        3. 使用回溯处理死胡同
        
        时间复杂度：O(E log E)
        空间复杂度：O(E)
        """
        # 构建邻接表，使用heapq管理每个机场的目的地
        graph = defaultdict(list)
        for src, dst in tickets:
            heapq.heappush(graph[src], dst)
        
        result = []
        
        def dfs(current: str):
            while graph[current]:
                destination = heapq.heappop(graph[current])
                dfs(destination)
            result.append(current)
        
        dfs("JFK")
        return result[::-1]
    
    def findItinerary_iterative(self, tickets: List[List[str]]) -> List[str]:
        """
        迭代解法：使用栈模拟递归
        
        解题思路：
        1. 使用栈模拟递归过程
        2. 使用贪心策略选择路径
        3. 处理死胡同情况
        
        时间复杂度：O(E log E)
        空间复杂度：O(E)
        """
        # 构建邻接表，使用heapq管理每个机场的目的地
        graph = defaultdict(list)
        for src, dst in tickets:
            heapq.heappush(graph[src], dst)
        
        stack = ["JFK"]
        result = []
        
        while stack:
            current = stack[-1]
            if graph[current]:
                # 选择字典序最小的目的地
                destination = heapq.heappop(graph[current])
                stack.append(destination)
            else:
                # 无法继续前进，将当前节点加入结果
                result.append(stack.pop())
        
        return result[::-1]


def main():
    """测试标准答案"""
    solution = Solution()
    
    # 测试用例1
    tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
    result = solution.findItinerary(tickets)
    expected = ["JFK","MUC","LHR","SFO","SJC"]
    assert result == expected
    
    # 测试用例2
    tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
    result = solution.findItinerary(tickets)
    expected = ["JFK","ATL","JFK","SFO","ATL","SFO"]
    assert result == expected
    
    # 测试用例3
    tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]
    result = solution.findItinerary(tickets)
    expected = ["JFK","NRT","JFK","KUL"]
    assert result == expected
    
    # 测试用例4
    tickets = [["JFK","ATL"],["ATL","JFK"]]
    result = solution.findItinerary(tickets)
    expected = ["JFK","ATL","JFK"]
    assert result == expected
    
    print("所有测试用例通过！")


if __name__ == "__main__":
    main()
