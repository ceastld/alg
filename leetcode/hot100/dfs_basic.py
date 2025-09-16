from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import sys

@dataclass
class TreeNode:
    """二叉树节点"""
    val: int
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None

@dataclass
class GraphNode:
    """图节点"""
    val: Any
    neighbors: List['GraphNode'] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = []

class BasicDFS:
    """最基础的DFS实现和示例 - Python 3 高级语法版本"""
    
    def __init__(self) -> None:
        pass
    
    def dfs_recursive(
        self, 
        graph: Dict[str, List[str]], 
        start: str, 
        visited: Optional[Set[str]] = None
    ) -> List[str]:
        """
        递归实现DFS
        
        Args:
            graph: 邻接表表示的图
            start: 起始节点
            visited: 访问标记集合
            
        Returns:
            访问顺序列表
        """
        if visited is None:
            visited = set()
        
        visited.add(start)
        result = [start]
        
        # 遍历当前节点的所有邻居
        for neighbor in graph.get(start, []):
            if neighbor not in visited:
                result.extend(self.dfs_recursive(graph, neighbor, visited))
        
        return result
    
    def dfs_iterative(
        self, 
        graph: Dict[str, List[str]], 
        start: str
    ) -> List[str]:
        """
        迭代实现DFS（使用栈）
        
        Args:
            graph: 邻接表表示的图
            start: 起始节点
            
        Returns:
            访问顺序列表
        """
        visited: Set[str] = set()
        stack: List[str] = [start]
        result: List[str] = []
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                result.append(node)
                
                # 将邻居节点加入栈（注意顺序，确保访问顺序一致）
                for neighbor in reversed(graph.get(node, [])):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result
    
    def dfs_path_finding(
        self, 
        graph: Dict[str, List[str]], 
        start: str, 
        end: str
    ) -> Optional[List[str]]:
        """
        使用DFS寻找路径
        
        Args:
            graph: 邻接表表示的图
            start: 起始节点
            end: 目标节点
            
        Returns:
            从start到end的路径，如果不存在返回None
        """
        def dfs_path(
            node: str, 
            target: str, 
            path: List[str], 
            visited: Set[str]
        ) -> Optional[List[str]]:
            if node == target:
                return path + [node]
            
            visited.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    result = dfs_path(neighbor, target, path + [node], visited)
                    if result:
                        return result
            
            visited.remove(node)  # 回溯
            return None
        
        return dfs_path(start, end, [], set())
    
    def dfs_cycle_detection(self, graph: Dict[str, List[str]]) -> bool:
        """
        使用DFS检测图中是否存在环
        
        Args:
            graph: 邻接表表示的图
            
        Returns:
            是否存在环
        """
        # 0: 未访问, 1: 正在访问, 2: 已访问
        state: Dict[str, int] = {node: 0 for node in graph}
        
        def has_cycle(node: str) -> bool:
            if state[node] == 1:  # 正在访问，发现后向边
                return True
            if state[node] == 2:  # 已访问
                return False
            
            state[node] = 1  # 标记为正在访问
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            state[node] = 2  # 标记为已访问
            return False
        
        # 检查每个未访问的节点
        for node in graph:
            if state[node] == 0:
                if has_cycle(node):
                    return True
        
        return False
    
    def dfs_connected_components(
        self, 
        graph: Dict[str, List[str]]
    ) -> List[List[str]]:
        """
        使用DFS找到所有连通分量
        
        Args:
            graph: 邻接表表示的图
            
        Returns:
            连通分量列表
        """
        visited: Set[str] = set()
        components: List[List[str]] = []
        
        def dfs_component(node: str, component: List[str]) -> None:
            visited.add(node)
            component.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs_component(neighbor, component)
        
        for node in graph:
            if node not in visited:
                component: List[str] = []
                dfs_component(node, component)
                components.append(component)
        
        return components
    
    def dfs_tree_traversal(
        self, 
        root: Optional[TreeNode]
    ) -> Dict[str, List[int]]:
        """
        二叉树DFS遍历（前序、中序、后序）
        
        Args:
            root: 二叉树根节点
            
        Returns:
            遍历结果字典
        """
        result: Dict[str, List[int]] = {
            'preorder': [],
            'inorder': [],
            'postorder': []
        }
        
        def traverse(node: Optional[TreeNode]) -> None:
            if not node:
                return
            
            # 前序遍历：根 -> 左 -> 右
            result['preorder'].append(node.val)
            
            # 遍历左子树
            traverse(node.left)
            
            # 中序遍历：左 -> 根 -> 右
            result['inorder'].append(node.val)
            
            # 遍历右子树
            traverse(node.right)
            
            # 后序遍历：左 -> 右 -> 根
            result['postorder'].append(node.val)
        
        traverse(root)
        return result

# 测试用例
def main() -> None:
    """主测试函数"""
    dfs = BasicDFS()
    
    # 测试图
    graph: Dict[str, List[str]] = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    
    print("=== 基础DFS测试 ===")
    print(f"图结构: {graph}")
    print()
    
    # 1. 递归DFS
    print("1. 递归DFS遍历:")
    recursive_result = dfs.dfs_recursive(graph, 'A')
    print(f"访问顺序: {recursive_result}")
    print()
    
    # 2. 迭代DFS
    print("2. 迭代DFS遍历:")
    iterative_result = dfs.dfs_iterative(graph, 'A')
    print(f"访问顺序: {iterative_result}")
    print()
    
    # 3. 路径查找
    print("3. 路径查找:")
    path = dfs.dfs_path_finding(graph, 'A', 'F')
    print(f"从A到F的路径: {path}")
    print()
    
    # 4. 环检测
    print("4. 环检测:")
    has_cycle = dfs.dfs_cycle_detection(graph)
    print(f"图中是否存在环: {has_cycle}")
    
    # 测试有环的图
    cyclic_graph: Dict[str, List[str]] = {
        'A': ['B'],
        'B': ['C'],
        'C': ['A']
    }
    has_cycle_cyclic = dfs.dfs_cycle_detection(cyclic_graph)
    print(f"有环图中是否存在环: {has_cycle_cyclic}")
    print()
    
    # 5. 连通分量
    print("5. 连通分量:")
    disconnected_graph: Dict[str, List[str]] = {
        'A': ['B'],
        'B': ['A'],
        'C': ['D'],
        'D': ['C'],
        'E': []
    }
    components = dfs.dfs_connected_components(disconnected_graph)
    print(f"连通分量: {components}")
    print()
    
    # 6. 二叉树遍历测试
    print("6. 二叉树遍历:")
    
    # 构建测试二叉树
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    tree_result = dfs.dfs_tree_traversal(root)
    print(f"前序遍历: {tree_result['preorder']}")
    print(f"中序遍历: {tree_result['inorder']}")
    print(f"后序遍历: {tree_result['postorder']}")

if __name__ == "__main__":
    main()

# 额外的实用DFS函数
class AdvancedDFS:
    """高级DFS应用 - Python 3 高级语法版本"""
    
    @staticmethod
    def dfs_maze_solver(
        maze: List[List[int]], 
        start: Tuple[int, int], 
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        使用DFS解决迷宫问题
        
        Args:
            maze: 二维数组，0表示可通行，1表示障碍
            start: 起始坐标 (row, col)
            end: 目标坐标 (row, col)
            
        Returns:
            路径坐标列表，如果无解返回None
        """
        rows, cols = len(maze), len(maze[0])
        visited: Set[Tuple[int, int]] = set()
        directions: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
        
        def dfs_maze(
            row: int, 
            col: int, 
            path: List[Tuple[int, int]]
        ) -> Optional[List[Tuple[int, int]]]:
            if (row, col) == end:
                return path + [(row, col)]
            
            if (row < 0 or row >= rows or col < 0 or col >= cols or
                maze[row][col] == 1 or (row, col) in visited):
                return None
            
            visited.add((row, col))
            
            for dr, dc in directions:
                result = dfs_maze(row + dr, col + dc, path + [(row, col)])
                if result:
                    return result
            
            visited.remove((row, col))  # 回溯
            return None
        
        return dfs_maze(start[0], start[1], [])
    
    @staticmethod
    def dfs_permutations(nums: List[int]) -> List[List[int]]:
        """
        使用DFS生成全排列
        
        Args:
            nums: 数字列表
            
        Returns:
            所有排列的列表
        """
        result: List[List[int]] = []
        
        def dfs_permute(current: List[int], remaining: List[int]) -> None:
            if not remaining:
                result.append(current[:])
                return
            
            for i in range(len(remaining)):
                current.append(remaining[i])
                dfs_permute(current, remaining[:i] + remaining[i+1:])
                current.pop()  # 回溯
        
        dfs_permute([], nums)
        return result
    
    @staticmethod
    def dfs_combinations(nums: List[int], k: int) -> List[List[int]]:
        """
        使用DFS生成组合
        
        Args:
            nums: 数字列表
            k: 组合长度
            
        Returns:
            所有组合的列表
        """
        result: List[List[int]] = []
        
        def dfs_combine(start: int, current: List[int]) -> None:
            if len(current) == k:
                result.append(current[:])
                return
            
            for i in range(start, len(nums)):
                current.append(nums[i])
                dfs_combine(i + 1, current)
                current.pop()  # 回溯
        
        dfs_combine(0, [])
        return result

# 高级DFS测试
def test_advanced_dfs() -> None:
    """测试高级DFS功能"""
    print("\n=== 高级DFS应用测试 ===")
    advanced = AdvancedDFS()
    
    # 迷宫求解测试
    print("1. 迷宫求解:")
    maze: List[List[int]] = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    start: Tuple[int, int] = (0, 0)
    end: Tuple[int, int] = (4, 4)
    path = advanced.dfs_maze_solver(maze, start, end)
    print(f"迷宫路径: {path}")
    print()
    
    # 全排列测试
    print("2. 全排列:")
    nums: List[int] = [1, 2, 3]
    permutations = advanced.dfs_permutations(nums)
    print(f"数字 {nums} 的全排列: {permutations}")
    print()
    
    # 组合测试
    print("3. 组合:")
    nums = [1, 2, 3, 4]
    k = 2
    combinations = advanced.dfs_combinations(nums, k)
    print(f"数字 {nums} 中长度为 {k} 的组合: {combinations}")

# 运行所有测试
if __name__ == "__main__":
    main()
    test_advanced_dfs()
