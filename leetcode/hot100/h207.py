"""
LeetCode 207. Course Schedule

题目描述：
你这个学期必须选修numCourses门课程，记为0到numCourses-1。
在选修某些课程之前需要一些先修课程。先修课程按数组prerequisites给出，其中prerequisites[i] = [ai, bi]，表示如果要学习课程ai则必须先学习课程bi。
例如，先修课程对[0, 1]表示：想要学习课程0，你需要先完成课程1。
请你判断是否可能完成所有课程的学习？如果可以，返回true；否则，返回false。

示例：
numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有2门课程。学习课程1之前，你需要完成课程0。这是可能的。

数据范围：
- 1 <= numCourses <= 10^5
- 0 <= prerequisites.length <= 5000
- prerequisites[i].length == 2
- 0 <= ai, bi < numCourses
- prerequisites[i]中的所有课程对互不相同
"""

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        # Method 1: Topological Sort using BFS (Kahn's Algorithm)
        # Build adjacency list and in-degree count
        graph = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses
        
        # Build graph and calculate in-degrees
        for course, prereq in prerequisites:
            graph[prereq].append(course)  # prereq -> course
            in_degree[course] += 1
        
        # Find all courses with no prerequisites (in-degree = 0)
        queue = []
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)
        
        # Process courses in topological order
        completed = 0
        while queue:
            course = queue.pop(0)
            completed += 1
            
            # Remove this course and update in-degrees of dependent courses
            for dependent in graph[course]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # If we can complete all courses, no cycle exists
        return completed == numCourses

    def canFinishDFS(self, numCourses, prerequisites):
        """
        Alternative method using DFS to detect cycles
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        # Build adjacency list
        graph = [[] for _ in range(numCourses)]
        for course, prereq in prerequisites:
            graph[prereq].append(course)
        
        # 0: unvisited, 1: visiting, 2: visited
        state = [0] * numCourses
        
        def hasCycle(course):
            if state[course] == 1:  # Currently visiting (back edge found)
                return True
            if state[course] == 2:  # Already visited
                return False
            
            state[course] = 1  # Mark as visiting
            
            # Check all dependent courses
            for dependent in graph[course]:
                if hasCycle(dependent):
                    return True
            
            state[course] = 2  # Mark as visited
            return False
        
        # Check each course for cycles
        for course in range(numCourses):
            if state[course] == 0:  # Unvisited
                if hasCycle(course):
                    return False
        
        return True

    def canFinishUnionFind(self, numCourses, prerequisites):
        """
        Alternative method using Union-Find (for reference)
        Note: This approach is less intuitive for this problem
        """
        # This is more complex and less efficient for cycle detection in DAG
        # The BFS and DFS approaches above are more suitable
        pass

# Test cases
if __name__ == "__main__":
    solution = Solution()
    
    # Test case 1: No cycle
    numCourses1 = 2
    prerequisites1 = [[1, 0]]
    print(f"Test 1 - Can finish: {solution.canFinish(numCourses1, prerequisites1)}")  # True
    print(f"Test 1 - DFS result: {solution.canFinishDFS(numCourses1, prerequisites1)}")  # True
    
    # Test case 2: Has cycle
    numCourses2 = 2
    prerequisites2 = [[1, 0], [0, 1]]
    print(f"Test 2 - Can finish: {solution.canFinish(numCourses2, prerequisites2)}")  # False
    print(f"Test 2 - DFS result: {solution.canFinishDFS(numCourses2, prerequisites2)}")  # False
    
    # Test case 3: Complex case
    numCourses3 = 4
    prerequisites3 = [[1, 0], [2, 0], [3, 1], [3, 2]]
    print(f"Test 3 - Can finish: {solution.canFinish(numCourses3, prerequisites3)}")  # True
    print(f"Test 3 - DFS result: {solution.canFinishDFS(numCourses3, prerequisites3)}")  # True
    
    # Test case 4: Single course
    numCourses4 = 1
    prerequisites4 = []
    print(f"Test 4 - Can finish: {solution.canFinish(numCourses4, prerequisites4)}")  # True
    print(f"Test 4 - DFS result: {solution.canFinishDFS(numCourses4, prerequisites4)}")  # True