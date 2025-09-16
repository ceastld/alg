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