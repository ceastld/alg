class Solution(object):
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        # Use monotonic stack to find next greater element
        # Stack stores indices of temperatures
        stack = []
        result = [0] * len(temperatures)
        
        for i, temp in enumerate(temperatures):
            # While current temperature is greater than stack top
            while stack and temperatures[stack[-1]] < temp:
                # Pop the index and calculate days difference
                prev_index = stack.pop()
                result[prev_index] = i - prev_index
            
            # Push current index to stack
            stack.append(i)
        
        return result
        