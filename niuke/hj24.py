def solve_chorus_formation(n: int, heights: list[int]) -> int:
    """
    Solve the chorus formation problem with O(n log n) time complexity.
    
    Args:
        n: Number of students
        heights: List of student heights
        
    Returns:
        Minimum number of students to remove
    """
    if n <= 2:
        return 0
    
    # Calculate longest increasing subsequence ending at each position using binary search
    lis = [1] * n
    tails = []  # tails[i] stores the smallest tail element of LIS of length i+1
    
    for i in range(n):
        # Binary search for the position to insert heights[i]
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < heights[i]:
                left = mid + 1
            else:
                right = mid
        
        # Update LIS length for position i
        lis[i] = left + 1
        
        # Update tails array
        if left == len(tails):
            tails.append(heights[i])
        else:
            tails[left] = heights[i]
    
    # Calculate longest decreasing subsequence starting at each position
    # We reverse the array and calculate LIS, then reverse the result
    lds = [1] * n
    tails = []
    
    for i in range(n - 1, -1, -1):
        # Binary search for the position to insert heights[i]
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < heights[i]:
                left = mid + 1
            else:
                right = mid
        
        # Update LDS length for position i
        lds[i] = left + 1
        
        # Update tails array
        if left == len(tails):
            tails.append(heights[i])
        else:
            tails[left] = heights[i]
    
    # Find maximum chorus formation length
    max_chorus = 0
    for i in range(n):
        # Chorus formation with i as the peak
        chorus_length = lis[i] + lds[i] - 1
        max_chorus = max(max_chorus, chorus_length)
    
    return n - max_chorus


def main():
    """Main function to handle input and output."""
    n = int(input())
    heights = list(map(int, input().split()))
    
    result = solve_chorus_formation(n, heights)
    print(result)


if __name__ == "__main__":
    main()
