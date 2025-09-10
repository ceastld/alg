def solve():
    """
    Remove duplicates and sort the numbers in ascending order.
    Use a boolean array of size 501 to track which numbers exist.
    """
    n = int(input())
    
    # Boolean array to mark which numbers exist (1-500)
    exists = [False] * 501
    
    # Read n numbers and mark them in the boolean array
    for _ in range(n):
        num = int(input())
        exists[num] = True
    
    # Output numbers in ascending order (1 to 500)
    for i in range(1, 501):
        if exists[i]:
            print(i)


if __name__ == "__main__":
    solve()
