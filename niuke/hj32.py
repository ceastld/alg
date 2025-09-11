def longest_palindrome(s: str) -> int:
    """
    Simple and intuitive center expansion algorithm
    Time: O(n²), Space: O(1)
    
    Args:
        s: Input string
        
    Returns:
        Length of the longest palindromic substring
    """
    if not s:
        return 0
    
    max_length = 1
    n = len(s)
    
    # Check odd length palindromes (center is a character)
    for i in range(n):
        left, right = i, i
        while left >= 0 and right < n and s[left] == s[right]:
            max_length = max(max_length, right - left + 1)
            left -= 1
            right += 1
    
    # Check even length palindromes (center is between characters)
    for i in range(n - 1):
        left, right = i, i + 1
        while left >= 0 and right < n and s[left] == s[right]:
            max_length = max(max_length, right - left + 1)
            left -= 1
            right += 1
    
    return max_length


if __name__ == "__main__":
    s = input().strip()
    result = longest_palindrome(s)
    print(result)
