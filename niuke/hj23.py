def solve(s: str) -> str:
    """
    Delete characters with minimum frequency from the string.
    
    Args:
        s: Input string containing only lowercase letters
        
    Returns:
        String with minimum frequency characters removed
    """
    # Count frequency of each character
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    # Find minimum frequency
    min_freq = min(freq.values())
    
    # Build result string excluding characters with minimum frequency
    result = ""
    for char in s:
        if freq[char] != min_freq:
            result += char
    
    return result


def main():
    """Main function to handle input and output."""
    s = input().strip()
    result = solve(s)
    # Ensure we always output at least one character as required
    if not result:
        # If all characters have the same frequency, keep the first character
        result = s[0] if s else ""
    print(result)


if __name__ == "__main__":
    main()
