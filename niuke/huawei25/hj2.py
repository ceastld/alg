def count_char_occurrences(s: str, c: str) -> int:
    """
    Count occurrences of character c in string s based on its type.
    
    Args:
        s: Input string containing letters, digits, and spaces
        c: Character to count (either letter or digit)
    
    Returns:
        Count of occurrences
    """
    if c.isalpha():
        # If c is a letter, count both uppercase and lowercase forms
        return s.count(c.upper()) + s.count(c.lower())
    elif c.isdigit():
        # If c is a digit, count its occurrences directly
        return s.count(c)
    else:
        return 0


def main():
    # Read input
    s = input().strip()
    c = input().strip()
    
    # Count occurrences and print result
    result = count_char_occurrences(s, c)
    print(result)


if __name__ == "__main__":
    main()
