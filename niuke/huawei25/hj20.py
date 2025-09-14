import re
from typing import List


def has_required_character_types(password: str) -> bool:
    """
    Check if password contains at least 3 types of characters:
    uppercase, lowercase, digit, special character
    """
    has_upper = bool(re.search(r'[A-Z]', password))
    has_lower = bool(re.search(r'[a-z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_special = bool(re.search(r'[^A-Za-z0-9]', password))
    
    types_count = sum([has_upper, has_lower, has_digit, has_special])
    return types_count >= 3


def has_repeated_substring(password: str) -> bool:
    """
    Check if password contains two independent substrings of length > 2
    that are identical
    """
    n = len(password)
    
    # Check all possible substring lengths from 3 to n//2
    for length in range(3, n // 2 + 1):
        # Check all possible starting positions
        for i in range(n - length + 1):
            substring = password[i:i + length]
            
            # Look for the same substring starting after the current one
            for j in range(i + length, n - length + 1):
                if password[j:j + length] == substring:
                    return True
    
    return False


def is_valid_password(password: str) -> bool:
    """
    Validate password according to all requirements
    """
    # Check length requirement
    if len(password) < 8:
        return False
    
    # Check character types requirement
    if not has_required_character_types(password):
        return False
    
    # Check repeated substring requirement
    if has_repeated_substring(password):
        return False
    
    return True


def solve():
    """
    Main function to handle input and output
    """
    try:
        while True:
            password = input().strip()
            if is_valid_password(password):
                print("OK")
            else:
                print("NG")
    except EOFError:
        pass


if __name__ == "__main__":
    solve()
