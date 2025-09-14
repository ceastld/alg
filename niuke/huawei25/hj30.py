def solve(s: str, t: str) -> str:
    """
    Solve the string processing problem
    
    Args:
        s: First input string
        t: Second input string
        
    Returns:
        Processed result string
    """
    # Merge phase
    u = s + t
    
    # Separate odd and even positions (1-indexed)
    odd_chars = [u[i] for i in range(0, len(u), 2)]  # positions 1, 3, 5, ...
    even_chars = [u[i] for i in range(1, len(u), 2)]  # positions 2, 4, 6, ...
    
    # Sort odd and even characters by ASCII
    odd_chars.sort()
    even_chars.sort()
    
    # Reconstruct u' by interleaving sorted odd and even characters
    u_prime = []
    odd_idx = even_idx = 0
    for i in range(len(u)):
        if i % 2 == 0:  # odd position (1-indexed)
            u_prime.append(odd_chars[odd_idx])
            odd_idx += 1
        else:  # even position (1-indexed)
            u_prime.append(even_chars[even_idx])
            even_idx += 1
    
    # Adjustment phase
    result = []
    for char in u_prime:
        if is_valid_hex(char):
            # Convert to decimal
            decimal = int(char, 16)
            # Convert to 4-bit binary with leading zeros
            binary = format(decimal, '04b')
            # Reverse the binary string
            reversed_binary = binary[::-1]
            # Convert back to uppercase hexadecimal
            hex_result = format(int(reversed_binary, 2), 'X')
            result.append(hex_result)
        else:
            # Keep original character if not valid hex
            result.append(char)
    
    return ''.join(result)


def is_valid_hex(char: str) -> bool:
    """
    Check if character is a valid hexadecimal character
    
    Args:
        char: Character to check
        
    Returns:
        True if valid hex character (0-9, a-f, A-F), False otherwise
    """
    return char.isdigit() or (char.lower() >= 'a' and char.lower() <= 'f')


def main():
    """Main function to handle input and output"""
    try:
        s, t = input().split()
        result = solve(s, t)
        print(result)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
