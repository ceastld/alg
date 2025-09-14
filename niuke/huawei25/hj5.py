def hex_to_decimal(hex_str: str) -> int:
    """
    Convert hexadecimal string to decimal integer.
    
    Args:
        hex_str: Hexadecimal string starting with '0x'
        
    Returns:
        Decimal integer value
    """
    # Remove '0x' prefix
    hex_digits = hex_str[2:]
    
    result = 0
    
    # Process from left to right using *16 + value method
    for digit in hex_digits:
        # Use ASCII encoding to convert hex digit to decimal value
        # More concise approach using ASCII arithmetic
        ascii_val = ord(digit.upper())
        if ascii_val <= ord('9'):
            # '0'-'9' -> 0-9
            value = ascii_val - ord('0')
        else:
            # 'A'-'F' -> 10-15
            value = ascii_val - ord('A') + 10
        
        result = result * 16 + value
    
    return result


def main():
    """Main function to handle input and output."""
    hex_input = input().strip()
    decimal_result = hex_to_decimal(hex_input)
    print(decimal_result)


if __name__ == "__main__":
    main()
