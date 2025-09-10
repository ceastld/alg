def password_transform(s: str) -> str:
    """
    Transform password according to the given rules:
    - Lowercase letters: map to numbers based on 9-key phone keyboard
    - Uppercase letters: convert to lowercase then shift by 1 position (Z->a, A->b, etc.)
    - Numbers: remain unchanged
    """
    # Dictionary for lowercase letter to number mapping (9-key phone keyboard)
    letter_to_number = {
        'a': '2', 'b': '2', 'c': '2',
        'd': '3', 'e': '3', 'f': '3',
        'g': '4', 'h': '4', 'i': '4',
        'j': '5', 'k': '5', 'l': '5',
        'm': '6', 'n': '6', 'o': '6',
        'p': '7', 'q': '7', 'r': '7', 's': '7',
        't': '8', 'u': '8', 'v': '8',
        'w': '9', 'x': '9', 'y': '9', 'z': '9'
    }
    
    # Dictionary for uppercase letter to lowercase shifted letter mapping
    uppercase_shift = {
        'A': 'b', 'B': 'c', 'C': 'd', 'D': 'e', 'E': 'f', 'F': 'g',
        'G': 'h', 'H': 'i', 'I': 'j', 'J': 'k', 'K': 'l', 'L': 'm',
        'M': 'n', 'N': 'o', 'O': 'p', 'P': 'q', 'Q': 'r', 'R': 's',
        'S': 't', 'T': 'u', 'U': 'v', 'V': 'w', 'W': 'x', 'X': 'y',
        'Y': 'z', 'Z': 'a'
    }
    
    result = []
    
    for char in s:
        if char.islower():
            # Lowercase letters: map to numbers
            result.append(letter_to_number[char])
        elif char.isupper():
            # Uppercase letters: convert to lowercase then shift by 1
            shifted_char = uppercase_shift[char]
            result.append(shifted_char)
        elif char.isdigit():
            # Numbers: remain unchanged
            result.append(char)
        else:
            # Other characters: remain unchanged
            result.append(char)
    
    return ''.join(result)


def main():
    """Main function to handle input and output"""
    s = input().strip()
    transformed = password_transform(s)
    print(transformed)


if __name__ == "__main__":
    main()
