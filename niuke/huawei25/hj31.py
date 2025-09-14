import re


def reverse_words(s: str) -> str:
    """
    Reverse the order of words in a string, keeping only alphabetic characters
    
    Args:
        s: Input string
        
    Returns:
        String with words in reversed order
    """
    # Split by non-alphabetic characters and filter out empty strings
    words = [word for word in re.split(r'[^a-zA-Z]', s) if word]
    return " ".join(reversed(words))


if __name__ == "__main__":
    s = input().strip()
    result = reverse_words(s)
    print(result)