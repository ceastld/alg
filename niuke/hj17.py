import re
from typing import Tuple


def parse_instructions(s: str) -> Tuple[int, int]:
    """
    Parse instruction sequence and return final coordinates.
    
    Valid instruction format: [ADWS] + 1-2 digits + semicolon
    - A: left, D: right, W: up, S: down
    - Distance must be 1-99
    
    Args:
        s: Instruction string containing commands like "A10;S20;W10;D30;"
        
    Returns:
        Tuple of (x, y) coordinates
    """
    x, y = 0, 0  # Initial position
    
    # Split by semicolon to get individual commands
    commands = s.split(';')
    
    for cmd in commands:
        if not cmd:  # Skip empty commands
            continue
            
        # Check if command matches pattern: [ADWS] + 1-2 digits
        if re.match(r'^[ADWS]\d{1,2}$', cmd):
            direction = cmd[0]
            distance_str = cmd[1:]
            distance = int(distance_str)
            
            # Check if distance is in valid range (1-99)
            if 1 <= distance <= 99:
                if direction == 'A':  # Left
                    x -= distance
                elif direction == 'D':  # Right
                    x += distance
                elif direction == 'W':  # Up
                    y += distance
                elif direction == 'S':  # Down
                    y -= distance
    
    return x, y


def main():
    """Main function to handle input and output."""
    s = input().strip()
    x, y = parse_instructions(s)
    print(f"{x},{y}")


if __name__ == "__main__":
    main()
