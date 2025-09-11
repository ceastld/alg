#!/usr/bin/env python3
"""
Example usage of the smart testing framework
"""

from .smart_test import test_cases, run_tests, create_test_suite, run_program_tests


# Method 1: Using decorator with function
@test_cases(
    ("4\n2 5 6 13", "2", "Example 1: [2,5,6,13] -> 2 pairs"),
    ("4\n1 2 2 2", "1", "Example 2: [1,2,2,2] -> 1 pair"),
    ("2\n3 6", "0", "Example 3: [3,6] -> 0 pairs")
)
def hj28_solution():
    """HJ28 Prime Partners solution with test cases"""
    # Import the solution function
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from niuke.hj28 import solve_prime_partners
    
    n = int(input())
    nums = list(map(int, input().split()))
    result = solve_prime_partners(nums)
    print(result)


# Method 2: Using test suite with program file
def test_hj28_program():
    """Test the actual program file"""
    suite = create_test_suite("HJ28 Program Tests")
    suite.add_cases([
        ("4\n2 5 6 13", "2", "Example 1"),
        ("4\n1 2 2 2", "1", "Example 2"),
        ("2\n3 6", "0", "Example 3")
    ])
    
    print("Testing HJ28 program file...")
    program_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'niuke', 'hj28.py')
    run_program_tests(program_path, suite)


# Method 3: Simple test cases for other problems
@test_cases(
    ("5", "3", "Simple input test"),
    ("10", "4", "Another test"),
    ("15", "5", "Third test")
)
def simple_solution():
    """Simple example solution"""
    n = int(input())
    result = n // 2 + 1  # Some calculation
    print(result)


def main():
    """Run all examples"""
    print("=" * 80)
    print("SMART TESTING FRAMEWORK EXAMPLES")
    print("=" * 80)
    
    print("\n1. Testing HJ28 solution function:")
    print("-" * 40)
    run_tests(hj28_solution)
    
    print("\n2. Testing HJ28 program file:")
    print("-" * 40)
    test_hj28_program()
    
    print("\n3. Testing simple solution:")
    print("-" * 40)
    run_tests(simple_solution)


if __name__ == "__main__":
    main()
