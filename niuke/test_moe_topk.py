"""
Test cases for MOE Top-k routing problem
"""

from moe_topk import solve_moe_topk


def test_example1():
    """Test example 1: 6 3 2 2 with probabilities [0.3, 0.1, 0.05, 0.6, 0.4, 0.2]"""
    n, m, p, k = 6, 3, 2, 2
    probabilities = [0.3, 0.1, 0.05, 0.6, 0.4, 0.2]
    
    result = solve_moe_topk(n, m, p, k, probabilities)
    expected = [3, 4]
    
    print(f"Example 1: {result}")
    print(f"Expected: {expected}")
    print(f"Correct: {result == expected}")
    print()
    
    # Manual verification:
    # Group 0: experts [0,1] -> max prob 0.3 at expert 0
    # Group 1: experts [2,3] -> max prob 0.6 at expert 3  
    # Group 2: experts [4,5] -> max prob 0.4 at expert 4
    # Top 2 groups by rep prob: Group 1 (0.6), Group 2 (0.4)
    # Experts in selected groups: [2,3,4,5]
    # Sort by prob desc, idx asc: (0.6,3), (0.4,4), (0.05,2), (0.2,5)
    # Top 2: [3,4] -> sort asc: [3,4] ✓


def test_example2():
    """Test example 2: 6 4 2 2 - should return error (n not divisible by m)"""
    n, m, p, k = 6, 4, 2, 2
    probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    result = solve_moe_topk(n, m, p, k, probabilities)
    expected = None  # error case
    
    print(f"Example 2: {result}")
    print(f"Expected: {expected}")
    print(f"Correct: {result == expected}")
    print()


def test_edge_cases():
    """Test various edge cases"""
    
    # Test case: p > m
    print("Test p > m:")
    result = solve_moe_topk(4, 2, 3, 2, [0.1, 0.2, 0.3, 0.4])
    print(f"Result: {result}, Expected: None, Correct: {result is None}")
    print()
    
    # Test case: not enough experts (p*g < k)
    print("Test p*g < k:")
    result = solve_moe_topk(4, 2, 1, 3, [0.1, 0.2, 0.3, 0.4])
    print(f"Result: {result}, Expected: None, Correct: {result is None}")
    print()
    
    # Test case: normal case
    print("Test normal case:")
    result = solve_moe_topk(4, 2, 2, 2, [0.1, 0.2, 0.3, 0.4])
    print(f"Result: {result}, Expected: [2, 3], Correct: {result == [2, 3]}")
    print()


def test_with_ties():
    """Test case with tied probabilities"""
    print("Test with tied probabilities:")
    n, m, p, k = 6, 2, 2, 3
    probabilities = [0.5, 0.5, 0.3, 0.3, 0.2, 0.2]
    
    result = solve_moe_topk(n, m, p, k, probabilities)
    print(f"Result: {result}")
    print()
    
    # Manual verification:
    # Group 0: experts [0,1,2] -> max prob 0.5 at expert 0 (smaller idx wins)
    # Group 1: experts [3,4,5] -> max prob 0.3 at expert 3
    # Top 2 groups: Group 0 (0.5), Group 1 (0.3)
    # Experts in selected groups: [0,1,2,3,4,5]
    # Sort by prob desc, idx asc: (0.5,0), (0.5,1), (0.3,2), (0.3,3), (0.2,4), (0.2,5)
    # Top 3: [0,1,2] -> sort asc: [0,1,2]


if __name__ == "__main__":
    test_example1()
    test_example2()
    test_edge_cases()
    test_with_ties()
