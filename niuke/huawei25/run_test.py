"""
Simple test runner for MOE Top-k problem
"""

from moe_topk import solve_moe_topk


def test_case(n, m, p, k, probabilities, expected):
    """Test a single case"""
    result = solve_moe_topk(n, m, p, k, probabilities)
    
    print(f"Input: n={n}, m={m}, p={p}, k={k}")
    print(f"Probabilities: {probabilities}")
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    
    if expected == "error":
        success = result is None
    else:
        success = result == expected
    
    print(f"Success: {success}")
    print("-" * 50)
    return success


def main():
    """Run all test cases"""
    print("Testing MOE Top-k Routing Problem")
    print("=" * 50)
    
    # Test case 1
    test_case(6, 3, 2, 2, [0.3, 0.1, 0.05, 0.6, 0.4, 0.2], [3, 4])
    
    # Test case 2  
    test_case(6, 4, 2, 2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "error")
    
    # Additional test cases
    test_case(4, 2, 2, 2, [0.1, 0.2, 0.3, 0.4], [2, 3])
    test_case(4, 2, 3, 2, [0.1, 0.2, 0.3, 0.4], "error")  # p > m
    test_case(4, 2, 1, 3, [0.1, 0.2, 0.3, 0.4], "error")  # p*g < k


if __name__ == "__main__":
    main()
