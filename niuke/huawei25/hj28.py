from math import sqrt
from typing import List


def get_primes(max_n: int) -> List[bool]:
    """Generate prime sieve up to max_n"""
    is_prime = [True] * (max_n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(sqrt(max_n)) + 1):
        if is_prime[i]:
            for j in range(i * i, max_n + 1, i):
                is_prime[j] = False
    return is_prime


def max_bipartite_matching(even_nums: List[int], odd_nums: List[int], is_connected) -> int:
    """Find maximum bipartite matching using DFS"""
    n = len(even_nums)
    m = len(odd_nums)
    match = [-1] * m  # match[j] = i means odd_nums[j] is matched with even_nums[i]
    count = 0
    
    def dfs(i: int, visited: List[bool]) -> bool:
        """DFS to find augmenting path"""
        for j in range(m):
            if is_connected(even_nums[i], odd_nums[j]) and not visited[j]:
                visited[j] = True
                if match[j] == -1 or dfs(match[j], visited):
                    match[j] = i
                    return True
        return False
    
    for i in range(n):
        visited = [False] * m
        if dfs(i, visited):
            count += 1
    
    return count


def solve_prime_partners(nums: List[int]) -> int:
    """Solve the prime partners problem"""
    # Separate even and odd numbers
    even_nums = [x for x in nums if x % 2 == 0]
    odd_nums = [x for x in nums if x % 2 == 1]
    
    # Generate prime sieve up to maximum possible sum
    max_sum = max(nums) * 2
    is_prime = get_primes(max_sum)
    
    # Create connection function
    def is_connected(a: int, b: int) -> bool:
        return is_prime[a + b]
    
    # Find maximum bipartite matching
    return max_bipartite_matching(even_nums, odd_nums, is_connected)


def main():
    """Main function to handle input and output"""
    n = int(input())
    nums = list(map(int, input().split()))
    result = solve_prime_partners(nums)
    print(result)


if __name__ == "__main__":
    main()
