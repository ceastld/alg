from math import sqrt

n = int(input().strip())

# Sieve of Eratosthenes for prime factorization
def get_primes(limit: int) -> list[int]:
    """Get all primes up to limit using sieve of Eratosthenes"""
    if limit < 2:
        return []
    
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i, prime in enumerate(is_prime) if prime]

# Prime factorization
factors = []
primes = get_primes(int(sqrt(n)))

for prime in primes:
    while n % prime == 0:
        factors.append(prime)
        n //= prime

if n > 1:
    factors.append(n)

print(" ".join(map(str, factors)))