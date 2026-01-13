import math

def is_prime(n: int) -> bool:
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if is_prime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1