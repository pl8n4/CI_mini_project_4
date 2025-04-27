# primes_serial.py
import math, sys, time

N = int(sys.argv[1])        # e.g. 10000000
t0 = time.perf_counter()

is_prime = bytearray(b'\x01') * (N+1)
is_prime[0:2] = b'\x00\x00'
for p in range(2, math.isqrt(N)+1):
    if is_prime[p]:
        # zero out p², p²+p, p²+2p, … up to N
        start = p*p
        step  = p
        count = (N - start)//step + 1
        is_prime[start:N+1:step] = b'\x00' * count

elapsed = time.perf_counter() - t0
print(f"{N} {elapsed:.3f}")
