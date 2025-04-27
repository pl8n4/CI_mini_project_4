# primes_mp_opt.py
import math, sys, time
from multiprocessing import Pool, cpu_count
import psutil

# Worker initializer: store small_primes in each child
def init_worker(sp):
    global small_primes
    small_primes = sp

def sieve_chunk(range_pair):
    low, high = range_pair
    size = high - low + 1
    arr = bytearray(b'\x01')*size
    for p in small_primes:
        start = max(p*p, ((low + p - 1)//p)*p)
        for m in range(start, high+1, p):
            arr[m-low] = 0
    if low <= 1:
        for i in range(min(2, high+1)-low):
            arr[i] = 0
    return arr

def main():
    N = int(sys.argv[1])
    t0 = time.perf_counter()

    # serial build of small primes
    limit = int(math.isqrt(N))
    small = bytearray(b'\x01')*(limit+1)
    small[0:2] = b'\x00\x00'
    for p in range(2, limit+1):
        if small[p]:
            for m in range(p*p, limit+1, p):
                small[m] = 0
    small_primes = [i for i,v in enumerate(small) if v]
    t1 = time.perf_counter()

    P = cpu_count()
    chunk = (N+1 + P -1)//P
    ranges = [(i*chunk, min((i+1)*chunk-1, N)) for i in range(P)]

    # create workers *once* with small_primes in their globals
    with Pool(P, initializer=init_worker, initargs=(small_primes,)) as pool:
        pool.map(sieve_chunk, ranges)
    t2 = time.perf_counter()

    mem_mb = psutil.Process().memory_info().rss/1e6
    print(f"N={N} cores={P} build_s={t1-t0:.3f} sieve_s={t2-t1:.3f} total_s={t2-t0:.3f} mem_MB={mem_mb:.1f}")

if __name__=="__main__":
    main()
