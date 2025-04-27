# primes_mpi.py
# MPI-based prime sieve using mpi4py

from mpi4py import MPI
import math, sys, time, psutil


def sieve_range(low, high, small_primes):
    """
    Sieve the interval [low..high] using the provided list of small_primes.
    Returns a bytearray where 1 indicates prime, 0 indicates composite.
    """
    size = high - low + 1
    is_prime = bytearray(b'\x01') * size

    # Mark composites by each small prime
    for p in small_primes:
        # Find first multiple >= low
        start = max(p*p, ((low + p - 1)//p)*p)
        for m in range(start, high + 1, p):
            is_prime[m - low] = 0

    # Handle 0 and 1 if they fall in this range
    if low <= 1:
        for i in range(min(2, high + 1) - low):
            is_prime[i] = 0
    return is_prime


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only rank 0 parses the command line
    if rank == 0:
        if len(sys.argv) != 2:
            print("Usage: python primes_mpi.py N", file=sys.stderr)
            sys.exit(1)
        N = int(sys.argv[1])
    else:
        N = None
    # Broadcast problem size N to all ranks
    N = comm.bcast(N, root=0)

    # Build small_primes (primes up to sqrt(N)) on rank 0
    if rank == 0:
        t_build_start = time.perf_counter()
        limit = int(math.isqrt(N))
        small = bytearray(b'\x01') * (limit + 1)
        small[0:2] = b'\x00\x00'
        for p in range(2, limit + 1):
            if small[p]:
                for m in range(p*p, limit + 1, p):
                    small[m] = 0
        small_primes = [i for i, v in enumerate(small) if v]
        t_build_end = time.perf_counter()
        build_time = t_build_end - t_build_start
    else:
        small_primes = None
        build_time = None
    # Broadcast small_primes list and build_time
    small_primes = comm.bcast(small_primes, root=0)
    build_time   = comm.bcast(build_time,   root=0)

    # Determine this rank's subrange [low..high]
    chunk_size = (N + 1 + size - 1) // size
    low  = rank * chunk_size
    high = min(low + chunk_size - 1, N)

    # Perform the sieve on this rank's subrange
    t_sieve_start = time.perf_counter()
    sieve_range(low, high, small_primes)
    t_sieve_end = time.perf_counter()
    sieve_time = t_sieve_end - t_sieve_start

    # Total time = build + sieve; take the max across ranks
    total_time = build_time + sieve_time
    max_total  = comm.reduce(total_time, op=MPI.MAX, root=0)

    # Rank 0 prints the final timing and memory usage
    if rank == 0:
        mem_mb = psutil.Process().memory_info().rss / 1e6
        print(f"N={N} cores={size} build_s={build_time:.3f} "
              f"sieve_s={sieve_time:.3f} total_s={max_total:.3f} "
              f"mem_MB={mem_mb:.1f}")


if __name__ == "__main__":
    main()
