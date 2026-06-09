# =============================================================================
# POC — NumPy reference side for benchmark/poc/npyiter_parity_poc.cs
# Identical methodology: preallocated outputs (out=), same shapes, same
# timing scheme. Fusion aspects let NumPy allocate its intermediate
# temporaries — eliminating those is precisely what fusion is.
# =============================================================================
import numpy as np
import time

def t_ms(f, iters, warm):
    for _ in range(warm):
        f()
    t0 = time.perf_counter()
    for _ in range(iters):
        f()
    return (time.perf_counter() - t0) * 1000.0 / iters

N1M = 1_000_000
N10M = 10_000_000

print(f"numpy {np.__version__}")
print()
print("aspect                                NumPy")
print("--------------------------------------------------")

# A. contiguous unary
a = (np.arange(N10M).astype(np.float32) + 1.0)
out10 = np.empty(N10M, np.float32)
t = t_ms(lambda: np.sqrt(a, out=out10), 80, 20)
print(f"A contig sqrt f32 10M               {t:8.2f} ms")

# B. contiguous binary
b = (np.arange(N10M).astype(np.float32) + 2.0)
t = t_ms(lambda: np.add(a, b, out=out10), 80, 20)
print(f"B contig add  f32 10M               {t:8.2f} ms")

# C. strided binary
wa = np.arange(2 * N1M).astype(np.float32) + 1.0
wb = np.arange(2 * N1M).astype(np.float32) + 2.0
sa, sb = wa[::2], wb[::2]
out1 = np.empty(N1M, np.float32)
t = t_ms(lambda: np.add(sa, sb, out=out1), 300, 60)
print(f"C strided add a[::2]+b[::2] f32 1M  {t*1000:8.0f} us")

# D. 2-D strided unary
big = (np.arange(4 * N1M).astype(np.float32) + 1.0).reshape(2000, 2000)
s2d = big[::2, ::2]
out2d = np.empty((1000, 1000), np.float32)
t = t_ms(lambda: np.sqrt(s2d, out=out2d), 300, 60)
print(f"D strided sqrt a[::2,::2] f32 1M    {t*1000:8.0f} us")

# E. strided reduction
we = (np.arange(2 * N1M).astype(np.float32) % 97.0) + 1.0
se = we[::2]
t = t_ms(lambda: np.sum(se), 300, 60)
print(f"E strided sum a[::2] f32 1M         {t*1000:8.0f} us")

# F. a*b + c — two passes + one temp (out= for the final result)
fa = (np.arange(N10M).astype(np.float32) % 13.0) + 1.0
fb = (np.arange(N10M).astype(np.float32) % 7.0) + 2.0
fc = (np.arange(N10M).astype(np.float32) % 5.0) + 3.0
t = t_ms(lambda: np.add(np.multiply(fa, fb), fc, out=out10), 60, 15)
print(f"F a*b+c f32 10M (2-pass + temp)     {t:8.2f} ms")

# G. (a-b)/(a+b) — three passes + two temps
ga = (np.arange(N10M).astype(np.float32) % 13.0) + 5.0
gb = (np.arange(N10M).astype(np.float32) % 7.0) + 1.0
t = t_ms(lambda: np.divide(np.subtract(ga, gb), np.add(ga, gb), out=out10), 60, 15)
print(f"G (a-b)/(a+b) f32 10M (3-pass)      {t:8.2f} ms")

# H. small-N per-call
h = np.arange(1000).astype(np.float32) + 1.0
outh = np.empty(1000, np.float32)
t = t_ms(lambda: np.sqrt(h, out=outh), 50_000, 5_000)
print(f"H small-N sqrt f32 1K               {t*1000:8.2f} us/call")
