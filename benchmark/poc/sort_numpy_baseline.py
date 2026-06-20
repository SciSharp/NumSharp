import numpy as np, time

def best_ms(fn, rounds=7):
    b = 1e18
    for _ in range(rounds):
        t = time.perf_counter(); fn(); b = min(b, time.perf_counter()-t)
    return b*1000

sizes = [1_000, 100_000, 1_000_000, 10_000_000]
dts = {'int32':np.int32,'int64':np.int64,'float32':np.float32,'float64':np.float64}

print("== np.sort (quicksort), 1-D ==")
for name,dt in dts.items():
    for n in sizes:
        if dt in (np.float32,np.float64):
            a = np.random.rand(n).astype(dt)
        else:
            a = np.random.randint(0, 1<<30, n).astype(dt)
        ms = best_ms(lambda: np.sort(a, kind='quicksort'))
        print(f"sort   {name:8} n={n:>9}  {ms:8.3f} ms  ({n/ms/1e3:8.1f} M/s)")

print("== np.argsort (quicksort), 1-D ==")
for name,dt in dts.items():
    for n in sizes:
        if dt in (np.float32,np.float64):
            a = np.random.rand(n).astype(dt)
        else:
            a = np.random.randint(0, 1<<30, n).astype(dt)
        ms = best_ms(lambda: np.argsort(a, kind='quicksort'))
        print(f"argsrt {name:8} n={n:>9}  {ms:8.3f} ms  ({n/ms/1e3:8.1f} M/s)")

print("== np.sort 2-D along axis (n=1,000,000 total) ==")
for shp in [(1000,1000),(1000000,1)]:
    a = np.random.rand(*shp)
    for ax in (0,1):
        ms = best_ms(lambda: np.sort(a, axis=ax, kind='quicksort'))
        print(f"sort2d shape={str(shp):>14} axis={ax}  {ms:8.3f} ms")
