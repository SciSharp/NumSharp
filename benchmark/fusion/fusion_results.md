# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NpyIter pass (no intermediates). Fixed-expression gate, not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    3.50 ms   unfused    6.39 ms   (1.83x)
  (a-b)/(a+b) fused    3.56 ms   unfused   15.61 ms   (4.38x)
  sum(a*b)    fused    2.70 ms   unfused    4.54 ms   (1.68x)
  sum(af*bf)  fused    1.76 ms   unfused    2.52 ms   (1.43x)  [f32]
  a*b+c out=  fused    3.94 ms
  i4*2+f8     fused    3.28 ms   unfused    4.67 ms   (1.42x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         12.79 ms
  (a-b)/(a+b)   19.99 ms
  sum(a*b)       8.93 ms
  sum(af*bf)     4.43 ms  [f32]
  a*b+c out=     4.97 ms  [two-pass with out=]
  i4*2+f8        9.89 ms
```
