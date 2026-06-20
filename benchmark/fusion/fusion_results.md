# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NpyIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    3.97 ms   unfused    6.97 ms   (1.76x)
  (a-b)/(a+b) fused    3.38 ms   unfused   15.76 ms   (4.66x)
  sum(a*b)    fused    2.54 ms   unfused    4.25 ms   (1.68x)
  sum(af*bf)  fused    1.51 ms   unfused    1.94 ms   (1.29x)  [f32]
  a*b+c out=  fused    3.81 ms
  i4*2+f8     fused    2.97 ms   unfused    4.97 ms   (1.67x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    4.41 ms   unfused    7.81 ms   (1.77x)
    [F      ] fused   64.28 ms   unfused    7.09 ms   (0.11x)
    [T      ] fused   63.72 ms   unfused    7.40 ms   (0.12x)
    [strided] fused    3.83 ms   unfused    5.30 ms   (1.38x)
    [bcast  ] fused    1.57 ms   unfused    4.57 ms   (2.91x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         13.78 ms
  (a-b)/(a+b)   20.78 ms
  sum(a*b)       8.99 ms
  sum(af*bf)     4.47 ms  [f32]
  a*b+c out=     5.58 ms  [two-pass with out=]
  i4*2+f8       10.15 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   14.18 ms
    [F      ]   19.11 ms
    [T      ]   15.05 ms
    [strided]    8.48 ms
    [bcast  ]   13.18 ms
```
