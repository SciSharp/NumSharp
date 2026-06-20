# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NpyIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    4.38 ms   unfused    7.08 ms   (1.62x)
  (a-b)/(a+b) fused    3.36 ms   unfused   14.21 ms   (4.23x)
  sum(a*b)    fused    2.68 ms   unfused    4.86 ms   (1.82x)
  sum(af*bf)  fused    1.77 ms   unfused    2.38 ms   (1.35x)  [f32]
  a*b+c out=  fused    4.22 ms
  i4*2+f8     fused    3.33 ms   unfused    5.02 ms   (1.51x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    4.37 ms   unfused    7.11 ms   (1.63x)
    [F      ] fused    4.33 ms   unfused    7.81 ms   (1.80x)
    [T      ] fused    4.28 ms   unfused    7.41 ms   (1.73x)
    [strided] fused    4.00 ms   unfused    5.67 ms   (1.42x)
    [bcast  ] fused    1.99 ms   unfused    5.04 ms   (2.54x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         15.10 ms
  (a-b)/(a+b)   21.07 ms
  sum(a*b)       9.39 ms
  sum(af*bf)     4.52 ms  [f32]
  a*b+c out=     5.90 ms  [two-pass with out=]
  i4*2+f8       10.72 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   14.02 ms
    [F      ]   13.72 ms
    [T      ]   13.79 ms
    [strided]    8.64 ms
    [bcast  ]   12.93 ms
```
