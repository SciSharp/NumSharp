# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    4.42 ms   unfused    7.42 ms   (1.68x)
  (a-b)/(a+b) fused    3.67 ms   unfused   14.74 ms   (4.02x)
  sum(a*b)    fused    2.63 ms   unfused    4.69 ms   (1.78x)
  sum(af*bf)  fused    1.63 ms   unfused    2.19 ms   (1.34x)  [f32]
  a*b+c out=  fused    4.15 ms   [1-pass fused-into-out]
  i4*2+f8     fused    3.31 ms   unfused    4.89 ms   (1.48x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    4.35 ms   unfused    7.44 ms   (1.71x)
    [F      ] fused    4.34 ms   unfused    7.39 ms   (1.70x)
    [T      ] fused    4.16 ms   unfused    7.40 ms   (1.78x)
    [strided] fused    3.89 ms   unfused    5.56 ms   (1.43x)
    [bcast  ] fused    1.56 ms   unfused    4.35 ms   (2.80x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         13.48 ms
  (a-b)/(a+b)   19.61 ms
  sum(a*b)       8.90 ms
  sum(af*bf)     4.42 ms  [f32]
  a*b+c out=     5.13 ms  [two-pass with out=]
  i4*2+f8       10.10 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   12.93 ms
    [F      ]   12.90 ms
    [T      ]   13.02 ms
    [strided]    7.94 ms
    [bcast  ]   12.48 ms
```
