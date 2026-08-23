# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    3.65 ms   unfused    6.65 ms   (1.82x)
  (a-b)/(a+b) fused    3.02 ms   unfused   12.63 ms   (4.18x)
  sum(a*b)    fused    2.60 ms   unfused    4.61 ms   (1.77x)
  sum(af*bf)  fused    1.57 ms   unfused    1.90 ms   (1.21x)  [f32]
  a*b+c out=  fused    3.84 ms   [1-pass fused-into-out]
  i4*2+f8     fused    3.07 ms   unfused    4.54 ms   (1.48x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    4.25 ms   unfused    6.65 ms   (1.56x)
    [F      ] fused    3.89 ms   unfused    6.53 ms   (1.68x)
    [T      ] fused    3.81 ms   unfused    6.61 ms   (1.74x)
    [strided] fused    3.66 ms   unfused    5.00 ms   (1.36x)
    [bcast  ] fused    1.86 ms   unfused    4.10 ms   (2.20x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         12.91 ms
  (a-b)/(a+b)   19.38 ms
  sum(a*b)       8.45 ms
  sum(af*bf)     4.31 ms  [f32]
  a*b+c out=     5.28 ms  [two-pass with out=]
  i4*2+f8       10.05 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   12.80 ms
    [F      ]   13.45 ms
    [T      ]   14.27 ms
    [strided]    8.58 ms
    [bcast  ]   13.10 ms
```
