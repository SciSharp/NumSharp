# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    3.56 ms   unfused    6.18 ms   (1.73x)
  (a-b)/(a+b) fused    2.88 ms   unfused   13.76 ms   (4.78x)
  sum(a*b)    fused    2.31 ms   unfused    3.90 ms   (1.69x)
  sum(af*bf)  fused    1.34 ms   unfused    1.69 ms   (1.26x)  [f32]
  a*b+c out=  fused    3.54 ms   [1-pass fused-into-out]
  i4*2+f8     fused    2.74 ms   unfused    3.91 ms   (1.43x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    3.40 ms   unfused    6.12 ms   (1.80x)
    [F      ] fused    3.52 ms   unfused    6.09 ms   (1.73x)
    [T      ] fused    3.44 ms   unfused    6.12 ms   (1.78x)
    [strided] fused    3.35 ms   unfused    4.69 ms   (1.40x)
    [bcast  ] fused    1.05 ms   unfused    3.44 ms   (3.27x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         13.01 ms
  (a-b)/(a+b)   18.89 ms
  sum(a*b)       8.96 ms
  sum(af*bf)     4.29 ms  [f32]
  a*b+c out=     4.90 ms  [two-pass with out=]
  i4*2+f8        9.78 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   13.33 ms
    [F      ]   13.04 ms
    [T      ]   13.53 ms
    [strided]    8.48 ms
    [bcast  ]   12.56 ms
```
