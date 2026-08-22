# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    4.25 ms   unfused    7.82 ms   (1.84x)
  (a-b)/(a+b) fused    3.67 ms   unfused   17.36 ms   (4.72x)
  sum(a*b)    fused    2.86 ms   unfused    5.15 ms   (1.80x)
  sum(af*bf)  fused    1.78 ms   unfused    2.64 ms   (1.49x)  [f32]
  a*b+c out=  fused    4.77 ms   [1-pass fused-into-out]
  i4*2+f8     fused    3.80 ms   unfused    5.69 ms   (1.50x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    5.30 ms   unfused    8.66 ms   (1.63x)
    [F      ] fused    5.17 ms   unfused    8.59 ms   (1.66x)
    [T      ] fused    4.72 ms   unfused    9.32 ms   (1.97x)
    [strided] fused    4.39 ms   unfused    7.12 ms   (1.62x)
    [bcast  ] fused    2.32 ms   unfused    5.99 ms   (2.58x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         20.43 ms
  (a-b)/(a+b)   34.27 ms
  sum(a*b)      13.21 ms
  sum(af*bf)     7.10 ms  [f32]
  a*b+c out=    10.52 ms  [two-pass with out=]
  i4*2+f8       14.97 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   22.68 ms
    [F      ]   23.92 ms
    [T      ]   17.55 ms
    [strided]   12.41 ms
    [bcast  ]   16.23 ms
```
