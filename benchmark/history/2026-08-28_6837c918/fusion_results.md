# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best window (min over ~200 ms):
  a*b+c       fused    3.38 ms   unfused    5.76 ms   (1.70x)
  (a-b)/(a+b) fused    2.83 ms   unfused   12.00 ms   (4.24x)
  sum(a*b)    fused    2.23 ms   unfused    3.64 ms   (1.63x)
  sum(af*bf)  fused    1.20 ms   unfused    1.45 ms   (1.21x)  [f32]
  a*b+c out=  fused    3.26 ms   [1-pass fused-into-out]
  i4*2+f8     fused    2.68 ms   unfused    3.79 ms   (1.41x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    3.38 ms   unfused    5.75 ms   (1.70x)
    [F      ] fused   11.74 ms   unfused   17.66 ms   (1.50x)
    [T      ] fused   11.71 ms   unfused   17.48 ms   (1.49x)
    [strided] fused   11.64 ms   unfused    4.76 ms   (0.41x)
    [bcast  ] fused    2.02 ms   unfused    7.72 ms   (3.81x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best window (min over ~200 ms):
  a*b+c         12.98 ms
  (a-b)/(a+b)   19.67 ms
  sum(a*b)       8.47 ms
  sum(af*bf)     4.15 ms  [f32]
  a*b+c out=     4.43 ms  [two-pass with out=]
  i4*2+f8       10.35 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   13.02 ms
    [F      ]   13.07 ms
    [T      ]   12.60 ms
    [strided]    7.73 ms
    [bcast  ]   12.14 ms
```
