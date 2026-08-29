# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best window (min over ~200 ms):
  a*b+c       fused    3.42 ms   unfused    5.68 ms   (1.66x)
  (a-b)/(a+b) fused    2.83 ms   unfused   11.66 ms   (4.12x)
  sum(a*b)    fused    2.17 ms   unfused    3.67 ms   (1.69x)
  sum(af*bf)  fused    1.22 ms   unfused    1.64 ms   (1.34x)  [f32]
  a*b+c out=  fused    3.33 ms   [1-pass fused-into-out]
  i4*2+f8     fused    2.73 ms   unfused    3.91 ms   (1.43x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    3.44 ms   unfused    5.90 ms   (1.72x)
    [F      ] fused   11.76 ms   unfused   17.82 ms   (1.52x)
    [T      ] fused   11.82 ms   unfused   17.38 ms   (1.47x)
    [strided] fused   11.84 ms   unfused   16.55 ms   (1.40x)
    [bcast  ] fused    2.23 ms   unfused    9.12 ms   (4.10x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best window (min over ~200 ms):
  a*b+c         12.68 ms
  (a-b)/(a+b)   19.13 ms
  sum(a*b)       8.42 ms
  sum(af*bf)     4.23 ms  [f32]
  a*b+c out=     5.08 ms  [two-pass with out=]
  i4*2+f8        9.93 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   12.83 ms
    [F      ]   12.50 ms
    [T      ]   12.59 ms
    [strided]    7.87 ms
    [bcast  ]   12.19 ms
```
