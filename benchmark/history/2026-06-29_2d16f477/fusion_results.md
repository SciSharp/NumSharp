# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    3.53 ms   unfused    6.08 ms   (1.72x)
  (a-b)/(a+b) fused    3.16 ms   unfused   12.32 ms   (3.90x)
  sum(a*b)    fused    2.30 ms   unfused    4.33 ms   (1.88x)
  sum(af*bf)  fused    1.43 ms   unfused    1.57 ms   (1.10x)  [f32]
  a*b+c out=  fused    3.56 ms   [1-pass fused-into-out]
  i4*2+f8     fused    2.78 ms   unfused    4.04 ms   (1.45x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    3.43 ms   unfused    6.05 ms   (1.77x)
    [F      ] fused    3.52 ms   unfused    6.18 ms   (1.75x)
    [T      ] fused    3.50 ms   unfused    6.27 ms   (1.79x)
    [strided] fused    3.31 ms   unfused    4.72 ms   (1.43x)
    [bcast  ] fused    1.06 ms   unfused    3.75 ms   (3.53x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         12.35 ms
  (a-b)/(a+b)   18.75 ms
  sum(a*b)       8.60 ms
  sum(af*bf)     4.14 ms  [f32]
  a*b+c out=     4.68 ms  [two-pass with out=]
  i4*2+f8        9.67 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   12.52 ms
    [F      ]   12.54 ms
    [T      ]   12.51 ms
    [strided]    7.91 ms
    [bcast  ]   11.95 ms
```
