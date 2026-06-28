# Fusion — np.evaluate vs unfused chains (and NumPy context)

`np.evaluate` runs a whole expression tree in one NDIter pass (no intermediates). Fixed-expression gate plus an operand-layout sweep of the flagship `a*b+c` (C/F/T/strided/bcast — does the fused single-pass win survive non-contiguous operands?), not a dtype/layout matrix — so reported as-is.

```
NumSharp — fused np.evaluate vs unfused np.* chains (4M elements, best-of-9; (Nx) = unfused ÷ fused, >1 = fusion faster):

correctness cross-checks ok

4M float64, best of 9:
  a*b+c       fused    4.48 ms   unfused    6.97 ms   (1.56x)
  (a-b)/(a+b) fused    3.26 ms   unfused   13.54 ms   (4.16x)
  sum(a*b)    fused    2.44 ms   unfused    3.90 ms   (1.60x)
  sum(af*bf)  fused    1.30 ms   unfused    1.68 ms   (1.29x)  [f32]
  a*b+c out=  fused    3.77 ms   [1-pass fused-into-out]
  i4*2+f8     fused    2.93 ms   unfused    4.18 ms   (1.43x)

  a*b+c across operand layouts (2-D 2000x2000, all 3 operands same layout):
    [C      ] fused    3.68 ms   unfused    6.43 ms   (1.75x)
    [F      ] fused    3.60 ms   unfused    6.67 ms   (1.85x)
    [T      ] fused    3.67 ms   unfused    6.37 ms   (1.74x)
    [strided] fused    3.49 ms   unfused    4.75 ms   (1.36x)
    [bcast  ] fused    1.11 ms   unfused    3.99 ms   (3.60x)

NumPy — absolutes on the same box (context for the unfused column):

numpy 2.4.2, 4M float64, best of 9:
  a*b+c         12.93 ms
  (a-b)/(a+b)   19.64 ms
  sum(a*b)       8.45 ms
  sum(af*bf)     4.19 ms  [f32]
  a*b+c out=     4.96 ms  [two-pass with out=]
  i4*2+f8        9.99 ms
  a*b+c across operand layouts (2-D 2000x2000, unfused):
    [C      ]   12.87 ms
    [F      ]   12.76 ms
    [T      ]   12.84 ms
    [strided]    7.87 ms
    [bcast  ]   12.36 ms
```
