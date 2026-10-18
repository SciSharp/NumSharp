# numpy.polynomial series classes (U10 / U11's class half) — NumSharp vs NumPy 2.4.2

252 cells, every result SHA-256-identical to NumPy's (checked before timing). Host: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 3 alternating run pair(s), each side's best window per cell. NPY/NS = NumPy time / NumSharp time (higher = NumSharp faster).

**All cells: min 0.24x, geomean 8.22x.**

| section | what | cells | min NPY/NS | geomean NPY/NS |
|---|---|---:|---:|---:|
| K | construction and the class methods: the constructor (8 / 1,000 coefficients), identity, basis(5), fromroots(5 roots), cast (16 coefficients, from another class) | 36 | 3.22 | 5.19 |
| A | arithmetic: + and * of two series (8 / 64 coefficients, + also at 1,000), + / * / true division by a Python float, // and divmod (16 by 4), ** 3 (4 coefficients), unary -, == | 78 | 4.20 | 18.19 |
| E | evaluation: p(x) at 1,000 and 100,000 points and at a Python float (8 coefficients), p(q) (4 and 4) | 24 | 2.91 | 9.81 |
| C | calculus and conversion: deriv, integ, integ with k and lbnd (64 coefficients), convert(kind=...) (16), linspace(100) | 30 | 3.84 | 8.74 |
| M | members: copy (1,000), trim, cutdeg(8), truncate(8) (64), mapparms, degree | 36 | 3.78 | 5.24 |
| P | printers: str (ascii), repr, _repr_latex_, format 'unicode' (8 coefficients) | 24 | 12.50 | 14.76 |
| S | pickling: __getstate__, __setstate__ (8 coefficients) — the section's floor: __getstate__ copies three arrays as NumPy does, at NumSharp's NDArray allocation floor (0.44 us a copy); __setstate__ stores a view per array (0.18 us each), so a series never disposes the caller's state arrays, where NumPy's is one `self.__dict__ = state` assignment (0.19 us in all) | 12 | 0.24 | 0.41 |
| L | LAPACK at one OpenBLAS thread: roots (8 coefficients), fit (1,000 points, degree 8) | 12 | 1.81 | 2.70 |

## Every cell

| cell | NumSharp us | NumPy us | NPY/NS |
|---|---:|---:|---:|
| K/ctor8/Polynomial | 1.452 | 5.379 | ✅ 3.70 |
| K/ctor1000/Polynomial | 1.656 | 5.853 | ✅ 3.53 |
| K/identity/Polynomial | 2.755 | 9.371 | ✅ 3.40 |
| K/basis5/Polynomial | 2.483 | 8.926 | ✅ 3.59 |
| K/fromroots5/Polynomial | 10.746 | 111.884 | ✅ 10.41 |
| K/cast16/Polynomial | 98.686 | 870.214 | ✅ 8.82 |
| A/add8/Polynomial | 1.605 | 20.349 | ✅ 12.68 |
| A/mul8/Polynomial | 1.803 | 21.335 | ✅ 11.83 |
| A/add64/Polynomial | 1.635 | 20.431 | ✅ 12.50 |
| A/mul64/Polynomial | 4.577 | 23.295 | ✅ 5.09 |
| A/add1000/Polynomial | 1.871 | 22.129 | ✅ 11.83 |
| A/add_float/Polynomial | 2.514 | 14.869 | ✅ 5.91 |
| A/mul_float/Polynomial | 2.623 | 25.026 | ✅ 9.54 |
| A/truediv_float/Polynomial | 4.054 | 23.629 | ✅ 5.83 |
| A/floordiv16x4/Polynomial | 4.194 | 814.697 | ✅ 194.25 |
| A/divmod16x4/Polynomial | 4.230 | 822.424 | ✅ 194.43 |
| A/pow3/Polynomial | 1.828 | 133.339 | ✅ 72.94 |
| A/neg/Polynomial | 1.433 | 7.946 | ✅ 5.55 |
| A/eq/Polynomial | 0.983 | 9.956 | ✅ 10.13 |
| E/call1000/Polynomial | 2.547 | 36.743 | ✅ 14.43 |
| E/call100000/Polynomial | 98.725 | 4472.800 | ✅ 45.31 |
| E/call_float/Polynomial | 1.034 | 4.916 | ✅ 4.75 |
| E/compose4/Polynomial | 23.671 | 324.949 | ✅ 13.73 |
| C/deriv64/Polynomial | 2.592 | 33.149 | ✅ 12.79 |
| C/integ64/Polynomial | 4.670 | 61.531 | ✅ 13.18 |
| C/integ64_k_lbnd/Polynomial | 8.312 | 99.985 | ✅ 12.03 |
| C/convert16/Polynomial | 80.488 | 1062.888 | ✅ 13.21 |
| C/linspace100/Polynomial | 6.360 | 33.840 | ✅ 5.32 |
| M/copy1000/Polynomial | 1.654 | 7.628 | ✅ 4.61 |
| M/trim64/Polynomial | 1.662 | 13.476 | ✅ 8.11 |
| M/cutdeg64/Polynomial | 2.007 | 7.583 | ✅ 3.78 |
| M/truncate64/Polynomial | 1.789 | 7.435 | ✅ 4.16 |
| M/mapparms/Polynomial | 0.276 | 1.196 | ✅ 4.33 |
| M/degree/Polynomial | 0.033 | 0.243 | ✅ 7.36 |
| P/str/Polynomial | 4.559 | 60.687 | ✅ 13.31 |
| P/repr/Polynomial | 7.633 | 159.631 | ✅ 20.91 |
| P/latex/Polynomial | 4.261 | 59.011 | ✅ 13.85 |
| P/format_unicode/Polynomial | 4.590 | 63.075 | ✅ 13.74 |
| S/getstate/Polynomial | 1.991 | 1.393 | 🟡 0.70 |
| S/setstate/Polynomial | 0.789 | 0.186 | 🟠 0.24 |
| L/roots8/Polynomial | 11.522 | 45.995 | ✅ 3.99 |
| L/fit1000x8/Polynomial | 88.371 | 159.815 | ✅ 1.81 |
| K/ctor8/Chebyshev | 1.450 | 5.376 | ✅ 3.71 |
| K/ctor1000/Chebyshev | 1.696 | 5.850 | ✅ 3.45 |
| K/identity/Chebyshev | 2.861 | 9.290 | ✅ 3.25 |
| K/basis5/Chebyshev | 2.263 | 9.006 | ✅ 3.98 |
| K/fromroots5/Chebyshev | 10.581 | 111.165 | ✅ 10.51 |
| K/cast16/Chebyshev | 82.071 | 890.279 | ✅ 10.85 |
| A/add8/Chebyshev | 1.606 | 20.761 | ✅ 12.93 |
| A/mul8/Chebyshev | 2.517 | 29.588 | ✅ 11.76 |
| A/add64/Chebyshev | 1.659 | 20.753 | ✅ 12.51 |
| A/mul64/Chebyshev | 8.303 | 34.873 | ✅ 4.20 |
| A/add1000/Chebyshev | 1.883 | 22.111 | ✅ 11.74 |
| A/add_float/Chebyshev | 2.516 | 14.558 | ✅ 5.79 |
| A/mul_float/Chebyshev | 2.853 | 24.626 | ✅ 8.63 |
| A/truediv_float/Chebyshev | 4.030 | 23.634 | ✅ 5.86 |
| A/floordiv16x4/Chebyshev | 4.096 | 811.397 | ✅ 198.09 |
| A/divmod16x4/Chebyshev | 4.171 | 819.583 | ✅ 196.50 |
| A/pow3/Chebyshev | 2.029 | 132.996 | ✅ 65.55 |
| A/neg/Chebyshev | 1.456 | 7.854 | ✅ 5.39 |
| A/eq/Chebyshev | 1.006 | 10.037 | ✅ 9.98 |
| E/call1000/Chebyshev | 2.744 | 36.926 | ✅ 13.46 |
| E/call100000/Chebyshev | 113.450 | 3612.325 | ✅ 31.84 |
| E/call_float/Chebyshev | 1.146 | 4.921 | ✅ 4.29 |
| E/compose4/Chebyshev | 24.845 | 332.485 | ✅ 13.38 |
| C/deriv64/Chebyshev | 2.730 | 32.889 | ✅ 12.05 |
| C/integ64/Chebyshev | 4.674 | 62.401 | ✅ 13.35 |
| C/integ64_k_lbnd/Chebyshev | 10.624 | 100.000 | ✅ 9.41 |
| C/convert16/Chebyshev | 97.727 | 1083.809 | ✅ 11.09 |
| C/linspace100/Chebyshev | 6.453 | 33.976 | ✅ 5.27 |
| M/copy1000/Chebyshev | 1.658 | 7.315 | ✅ 4.41 |
| M/trim64/Chebyshev | 1.634 | 13.434 | ✅ 8.22 |
| M/cutdeg64/Chebyshev | 1.966 | 7.522 | ✅ 3.83 |
| M/truncate64/Chebyshev | 1.805 | 7.468 | ✅ 4.14 |
| M/mapparms/Chebyshev | 0.269 | 1.191 | ✅ 4.43 |
| M/degree/Chebyshev | 0.031 | 0.241 | ✅ 7.77 |
| P/str/Chebyshev | 4.782 | 62.490 | ✅ 13.07 |
| P/repr/Chebyshev | 7.668 | 159.339 | ✅ 20.78 |
| P/latex/Chebyshev | 4.464 | 58.995 | ✅ 13.22 |
| P/format_unicode/Chebyshev | 4.874 | 62.570 | ✅ 12.84 |
| S/getstate/Chebyshev | 1.972 | 1.374 | 🟡 0.70 |
| S/setstate/Chebyshev | 0.782 | 0.185 | 🟠 0.24 |
| L/roots8/Chebyshev | 11.802 | 44.922 | ✅ 3.81 |
| L/fit1000x8/Chebyshev | 81.740 | 159.705 | ✅ 1.95 |
| K/ctor8/Legendre | 1.425 | 5.221 | ✅ 3.66 |
| K/ctor1000/Legendre | 1.636 | 5.914 | ✅ 3.61 |
| K/identity/Legendre | 2.701 | 8.995 | ✅ 3.33 |
| K/basis5/Legendre | 2.209 | 8.738 | ✅ 3.96 |
| K/fromroots5/Legendre | 10.374 | 111.956 | ✅ 10.79 |
| K/cast16/Legendre | 79.957 | 889.332 | ✅ 11.12 |
| A/add8/Legendre | 1.623 | 20.295 | ✅ 12.50 |
| A/mul8/Legendre | 3.350 | 208.996 | ✅ 62.39 |
| A/add64/Legendre | 1.646 | 20.224 | ✅ 12.29 |
| A/mul64/Legendre | 45.067 | 5571.500 | ✅ 123.63 |
| A/add1000/Legendre | 1.857 | 22.004 | ✅ 11.85 |
| A/add_float/Legendre | 2.482 | 14.610 | ✅ 5.89 |
| A/mul_float/Legendre | 2.782 | 24.528 | ✅ 8.82 |
| A/truediv_float/Legendre | 4.058 | 23.361 | ✅ 5.76 |
| A/floordiv16x4/Legendre | 11.524 | 788.860 | ✅ 68.45 |
| A/divmod16x4/Legendre | 11.410 | 794.614 | ✅ 69.64 |
| A/pow3/Legendre | 3.029 | 128.351 | ✅ 42.37 |
| A/neg/Legendre | 1.416 | 7.811 | ✅ 5.52 |
| A/eq/Legendre | 0.988 | 9.924 | ✅ 10.04 |
| E/call1000/Legendre | 3.178 | 36.924 | ✅ 11.62 |
| E/call100000/Legendre | 159.833 | 3258.750 | ✅ 20.39 |
| E/call_float/Legendre | 1.321 | 4.878 | ✅ 3.69 |
| E/compose4/Legendre | 30.373 | 321.175 | ✅ 10.57 |
| C/deriv64/Legendre | 2.597 | 33.386 | ✅ 12.86 |
| C/integ64/Legendre | 4.658 | 61.820 | ✅ 13.27 |
| C/integ64_k_lbnd/Legendre | 14.557 | 100.051 | ✅ 6.87 |
| C/convert16/Legendre | 164.713 | 1081.720 | ✅ 6.57 |
| C/linspace100/Legendre | 6.463 | 33.593 | ✅ 5.20 |
| M/copy1000/Legendre | 1.642 | 7.539 | ✅ 4.59 |
| M/trim64/Legendre | 1.641 | 13.427 | ✅ 8.18 |
| M/cutdeg64/Legendre | 1.982 | 7.617 | ✅ 3.84 |
| M/truncate64/Legendre | 1.767 | 7.480 | ✅ 4.23 |
| M/mapparms/Legendre | 0.269 | 1.200 | ✅ 4.46 |
| M/degree/Legendre | 0.031 | 0.243 | ✅ 7.84 |
| P/str/Legendre | 4.754 | 61.016 | ✅ 12.83 |
| P/repr/Legendre | 7.717 | 160.518 | ✅ 20.80 |
| P/latex/Legendre | 4.598 | 59.457 | ✅ 12.93 |
| P/format_unicode/Legendre | 4.843 | 62.899 | ✅ 12.99 |
| S/getstate/Legendre | 1.958 | 1.366 | 🟡 0.70 |
| S/setstate/Legendre | 0.771 | 0.186 | 🟠 0.24 |
| L/roots8/Legendre | 13.271 | 46.027 | ✅ 3.47 |
| L/fit1000x8/Legendre | 84.834 | 159.592 | ✅ 1.88 |
| K/ctor8/Laguerre | 1.444 | 5.431 | ✅ 3.76 |
| K/ctor1000/Laguerre | 1.634 | 5.887 | ✅ 3.60 |
| K/identity/Laguerre | 2.896 | 9.321 | ✅ 3.22 |
| K/basis5/Laguerre | 2.231 | 8.962 | ✅ 4.02 |
| K/fromroots5/Laguerre | 11.165 | 112.226 | ✅ 10.05 |
| K/cast16/Laguerre | 81.839 | 893.268 | ✅ 10.91 |
| A/add8/Laguerre | 1.619 | 20.339 | ✅ 12.56 |
| A/mul8/Laguerre | 3.808 | 265.815 | ✅ 69.80 |
| A/add64/Laguerre | 1.672 | 20.292 | ✅ 12.14 |
| A/mul64/Laguerre | 53.350 | 7350.100 | ✅ 137.77 |
| A/add1000/Laguerre | 1.836 | 22.175 | ✅ 12.08 |
| A/add_float/Laguerre | 2.520 | 14.384 | ✅ 5.71 |
| A/mul_float/Laguerre | 2.828 | 24.419 | ✅ 8.63 |
| A/truediv_float/Laguerre | 4.093 | 23.733 | ✅ 5.80 |
| A/floordiv16x4/Laguerre | 13.481 | 796.769 | ✅ 59.10 |
| A/divmod16x4/Laguerre | 13.506 | 798.831 | ✅ 59.15 |
| A/pow3/Laguerre | 3.345 | 129.406 | ✅ 38.69 |
| A/neg/Laguerre | 1.422 | 7.790 | ✅ 5.48 |
| A/eq/Laguerre | 0.928 | 10.023 | ✅ 10.80 |
| E/call1000/Laguerre | 10.320 | 37.062 | ✅ 3.59 |
| E/call100000/Laguerre | 868.767 | 3257.200 | ✅ 3.75 |
| E/call_float/Laguerre | 1.673 | 4.869 | ✅ 2.91 |
| E/compose4/Laguerre | 45.016 | 324.225 | ✅ 7.20 |
| C/deriv64/Laguerre | 2.596 | 33.403 | ✅ 12.87 |
| C/integ64/Laguerre | 7.040 | 61.795 | ✅ 8.78 |
| C/integ64_k_lbnd/Laguerre | 21.240 | 100.613 | ✅ 4.74 |
| C/convert16/Laguerre | 278.520 | 1070.720 | ✅ 3.84 |
| C/linspace100/Laguerre | 7.099 | 33.860 | ✅ 4.77 |
| M/copy1000/Laguerre | 1.645 | 7.633 | ✅ 4.64 |
| M/trim64/Laguerre | 1.623 | 13.471 | ✅ 8.30 |
| M/cutdeg64/Laguerre | 1.983 | 7.601 | ✅ 3.83 |
| M/truncate64/Laguerre | 1.784 | 7.501 | ✅ 4.20 |
| M/mapparms/Laguerre | 0.270 | 1.203 | ✅ 4.46 |
| M/degree/Laguerre | 0.031 | 0.241 | ✅ 7.77 |
| P/str/Laguerre | 4.745 | 60.918 | ✅ 12.84 |
| P/repr/Laguerre | 7.658 | 159.183 | ✅ 20.79 |
| P/latex/Laguerre | 4.470 | 59.621 | ✅ 13.34 |
| P/format_unicode/Laguerre | 5.010 | 62.632 | ✅ 12.50 |
| S/getstate/Laguerre | 1.951 | 1.374 | 🟡 0.70 |
| S/setstate/Laguerre | 0.772 | 0.186 | 🟠 0.24 |
| L/roots8/Laguerre | 12.176 | 44.785 | ✅ 3.68 |
| L/fit1000x8/Laguerre | 79.314 | 158.178 | ✅ 1.99 |
| K/ctor8/Hermite | 1.425 | 5.396 | ✅ 3.79 |
| K/ctor1000/Hermite | 1.620 | 5.970 | ✅ 3.69 |
| K/identity/Hermite | 2.787 | 9.178 | ✅ 3.29 |
| K/basis5/Hermite | 2.217 | 8.995 | ✅ 4.06 |
| K/fromroots5/Hermite | 10.374 | 111.892 | ✅ 10.79 |
| K/cast16/Hermite | 80.859 | 895.709 | ✅ 11.08 |
| A/add8/Hermite | 1.625 | 20.124 | ✅ 12.38 |
| A/mul8/Hermite | 3.179 | 185.245 | ✅ 58.27 |
| A/add64/Hermite | 1.643 | 20.525 | ✅ 12.49 |
| A/mul64/Hermite | 31.400 | 4468.425 | ✅ 142.31 |
| A/add1000/Hermite | 1.853 | 22.166 | ✅ 11.96 |
| A/add_float/Hermite | 2.491 | 14.755 | ✅ 5.92 |
| A/mul_float/Hermite | 2.794 | 24.922 | ✅ 8.92 |
| A/truediv_float/Hermite | 4.040 | 23.739 | ✅ 5.88 |
| A/floordiv16x4/Hermite | 10.865 | 801.600 | ✅ 73.78 |
| A/divmod16x4/Hermite | 10.959 | 798.762 | ✅ 72.89 |
| A/pow3/Hermite | 2.911 | 129.415 | ✅ 44.46 |
| A/neg/Hermite | 1.432 | 7.826 | ✅ 5.47 |
| A/eq/Hermite | 0.927 | 9.976 | ✅ 10.76 |
| E/call1000/Hermite | 2.902 | 37.153 | ✅ 12.80 |
| E/call100000/Hermite | 134.725 | 3249.767 | ✅ 24.12 |
| E/call_float/Hermite | 1.271 | 4.830 | ✅ 3.80 |
| E/compose4/Hermite | 27.678 | 322.750 | ✅ 11.66 |
| C/deriv64/Hermite | 2.580 | 33.187 | ✅ 12.86 |
| C/integ64/Hermite | 4.575 | 62.344 | ✅ 13.63 |
| C/integ64_k_lbnd/Hermite | 13.236 | 100.199 | ✅ 7.57 |
| C/convert16/Hermite | 132.528 | 1090.233 | ✅ 8.23 |
| C/linspace100/Hermite | 6.334 | 33.914 | ✅ 5.35 |
| M/copy1000/Hermite | 1.637 | 7.617 | ✅ 4.65 |
| M/trim64/Hermite | 1.619 | 13.505 | ✅ 8.34 |
| M/cutdeg64/Hermite | 1.947 | 7.560 | ✅ 3.88 |
| M/truncate64/Hermite | 1.770 | 7.488 | ✅ 4.23 |
| M/mapparms/Hermite | 0.266 | 1.194 | ✅ 4.49 |
| M/degree/Hermite | 0.032 | 0.242 | ✅ 7.56 |
| P/str/Hermite | 4.777 | 61.575 | ✅ 12.89 |
| P/repr/Hermite | 7.446 | 162.740 | ✅ 21.86 |
| P/latex/Hermite | 4.557 | 59.628 | ✅ 13.08 |
| P/format_unicode/Hermite | 4.841 | 63.728 | ✅ 13.16 |
| S/getstate/Hermite | 2.004 | 1.383 | 🟡 0.69 |
| S/setstate/Hermite | 0.772 | 0.186 | 🟠 0.24 |
| L/roots8/Hermite | 12.603 | 45.306 | ✅ 3.59 |
| L/fit1000x8/Hermite | 76.941 | 159.899 | ✅ 2.08 |
| K/ctor8/HermiteE | 1.440 | 5.388 | ✅ 3.74 |
| K/ctor1000/HermiteE | 1.616 | 5.888 | ✅ 3.64 |
| K/identity/HermiteE | 2.723 | 9.371 | ✅ 3.44 |
| K/basis5/HermiteE | 2.228 | 8.989 | ✅ 4.03 |
| K/fromroots5/HermiteE | 10.338 | 112.164 | ✅ 10.85 |
| K/cast16/HermiteE | 80.426 | 892.005 | ✅ 11.09 |
| A/add8/HermiteE | 1.628 | 19.632 | ✅ 12.06 |
| A/mul8/HermiteE | 3.026 | 169.457 | ✅ 56.00 |
| A/add64/HermiteE | 1.639 | 20.396 | ✅ 12.44 |
| A/mul64/HermiteE | 28.700 | 3925.360 | ✅ 136.77 |
| A/add1000/HermiteE | 1.860 | 22.253 | ✅ 11.96 |
| A/add_float/HermiteE | 2.495 | 14.822 | ✅ 5.94 |
| A/mul_float/HermiteE | 2.775 | 24.400 | ✅ 8.79 |
| A/truediv_float/HermiteE | 4.068 | 23.323 | ✅ 5.73 |
| A/floordiv16x4/HermiteE | 10.312 | 794.367 | ✅ 77.03 |
| A/divmod16x4/HermiteE | 10.232 | 798.084 | ✅ 78.00 |
| A/pow3/HermiteE | 2.787 | 129.947 | ✅ 46.63 |
| A/neg/HermiteE | 1.445 | 7.859 | ✅ 5.44 |
| A/eq/HermiteE | 0.945 | 9.990 | ✅ 10.57 |
| E/call1000/HermiteE | 2.871 | 37.761 | ✅ 13.15 |
| E/call100000/HermiteE | 129.750 | 3229.650 | ✅ 24.89 |
| E/call_float/HermiteE | 1.239 | 4.862 | ✅ 3.92 |
| E/compose4/HermiteE | 24.634 | 323.492 | ✅ 13.13 |
| C/deriv64/HermiteE | 2.606 | 33.333 | ✅ 12.79 |
| C/integ64/HermiteE | 4.539 | 61.683 | ✅ 13.59 |
| C/integ64_k_lbnd/HermiteE | 13.269 | 100.243 | ✅ 7.55 |
| C/convert16/HermiteE | 129.511 | 1120.006 | ✅ 8.65 |
| C/linspace100/HermiteE | 6.387 | 34.424 | ✅ 5.39 |
| M/copy1000/HermiteE | 1.651 | 7.412 | ✅ 4.49 |
| M/trim64/HermiteE | 1.620 | 13.549 | ✅ 8.36 |
| M/cutdeg64/HermiteE | 1.959 | 7.578 | ✅ 3.87 |
| M/truncate64/HermiteE | 1.769 | 7.510 | ✅ 4.25 |
| M/mapparms/HermiteE | 0.269 | 1.210 | ✅ 4.50 |
| M/degree/HermiteE | 0.031 | 0.240 | ✅ 7.74 |
| P/str/HermiteE | 4.812 | 61.150 | ✅ 12.71 |
| P/repr/HermiteE | 7.443 | 162.198 | ✅ 21.79 |
| P/latex/HermiteE | 4.553 | 60.431 | ✅ 13.27 |
| P/format_unicode/HermiteE | 4.874 | 64.119 | ✅ 13.16 |
| S/getstate/HermiteE | 1.946 | 1.383 | 🟡 0.71 |
| S/setstate/HermiteE | 0.772 | 0.184 | 🟠 0.24 |
| L/roots8/HermiteE | 12.523 | 45.847 | ✅ 3.66 |
| L/fit1000x8/HermiteE | 75.392 | 159.263 | ✅ 2.11 |

## Cells below 1.5x

- S/setstate/Polynomial: 0.24x (NumSharp 0.789 us, NumPy 0.186 us)
- S/setstate/Chebyshev: 0.24x (NumSharp 0.782 us, NumPy 0.185 us)
- S/setstate/HermiteE: 0.24x (NumSharp 0.772 us, NumPy 0.184 us)
- S/setstate/Hermite: 0.24x (NumSharp 0.772 us, NumPy 0.186 us)
- S/setstate/Laguerre: 0.24x (NumSharp 0.772 us, NumPy 0.186 us)
- S/setstate/Legendre: 0.24x (NumSharp 0.771 us, NumPy 0.186 us)
- S/getstate/Hermite: 0.69x (NumSharp 2.004 us, NumPy 1.383 us)
- S/getstate/Chebyshev: 0.70x (NumSharp 1.972 us, NumPy 1.374 us)
- S/getstate/Legendre: 0.70x (NumSharp 1.958 us, NumPy 1.366 us)
- S/getstate/Polynomial: 0.70x (NumSharp 1.991 us, NumPy 1.393 us)
- S/getstate/Laguerre: 0.70x (NumSharp 1.951 us, NumPy 1.374 us)
- S/getstate/HermiteE: 0.71x (NumSharp 1.946 us, NumPy 1.383 us)
