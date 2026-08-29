# Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-c3b80689652f6d539fa08543e6c9e4277204935d5b149c15f879e1ce51c9d9b5\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b) [ndarray]` | 1,000 | 0.0004 ms | 0.0006 ms | 0.0004 | 1.000x |
| `a.dot(b) [ndarray]` | 100,000 | 0.0122 ms | 0.0122 ms | 0.0068 | 0.557x |
| `a.dot(b) [ndarray]` | 10,000,000 | 7.5925 ms | 8.3942 ms | 8.3031 | 1.094x |
| `np.convolve(a, kernel)` | 1,000 | 0.0131 ms | 0.0424 ms | 0.0235 | 1.794x |
| `np.convolve(a, kernel)` | 100,000 | 0.3791 ms | 1.9148 ms | 2.2177 | 5.850x |
| `np.convolve(a, kernel)` | 10,000,000 | 46.4639 ms | 206.6472 ms | 199.3455 | 4.290x |
| `np.correlate(a, kernel)` | 1,000 | 0.0043 ms | 0.0332 ms | 0.0224 | 5.209x |
| `np.correlate(a, kernel)` | 100,000 | 0.3706 ms | 1.9150 ms | 2.1965 | 5.927x |
| `np.correlate(a, kernel)` | 10,000,000 | 46.2195 ms | 166.2216 ms | 206.0500 | 4.458x |
| `np.dot(a, b)` | 1,000 | 0.0004 ms | 0.0007 ms | 0.0005 | 1.250x |
| `np.dot(a, b)` | 100,000 | 0.0122 ms | 0.0122 ms | 0.0069 | 0.566x |
| `np.dot(a, b)` | 10,000,000 | 7.9269 ms | 4.4499 ms | 7.4223 | 1.668x |
| `np.einsum('ij,jk->ik', a, b)` | 1,000 | 0.0039 ms | 0.0021 ms | 0.0061 | 2.905x |
| `np.einsum('ij,jk->ik', a, b)` | 100,000 | 3.8778 ms | 2.2524 ms | 5.1769 | 2.298x |
| `np.einsum('ij,jk->ik', a, b)` | 10,000,000 | 6.4587 ms | 4.0000 ms | 9.6582 | 2.415x |
| `np.einsum(subscripts, operands)` | 1,000 | 0.0024 ms | 0.0011 ms | 0.0014 | 1.273x |
| `np.einsum(subscripts, operands)` | 100,000 | 0.0134 ms | 0.0134 ms | 0.0394 | 2.940x |
| `np.einsum(subscripts, operands)` | 10,000,000 | 8.4578 ms | 8.4570 ms | 9.2857 | 1.098x |
| `np.inner(a, b)` | 1,000 | 0.0004 ms | 0.0007 ms | 0.0006 | 1.500x |
| `np.inner(a, b)` | 100,000 | 0.0122 ms | 0.0122 ms | 0.0070 | 0.574x |
| `np.inner(a, b)` | 10,000,000 | 7.9907 ms | 7.6714 ms | 8.3426 | 1.087x |
| `np.linalg.cholesky(a)` | 1,000 | missing_backend | 0.0034 ms | 0.0043 | 1.265x |
| `np.linalg.cholesky(a)` | 100,000 | missing_backend | 0.0326 ms | 0.0174 | 0.534x |
| `np.linalg.cholesky(a)` | 10,000,000 | missing_backend | 0.0680 ms | 0.1038 | 1.526x |
| `np.linalg.cond(a)` | 1,000 | missing_backend | 0.0533 ms | 0.0349 | 0.655x |
| `np.linalg.cond(a)` | 100,000 | missing_backend | 0.4661 ms | 0.4757 | 1.021x |
| `np.linalg.cond(a)` | 10,000,000 | missing_backend | 0.9824 ms | 0.9891 | 1.007x |
| `np.linalg.det(a)` | 1,000 | 0.0043 ms | 0.0080 ms | 0.0126 | 2.930x |
| `np.linalg.det(a)` | 100,000 | 0.0728 ms | 0.0543 ms | 0.0587 | 1.081x |
| `np.linalg.det(a)` | 10,000,000 | 0.1875 ms | 0.1129 ms | 0.1187 | 1.051x |
| `np.linalg.eig(a)` | 1,000 | missing_backend | 0.1941 ms | 0.2033 | 1.047x |
| `np.linalg.eig(a)` | 100,000 | missing_backend | 2.6725 ms | 2.6986 | 1.010x |
| `np.linalg.eig(a)` | 10,000,000 | missing_backend | 5.8937 ms | 5.8331 | 0.990x |
| `np.linalg.eigh(a)` | 1,000 | missing_backend | 0.0847 ms | 0.0463 | 0.547x |
| `np.linalg.eigh(a)` | 100,000 | missing_backend | 0.8180 ms | 0.4004 | 0.489x |
| `np.linalg.eigh(a)` | 10,000,000 | missing_backend | 1.5346 ms | 1.5458 | 1.007x |
| `np.linalg.eigvals(a)` | 1,000 | missing_backend | 0.1109 ms | 0.1279 | 1.153x |
| `np.linalg.eigvals(a)` | 100,000 | missing_backend | 1.5514 ms | 1.5435 | 0.995x |
| `np.linalg.eigvals(a)` | 10,000,000 | missing_backend | 3.2028 ms | 3.0426 | 0.950x |
| `np.linalg.eigvalsh(a)` | 1,000 | missing_backend | 0.0294 ms | 0.0366 | 1.245x |
| `np.linalg.eigvalsh(a)` | 100,000 | missing_backend | 0.2839 ms | 0.2893 | 1.019x |
| `np.linalg.eigvalsh(a)` | 10,000,000 | missing_backend | 0.5183 ms | 0.5261 | 1.015x |
| `np.linalg.inv(a)` | 1,000 | 0.0125 ms | 0.0157 ms | 0.0239 | 1.912x |
| `np.linalg.inv(a)` | 100,000 | 0.2485 ms | 0.1604 ms | 0.1823 | 1.137x |
| `np.linalg.inv(a)` | 10,000,000 | 0.5972 ms | 0.3403 ms | 0.3679 | 1.081x |
| `np.linalg.lstsq(a, b)` | 1,000 | missing_backend | 0.1518 ms | 0.1637 | 1.078x |
| `np.linalg.lstsq(a, b)` | 100,000 | missing_backend | 1.6220 ms | 1.6285 | 1.004x |
| `np.linalg.lstsq(a, b)` | 10,000,000 | missing_backend | 2.9606 ms | 2.9627 | 1.001x |
| `np.linalg.matmul(a, b)` | 1,000 | 0.0036 ms | 0.0016 ms | 0.0023 | 1.437x |
| `np.linalg.matmul(a, b)` | 100,000 | 3.8721 ms | 2.2462 ms | 2.2518 | 1.002x |
| `np.linalg.matmul(a, b)` | 10,000,000 | 6.4165 ms | 3.9929 ms | 4.2537 | 1.065x |
| `np.linalg.matrix_norm(a, ord=2)` | 1,000 | missing_backend | 0.0532 ms | 0.0695 | 1.306x |
| `np.linalg.matrix_norm(a, ord=2)` | 100,000 | missing_backend | 0.4653 ms | 0.4811 | 1.034x |
| `np.linalg.matrix_norm(a, ord=2)` | 10,000,000 | missing_backend | 0.9814 ms | 0.9909 | 1.010x |
| `np.linalg.matrix_power(a, -1)` | 1,000 | 0.0133 ms | 0.0165 ms | 0.0257 | 1.932x |
| `np.linalg.matrix_power(a, -1)` | 100,000 | 0.2505 ms | 0.1635 ms | 0.1843 | 1.127x |
| `np.linalg.matrix_power(a, -1)` | 10,000,000 | 0.6002 ms | 0.3473 ms | 0.3715 | 1.070x |
| `np.linalg.matrix_power(a, 3)` | 1,000 | 0.0155 ms | 0.0078 ms | 0.0052 | 0.667x |
| `np.linalg.matrix_power(a, 3)` | 100,000 | 0.7173 ms | 0.3164 ms | 0.3235 | 1.022x |
| `np.linalg.matrix_power(a, 3)` | 10,000,000 | 0.7173 ms | 0.3163 ms | 0.3239 | 1.024x |
| `np.linalg.matrix_rank(a)` | 1,000 | missing_backend | 0.0559 ms | 0.0717 | 1.283x |
| `np.linalg.matrix_rank(a)` | 100,000 | missing_backend | 0.4716 ms | 0.4822 | 1.022x |
| `np.linalg.matrix_rank(a)` | 10,000,000 | missing_backend | 0.9822 ms | 0.9950 | 1.013x |
| `np.linalg.multi_dot(arrays)` | 1,000 | 0.0157 ms | 0.0078 ms | 0.0051 | 0.654x |
| `np.linalg.multi_dot(arrays)` | 100,000 | 0.7175 ms | 0.3163 ms | 0.3291 | 1.040x |
| `np.linalg.multi_dot(arrays)` | 10,000,000 | 0.7174 ms | 0.3166 ms | 0.3290 | 1.039x |
| `np.linalg.norm(a, ord=2)` | 1,000 | missing_backend | 0.0319 ms | 0.0696 | 2.182x |
| `np.linalg.norm(a, ord=2)` | 100,000 | missing_backend | 0.4662 ms | 0.4809 | 1.032x |
| `np.linalg.norm(a, ord=2)` | 10,000,000 | missing_backend | 0.9787 ms | 0.9930 | 1.015x |
| `np.linalg.pinv(a)` | 1,000 | missing_backend | 0.1601 ms | 0.0845 | 0.528x |
| `np.linalg.pinv(a)` | 100,000 | missing_backend | 1.5796 ms | 1.5793 | 1.000x |
| `np.linalg.pinv(a)` | 10,000,000 | missing_backend | 3.0850 ms | 3.2965 | 1.069x |
| `np.linalg.qr(a)` | 1,000 | missing_backend | 0.0333 ms | 0.0225 | 0.676x |
| `np.linalg.qr(a)` | 100,000 | missing_backend | 0.3141 ms | 0.3839 | 1.222x |
| `np.linalg.qr(a)` | 10,000,000 | missing_backend | 0.6888 ms | 0.7958 | 1.155x |
| `np.linalg.slogdet(a)` | 1,000 | 0.0046 ms | 0.0080 ms | 0.0144 | 3.130x |
| `np.linalg.slogdet(a)` | 100,000 | 0.0322 ms | 0.0548 ms | 0.0604 | 1.876x |
| `np.linalg.slogdet(a)` | 10,000,000 | 0.1889 ms | 0.1133 ms | 0.1195 | 1.055x |
| `np.linalg.solve(a, b)` | 1,000 | 0.0071 ms | 0.0083 ms | 0.0071 | 1.000x |
| `np.linalg.solve(a, b)` | 100,000 | 0.0464 ms | 0.0579 ms | 0.0640 | 1.379x |
| `np.linalg.solve(a, b)` | 10,000,000 | 0.2406 ms | 0.1187 ms | 0.1247 | 1.051x |
| `np.linalg.svd(a)` | 1,000 | missing_backend | 0.1357 ms | 0.1490 | 1.098x |
| `np.linalg.svd(a)` | 100,000 | missing_backend | 1.4284 ms | 1.4689 | 1.028x |
| `np.linalg.svd(a)` | 10,000,000 | missing_backend | 2.7944 ms | 2.9843 | 1.068x |
| `np.linalg.svdvals(a)` | 1,000 | missing_backend | 0.0474 ms | 0.0558 | 1.177x |
| `np.linalg.svdvals(a)` | 100,000 | missing_backend | 0.4632 ms | 0.4667 | 1.008x |
| `np.linalg.svdvals(a)` | 10,000,000 | missing_backend | 0.9755 ms | 0.9773 | 1.002x |
| `np.linalg.tensordot(a, b)` | 1,000 | 0.0041 ms | 0.0048 ms | 0.0056 | 1.366x |
| `np.linalg.tensordot(a, b)` | 100,000 | 0.3594 ms | 0.1591 ms | 0.1721 | 1.082x |
| `np.linalg.tensordot(a, b)` | 10,000,000 | 0.3621 ms | 0.1591 ms | 0.1721 | 1.082x |
| `np.linalg.tensorinv(a)` | 1,000 | 0.0128 ms | 0.0158 ms | 0.0259 | 2.023x |
| `np.linalg.tensorinv(a)` | 100,000 | 0.2490 ms | 0.1609 ms | 0.1845 | 1.147x |
| `np.linalg.tensorinv(a)` | 10,000,000 | 0.5967 ms | 0.3431 ms | 0.3720 | 1.084x |
| `np.linalg.tensorsolve(a, b)` | 1,000 | 0.0076 ms | 0.0089 ms | 0.0184 | 2.421x |
| `np.linalg.tensorsolve(a, b)` | 100,000 | 0.1032 ms | 0.0587 ms | 0.0667 | 1.136x |
| `np.linalg.tensorsolve(a, b)` | 10,000,000 | 0.2408 ms | 0.1198 ms | 0.1276 | 1.065x |
| `np.linalg.vecdot(a, b)` | 1,000 | 0.0018 ms | 0.0008 ms | 0.0008 | 1.000x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0341 ms | 0.0123 ms | 0.0073 | 0.593x |
| `np.linalg.vecdot(a, b)` | 10,000,000 | 29.2990 ms | 8.4696 ms | 8.4273 | 0.995x |
| `np.matmul(a, b)` | 1,000 | 0.0036 ms | 0.0016 ms | 0.0022 | 1.375x |
| `np.matmul(a, b)` | 100,000 | 3.8830 ms | 2.2470 ms | 2.2609 | 1.006x |
| `np.matmul(a, b)` | 10,000,000 | 6.5035 ms | 3.9891 ms | 4.2523 | 1.066x |
| `np.matvec(a, b)` | 1,000 | 0.0010 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.matvec(a, b)` | 100,000 | 0.0155 ms | 0.0098 ms | 0.0116 | 1.184x |
| `np.matvec(a, b)` | 10,000,000 | 0.0216 ms | 0.0142 ms | 0.0162 | 1.141x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.0660 ms | 0.0448 | 0.679x |
| `np.polyfit(x, y, 3)` | 100,000 | missing_backend | 1.1174 ms | 0.8973 | 0.803x |
| `np.polyfit(x, y, 3)` | 10,000,000 | missing_backend | 7.0263 ms | 8.5729 | 1.220x |
| `np.roots(p)` | 1,000 | missing_backend | 0.0736 ms | 0.0878 | 1.193x |
| `np.roots(p)` | 100,000 | missing_backend | 0.1891 ms | 0.2035 | 1.076x |
| `np.roots(p)` | 10,000,000 | missing_backend | 0.7742 ms | 0.7704 | 0.995x |
| `np.tensordot(a, b)` | 1,000 | 0.0013 ms | 0.0013 ms | 0.0036 | 2.769x |
| `np.tensordot(a, b)` | 100,000 | 0.0134 ms | 0.0134 ms | 0.0209 | 1.560x |
| `np.tensordot(a, b)` | 10,000,000 | 7.2145 ms | 7.7939 ms | 8.5836 | 1.190x |
| `np.tensordot(a, b, axes=1)` | 1,000 | 0.0041 ms | 0.0048 ms | 0.0054 | 1.317x |
| `np.tensordot(a, b, axes=1)` | 100,000 | 3.8851 ms | 2.2485 ms | 2.2798 | 1.014x |
| `np.tensordot(a, b, axes=1)` | 10,000,000 | 6.4310 ms | 3.9968 ms | 4.2931 | 1.074x |
| `np.vdot(a, b)` | 1,000 | 0.0010 ms | 0.0007 ms | 0.0005 | 0.714x |
| `np.vdot(a, b)` | 100,000 | 0.0125 ms | 0.0122 ms | 0.0070 | 0.574x |
| `np.vdot(a, b)` | 10,000,000 | 7.9176 ms | 8.2873 ms | 8.3329 | 1.052x |
| `np.vecdot(a, b)` | 1,000 | 0.0026 ms | 0.0008 ms | 0.0006 | 0.750x |
| `np.vecdot(a, b)` | 100,000 | 0.0341 ms | 0.0123 ms | 0.0070 | 0.569x |
| `np.vecdot(a, b)` | 10,000,000 | 29.1383 ms | 8.4108 ms | 8.2750 | 0.984x |
| `np.vecmat(a, b)` | 1,000 | 0.0009 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.vecmat(a, b)` | 100,000 | 0.0098 ms | 0.0117 ms | 0.0136 | 1.388x |
| `np.vecmat(a, b)` | 10,000,000 | 0.0272 ms | 0.0176 ms | 0.0194 | 1.102x |
