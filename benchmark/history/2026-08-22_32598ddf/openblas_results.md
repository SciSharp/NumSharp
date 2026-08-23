# Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-731fd42e1dd9378fe3a752f5d05d483a3188d66c472f3d268b9febfc8d1b8f5a\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b)` | 100,000 | 0.0097 ms | 0.0077 ms | 0.0087 | 1.130x |
| `np.convolve(a, v, mode='same')` | 100,000 | 0.1257 ms | 2.4239 ms | 1.0977 | 8.733x |
| `np.correlate(a, v, mode='same')` | 100,000 | 0.1179 ms | 2.4322 ms | 1.2291 | 10.425x |
| `np.dot(a, b)` | 100,000 | 0.0092 ms | 0.0076 ms | 0.0083 | 1.092x |
| `np.einsum('ij,jk->ik', a, b)` | 128 | 0.2206 ms | 0.0939 ms | 0.2651 | 2.823x |
| `np.inner(a, b)` | 100,000 | 0.0075 ms | 0.0078 ms | 0.0076 | 1.013x |
| `np.linalg.cholesky(a)` | 32 | missing_backend | 0.0166 ms | 0.0051 | 0.307x |
| `np.linalg.cholesky(a)` | 96 | missing_backend | 0.0414 ms | 0.0229 | 0.553x |
| `np.linalg.cond(a)` | 32 | missing_backend | 0.0454 ms | 0.0424 | 0.934x |
| `np.linalg.cond(a)` | 96 | missing_backend | 0.3186 ms | 0.3338 | 1.048x |
| `np.linalg.det(a)` | 32 | missing_backend | 0.0100 ms | 0.0070 | 0.700x |
| `np.linalg.det(a)` | 96 | missing_backend | 0.0322 ms | 0.0321 | 0.997x |
| `np.linalg.eig(a)` | 32 | missing_backend | 0.1395 ms | 0.1148 | 0.823x |
| `np.linalg.eig(a)` | 96 | missing_backend | 1.4870 ms | 1.5073 | 1.014x |
| `np.linalg.eigh(a)` | 32 | missing_backend | 0.0577 ms | 0.0535 | 0.927x |
| `np.linalg.eigh(a)` | 96 | missing_backend | 0.4273 ms | 0.4739 | 1.109x |
| `np.linalg.eigvals(a)` | 32 | missing_backend | 0.0800 ms | 0.0700 | 0.875x |
| `np.linalg.eigvals(a)` | 96 | missing_backend | 0.8608 ms | 0.9121 | 1.060x |
| `np.linalg.eigvalsh(a)` | 32 | missing_backend | 0.0245 ms | 0.0239 | 0.976x |
| `np.linalg.eigvalsh(a)` | 96 | missing_backend | 0.1664 ms | 0.1824 | 1.096x |
| `np.linalg.inv(a)` | 32 | missing_backend | 0.0209 ms | 0.0115 | 0.550x |
| `np.linalg.inv(a)` | 96 | missing_backend | 0.0881 ms | 0.0883 | 1.002x |
| `np.linalg.lstsq(a, b)` | 32 | missing_backend | 0.0946 ms | 0.0918 | 0.970x |
| `np.linalg.lstsq(a, b)` | 96 | missing_backend | 0.8382 ms | 0.8770 | 1.046x |
| `np.linalg.matmul(a, b)` | 128 | 0.2107 ms | 0.0915 ms | 0.0670 | 0.732x |
| `np.linalg.matrix_norm(a, ord=2)` | 32 | missing_backend | 0.0364 ms | 0.0467 | 1.283x |
| `np.linalg.matrix_norm(a, ord=2)` | 96 | missing_backend | 0.3064 ms | 0.3300 | 1.077x |
| `np.linalg.matrix_power(a, -1)` | 32 | missing_backend | 0.0164 ms | 0.0134 | 0.817x |
| `np.linalg.matrix_power(a, -1)` | 96 | missing_backend | 0.0919 ms | 0.0970 | 1.055x |
| `np.linalg.matrix_rank(a)` | 32 | missing_backend | 0.0446 ms | 0.0421 | 0.944x |
| `np.linalg.matrix_rank(a)` | 96 | missing_backend | 0.3003 ms | 0.3034 | 1.010x |
| `np.linalg.multi_dot(arrays)` | 128 | 0.3759 ms | 0.1970 ms | 0.1324 | 0.672x |
| `np.linalg.norm(a, ord=2)` | 32 | missing_backend | 0.0356 ms | 0.0468 | 1.315x |
| `np.linalg.norm(a, ord=2)` | 96 | missing_backend | 0.2922 ms | 0.3086 | 1.056x |
| `np.linalg.pinv(a)` | 32 | missing_backend | 0.1116 ms | 0.0982 | 0.880x |
| `np.linalg.pinv(a)` | 96 | missing_backend | 0.8636 ms | 0.8362 | 0.968x |
| `np.linalg.qr(a)` | 32 | missing_backend | 0.0360 ms | 0.0263 | 0.731x |
| `np.linalg.qr(a)` | 96 | missing_backend | 0.1876 ms | 0.1878 | 1.001x |
| `np.linalg.slogdet(a)` | 32 | missing_backend | 0.0088 ms | 0.0079 | 0.898x |
| `np.linalg.slogdet(a)` | 96 | missing_backend | 0.0319 ms | 0.0328 | 1.028x |
| `np.linalg.solve(a, b)` | 32 | missing_backend | 0.0089 ms | 0.0107 | 1.202x |
| `np.linalg.solve(a, b)` | 96 | missing_backend | 0.0320 ms | 0.0335 | 1.047x |
| `np.linalg.svd(a)` | 32 | missing_backend | 0.0857 ms | 0.0934 | 1.090x |
| `np.linalg.svd(a)` | 96 | missing_backend | 0.7232 ms | 0.8002 | 1.106x |
| `np.linalg.svdvals(a)` | 32 | missing_backend | 0.0329 ms | 0.0394 | 1.198x |
| `np.linalg.svdvals(a)` | 96 | missing_backend | 0.2928 ms | 0.3313 | 1.131x |
| `np.linalg.tensordot(a, b, axes=1)` | 128 | 0.2046 ms | 0.0895 ms | 0.0663 | 0.741x |
| `np.linalg.tensorinv(a)` | 32 | missing_backend | 0.0165 ms | 0.0147 | 0.891x |
| `np.linalg.tensorinv(a)` | 96 | missing_backend | 0.0869 ms | 0.0988 | 1.137x |
| `np.linalg.tensorsolve(a, b)` | 32 | missing_backend | 0.0094 ms | 0.0092 | 0.979x |
| `np.linalg.tensorsolve(a, b)` | 96 | missing_backend | 0.0326 ms | 0.0391 | 1.199x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0722 ms | 0.0145 ms | 0.0079 | 0.545x |
| `np.matmul(a, b)` | 128 | 0.2112 ms | 0.0933 ms | 0.0636 | 0.682x |
| `np.matvec(a, b)` | 128 | 0.0231 ms | 0.0022 ms | 0.0022 | 1.000x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.2248 ms | 0.0520 | 0.231x |
| `np.roots(p)` | 16 | missing_backend | 0.0606 ms | 0.0442 | 0.729x |
| `np.tensordot(a, b, axes=1)` | 128 | 0.2107 ms | 0.0943 ms | 0.0664 | 0.704x |
| `np.vdot(a, b)` | 100,000 | 0.0079 ms | 0.0083 ms | 0.0075 | 0.949x |
| `np.vecdot(a, b)` | 100,000 | 0.1582 ms | 0.0078 ms | 0.0076 | 0.974x |
| `np.vecmat(a, b)` | 128 | 0.0036 ms | 0.0022 ms | 0.0017 | 0.773x |
