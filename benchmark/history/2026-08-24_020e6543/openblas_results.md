# Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-7ec7c6ad0c58f826c3d04fc8c1f325edd5ddd512868ff2c0d77027c2d96ba979\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b)` | 100,000 | 0.0071 ms | 0.0093 ms | 0.0079 | 1.113x |
| `np.convolve(a, v, mode='same')` | 100,000 | 0.1170 ms | 0.9291 ms | 1.1340 | 9.692x |
| `np.correlate(a, v, mode='same')` | 100,000 | 0.1155 ms | 0.9218 ms | 1.2404 | 10.739x |
| `np.dot(a, b)` | 100,000 | 0.0072 ms | 0.0074 ms | 0.0076 | 1.056x |
| `np.einsum('ij,jk->ik', a, b)` | 128 | 0.2096 ms | 0.0619 ms | 0.2650 | 4.281x |
| `np.inner(a, b)` | 100,000 | 0.0070 ms | 0.0075 ms | 0.0081 | 1.157x |
| `np.linalg.cholesky(a)` | 32 | missing_backend | 0.0026 ms | 0.0054 | 2.077x |
| `np.linalg.cholesky(a)` | 96 | missing_backend | 0.0356 ms | 0.0297 | 0.834x |
| `np.linalg.cond(a)` | 32 | missing_backend | 0.0378 ms | 0.0401 | 1.061x |
| `np.linalg.cond(a)` | 96 | missing_backend | 0.5259 ms | 0.3198 | 0.608x |
| `np.linalg.det(a)` | 32 | missing_backend | 0.0052 ms | 0.0066 | 1.269x |
| `np.linalg.det(a)` | 96 | missing_backend | 0.0600 ms | 0.0300 | 0.500x |
| `np.linalg.eig(a)` | 32 | missing_backend | 0.1160 ms | 0.1194 | 1.029x |
| `np.linalg.eig(a)` | 96 | missing_backend | 2.7234 ms | 1.5348 | 0.564x |
| `np.linalg.eigh(a)` | 32 | missing_backend | 0.0468 ms | 0.0641 | 1.370x |
| `np.linalg.eigh(a)` | 96 | missing_backend | 0.8513 ms | 0.4429 | 0.520x |
| `np.linalg.eigvals(a)` | 32 | missing_backend | 0.0675 ms | 0.0746 | 1.105x |
| `np.linalg.eigvals(a)` | 96 | missing_backend | 1.5264 ms | 0.8668 | 0.568x |
| `np.linalg.eigvalsh(a)` | 32 | missing_backend | 0.0209 ms | 0.0252 | 1.206x |
| `np.linalg.eigvalsh(a)` | 96 | missing_backend | 0.2885 ms | 0.2323 | 0.805x |
| `np.linalg.inv(a)` | 32 | missing_backend | 0.0094 ms | 0.0126 | 1.340x |
| `np.linalg.inv(a)` | 96 | missing_backend | 0.1642 ms | 0.1231 | 0.750x |
| `np.linalg.lstsq(a, b)` | 32 | missing_backend | 0.0834 ms | 0.1046 | 1.254x |
| `np.linalg.lstsq(a, b)` | 96 | missing_backend | 1.6352 ms | 1.1186 | 0.684x |
| `np.linalg.matmul(a, b)` | 128 | 0.2078 ms | 0.0635 ms | 0.0675 | 1.063x |
| `np.linalg.matrix_norm(a, ord=2)` | 32 | missing_backend | 0.0581 ms | 0.0442 | 0.761x |
| `np.linalg.matrix_norm(a, ord=2)` | 96 | missing_backend | 0.4981 ms | 0.3155 | 0.633x |
| `np.linalg.matrix_power(a, -1)` | 32 | missing_backend | 0.0214 ms | 0.0132 | 0.617x |
| `np.linalg.matrix_power(a, -1)` | 96 | missing_backend | 0.1657 ms | 0.0885 | 0.534x |
| `np.linalg.matrix_rank(a)` | 32 | missing_backend | 0.0372 ms | 0.0417 | 1.121x |
| `np.linalg.matrix_rank(a)` | 96 | missing_backend | 0.5264 ms | 0.4112 | 0.781x |
| `np.linalg.multi_dot(arrays)` | 128 | 0.4100 ms | 0.1199 ms | 0.1342 | 1.119x |
| `np.linalg.norm(a, ord=2)` | 32 | missing_backend | 0.0645 ms | 0.0471 | 0.730x |
| `np.linalg.norm(a, ord=2)` | 96 | missing_backend | 0.4958 ms | 0.3085 | 0.622x |
| `np.linalg.pinv(a)` | 32 | missing_backend | 0.0967 ms | 0.1006 | 1.040x |
| `np.linalg.pinv(a)` | 96 | missing_backend | 1.6728 ms | 1.0330 | 0.618x |
| `np.linalg.qr(a)` | 32 | missing_backend | 0.0212 ms | 0.0297 | 1.401x |
| `np.linalg.qr(a)` | 96 | missing_backend | 0.3196 ms | 0.1882 | 0.589x |
| `np.linalg.slogdet(a)` | 32 | missing_backend | 0.0058 ms | 0.0077 | 1.328x |
| `np.linalg.slogdet(a)` | 96 | missing_backend | 0.0585 ms | 0.0315 | 0.538x |
| `np.linalg.solve(a, b)` | 32 | missing_backend | 0.0053 ms | 0.0087 | 1.641x |
| `np.linalg.solve(a, b)` | 96 | missing_backend | 0.0609 ms | 0.0331 | 0.544x |
| `np.linalg.svd(a)` | 32 | missing_backend | 0.0747 ms | 0.0872 | 1.167x |
| `np.linalg.svd(a)` | 96 | missing_backend | 1.4758 ms | 0.9363 | 0.634x |
| `np.linalg.svdvals(a)` | 32 | missing_backend | 0.0307 ms | 0.0350 | 1.140x |
| `np.linalg.svdvals(a)` | 96 | missing_backend | 0.4991 ms | 0.3091 | 0.619x |
| `np.linalg.tensordot(a, b, axes=1)` | 128 | 0.2149 ms | 0.0614 ms | 0.0695 | 1.132x |
| `np.linalg.tensorinv(a)` | 32 | missing_backend | 0.0100 ms | 0.0127 | 1.270x |
| `np.linalg.tensorinv(a)` | 96 | missing_backend | 0.1621 ms | 0.0919 | 0.567x |
| `np.linalg.tensorsolve(a, b)` | 32 | missing_backend | 0.0065 ms | 0.0099 | 1.523x |
| `np.linalg.tensorsolve(a, b)` | 96 | missing_backend | 0.0593 ms | 0.0344 | 0.580x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0339 ms | 0.0075 ms | 0.0108 | 1.440x |
| `np.matmul(a, b)` | 128 | 0.2122 ms | 0.0638 ms | 0.0644 | 1.009x |
| `np.matvec(a, b)` | 128 | 0.0232 ms | 0.0021 ms | 0.0025 | 1.191x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.1007 ms | 0.0561 | 0.557x |
| `np.roots(p)` | 16 | missing_backend | 0.0570 ms | 0.0416 | 0.730x |
| `np.tensordot(a, b, axes=1)` | 128 | 0.2118 ms | 0.0614 ms | 0.0713 | 1.161x |
| `np.vdot(a, b)` | 100,000 | 0.0077 ms | 0.0081 ms | 0.0096 | 1.247x |
| `np.vecdot(a, b)` | 100,000 | 0.0380 ms | 0.0073 ms | 0.0140 | 1.918x |
| `np.vecmat(a, b)` | 128 | 0.0031 ms | 0.0021 ms | 0.0020 | 0.952x |
