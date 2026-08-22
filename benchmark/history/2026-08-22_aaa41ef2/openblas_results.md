# Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-d37087cf9fdda14af443dc52c5a0ebc7f8066d4f752a929e00b12ec4c6ab1fb4\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b)` | 100,000 | 0.0097 ms | 0.0103 ms | 0.0089 | 0.918x |
| `np.convolve(a, v, mode='same')` | 100,000 | 0.1224 ms | 2.8086 ms | 1.1405 | 9.318x |
| `np.correlate(a, v, mode='same')` | 100,000 | 0.1176 ms | 2.6770 ms | 1.0250 | 8.716x |
| `np.dot(a, b)` | 100,000 | 0.0086 ms | 0.0082 ms | 0.0091 | 1.110x |
| `np.einsum('ij,jk->ik', a, b)` | 128 | 0.2230 ms | 0.1030 ms | 0.2479 | 2.407x |
| `np.inner(a, b)` | 100,000 | 0.0087 ms | 0.0089 ms | 0.0092 | 1.057x |
| `np.linalg.cholesky(a)` | 32 | missing_backend | 0.0159 ms | 0.0049 | 0.308x |
| `np.linalg.cholesky(a)` | 96 | missing_backend | 0.0624 ms | 0.0200 | 0.321x |
| `np.linalg.cond(a)` | 32 | missing_backend | 0.0525 ms | 0.0371 | 0.707x |
| `np.linalg.cond(a)` | 96 | missing_backend | 0.3435 ms | 0.2975 | 0.866x |
| `np.linalg.det(a)` | 32 | missing_backend | 0.0100 ms | 0.0067 | 0.670x |
| `np.linalg.det(a)` | 96 | missing_backend | 0.0351 ms | 0.0417 | 1.188x |
| `np.linalg.eig(a)` | 32 | missing_backend | 0.1592 ms | 0.1117 | 0.702x |
| `np.linalg.eig(a)` | 96 | missing_backend | 1.6014 ms | 1.4666 | 0.916x |
| `np.linalg.eigh(a)` | 32 | missing_backend | 0.0634 ms | 0.0499 | 0.787x |
| `np.linalg.eigh(a)` | 96 | missing_backend | 0.4731 ms | 0.4264 | 0.901x |
| `np.linalg.eigvals(a)` | 32 | missing_backend | 0.1330 ms | 0.0716 | 0.538x |
| `np.linalg.eigvals(a)` | 96 | missing_backend | 1.0080 ms | 0.8309 | 0.824x |
| `np.linalg.eigvalsh(a)` | 32 | missing_backend | 0.0280 ms | 0.0231 | 0.825x |
| `np.linalg.eigvalsh(a)` | 96 | missing_backend | 0.1874 ms | 0.1777 | 0.948x |
| `np.linalg.inv(a)` | 32 | missing_backend | 0.0231 ms | 0.0120 | 0.519x |
| `np.linalg.inv(a)` | 96 | missing_backend | 0.0976 ms | 0.0919 | 0.942x |
| `np.linalg.lstsq(a, b)` | 32 | missing_backend | 0.0992 ms | 0.0837 | 0.844x |
| `np.linalg.lstsq(a, b)` | 96 | missing_backend | 0.9049 ms | 0.9046 | 1.000x |
| `np.linalg.matmul(a, b)` | 128 | 0.2218 ms | 0.1032 ms | 0.0607 | 0.588x |
| `np.linalg.matrix_norm(a, ord=2)` | 32 | missing_backend | 0.0399 ms | 0.0385 | 0.965x |
| `np.linalg.matrix_norm(a, ord=2)` | 96 | missing_backend | 0.3253 ms | 0.4574 | 1.406x |
| `np.linalg.matrix_power(a, -1)` | 32 | missing_backend | 0.0205 ms | 0.0121 | 0.590x |
| `np.linalg.matrix_power(a, -1)` | 96 | missing_backend | 0.1032 ms | 0.0879 | 0.852x |
| `np.linalg.matrix_rank(a)` | 32 | missing_backend | 0.0502 ms | 0.0384 | 0.765x |
| `np.linalg.matrix_rank(a)` | 96 | missing_backend | 0.3256 ms | 0.3217 | 0.988x |
| `np.linalg.multi_dot(arrays)` | 128 | 0.3926 ms | 0.2166 ms | 0.1271 | 0.587x |
| `np.linalg.norm(a, ord=2)` | 32 | missing_backend | 0.0400 ms | 0.0384 | 0.960x |
| `np.linalg.norm(a, ord=2)` | 96 | missing_backend | 0.3202 ms | 0.3372 | 1.053x |
| `np.linalg.pinv(a)` | 32 | missing_backend | 0.1311 ms | 0.0940 | 0.717x |
| `np.linalg.pinv(a)` | 96 | missing_backend | 0.9779 ms | 0.8117 | 0.830x |
| `np.linalg.qr(a)` | 32 | missing_backend | 0.0390 ms | 0.0249 | 0.638x |
| `np.linalg.qr(a)` | 96 | missing_backend | 0.2086 ms | 0.1827 | 0.876x |
| `np.linalg.slogdet(a)` | 32 | missing_backend | 0.0099 ms | 0.0071 | 0.717x |
| `np.linalg.slogdet(a)` | 96 | missing_backend | 0.0357 ms | 0.0317 | 0.888x |
| `np.linalg.solve(a, b)` | 32 | missing_backend | 0.0096 ms | 0.0087 | 0.906x |
| `np.linalg.solve(a, b)` | 96 | missing_backend | 0.0350 ms | 0.0321 | 0.917x |
| `np.linalg.svd(a)` | 32 | missing_backend | 0.0959 ms | 0.0821 | 0.856x |
| `np.linalg.svd(a)` | 96 | missing_backend | 0.8168 ms | 0.7335 | 0.898x |
| `np.linalg.svdvals(a)` | 32 | missing_backend | 0.0371 ms | 0.0332 | 0.895x |
| `np.linalg.svdvals(a)` | 96 | missing_backend | 0.3204 ms | 0.2975 | 0.929x |
| `np.linalg.tensordot(a, b, axes=1)` | 128 | 0.2095 ms | 0.1074 ms | 0.0662 | 0.616x |
| `np.linalg.tensorinv(a)` | 32 | missing_backend | 0.0187 ms | 0.0129 | 0.690x |
| `np.linalg.tensorinv(a)` | 96 | missing_backend | 0.0993 ms | 0.0866 | 0.872x |
| `np.linalg.tensorsolve(a, b)` | 32 | missing_backend | 0.0104 ms | 0.0093 | 0.894x |
| `np.linalg.tensorsolve(a, b)` | 96 | missing_backend | 0.0360 ms | 0.0348 | 0.967x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0772 ms | 0.0163 ms | 0.0096 | 0.589x |
| `np.matmul(a, b)` | 128 | 0.2164 ms | 0.1024 ms | 0.0600 | 0.586x |
| `np.matvec(a, b)` | 128 | 0.0244 ms | 0.0024 ms | 0.0023 | 0.958x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.2522 ms | 0.0533 | 0.211x |
| `np.roots(p)` | 16 | missing_backend | 0.0652 ms | 0.0566 | 0.868x |
| `np.tensordot(a, b, axes=1)` | 128 | 0.2113 ms | 0.1136 ms | 0.0668 | 0.588x |
| `np.vdot(a, b)` | 100,000 | 0.0100 ms | 0.0089 ms | 0.0091 | 1.022x |
| `np.vecdot(a, b)` | 100,000 | 0.1569 ms | 0.0080 ms | 0.0092 | 1.150x |
| `np.vecmat(a, b)` | 128 | 0.0036 ms | 0.0024 ms | 0.0019 | 0.792x |
