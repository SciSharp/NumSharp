# Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-f51086509ee9ad51f55da76c42db6bc0fa2231c6d2afed4b238a29428db980b1\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b)` | 100,000 | 0.0090 ms | 0.0126 ms | 0.0092 | 1.022x |
| `np.convolve(a, v, mode='same')` | 100,000 | 0.1190 ms | 2.5321 ms | 0.9955 | 8.366x |
| `np.correlate(a, v, mode='same')` | 100,000 | 0.1190 ms | 2.5900 ms | 1.1121 | 9.345x |
| `np.dot(a, b)` | 100,000 | 0.0092 ms | 0.0099 ms | 0.0093 | 1.011x |
| `np.einsum('ij,jk->ik', a, b)` | 128 | 0.2398 ms | 0.0609 ms | 0.2383 | 3.913x |
| `np.inner(a, b)` | 100,000 | 0.0073 ms | 0.0100 ms | 0.0096 | 1.315x |
| `np.linalg.cholesky(a)` | 32 | missing_backend | 0.0164 ms | 0.0055 | 0.335x |
| `np.linalg.cholesky(a)` | 96 | missing_backend | 0.0312 ms | 0.0203 | 0.651x |
| `np.linalg.cond(a)` | 32 | missing_backend | 0.0477 ms | 0.0366 | 0.767x |
| `np.linalg.cond(a)` | 96 | missing_backend | 0.3018 ms | 0.3034 | 1.005x |
| `np.linalg.det(a)` | 32 | missing_backend | 0.0094 ms | 0.0066 | 0.702x |
| `np.linalg.det(a)` | 96 | missing_backend | 0.0310 ms | 0.0307 | 0.990x |
| `np.linalg.eig(a)` | 32 | missing_backend | 0.1386 ms | 0.1109 | 0.800x |
| `np.linalg.eig(a)` | 96 | missing_backend | 1.5441 ms | 1.4065 | 0.911x |
| `np.linalg.eigh(a)` | 32 | missing_backend | 0.0573 ms | 0.0539 | 0.941x |
| `np.linalg.eigh(a)` | 96 | missing_backend | 0.4227 ms | 0.5738 | 1.357x |
| `np.linalg.eigvals(a)` | 32 | missing_backend | 0.0797 ms | 0.0755 | 0.947x |
| `np.linalg.eigvals(a)` | 96 | missing_backend | 0.8621 ms | 0.8534 | 0.990x |
| `np.linalg.eigvalsh(a)` | 32 | missing_backend | 0.0249 ms | 0.0244 | 0.980x |
| `np.linalg.eigvalsh(a)` | 96 | missing_backend | 0.1695 ms | 0.1658 | 0.978x |
| `np.linalg.inv(a)` | 32 | missing_backend | 0.0204 ms | 0.0121 | 0.593x |
| `np.linalg.inv(a)` | 96 | missing_backend | 0.0880 ms | 0.0829 | 0.942x |
| `np.linalg.lstsq(a, b)` | 32 | missing_backend | 0.0895 ms | 0.0859 | 0.960x |
| `np.linalg.lstsq(a, b)` | 96 | missing_backend | 0.8308 ms | 0.8393 | 1.010x |
| `np.linalg.matmul(a, b)` | 128 | 0.2455 ms | 0.0592 ms | 0.0612 | 1.034x |
| `np.linalg.matrix_norm(a, ord=2)` | 32 | missing_backend | 0.0360 ms | 0.0384 | 1.067x |
| `np.linalg.matrix_norm(a, ord=2)` | 96 | missing_backend | 0.3013 ms | 0.3037 | 1.008x |
| `np.linalg.matrix_power(a, -1)` | 32 | missing_backend | 0.0169 ms | 0.0122 | 0.722x |
| `np.linalg.matrix_power(a, -1)` | 96 | missing_backend | 0.0869 ms | 0.0812 | 0.934x |
| `np.linalg.matrix_rank(a)` | 32 | missing_backend | 0.0440 ms | 0.0387 | 0.880x |
| `np.linalg.matrix_rank(a)` | 96 | missing_backend | 0.2998 ms | 0.3286 | 1.096x |
| `np.linalg.multi_dot(arrays)` | 128 | 0.4478 ms | 0.1288 ms | 0.1252 | 0.972x |
| `np.linalg.norm(a, ord=2)` | 32 | missing_backend | 0.0369 ms | 0.0382 | 1.035x |
| `np.linalg.norm(a, ord=2)` | 96 | missing_backend | 0.2949 ms | 0.3021 | 1.024x |
| `np.linalg.pinv(a)` | 32 | missing_backend | 0.1106 ms | 0.0896 | 0.810x |
| `np.linalg.pinv(a)` | 96 | missing_backend | 0.9038 ms | 0.9667 | 1.070x |
| `np.linalg.qr(a)` | 32 | missing_backend | 0.0371 ms | 0.0239 | 0.644x |
| `np.linalg.qr(a)` | 96 | missing_backend | 0.1843 ms | 0.2218 | 1.203x |
| `np.linalg.slogdet(a)` | 32 | missing_backend | 0.0093 ms | 0.0072 | 0.774x |
| `np.linalg.slogdet(a)` | 96 | missing_backend | 0.0318 ms | 0.0330 | 1.038x |
| `np.linalg.solve(a, b)` | 32 | missing_backend | 0.0091 ms | 0.0078 | 0.857x |
| `np.linalg.solve(a, b)` | 96 | missing_backend | 0.0315 ms | 0.0324 | 1.029x |
| `np.linalg.svd(a)` | 32 | missing_backend | 0.0938 ms | 0.0782 | 0.834x |
| `np.linalg.svd(a)` | 96 | missing_backend | 0.7225 ms | 0.9305 | 1.288x |
| `np.linalg.svdvals(a)` | 32 | missing_backend | 0.0330 ms | 0.0337 | 1.021x |
| `np.linalg.svdvals(a)` | 96 | missing_backend | 0.2944 ms | 0.2945 | 1.000x |
| `np.linalg.tensordot(a, b, axes=1)` | 128 | 0.2253 ms | 0.0979 ms | 0.0674 | 0.688x |
| `np.linalg.tensorinv(a)` | 32 | missing_backend | 0.0161 ms | 0.0123 | 0.764x |
| `np.linalg.tensorinv(a)` | 96 | missing_backend | 0.0868 ms | 0.0850 | 0.979x |
| `np.linalg.tensorsolve(a, b)` | 32 | missing_backend | 0.0094 ms | 0.0091 | 0.968x |
| `np.linalg.tensorsolve(a, b)` | 96 | missing_backend | 0.0335 ms | 0.0329 | 0.982x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0559 ms | 0.0099 ms | 0.0080 | 0.808x |
| `np.matmul(a, b)` | 128 | 0.2400 ms | 0.0591 ms | 0.0592 | 1.002x |
| `np.matvec(a, b)` | 128 | 0.0237 ms | 0.0023 ms | 0.0021 | 0.913x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.2138 ms | 0.0494 | 0.231x |
| `np.roots(p)` | 16 | missing_backend | 0.0596 ms | 0.0413 | 0.693x |
| `np.tensordot(a, b, axes=1)` | 128 | 0.1831 ms | 0.0982 ms | 0.0681 | 0.693x |
| `np.vdot(a, b)` | 100,000 | 0.0080 ms | 0.0106 ms | 0.0075 | 0.937x |
| `np.vecdot(a, b)` | 100,000 | 0.1818 ms | 0.0098 ms | 0.0076 | 0.776x |
| `np.vecmat(a, b)` | 128 | 0.0042 ms | 0.0020 ms | 0.0021 | 1.050x |
