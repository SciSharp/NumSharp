# Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-6f2338a9b13e5e0c63650ad0eac116091195bf72a3489e1e18824d08b5dda7c1\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b)` | 100,000 | 0.0070 ms | 0.0073 ms | 0.0069 | 0.986x |
| `np.convolve(a, v, mode='same')` | 100,000 | 0.1165 ms | 0.9656 ms | 0.9941 | 8.533x |
| `np.correlate(a, v, mode='same')` | 100,000 | 0.1173 ms | 0.9917 ms | 1.1135 | 9.493x |
| `np.dot(a, b)` | 100,000 | 0.0070 ms | 0.0074 ms | 0.0070 | 1.000x |
| `np.einsum('ij,jk->ik', a, b)` | 128 | 0.2151 ms | 0.0605 ms | 0.2416 | 3.993x |
| `np.inner(a, b)` | 100,000 | 0.0070 ms | 0.0074 ms | 0.0071 | 1.014x |
| `np.linalg.cholesky(a)` | 32 | missing_backend | 0.0025 ms | 0.0047 | 1.880x |
| `np.linalg.cholesky(a)` | 96 | missing_backend | 0.0160 ms | 0.0183 | 1.144x |
| `np.linalg.cond(a)` | 32 | missing_backend | 0.0390 ms | 0.0353 | 0.905x |
| `np.linalg.cond(a)` | 96 | missing_backend | 0.5162 ms | 0.2785 | 0.540x |
| `np.linalg.det(a)` | 32 | missing_backend | 0.0052 ms | 0.0061 | 1.173x |
| `np.linalg.det(a)` | 96 | missing_backend | 0.0578 ms | 0.0272 | 0.471x |
| `np.linalg.eig(a)` | 32 | missing_backend | 0.1146 ms | 0.1032 | 0.901x |
| `np.linalg.eig(a)` | 96 | missing_backend | 2.6840 ms | 1.3641 | 0.508x |
| `np.linalg.eigh(a)` | 32 | missing_backend | 0.0462 ms | 0.0487 | 1.054x |
| `np.linalg.eigh(a)` | 96 | missing_backend | 0.8220 ms | 0.3986 | 0.485x |
| `np.linalg.eigvals(a)` | 32 | missing_backend | 0.0670 ms | 0.0664 | 0.991x |
| `np.linalg.eigvals(a)` | 96 | missing_backend | 1.5275 ms | 0.7744 | 0.507x |
| `np.linalg.eigvalsh(a)` | 32 | missing_backend | 0.0207 ms | 0.0228 | 1.101x |
| `np.linalg.eigvalsh(a)` | 96 | missing_backend | 0.2832 ms | 0.1599 | 0.565x |
| `np.linalg.inv(a)` | 32 | missing_backend | 0.0090 ms | 0.0111 | 1.233x |
| `np.linalg.inv(a)` | 96 | missing_backend | 0.1619 ms | 0.0787 | 0.486x |
| `np.linalg.lstsq(a, b)` | 32 | missing_backend | 0.0832 ms | 0.0802 | 0.964x |
| `np.linalg.lstsq(a, b)` | 96 | missing_backend | 1.6242 ms | 0.7920 | 0.488x |
| `np.linalg.matmul(a, b)` | 128 | 0.2128 ms | 0.0586 ms | 0.0587 | 1.002x |
| `np.linalg.matrix_norm(a, ord=2)` | 32 | missing_backend | 0.0328 ms | 0.0369 | 1.125x |
| `np.linalg.matrix_norm(a, ord=2)` | 96 | missing_backend | 0.4903 ms | 0.2894 | 0.590x |
| `np.linalg.matrix_power(a, -1)` | 32 | missing_backend | 0.0106 ms | 0.0115 | 1.085x |
| `np.linalg.matrix_power(a, -1)` | 96 | missing_backend | 0.1651 ms | 0.0803 | 0.486x |
| `np.linalg.matrix_rank(a)` | 32 | missing_backend | 0.0365 ms | 0.0370 | 1.014x |
| `np.linalg.matrix_rank(a)` | 96 | missing_backend | 0.4982 ms | 0.2853 | 0.573x |
| `np.linalg.multi_dot(arrays)` | 128 | 0.3826 ms | 0.1175 ms | 0.1212 | 1.031x |
| `np.linalg.norm(a, ord=2)` | 32 | missing_backend | 0.0332 ms | 0.0368 | 1.108x |
| `np.linalg.norm(a, ord=2)` | 96 | missing_backend | 0.4903 ms | 0.2827 | 0.577x |
| `np.linalg.pinv(a)` | 32 | missing_backend | 0.0956 ms | 0.0871 | 0.911x |
| `np.linalg.pinv(a)` | 96 | missing_backend | 1.5979 ms | 0.7379 | 0.462x |
| `np.linalg.qr(a)` | 32 | missing_backend | 0.0203 ms | 0.0236 | 1.163x |
| `np.linalg.qr(a)` | 96 | missing_backend | 0.3164 ms | 0.1607 | 0.508x |
| `np.linalg.slogdet(a)` | 32 | missing_backend | 0.0057 ms | 0.0068 | 1.193x |
| `np.linalg.slogdet(a)` | 96 | missing_backend | 0.0564 ms | 0.0282 | 0.500x |
| `np.linalg.solve(a, b)` | 32 | missing_backend | 0.0053 ms | 0.0077 | 1.453x |
| `np.linalg.solve(a, b)` | 96 | missing_backend | 0.0592 ms | 0.0299 | 0.505x |
| `np.linalg.svd(a)` | 32 | missing_backend | 0.0750 ms | 0.0771 | 1.028x |
| `np.linalg.svd(a)` | 96 | missing_backend | 1.4331 ms | 0.6929 | 0.483x |
| `np.linalg.svdvals(a)` | 32 | missing_backend | 0.0298 ms | 0.0320 | 1.074x |
| `np.linalg.svdvals(a)` | 96 | missing_backend | 0.4894 ms | 0.2761 | 0.564x |
| `np.linalg.tensordot(a, b, axes=1)` | 128 | 0.2166 ms | 0.0606 ms | 0.0635 | 1.048x |
| `np.linalg.tensorinv(a)` | 32 | missing_backend | 0.0096 ms | 0.0118 | 1.229x |
| `np.linalg.tensorinv(a)` | 96 | missing_backend | 0.1621 ms | 0.0801 | 0.494x |
| `np.linalg.tensorsolve(a, b)` | 32 | missing_backend | 0.0067 ms | 0.0086 | 1.284x |
| `np.linalg.tensorsolve(a, b)` | 96 | missing_backend | 0.0604 ms | 0.0309 | 0.512x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0329 ms | 0.0073 ms | 0.0074 | 1.014x |
| `np.matmul(a, b)` | 128 | 0.2133 ms | 0.0587 ms | 0.0585 | 0.997x |
| `np.matvec(a, b)` | 128 | 0.0227 ms | 0.0016 ms | 0.0020 | 1.250x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.1084 ms | 0.0454 | 0.419x |
| `np.roots(p)` | 16 | missing_backend | 0.0515 ms | 0.0374 | 0.726x |
| `np.tensordot(a, b, axes=1)` | 128 | 0.2149 ms | 0.0608 ms | 0.0634 | 1.043x |
| `np.vdot(a, b)` | 100,000 | 0.0076 ms | 0.0081 ms | 0.0071 | 0.934x |
| `np.vecdot(a, b)` | 100,000 | 0.0333 ms | 0.0073 ms | 0.0071 | 0.973x |
| `np.vecmat(a, b)` | 128 | 0.0027 ms | 0.0017 ms | 0.0015 | 0.882x |
