# Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-6702e17e3d34774811ed31feb79b1be9233097d85018996ddc811efc382c4e33\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b) [ndarray]` | 1,000 | 0.0003 ms | 0.0006 ms | 0.0004 | 1.333x |
| `a.dot(b) [ndarray]` | 100,000 | 0.0090 ms | 0.0090 ms | 0.0068 | 0.756x |
| `a.dot(b) [ndarray]` | 10,000,000 | 3.0736 ms | 2.9441 ms | 3.0018 | 1.020x |
| `np.convolve(a, kernel)` | 1,000 | 0.0036 ms | 0.0156 ms | 0.0110 | 3.056x |
| `np.convolve(a, kernel)` | 100,000 | 0.1138 ms | 0.9192 ms | 0.9811 | 8.621x |
| `np.convolve(a, kernel)` | 10,000,000 | 19.3707 ms | 100.8334 ms | 109.7462 | 5.666x |
| `np.correlate(a, kernel)` | 1,000 | 0.0014 ms | 0.0132 ms | 0.0105 | 7.500x |
| `np.correlate(a, kernel)` | 100,000 | 0.1113 ms | 0.9014 ms | 0.9719 | 8.732x |
| `np.correlate(a, kernel)` | 10,000,000 | 19.3109 ms | 100.8807 ms | 109.5148 | 5.671x |
| `np.dot(a, b)` | 1,000 | 0.0004 ms | 0.0007 ms | 0.0005 | 1.250x |
| `np.dot(a, b)` | 100,000 | 0.0091 ms | 0.0090 ms | 0.0069 | 0.767x |
| `np.dot(a, b)` | 10,000,000 | 3.0302 ms | 2.9299 ms | 3.0031 | 1.025x |
| `np.einsum('ij,jk->ik', a, b)` | 1,000 | 0.0040 ms | 0.0030 ms | 0.0062 | 2.067x |
| `np.einsum('ij,jk->ik', a, b)` | 100,000 | 1.8205 ms | 0.7969 ms | 3.5330 | 4.433x |
| `np.einsum('ij,jk->ik', a, b)` | 10,000,000 | 2.3765 ms | 1.4162 ms | 6.4002 | 4.519x |
| `np.einsum(subscripts, operands)` | 1,000 | 0.0086 ms | 0.0021 ms | 0.0014 | 0.667x |
| `np.einsum(subscripts, operands)` | 100,000 | 0.6336 ms | 0.0098 ms | 0.0196 | 2.000x |
| `np.einsum(subscripts, operands)` | 10,000,000 | 63.5258 ms | 2.9374 ms | 3.7541 | 1.278x |
| `np.inner(a, b)` | 1,000 | 0.0003 ms | 0.0007 ms | 0.0006 | 2.000x |
| `np.inner(a, b)` | 100,000 | 0.0090 ms | 0.0090 ms | 0.0070 | 0.778x |
| `np.inner(a, b)` | 10,000,000 | 2.9977 ms | 2.9246 ms | 3.0051 | 1.028x |
| `np.linalg.cholesky(a)` | 1,000 | missing_backend | 0.0017 ms | 0.0046 | 2.706x |
| `np.linalg.cholesky(a)` | 100,000 | missing_backend | 0.0144 ms | 0.0188 | 1.306x |
| `np.linalg.cholesky(a)` | 10,000,000 | missing_backend | 0.0315 ms | 0.0572 | 1.816x |
| `np.linalg.cond(a)` | 1,000 | missing_backend | 0.0318 ms | 0.0351 | 1.104x |
| `np.linalg.cond(a)` | 100,000 | missing_backend | 0.2608 ms | 0.2590 | 0.993x |
| `np.linalg.cond(a)` | 10,000,000 | missing_backend | 0.5530 ms | 0.5514 | 0.997x |
| `np.linalg.det(a)` | 1,000 | missing_backend | 0.0044 ms | 0.0058 | 1.318x |
| `np.linalg.det(a)` | 100,000 | missing_backend | 0.0256 ms | 0.0268 | 1.047x |
| `np.linalg.det(a)` | 10,000,000 | missing_backend | 0.0531 ms | 0.0543 | 1.023x |
| `np.linalg.eig(a)` | 1,000 | missing_backend | 0.1034 ms | 0.1003 | 0.970x |
| `np.linalg.eig(a)` | 100,000 | missing_backend | 1.3699 ms | 1.3776 | 1.006x |
| `np.linalg.eig(a)` | 10,000,000 | missing_backend | 2.8455 ms | 3.0686 | 1.078x |
| `np.linalg.eigh(a)` | 1,000 | missing_backend | 0.0451 ms | 0.0478 | 1.060x |
| `np.linalg.eigh(a)` | 100,000 | missing_backend | 0.3887 ms | 0.3925 | 1.010x |
| `np.linalg.eigh(a)` | 10,000,000 | missing_backend | 0.7142 ms | 0.7313 | 1.024x |
| `np.linalg.eigvals(a)` | 1,000 | missing_backend | 0.0599 ms | 0.0639 | 1.067x |
| `np.linalg.eigvals(a)` | 100,000 | missing_backend | 0.8128 ms | 0.7854 | 0.966x |
| `np.linalg.eigvals(a)` | 10,000,000 | missing_backend | 1.5956 ms | 1.5325 | 0.960x |
| `np.linalg.eigvalsh(a)` | 1,000 | missing_backend | 0.0189 ms | 0.0212 | 1.122x |
| `np.linalg.eigvalsh(a)` | 100,000 | missing_backend | 0.1581 ms | 0.1584 | 1.002x |
| `np.linalg.eigvalsh(a)` | 10,000,000 | missing_backend | 0.2950 ms | 0.2934 | 0.995x |
| `np.linalg.inv(a)` | 1,000 | missing_backend | 0.0083 ms | 0.0110 | 1.325x |
| `np.linalg.inv(a)` | 100,000 | missing_backend | 0.0756 ms | 0.0786 | 1.040x |
| `np.linalg.inv(a)` | 10,000,000 | missing_backend | 0.1608 ms | 0.1821 | 1.132x |
| `np.linalg.lstsq(a, b)` | 1,000 | missing_backend | 0.0772 ms | 0.0793 | 1.027x |
| `np.linalg.lstsq(a, b)` | 100,000 | missing_backend | 0.7965 ms | 0.7911 | 0.993x |
| `np.linalg.lstsq(a, b)` | 10,000,000 | missing_backend | 1.4325 ms | 1.4275 | 0.997x |
| `np.linalg.matmul(a, b)` | 1,000 | 0.0037 ms | 0.0019 ms | 0.0024 | 1.263x |
| `np.linalg.matmul(a, b)` | 100,000 | 1.8253 ms | 0.7939 ms | 0.8002 | 1.008x |
| `np.linalg.matmul(a, b)` | 10,000,000 | 2.3761 ms | 1.4111 ms | 1.6060 | 1.138x |
| `np.linalg.matrix_norm(a, ord=2)` | 1,000 | missing_backend | 0.0313 ms | 0.0366 | 1.169x |
| `np.linalg.matrix_norm(a, ord=2)` | 100,000 | missing_backend | 0.2576 ms | 0.2602 | 1.010x |
| `np.linalg.matrix_norm(a, ord=2)` | 10,000,000 | missing_backend | 0.5521 ms | 0.5498 | 0.996x |
| `np.linalg.matrix_power(a, -1)` | 1,000 | missing_backend | 0.0088 ms | 0.0113 | 1.284x |
| `np.linalg.matrix_power(a, -1)` | 100,000 | missing_backend | 0.0774 ms | 0.0787 | 1.017x |
| `np.linalg.matrix_power(a, -1)` | 10,000,000 | missing_backend | 0.1628 ms | 0.1825 | 1.121x |
| `np.linalg.matrix_power(a, 3)` | 1,000 | 0.0071 ms | 0.0034 ms | 0.0052 | 1.529x |
| `np.linalg.matrix_power(a, 3)` | 100,000 | 0.3497 ms | 0.1158 ms | 0.1179 | 1.018x |
| `np.linalg.matrix_power(a, 3)` | 10,000,000 | 0.3465 ms | 0.1159 ms | 0.1176 | 1.015x |
| `np.linalg.matrix_rank(a)` | 1,000 | missing_backend | 0.0323 ms | 0.0368 | 1.139x |
| `np.linalg.matrix_rank(a)` | 100,000 | missing_backend | 0.2600 ms | 0.2617 | 1.007x |
| `np.linalg.matrix_rank(a)` | 10,000,000 | missing_backend | 0.5533 ms | 0.5506 | 0.995x |
| `np.linalg.multi_dot(arrays)` | 1,000 | 0.0071 ms | 0.0034 ms | 0.0051 | 1.500x |
| `np.linalg.multi_dot(arrays)` | 100,000 | 0.3509 ms | 0.1159 ms | 0.1207 | 1.041x |
| `np.linalg.multi_dot(arrays)` | 10,000,000 | 0.3066 ms | 0.1159 ms | 0.1203 | 1.038x |
| `np.linalg.norm(a, ord=2)` | 1,000 | missing_backend | 0.0317 ms | 0.0367 | 1.158x |
| `np.linalg.norm(a, ord=2)` | 100,000 | missing_backend | 0.2578 ms | 0.2619 | 1.016x |
| `np.linalg.norm(a, ord=2)` | 10,000,000 | missing_backend | 0.5519 ms | 0.5515 | 0.999x |
| `np.linalg.pinv(a)` | 1,000 | missing_backend | 0.0804 ms | 0.0850 | 1.057x |
| `np.linalg.pinv(a)` | 100,000 | missing_backend | 0.7248 ms | 0.7320 | 1.010x |
| `np.linalg.pinv(a)` | 10,000,000 | missing_backend | 1.4073 ms | 1.6035 | 1.139x |
| `np.linalg.qr(a)` | 1,000 | missing_backend | 0.0186 ms | 0.0229 | 1.231x |
| `np.linalg.qr(a)` | 100,000 | missing_backend | 0.1474 ms | 0.1609 | 1.092x |
| `np.linalg.qr(a)` | 10,000,000 | missing_backend | 0.3365 ms | 0.3988 | 1.185x |
| `np.linalg.slogdet(a)` | 1,000 | missing_backend | 0.0044 ms | 0.0066 | 1.500x |
| `np.linalg.slogdet(a)` | 100,000 | missing_backend | 0.0258 ms | 0.0277 | 1.074x |
| `np.linalg.slogdet(a)` | 10,000,000 | missing_backend | 0.0533 ms | 0.0552 | 1.036x |
| `np.linalg.solve(a, b)` | 1,000 | missing_backend | 0.0046 ms | 0.0075 | 1.630x |
| `np.linalg.solve(a, b)` | 100,000 | missing_backend | 0.0273 ms | 0.0295 | 1.081x |
| `np.linalg.solve(a, b)` | 10,000,000 | missing_backend | 0.0552 ms | 0.0576 | 1.043x |
| `np.linalg.svd(a)` | 1,000 | missing_backend | 0.0716 ms | 0.0749 | 1.046x |
| `np.linalg.svd(a)` | 100,000 | missing_backend | 0.6798 ms | 0.6804 | 1.001x |
| `np.linalg.svd(a)` | 10,000,000 | missing_backend | 1.2957 ms | 1.4826 | 1.144x |
| `np.linalg.svdvals(a)` | 1,000 | missing_backend | 0.0294 ms | 0.0319 | 1.085x |
| `np.linalg.svdvals(a)` | 100,000 | missing_backend | 0.2572 ms | 0.2569 | 0.999x |
| `np.linalg.svdvals(a)` | 10,000,000 | missing_backend | 0.5506 ms | 0.5445 | 0.989x |
| `np.linalg.tensordot(a, b)` | 1,000 | 0.0043 ms | 0.0023 ms | 0.0056 | 2.435x |
| `np.linalg.tensordot(a, b)` | 100,000 | 0.1702 ms | 0.0587 ms | 0.0633 | 1.078x |
| `np.linalg.tensordot(a, b)` | 10,000,000 | 0.1377 ms | 0.0584 ms | 0.0633 | 1.084x |
| `np.linalg.tensorinv(a)` | 1,000 | missing_backend | 0.0085 ms | 0.0115 | 1.353x |
| `np.linalg.tensorinv(a)` | 100,000 | missing_backend | 0.0757 ms | 0.0790 | 1.044x |
| `np.linalg.tensorinv(a)` | 10,000,000 | missing_backend | 0.1606 ms | 0.1829 | 1.139x |
| `np.linalg.tensorsolve(a, b)` | 1,000 | missing_backend | 0.0050 ms | 0.0083 | 1.660x |
| `np.linalg.tensorsolve(a, b)` | 100,000 | missing_backend | 0.0277 ms | 0.0303 | 1.094x |
| `np.linalg.tensorsolve(a, b)` | 10,000,000 | missing_backend | 0.0557 ms | 0.0586 | 1.052x |
| `np.linalg.vecdot(a, b)` | 1,000 | 0.0024 ms | 0.0007 ms | 0.0008 | 1.143x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0196 ms | 0.0090 ms | 0.0073 | 0.811x |
| `np.linalg.vecdot(a, b)` | 10,000,000 | 18.4952 ms | 2.9730 ms | 3.0120 | 1.013x |
| `np.matmul(a, b)` | 1,000 | 0.0041 ms | 0.0023 ms | 0.0022 | 0.957x |
| `np.matmul(a, b)` | 100,000 | 1.8243 ms | 0.7937 ms | 0.8000 | 1.008x |
| `np.matmul(a, b)` | 10,000,000 | 2.3773 ms | 1.4120 ms | 1.6071 | 1.138x |
| `np.matvec(a, b)` | 1,000 | 0.0015 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.matvec(a, b)` | 100,000 | 0.0692 ms | 0.0068 ms | 0.0072 | 1.059x |
| `np.matvec(a, b)` | 10,000,000 | 0.1008 ms | 0.0069 ms | 0.0071 | 1.029x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.0663 ms | 0.0449 | 0.677x |
| `np.polyfit(x, y, 3)` | 100,000 | missing_backend | 0.6043 ms | 0.4632 | 0.767x |
| `np.polyfit(x, y, 3)` | 10,000,000 | missing_backend | 4.2502 ms | 5.1652 | 1.215x |
| `np.roots(p)` | 1,000 | missing_backend | 0.0341 ms | 0.0366 | 1.073x |
| `np.roots(p)` | 100,000 | missing_backend | 0.0947 ms | 0.0977 | 1.032x |
| `np.roots(p)` | 10,000,000 | missing_backend | 0.4145 ms | 0.4097 | 0.988x |
| `np.tensordot(a, b)` | 1,000 | 0.0076 ms | 0.0013 ms | 0.0036 | 2.769x |
| `np.tensordot(a, b)` | 100,000 | 0.6327 ms | 0.0097 ms | 0.0101 | 1.041x |
| `np.tensordot(a, b)` | 10,000,000 | 63.5058 ms | 3.1128 ms | 3.0330 | 0.974x |
| `np.tensordot(a, b, axes=1)` | 1,000 | 0.0043 ms | 0.0023 ms | 0.0054 | 2.348x |
| `np.tensordot(a, b, axes=1)` | 100,000 | 3.4449 ms | 0.7964 ms | 0.8139 | 1.022x |
| `np.tensordot(a, b, axes=1)` | 10,000,000 | 2.3840 ms | 1.4137 ms | 1.6202 | 1.146x |
| `np.vdot(a, b)` | 1,000 | 0.0009 ms | 0.0006 ms | 0.0005 | 0.833x |
| `np.vdot(a, b)` | 100,000 | 0.0092 ms | 0.0090 ms | 0.0070 | 0.778x |
| `np.vdot(a, b)` | 10,000,000 | 2.9873 ms | 2.9578 ms | 2.9930 | 1.012x |
| `np.vecdot(a, b)` | 1,000 | 0.0026 ms | 0.0007 ms | 0.0006 | 0.857x |
| `np.vecdot(a, b)` | 100,000 | 0.0198 ms | 0.0090 ms | 0.0070 | 0.778x |
| `np.vecdot(a, b)` | 10,000,000 | 18.3500 ms | 2.9600 ms | 3.0103 | 1.017x |
| `np.vecmat(a, b)` | 1,000 | 0.0007 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.vecmat(a, b)` | 100,000 | 0.0311 ms | 0.0069 ms | 0.0072 | 1.043x |
| `np.vecmat(a, b)` | 10,000,000 | 0.0482 ms | 0.0069 ms | 0.0067 | 0.971x |
