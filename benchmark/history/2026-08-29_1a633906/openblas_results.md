# Backend profiles — Managed C# and OpenBLAS

Both profiles use the same operation/dtype/N/scenario schema and the same single-threaded NumPy baseline.
`missing_backend` and `not_supported` are availability outcomes, never timed failures.

OpenBLAS backend: `C:\Users\ELI\AppData\Local\Temp\dotnet\runfile\app-85577fbdbca7e757a9c92c53ae14617934494c5e5b495c5196c0ca0a1ff07af1\bin\release\runtimes\win-x64\native\libscipy_openblas64_.dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`

| Operation | Size | Managed C# | OpenBLAS | NumPy ms | Best NPY/NS |
|---|---:|---:|---:|---:|---:|
| `a.dot(b) [ndarray]` | 1,000 | 0.0003 ms | 0.0006 ms | 0.0004 | 1.333x |
| `a.dot(b) [ndarray]` | 100,000 | 0.0097 ms | 0.0097 ms | 0.0098 | 1.010x |
| `a.dot(b) [ndarray]` | 10,000,000 | 0.1695 ms | 0.1731 ms | 0.1787 | 1.054x |
| `np.convolve(a, kernel)` | 1,000 | 0.0037 ms | 0.0156 ms | 0.0106 | 2.865x |
| `np.convolve(a, kernel)` | 100,000 | 0.1140 ms | 0.9208 ms | 1.0843 | 9.511x |
| `np.convolve(a, kernel)` | 10,000,000 | 1.1453 ms | 9.7410 ms | 10.5980 | 9.253x |
| `np.correlate(a, kernel)` | 1,000 | 0.0014 ms | 0.0133 ms | 0.0102 | 7.286x |
| `np.correlate(a, kernel)` | 100,000 | 0.1116 ms | 0.9140 ms | 1.0847 | 9.720x |
| `np.correlate(a, kernel)` | 10,000,000 | 1.1316 ms | 9.2459 ms | 10.5508 | 9.324x |
| `np.dot(a, b)` | 1,000 | 0.0004 ms | 0.0007 ms | 0.0005 | 1.250x |
| `np.dot(a, b)` | 100,000 | 0.0067 ms | 0.0067 ms | 0.0069 | 1.030x |
| `np.dot(a, b)` | 10,000,000 | 2.9704 ms | 3.3376 ms | 2.9709 | 1.000x |
| `np.einsum('ij,jk->ik', a, b)` | 1,000 | 0.0041 ms | 0.0028 ms | 0.0062 | 2.214x |
| `np.einsum('ij,jk->ik', a, b)` | 100,000 | 1.4043 ms | 0.7998 ms | 3.5304 | 4.414x |
| `np.einsum('ij,jk->ik', a, b)` | 10,000,000 | 3.1350 ms | 1.4120 ms | 6.5147 | 4.614x |
| `np.einsum(subscripts, operands)` | 1,000 | 0.0010 ms | 0.0019 ms | 0.0013 | 1.300x |
| `np.einsum(subscripts, operands)` | 100,000 | 0.0075 ms | 0.0074 ms | 0.0196 | 2.649x |
| `np.einsum(subscripts, operands)` | 10,000,000 | 3.1546 ms | 2.9748 ms | 3.7698 | 1.267x |
| `np.inner(a, b)` | 1,000 | 0.0003 ms | 0.0007 ms | 0.0006 | 2.000x |
| `np.inner(a, b)` | 100,000 | 0.0097 ms | 0.0097 ms | 0.0100 | 1.031x |
| `np.inner(a, b)` | 10,000,000 | 0.1678 ms | 0.1724 ms | 0.1755 | 1.046x |
| `np.linalg.cholesky(a)` | 1,000 | missing_backend | 0.0017 ms | 0.0044 | 2.588x |
| `np.linalg.cholesky(a)` | 100,000 | missing_backend | 0.0147 ms | 0.0183 | 1.245x |
| `np.linalg.cholesky(a)` | 10,000,000 | missing_backend | 0.0310 ms | 0.0568 | 1.832x |
| `np.linalg.cond(a)` | 1,000 | missing_backend | 0.0317 ms | 0.0351 | 1.107x |
| `np.linalg.cond(a)` | 100,000 | missing_backend | 0.2608 ms | 0.2613 | 1.002x |
| `np.linalg.cond(a)` | 10,000,000 | missing_backend | 0.5533 ms | 0.5569 | 1.007x |
| `np.linalg.det(a)` | 1,000 | 0.0020 ms | 0.0045 ms | 0.0058 | 2.900x |
| `np.linalg.det(a)` | 100,000 | 0.0301 ms | 0.0257 ms | 0.0269 | 1.047x |
| `np.linalg.det(a)` | 10,000,000 | 0.0860 ms | 0.0534 ms | 0.0544 | 1.019x |
| `np.linalg.eig(a)` | 1,000 | missing_backend | 0.1017 ms | 0.1007 | 0.990x |
| `np.linalg.eig(a)` | 100,000 | missing_backend | 1.3708 ms | 1.3923 | 1.016x |
| `np.linalg.eig(a)` | 10,000,000 | missing_backend | 2.8319 ms | 2.9327 | 1.036x |
| `np.linalg.eigh(a)` | 1,000 | missing_backend | 0.0449 ms | 0.0479 | 1.067x |
| `np.linalg.eigh(a)` | 100,000 | missing_backend | 0.3893 ms | 0.3943 | 1.013x |
| `np.linalg.eigh(a)` | 10,000,000 | missing_backend | 0.7102 ms | 0.7346 | 1.034x |
| `np.linalg.eigvals(a)` | 1,000 | missing_backend | 0.0598 ms | 0.0640 | 1.070x |
| `np.linalg.eigvals(a)` | 100,000 | missing_backend | 0.7965 ms | 0.8132 | 1.021x |
| `np.linalg.eigvals(a)` | 10,000,000 | missing_backend | 1.5520 ms | 1.5368 | 0.990x |
| `np.linalg.eigvalsh(a)` | 1,000 | missing_backend | 0.0188 ms | 0.0213 | 1.133x |
| `np.linalg.eigvalsh(a)` | 100,000 | missing_backend | 0.1593 ms | 0.1598 | 1.003x |
| `np.linalg.eigvalsh(a)` | 10,000,000 | missing_backend | 0.2925 ms | 0.2941 | 1.005x |
| `np.linalg.inv(a)` | 1,000 | 0.0051 ms | 0.0082 ms | 0.0108 | 2.118x |
| `np.linalg.inv(a)` | 100,000 | 0.0947 ms | 0.0758 ms | 0.0783 | 1.033x |
| `np.linalg.inv(a)` | 10,000,000 | 0.2987 ms | 0.1602 ms | 0.1819 | 1.135x |
| `np.linalg.lstsq(a, b)` | 1,000 | missing_backend | 0.0770 ms | 0.0795 | 1.032x |
| `np.linalg.lstsq(a, b)` | 100,000 | missing_backend | 0.8111 ms | 0.7961 | 0.982x |
| `np.linalg.lstsq(a, b)` | 10,000,000 | missing_backend | 1.4309 ms | 1.4468 | 1.011x |
| `np.linalg.matmul(a, b)` | 1,000 | 0.0036 ms | 0.0018 ms | 0.0024 | 1.333x |
| `np.linalg.matmul(a, b)` | 100,000 | 1.4014 ms | 0.7991 ms | 0.7960 | 0.996x |
| `np.linalg.matmul(a, b)` | 10,000,000 | 3.1470 ms | 1.4118 ms | 1.6112 | 1.141x |
| `np.linalg.matrix_norm(a, ord=2)` | 1,000 | missing_backend | 0.0315 ms | 0.0367 | 1.165x |
| `np.linalg.matrix_norm(a, ord=2)` | 100,000 | missing_backend | 0.2561 ms | 0.2640 | 1.031x |
| `np.linalg.matrix_norm(a, ord=2)` | 10,000,000 | missing_backend | 0.5589 ms | 0.5548 | 0.993x |
| `np.linalg.matrix_power(a, -1)` | 1,000 | 0.0053 ms | 0.0086 ms | 0.0114 | 2.151x |
| `np.linalg.matrix_power(a, -1)` | 100,000 | 0.1194 ms | 0.0773 ms | 0.0788 | 1.019x |
| `np.linalg.matrix_power(a, -1)` | 10,000,000 | 0.2419 ms | 0.1629 ms | 0.1829 | 1.123x |
| `np.linalg.matrix_power(a, 3)` | 1,000 | 0.0072 ms | 0.0034 ms | 0.0052 | 1.529x |
| `np.linalg.matrix_power(a, 3)` | 100,000 | 0.3514 ms | 0.1161 ms | 0.1180 | 1.016x |
| `np.linalg.matrix_power(a, 3)` | 10,000,000 | 0.3858 ms | 0.1160 ms | 0.1179 | 1.016x |
| `np.linalg.matrix_rank(a)` | 1,000 | missing_backend | 0.0325 ms | 0.0366 | 1.126x |
| `np.linalg.matrix_rank(a)` | 100,000 | missing_backend | 0.2619 ms | 0.2635 | 1.006x |
| `np.linalg.matrix_rank(a)` | 10,000,000 | missing_backend | 0.5586 ms | 0.5537 | 0.991x |
| `np.linalg.multi_dot(arrays)` | 1,000 | 0.0073 ms | 0.0034 ms | 0.0051 | 1.500x |
| `np.linalg.multi_dot(arrays)` | 100,000 | 0.3583 ms | 0.1161 ms | 0.1207 | 1.040x |
| `np.linalg.multi_dot(arrays)` | 10,000,000 | 0.3775 ms | 0.1159 ms | 0.1207 | 1.041x |
| `np.linalg.norm(a, ord=2)` | 1,000 | missing_backend | 0.0303 ms | 0.0366 | 1.208x |
| `np.linalg.norm(a, ord=2)` | 100,000 | missing_backend | 0.2570 ms | 0.2639 | 1.027x |
| `np.linalg.norm(a, ord=2)` | 10,000,000 | missing_backend | 0.5601 ms | 0.5526 | 0.987x |
| `np.linalg.pinv(a)` | 1,000 | missing_backend | 0.0809 ms | 0.0852 | 1.053x |
| `np.linalg.pinv(a)` | 100,000 | missing_backend | 0.7224 ms | 0.7352 | 1.018x |
| `np.linalg.pinv(a)` | 10,000,000 | missing_backend | 1.4111 ms | 1.4216 | 1.007x |
| `np.linalg.qr(a)` | 1,000 | missing_backend | 0.0186 ms | 0.0228 | 1.226x |
| `np.linalg.qr(a)` | 100,000 | missing_backend | 0.1455 ms | 0.1621 | 1.114x |
| `np.linalg.qr(a)` | 10,000,000 | missing_backend | 0.3389 ms | 0.4030 | 1.189x |
| `np.linalg.slogdet(a)` | 1,000 | 0.0020 ms | 0.0045 ms | 0.0065 | 3.250x |
| `np.linalg.slogdet(a)` | 100,000 | 0.0319 ms | 0.0260 ms | 0.0277 | 1.065x |
| `np.linalg.slogdet(a)` | 10,000,000 | 0.0865 ms | 0.0533 ms | 0.0557 | 1.045x |
| `np.linalg.solve(a, b)` | 1,000 | 0.0035 ms | 0.0047 ms | 0.0073 | 2.086x |
| `np.linalg.solve(a, b)` | 100,000 | 0.0464 ms | 0.0271 ms | 0.0295 | 1.089x |
| `np.linalg.solve(a, b)` | 10,000,000 | 0.1125 ms | 0.0554 ms | 0.0582 | 1.051x |
| `np.linalg.svd(a)` | 1,000 | missing_backend | 0.0721 ms | 0.0746 | 1.035x |
| `np.linalg.svd(a)` | 100,000 | missing_backend | 0.6822 ms | 0.6831 | 1.001x |
| `np.linalg.svd(a)` | 10,000,000 | missing_backend | 1.2951 ms | 1.3340 | 1.030x |
| `np.linalg.svdvals(a)` | 1,000 | missing_backend | 0.0295 ms | 0.0319 | 1.081x |
| `np.linalg.svdvals(a)` | 100,000 | missing_backend | 0.2589 ms | 0.2579 | 0.996x |
| `np.linalg.svdvals(a)` | 10,000,000 | missing_backend | 0.5574 ms | 0.5469 | 0.981x |
| `np.linalg.tensordot(a, b)` | 1,000 | 0.0043 ms | 0.0024 ms | 0.0056 | 2.333x |
| `np.linalg.tensordot(a, b)` | 100,000 | 0.1709 ms | 0.0585 ms | 0.0634 | 1.084x |
| `np.linalg.tensordot(a, b)` | 10,000,000 | 0.2136 ms | 0.0585 ms | 0.0633 | 1.082x |
| `np.linalg.tensorinv(a)` | 1,000 | 0.0052 ms | 0.0086 ms | 0.0115 | 2.212x |
| `np.linalg.tensorinv(a)` | 100,000 | 0.0949 ms | 0.0761 ms | 0.0791 | 1.039x |
| `np.linalg.tensorinv(a)` | 10,000,000 | 0.2988 ms | 0.1609 ms | 0.1829 | 1.137x |
| `np.linalg.tensorsolve(a, b)` | 1,000 | 0.0039 ms | 0.0050 ms | 0.0083 | 2.128x |
| `np.linalg.tensorsolve(a, b)` | 100,000 | 0.0467 ms | 0.0275 ms | 0.0305 | 1.109x |
| `np.linalg.tensorsolve(a, b)` | 10,000,000 | 0.1126 ms | 0.0556 ms | 0.0596 | 1.072x |
| `np.linalg.vecdot(a, b)` | 1,000 | 0.0020 ms | 0.0007 ms | 0.0008 | 1.143x |
| `np.linalg.vecdot(a, b)` | 100,000 | 0.0273 ms | 0.0067 ms | 0.0073 | 1.090x |
| `np.linalg.vecdot(a, b)` | 10,000,000 | 19.0396 ms | 3.2513 ms | 3.0080 | 0.925x |
| `np.matmul(a, b)` | 1,000 | 0.0039 ms | 0.0019 ms | 0.0022 | 1.158x |
| `np.matmul(a, b)` | 100,000 | 1.4009 ms | 0.8000 ms | 0.7956 | 0.995x |
| `np.matmul(a, b)` | 10,000,000 | 3.1524 ms | 1.4228 ms | 1.6034 | 1.127x |
| `np.matvec(a, b)` | 1,000 | 0.0008 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.matvec(a, b)` | 100,000 | 0.0097 ms | 0.0068 ms | 0.0072 | 1.059x |
| `np.matvec(a, b)` | 10,000,000 | 0.0096 ms | 0.0069 ms | 0.0071 | 1.029x |
| `np.polyfit(x, y, 3)` | 1,000 | missing_backend | 0.0655 ms | 0.0451 | 0.689x |
| `np.polyfit(x, y, 3)` | 100,000 | missing_backend | 0.6064 ms | 0.5296 | 0.873x |
| `np.polyfit(x, y, 3)` | 10,000,000 | missing_backend | 4.2946 ms | 5.2731 | 1.228x |
| `np.roots(p)` | 1,000 | missing_backend | 0.0344 ms | 0.0366 | 1.064x |
| `np.roots(p)` | 100,000 | missing_backend | 0.0953 ms | 0.0977 | 1.025x |
| `np.roots(p)` | 10,000,000 | missing_backend | 0.4132 ms | 0.4110 | 0.995x |
| `np.tensordot(a, b)` | 1,000 | 0.0011 ms | 0.0013 ms | 0.0037 | 3.364x |
| `np.tensordot(a, b)` | 100,000 | 0.0074 ms | 0.0074 ms | 0.0101 | 1.365x |
| `np.tensordot(a, b)` | 10,000,000 | 3.0513 ms | 2.9836 ms | 3.0781 | 1.032x |
| `np.tensordot(a, b, axes=1)` | 1,000 | 0.0043 ms | 0.0023 ms | 0.0054 | 2.348x |
| `np.tensordot(a, b, axes=1)` | 100,000 | 1.4011 ms | 0.7973 ms | 0.8087 | 1.014x |
| `np.tensordot(a, b, axes=1)` | 10,000,000 | 3.1346 ms | 1.4118 ms | 1.6270 | 1.152x |
| `np.vdot(a, b)` | 1,000 | 0.0006 ms | 0.0007 ms | 0.0005 | 0.833x |
| `np.vdot(a, b)` | 100,000 | 0.0068 ms | 0.0067 ms | 0.0070 | 1.045x |
| `np.vdot(a, b)` | 10,000,000 | 2.9826 ms | 3.2576 ms | 3.0013 | 1.006x |
| `np.vecdot(a, b)` | 1,000 | 0.0023 ms | 0.0007 ms | 0.0006 | 0.857x |
| `np.vecdot(a, b)` | 100,000 | 0.0275 ms | 0.0067 ms | 0.0070 | 1.045x |
| `np.vecdot(a, b)` | 10,000,000 | 18.7395 ms | 3.1114 ms | 3.0119 | 0.968x |
| `np.vecmat(a, b)` | 1,000 | 0.0008 ms | 0.0004 ms | 0.0007 | 1.750x |
| `np.vecmat(a, b)` | 100,000 | 0.0086 ms | 0.0073 ms | 0.0073 | 1.000x |
| `np.vecmat(a, b)` | 10,000,000 | 0.0102 ms | 0.0064 ms | 0.0067 | 1.047x |
