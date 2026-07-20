# NumPy ↔ NumSharp API coverage

Compared with NumPy **2.4.2** using NumSharp assembly **0.60.0.0**.

Headline API availability: **60.4%** (338 of 560 default-scope APIs). Including partial mappings, **60.9%** are addressed.

| Surface | Available | Partial | Unsupported | Missing | Total | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| np.fft.* | 0 | 0 | 0 | 18 | 18 | 0.0% |
| np.linalg.* | 6 | 0 | 0 | 25 | 31 | 19.4% |
| ndarray.* | 50 | 1 | 0 | 19 | 70 | 71.4% |
| np.* | 234 | 1 | 0 | 155 | 390 | 60.0% |
| np.random.* | 48 | 1 | 0 | 2 | 51 | 94.1% |

> Availability is based on the compiled public API. It is not a blanket behavioral-parity claim; dtype, layout, signature, and edge-case parity require differential tests.

## Highest-priority gaps

| API | Surface | Status | Category |
|---|---|---|---|
| [`ndarray.data`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.data.html) | ndarray | partial | Array attributes |
| [`np.shape`](https://numpy.org/doc/stable/reference/generated/numpy.shape.html) | np | partial | Shape manipulation |
| [`np.random.random_integers`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_integers.html) | random | partial | Random |
| [`np.fft.fft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html) | fft | missing | Fourier transforms |
| [`np.fft.fft2`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html) | fft | missing | Fourier transforms |
| [`np.fft.fftfreq`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html) | fft | missing | Fourier transforms |
| [`np.fft.fftn`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html) | fft | missing | Fourier transforms |
| [`np.fft.fftshift`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html) | fft | missing | Fourier transforms |
| [`np.fft.hfft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html) | fft | missing | Fourier transforms |
| [`np.fft.ifft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html) | fft | missing | Fourier transforms |
| [`np.fft.ifft2`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html) | fft | missing | Fourier transforms |
| [`np.fft.ifftn`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftn.html) | fft | missing | Fourier transforms |
| [`np.fft.ifftshift`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html) | fft | missing | Fourier transforms |
| [`np.fft.ihfft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ihfft.html) | fft | missing | Fourier transforms |
| [`np.fft.irfft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html) | fft | missing | Fourier transforms |
| [`np.fft.irfft2`](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html) | fft | missing | Fourier transforms |
| [`np.fft.irfftn`](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfftn.html) | fft | missing | Fourier transforms |
| [`np.fft.rfft`](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html) | fft | missing | Fourier transforms |
| [`np.fft.rfft2`](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft2.html) | fft | missing | Fourier transforms |
| [`np.fft.rfftfreq`](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html) | fft | missing | Fourier transforms |
| [`np.fft.rfftn`](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html) | fft | missing | Fourier transforms |
| [`np.linalg.cholesky`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html) | linalg | missing | Linear algebra |
| [`np.linalg.cond`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html) | linalg | missing | Linear algebra |
| [`np.linalg.cross`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cross.html) | linalg | missing | Linear algebra |
| [`np.linalg.det`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html) | linalg | missing | Linear algebra |
| [`np.linalg.eig`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) | linalg | missing | Linear algebra |
| [`np.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) | linalg | missing | Linear algebra |
| [`np.linalg.eigvals`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html) | linalg | missing | Linear algebra |
| [`np.linalg.eigvalsh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html) | linalg | missing | Linear algebra |
| [`np.linalg.inv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html) | linalg | missing | Linear algebra |
| [`np.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) | linalg | missing | Linear algebra |
| [`np.linalg.matrix_norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_norm.html) | linalg | missing | Linear algebra |
| [`np.linalg.matrix_rank`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html) | linalg | missing | Linear algebra |
| [`np.linalg.multi_dot`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html) | linalg | missing | Linear algebra |
| [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) | linalg | missing | Linear algebra |
| [`np.linalg.pinv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html) | linalg | missing | Linear algebra |
| [`np.linalg.qr`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html) | linalg | missing | Linear algebra |
| [`np.linalg.slogdet`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html) | linalg | missing | Linear algebra |
| [`np.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html) | linalg | missing | Linear algebra |
| [`np.linalg.svd`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) | linalg | missing | Linear algebra |
| [`np.linalg.svdvals`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svdvals.html) | linalg | missing | Linear algebra |
| [`np.linalg.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html) | linalg | missing | Linear algebra |
| [`np.linalg.tensorinv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorinv.html) | linalg | missing | Linear algebra |
| [`np.linalg.tensorsolve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorsolve.html) | linalg | missing | Linear algebra |
| [`np.linalg.vecdot`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.vecdot.html) | linalg | missing | Linear algebra |
| [`np.linalg.vector_norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.vector_norm.html) | linalg | missing | Linear algebra |
| [`ndarray.argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argpartition.html) | ndarray | missing | Sorting & searching |
| [`ndarray.byteswap`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.byteswap.html) | ndarray | missing | Array methods |
| [`ndarray.choose`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.choose.html) | ndarray | missing | Array methods |
| [`ndarray.conj`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conj.html) | ndarray | missing | Array methods |

## Counting rules

The default scope is NumPy top-level callables, ndarray public methods/properties, and callables in numpy.random, numpy.linalg, and numpy.fft. Types, constants, modules, and NumSharp-only APIs remain searchable in the JSON artifact but do not affect the headline percentage.
