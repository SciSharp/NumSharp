# NumPy ↔ NumSharp API coverage

Compared with NumPy **2.4.2** using NumSharp assembly **0.60.0.0**.

Headline API availability: **76.8%** (430 of 560 default-scope APIs). Including partial mappings, **77.3%** are addressed.

| Surface | Available | Partial | Unsupported | Missing | Total | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| np.fft.* | 18 | 0 | 0 | 0 | 18 | 100.0% |
| np.linalg.* | 31 | 0 | 0 | 0 | 31 | 100.0% |
| ndarray.* | 54 | 1 | 0 | 15 | 70 | 77.1% |
| np.* | 279 | 1 | 0 | 110 | 390 | 71.5% |
| np.random.* | 48 | 1 | 0 | 2 | 51 | 94.1% |

> Availability is based on the compiled public API. It is not a blanket behavioral-parity claim; dtype, layout, signature, and edge-case parity require differential tests.

## Highest-priority gaps

| API | Surface | Status | Category |
|---|---|---|---|
| [`ndarray.data`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.data.html) | ndarray | partial | Array attributes |
| [`np.shape`](https://numpy.org/doc/stable/reference/generated/numpy.shape.html) | np | partial | Shape manipulation |
| [`np.random.random_integers`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_integers.html) | random | partial | Random |
| [`ndarray.byteswap`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.byteswap.html) | ndarray | missing | Array methods |
| [`ndarray.choose`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.choose.html) | ndarray | missing | Array methods |
| [`ndarray.ctypes`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html) | ndarray | missing | Array attributes |
| [`ndarray.device`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.device.html) | ndarray | missing | Array attributes |
| [`ndarray.dump`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dump.html) | ndarray | missing | Array methods |
| [`ndarray.dumps`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dumps.html) | ndarray | missing | Array methods |
| [`ndarray.fill`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html) | ndarray | missing | Array methods |
| [`ndarray.flags`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html) | ndarray | missing | Array attributes |
| [`ndarray.getfield`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.getfield.html) | ndarray | missing | Array methods |
| [`ndarray.imag`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.imag.html) | ndarray | missing | Array attributes |
| [`ndarray.nbytes`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html) | ndarray | missing | Array attributes |
| [`ndarray.real`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.real.html) | ndarray | missing | Array attributes |
| [`ndarray.setfield`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setfield.html) | ndarray | missing | Array methods |
| [`ndarray.setflags`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setflags.html) | ndarray | missing | Array methods |
| [`ndarray.to_device`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.to_device.html) | ndarray | missing | Array methods |
| [`np.acosh`](https://numpy.org/doc/stable/reference/generated/numpy.acosh.html) | np | missing | Math |
| [`np.apply_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html) | np | missing | Shape manipulation |
| [`np.apply_over_axes`](https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html) | np | missing | Shape manipulation |
| [`np.arccosh`](https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html) | np | missing | Math |
| [`np.arcsinh`](https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html) | np | missing | Math |
| [`np.arctanh`](https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html) | np | missing | Math |
| [`np.array_equiv`](https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html) | np | missing | Logic & comparison |
| [`np.asinh`](https://numpy.org/doc/stable/reference/generated/numpy.asinh.html) | np | missing | Math |
| [`np.atanh`](https://numpy.org/doc/stable/reference/generated/numpy.atanh.html) | np | missing | Math |
| [`np.bartlett`](https://numpy.org/doc/stable/reference/generated/numpy.bartlett.html) | np | missing | Window functions |
| [`np.base_repr`](https://numpy.org/doc/stable/reference/generated/numpy.base_repr.html) | np | missing | Text & formatting |
| [`np.binary_repr`](https://numpy.org/doc/stable/reference/generated/numpy.binary_repr.html) | np | missing | Text & formatting |
| [`np.bincount`](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) | np | missing | Statistics & histograms |
| [`np.bitwise_count`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_count.html) | np | missing | Math |
| [`np.blackman`](https://numpy.org/doc/stable/reference/generated/numpy.blackman.html) | np | missing | Window functions |
| [`np.bmat`](https://numpy.org/doc/stable/reference/generated/numpy.bmat.html) | np | missing | Linear algebra |
| [`np.broadcast_shapes`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_shapes.html) | np | missing | Shape manipulation |
| [`np.busday_count`](https://numpy.org/doc/stable/reference/generated/numpy.busday_count.html) | np | missing | Date & time |
| [`np.busday_offset`](https://numpy.org/doc/stable/reference/generated/numpy.busday_offset.html) | np | missing | Date & time |
| [`np.choose`](https://numpy.org/doc/stable/reference/generated/numpy.choose.html) | np | missing | Indexing & selection |
| [`np.copysign`](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html) | np | missing | Math |
| [`np.corrcoef`](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html) | np | missing | Statistics & histograms |
| [`np.correlate`](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html) | np | missing | Statistics & histograms |
| [`np.cov`](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) | np | missing | Statistics & histograms |
| [`np.cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html) | np | missing | Math |
| [`np.datetime_as_string`](https://numpy.org/doc/stable/reference/generated/numpy.datetime_as_string.html) | np | missing | Date & time |
| [`np.datetime_data`](https://numpy.org/doc/stable/reference/generated/numpy.datetime_data.html) | np | missing | Date & time |
| [`np.digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) | np | missing | Statistics & histograms |
| [`np.divmod`](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html) | np | missing | Math |
| [`np.einsum_path`](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html) | np | missing | Linear algebra |
| [`np.fabs`](https://numpy.org/doc/stable/reference/generated/numpy.fabs.html) | np | missing | Math |
| [`np.fix`](https://numpy.org/doc/stable/reference/generated/numpy.fix.html) | np | missing | Math |

## Counting rules

The default scope is NumPy top-level callables, ndarray public methods/properties, and callables in numpy.random, numpy.linalg, and numpy.fft. Types, constants, modules, and NumSharp-only APIs remain searchable in the JSON artifact but do not affect the headline percentage.
