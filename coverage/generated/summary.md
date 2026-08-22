# NumPy ↔ NumSharp API coverage

Compared with NumPy **2.4.2** using NumSharp assembly **0.60.0.0**.

Headline API availability: **84.3%** (472 of 560 default-scope APIs). Including partial mappings, **84.8%** are addressed.

| Surface | Available | Partial | Unsupported | Missing | Total | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| np.fft.* | 18 | 0 | 0 | 0 | 18 | 100.0% |
| np.linalg.* | 31 | 0 | 0 | 0 | 31 | 100.0% |
| ndarray.* | 66 | 1 | 0 | 3 | 70 | 94.3% |
| np.* | 309 | 1 | 0 | 80 | 390 | 79.2% |
| np.random.* | 48 | 1 | 0 | 2 | 51 | 94.1% |

> Availability is based on the compiled public API. It is not a blanket behavioral-parity claim; dtype, layout, signature, and edge-case parity require differential tests.

## Highest-priority gaps

| API | Surface | Status | Category |
|---|---|---|---|
| [`ndarray.data`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.data.html) | ndarray | partial | Array attributes |
| [`np.shape`](https://numpy.org/doc/stable/reference/generated/numpy.shape.html) | np | partial | Shape manipulation |
| [`np.random.random_integers`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_integers.html) | random | partial | Random |
| [`ndarray.ctypes`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html) | ndarray | missing | Array attributes |
| [`ndarray.dump`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dump.html) | ndarray | missing | Array methods |
| [`ndarray.dumps`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dumps.html) | ndarray | missing | Array methods |
| [`np.apply_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html) | np | missing | Shape manipulation |
| [`np.apply_over_axes`](https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html) | np | missing | Shape manipulation |
| [`np.array_equiv`](https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html) | np | missing | Logic & comparison |
| [`np.bartlett`](https://numpy.org/doc/stable/reference/generated/numpy.bartlett.html) | np | missing | Window functions |
| [`np.base_repr`](https://numpy.org/doc/stable/reference/generated/numpy.base_repr.html) | np | missing | Text & formatting |
| [`np.binary_repr`](https://numpy.org/doc/stable/reference/generated/numpy.binary_repr.html) | np | missing | Text & formatting |
| [`np.bitwise_count`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_count.html) | np | missing | Math |
| [`np.blackman`](https://numpy.org/doc/stable/reference/generated/numpy.blackman.html) | np | missing | Window functions |
| [`np.bmat`](https://numpy.org/doc/stable/reference/generated/numpy.bmat.html) | np | missing | Linear algebra |
| [`np.broadcast_shapes`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_shapes.html) | np | missing | Shape manipulation |
| [`np.busday_count`](https://numpy.org/doc/stable/reference/generated/numpy.busday_count.html) | np | missing | Date & time |
| [`np.busday_offset`](https://numpy.org/doc/stable/reference/generated/numpy.busday_offset.html) | np | missing | Date & time |
| [`np.copysign`](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html) | np | missing | Math |
| [`np.datetime_as_string`](https://numpy.org/doc/stable/reference/generated/numpy.datetime_as_string.html) | np | missing | Date & time |
| [`np.datetime_data`](https://numpy.org/doc/stable/reference/generated/numpy.datetime_data.html) | np | missing | Date & time |
| [`np.divmod`](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html) | np | missing | Math |
| [`np.fabs`](https://numpy.org/doc/stable/reference/generated/numpy.fabs.html) | np | missing | Math |
| [`np.fix`](https://numpy.org/doc/stable/reference/generated/numpy.fix.html) | np | missing | Math |
| [`np.float_power`](https://numpy.org/doc/stable/reference/generated/numpy.float_power.html) | np | missing | Math |
| [`np.fmod`](https://numpy.org/doc/stable/reference/generated/numpy.fmod.html) | np | missing | Math |
| [`np.frexp`](https://numpy.org/doc/stable/reference/generated/numpy.frexp.html) | np | missing | Math |
| [`np.from_dlpack`](https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html) | np | missing | Array creation |
| [`np.fromfunction`](https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html) | np | missing | Array creation |
| [`np.fromiter`](https://numpy.org/doc/stable/reference/generated/numpy.fromiter.html) | np | missing | Array creation |
| [`np.frompyfunc`](https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html) | np | missing | Math |
| [`np.fromregex`](https://numpy.org/doc/stable/reference/generated/numpy.fromregex.html) | np | missing | Array creation |
| [`np.gcd`](https://numpy.org/doc/stable/reference/generated/numpy.gcd.html) | np | missing | Math |
| [`np.genfromtxt`](https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html) | np | missing | Array creation |
| [`np.geomspace`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html) | np | missing | Array creation |
| [`np.get_include`](https://numpy.org/doc/stable/reference/generated/numpy.get_include.html) | np | missing | Runtime & diagnostics |
| [`np.getbufsize`](https://numpy.org/doc/stable/reference/generated/numpy.getbufsize.html) | np | missing | Floating-point handling |
| [`np.geterr`](https://numpy.org/doc/stable/reference/generated/numpy.geterr.html) | np | missing | Floating-point handling |
| [`np.geterrcall`](https://numpy.org/doc/stable/reference/generated/numpy.geterrcall.html) | np | missing | Floating-point handling |
| [`np.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) | np | missing | Math |
| [`np.hamming`](https://numpy.org/doc/stable/reference/generated/numpy.hamming.html) | np | missing | Window functions |
| [`np.hanning`](https://numpy.org/doc/stable/reference/generated/numpy.hanning.html) | np | missing | Window functions |
| [`np.heaviside`](https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html) | np | missing | Math |
| [`np.histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) | np | missing | Statistics & histograms |
| [`np.histogram2d`](https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html) | np | missing | Statistics & histograms |
| [`np.histogram_bin_edges`](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html) | np | missing | Statistics & histograms |
| [`np.histogramdd`](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html) | np | missing | Statistics & histograms |
| [`np.hypot`](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html) | np | missing | Math |
| [`np.i0`](https://numpy.org/doc/stable/reference/generated/numpy.i0.html) | np | missing | Math |
| [`np.info`](https://numpy.org/doc/stable/reference/generated/numpy.info.html) | np | missing | Runtime & diagnostics |

## Case-insensitive near-misses

NumPy's public API is case-sensitive, so a NumSharp member is credited only when the spelling matches exactly. The generator additionally folds case to surface near-misses — in-scope NumPy APIs left *missing* for which NumSharp exposes a same-surface member differing only by case. These are **not** counted as available; rename to the exact NumPy spelling (or record a reviewed alias) to close them.

_None detected._

## Counting rules

The default scope is NumPy top-level callables, ndarray public methods/properties, and callables in numpy.random, numpy.linalg, and numpy.fft. Types, constants, modules, and NumSharp-only APIs remain searchable in the JSON artifact but do not affect the headline percentage.
