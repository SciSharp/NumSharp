using System;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    // ============================== np.gradient ==============================
    // Return the gradient of an N-dimensional array, computed with second-order
    // accurate central differences in the interior and first/second-order
    // one-sided differences at the boundaries. The result has the SAME shape as
    // the input (unlike np.diff, which shrinks the axis).
    //
    // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::gradient
    //
    // This is a faithful port of NumPy's own implementation, which is a pure
    // composition of slicing + arithmetic — no per-element loop of our own; the
    // loops all live in the NumSharp arithmetic operators (NDIter / IL kernels).
    // Each output axis is a fresh `otype` array whose interior and two edges are
    // filled by slice-assignment of the difference expressions.
    //
    // Two dtype/precision rules are ported EXACTLY so the result is bit-identical
    // to NumPy:
    //   * Output dtype (`otype`): a float-family input keeps its dtype
    //     (Half/Single/Double/Complex; NumSharp-only Decimal too); an
    //     integer/char input is promoted to float64 (NumPy converts f to float64
    //     first, matching np.diff-free arithmetic); a BOOLEAN input raises the
    //     same "numpy boolean subtract" TypeError NumPy raises (bool is not an
    //     integer subtype, so NumPy leaves it bool and the internal subtract
    //     fails).
    //   * Spacing precision: UNIFORM spacing feeds the stencil WEAK C# double
    //     coefficients (NEP50 weak scalars, so a float32 input stays float32),
    //     while NON-UNIFORM spacing (a coordinate array) feeds STRONG float64
    //     coefficients (NumPy converts the coordinates to float64), so the
    //     stencil runs in float64 and is cast down to `otype` on store — exactly
    //     as NumPy pre-allocates `out` and assigns the computed slices into it.
    //
    // The API mirrors NumPy's `gradient(f, *varargs, axis=None, edge_order=1)`.
    // C# cannot express NumPy's keyword-only `axis`/`edge_order` alongside the
    // positional `*varargs`, so the spacing is spread over arity overloads
    // (0..3 positional + an explicit object[] for N) and `axis`/`edge_order` are
    // named. Two documented C# resolution quirks fall out of this (both because
    // NumPy makes `axis` keyword-only and C# cannot):
    //   * `gradient(f, 2, 3)` — two BARE INT positional spacings — binds
    //     spacing=2, axis=3 (a single spacing with an integer axis). For two
    //     scalar spacings write doubles `gradient(f, 2.0, 3.0)` or an explicit
    //     `gradient(f, new object[]{2, 3})`.
    //   * A tuple `axis` with NO spacing needs the explicit-array form:
    //     `gradient(f, Array.Empty<object>(), axis: new[]{0, 1})` (a scalar axis
    //     `gradient(f, axis: 0)` works directly).
    public static partial class np
    {
        /// <summary>
        ///     The result of <see cref="gradient(NDArray, object, int?, int)"/>: NumPy returns a
        ///     single ndarray when the gradient is taken along ONE axis and a tuple of ndarrays
        ///     (one per axis) otherwise. C# cannot pick a return type from a runtime count, so this
        ///     value stands in for both — it converts implicitly to <see cref="NDArray"/> (valid only
        ///     for the single-axis result), to <see cref="NDArray"/><c>[]</c> (always),
        ///     <c>Deconstruct</c>s (<c>var (gy, gx) = np.gradient(a);</c>) and indexes (<c>[k]</c>).
        /// </summary>
        public readonly struct GradientResult : INDArrayCarrier
        {
            private readonly NDArray[] _grads;
            private readonly bool _single;

            internal GradientResult(NDArray[] grads, bool single)
            {
                _grads = grads ?? Array.Empty<NDArray>();
                _single = single;
            }

            /// <summary>Number of gradient components (one per requested axis).</summary>
            public int Length => _grads?.Length ?? 0;

            /// <summary>
            ///     True when the gradient was taken along a single axis — NumPy returns a bare
            ///     ndarray then (convertible via the implicit <see cref="NDArray"/> cast), and a
            ///     tuple otherwise.
            /// </summary>
            public bool IsSingle => _single;

            /// <summary>The gradient component along the k-th requested axis.</summary>
            public NDArray this[int index] => (_grads ?? Array.Empty<NDArray>())[index];

            /// <summary>Returns the gradient components as an <see cref="NDArray"/><c>[]</c>.</summary>
            public NDArray[] ToArray() => _grads ?? Array.Empty<NDArray>();

            /// <summary>
            ///     The single-axis gradient. Matches NumPy's bare-ndarray return, valid only when
            ///     exactly one axis was requested; otherwise throws (use the array conversion).
            /// </summary>
            public static implicit operator NDArray(GradientResult result)
            {
                if (!result._single || result.Length != 1)
                    throw new InvalidOperationException(
                        $"np.gradient produced {result.Length} components; it is a tuple, not a single " +
                        "array. Deconstruct it, index it, or convert to NDArray[].");
                return result._grads[0];
            }

            /// <summary>Exposes all gradient components — the tuple NumPy's <c>gradient</c> returns.</summary>
            public static implicit operator NDArray[](GradientResult result) => result.ToArray();

            /// <summary>Deconstructs a two-component result: <c>var (gy, gx) = np.gradient(a);</c></summary>
            public void Deconstruct(out NDArray item1, out NDArray item2)
            {
                EnsureArity(2);
                item1 = _grads[0];
                item2 = _grads[1];
            }

            /// <summary>Deconstructs a three-component result.</summary>
            public void Deconstruct(out NDArray item1, out NDArray item2, out NDArray item3)
            {
                EnsureArity(3);
                item1 = _grads[0];
                item2 = _grads[1];
                item3 = _grads[2];
            }

            private void EnsureArity(int n)
            {
                int have = Length;
                if (have != n)
                    throw new InvalidOperationException(
                        $"np.gradient produced {have} components; cannot deconstruct into {n}. Use " +
                        "indexing or the NDArray[] conversion.");
            }

            void INDArrayCarrier.YieldTo(NDScope scope) => scope.Returns(_grads);
        }

        // ---- Public overloads (spacing spread over arities; axis/edge_order named) ----

        /// <summary>
        ///     Return the gradient of an N-dimensional array using second-order central differences
        ///     in the interior and first/second-order one-sided differences at the boundaries.
        /// </summary>
        /// <param name="f">Input array (samples of a scalar function).</param>
        /// <param name="spacing">
        ///     Optional spacing between samples: a scalar sample distance applied to every requested
        ///     axis, or a 1-D coordinate array giving the sample positions along the (single) requested
        ///     axis. <c>null</c> (the default) means unit spacing.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to take the gradient; <c>null</c> (default) computes it for every
        ///     axis (returning a tuple). Negative axes count from the end.
        /// </param>
        /// <param name="edge_order">Boundary difference order, 1 (default) or 2.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.gradient.html</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="f"/> is <c>null</c>.</exception>
        /// <exception cref="AxisError">An <paramref name="axis"/> is outside <c>[-f.ndim, f.ndim-1]</c>.</exception>
        /// <exception cref="ValueError">
        ///     A repeated <paramref name="axis"/>; <paramref name="edge_order"/> &gt; 2; an axis is shorter
        ///     than <c>edge_order + 1</c>; or a 1-D coordinate spacing whose length or rank does not match
        ///     its axis ("distances must be either scalars or 1d" / "… must match the length …").
        /// </exception>
        /// <exception cref="TypeError">
        ///     The number of positional spacings equals neither 0, 1, nor the number of requested axes
        ///     ("invalid number of arguments").
        /// </exception>
        /// <exception cref="NotSupportedException">
        ///     <paramref name="f"/> is boolean (NumPy rejects the internal subtract).
        /// </exception>
        public static GradientResult gradient(NDArray f, object spacing = null, int? axis = null, int edge_order = 1)
            => GradientImpl(f,
                            spacing is null ? Array.Empty<object>() : new[] { spacing },
                            axis.HasValue ? new[] { axis.Value } : null,
                            edge_order);

        /// <summary>Single spacing with a tuple <paramref name="axis"/> — see the main overload.</summary>
        /// <inheritdoc cref="gradient(NDArray, object, int?, int)" path="/exception"/>
        public static GradientResult gradient(NDArray f, object spacing, int[] axis, int edge_order = 1)
            => GradientImpl(f, new[] { spacing }, axis, edge_order);

        /// <summary>Two per-axis spacings — see the main overload.</summary>
        /// <inheritdoc cref="gradient(NDArray, object, int?, int)" path="/exception"/>
        public static GradientResult gradient(NDArray f, object spacing0, object spacing1, int[] axis = null, int edge_order = 1)
            => GradientImpl(f, new[] { spacing0, spacing1 }, axis, edge_order);

        /// <summary>Three per-axis spacings — see the main overload.</summary>
        /// <inheritdoc cref="gradient(NDArray, object, int?, int)" path="/exception"/>
        public static GradientResult gradient(NDArray f, object spacing0, object spacing1, object spacing2, int[] axis = null, int edge_order = 1)
            => GradientImpl(f, new[] { spacing0, spacing1, spacing2 }, axis, edge_order);

        /// <summary>
        ///     N per-axis spacings passed explicitly (each a scalar or a 1-D coordinate array). Also the
        ///     form for a tuple <paramref name="axis"/> with no spacing: pass an empty array.
        /// </summary>
        /// <inheritdoc cref="gradient(NDArray, object, int?, int)" path="/exception"/>
        public static GradientResult gradient(NDArray f, object[] varargs, int[] axis = null, int edge_order = 1)
            => GradientImpl(f, varargs ?? Array.Empty<object>(), axis, edge_order);

        // ---- Core algorithm (port of numpy.gradient) ----

        [NDScoped]
        private static GradientResult GradientImpl(NDArray f, object[] varargs, int[] axisArg, int edge_order)
        {
            if (f is null) throw new ArgumentNullException(nameof(f));

            int N = f.ndim;

            // axes = normalize_axis_tuple(axis, N)  (all axes when axis is null). Done inline —
            // NumPy's normalize_axis_tuple REJECTS a repeated axis ("repeated axis"), while the
            // shared engine helper silently de-duplicates, so this matches NumPy exactly.
            int[] axes;
            if (axisArg is null)
            {
                axes = new int[N];
                for (int i = 0; i < N; i++) axes[i] = i;
            }
            else
            {
                axes = new int[axisArg.Length];
                for (int i = 0; i < axisArg.Length; i++)
                {
                    int ax = axisArg[i];
                    int adj = ax >= 0 ? ax : ax + N;        // reports the ORIGINAL axis on error
                    if (adj < 0 || adj >= N)
                        throw new AxisError(ax, N);
                    axes[i] = adj;
                }
                var seen = new HashSet<int>();
                foreach (int a in axes)
                    if (!seen.Add(a))
                        throw new ValueError("repeated axis");
            }
            int len_axes = axes.Length;

            // Resolve the spacing for each axis. Each entry is either a boxed double (uniform
            // spacing → weak coefficient) or a float64 1-D NDArray of successive differences
            // (non-uniform spacing → strong coefficients).
            int n = varargs.Length;
            object[] dx;
            if (n == 0)
            {
                dx = new object[len_axes];
                for (int i = 0; i < len_axes; i++) dx[i] = 1.0;              // unit spacing
            }
            else if (n == 1 && GradSpacingNdim(varargs[0]) == 0)
            {
                double s = GradToDouble(varargs[0]);                          // one scalar for all axes
                dx = new object[len_axes];
                for (int i = 0; i < len_axes; i++) dx[i] = s;
            }
            else if (n == len_axes)
            {
                dx = new object[len_axes];
                for (int i = 0; i < len_axes; i++)
                {
                    object v = varargs[i];
                    if (GradSpacingNdim(v) == 0)
                    {
                        dx[i] = GradToDouble(v);                              // per-axis scalar
                        continue;
                    }
                    NDArray distances = np.asanyarray(v);
                    if (distances.ndim != 1)
                        throw new ValueError("distances must be either scalars or 1d");
                    if (distances.shape[0] != f.shape[axes[i]])
                        throw new ValueError(
                            "when 1d, distances must match the length of the corresponding dimension");
                    // Integer coordinates are converted to float64 to avoid modular arithmetic in diff.
                    if (IsIntegerLike(distances.GetTypeCode))
                        distances = distances.astype(np.float64);
                    NDArray diffx = np.diff(distances);
                    // Constant spacing collapses to the scalar (uniform) case — a real speedup and
                    // it matches NumPy bit-for-bit (the whole-array uniform formula runs instead).
                    if (IsConstant(diffx))
                        dx[i] = diffx.GetDouble(0);
                    else
                        dx[i] = diffx;                                        // float64 diffs
                }
            }
            else
            {
                throw new TypeError("invalid number of arguments");
            }

            if (edge_order > 2)
                throw new ValueError("'edge_order' greater than 2 not supported");

            // Output dtype (otype). Integer/char → compute in float64. Boolean → NumPy's subtract
            // TypeError. Float-family (incl. Decimal) keeps its dtype.
            NPTypeCode ftc = f.GetTypeCode;
            if (ftc == NPTypeCode.Boolean)
                throw new NotSupportedException(
                    "numpy boolean subtract, the `-` operator, is not supported, " +
                    "use the bitwise_xor, the `^` operator, or the logical_xor function instead.");
            NDArray work = IsIntegerLike(ftc) ? f.astype(np.float64) : f;
            DType otype = work.dtype;

            var outvals = new NDArray[len_axes];
            for (int ai = 0; ai < len_axes; ai++)
            {
                int axis = axes[ai];
                object ax_dx = dx[ai];
                long M = work.shape[axis];
                if (M < edge_order + 1)
                    throw new ValueError(
                        "Shape of array too small to calculate a numerical gradient, " +
                        "at least (edge_order + 1) elements are required.");

                // Fresh C-contiguous output of `work`'s DIMENSIONS (never `work.Shape`, which would
                // inherit a strided view's strides/offset and mis-place the assigned slices).
                long[] outDims = new long[N];
                for (int d = 0; d < N; d++) outDims[d] = work.shape[d];
                NDArray outp = np.empty(new Shape(outDims), otype);
                bool uniform = ax_dx is double;

                // ---- interior: 2nd-order central differences over [1 : M-1] ----
                //   uniform     : out[1:-1] = (f[2:] - f[:-2]) / (2*dx)
                //   non-uniform : out[1:-1] = a*f[:-2] + b*f[1:-1] + c*f[2:]
                {
                    NDArray fm2 = SliceAlongAxis(work, axis, 0, M - 2);      // f[:-2]
                    NDArray fmid = SliceAlongAxis(work, axis, 1, M - 1);     // f[1:-1]
                    NDArray fp2 = SliceAlongAxis(work, axis, 2, M);          // f[2:]
                    if (uniform)
                    {
                        // Compute straight into the out view via ufunc out= — no intermediate
                        // full-size temp, no slice-copy. dtype:otype forces the loop to `otype`
                        // (so a float32 input divides in float32, matching NumPy's weak scalar).
                        NDArray ov = outp[BuildAxisSlices(N, axis, 1, M - 1)];
                        np.subtract(fp2, fm2, @out: ov, dtype: otype);       // ov = f[2:] - f[:-2]
                        using var half2 = NDArray.Scalar(2.0 * (double)ax_dx);
                        np.divide(ov, half2, @out: ov, dtype: otype);        // ov /= (2*dx)
                        ov.Dispose();
                    }
                    else
                    {
                        NDArray axd = (NDArray)ax_dx;                        // float64 diffs (len M-1)
                        NDArray dx1 = SliceAlongAxis(axd, 0, 0, M - 2);      // ax_dx[:-1]
                        NDArray dx2 = SliceAlongAxis(axd, 0, 1, M - 1);      // ax_dx[1:]
                        NDArray sum = dx1 + dx2;
                        // a = -dx2/(dx1*(dx1+dx2)); b = (dx2-dx1)/(dx1*dx2); c = dx1/(dx2*(dx1+dx2))
                        NDArray a = ReshapeAlong(-(dx2) / (dx1 * sum), axis, N);
                        NDArray b = ReshapeAlong((dx2 - dx1) / (dx1 * dx2), axis, N);
                        NDArray c = ReshapeAlong(dx1 / (dx2 * sum), axis, N);
                        // a*f[:-2] + b*f[1:-1] + c*f[2:] as ONE fused pass straight into the out view
                        // (same left-to-right elementwise order as NumPy, so bit-identical; the f64
                        // result is cast down to `otype` by the out= store for a float32/float16 input).
                        NDArray ov = outp[BuildAxisSlices(N, axis, 1, M - 1)];
                        var expr = NDExpr.Arr(a) * NDExpr.Arr(fm2)
                                 + NDExpr.Arr(b) * NDExpr.Arr(fmid)
                                 + NDExpr.Arr(c) * NDExpr.Arr(fp2);
                        np.evaluate(expr, @out: ov);
                        ov.Dispose();
                        dx1.Dispose(); dx2.Dispose(); sum.Dispose();
                        a.Dispose(); b.Dispose(); c.Dispose();
                    }
                    fm2.Dispose(); fmid.Dispose(); fp2.Dispose();
                }

                // ---- edges ----
                if (edge_order == 1)
                {
                    // out[0]  = (f[1]  - f[0])  / dx_0     (dx_0 = ax_dx    or ax_dx[0])
                    // out[-1] = (f[-1] - f[-2]) / dx_n     (dx_n = ax_dx    or ax_dx[-1])
                    double dx0v = uniform ? (double)ax_dx : ((NDArray)ax_dx).GetDouble(0);
                    double dxnv = uniform ? (double)ax_dx : ((NDArray)ax_dx).GetDouble(((NDArray)ax_dx).size - 1);
                    WriteEdge(outp, work, axis, N, 0,
                        Edge1(SliceAlongAxis(work, axis, 1, 2), SliceAlongAxis(work, axis, 0, 1), dx0v, uniform));
                    WriteEdge(outp, work, axis, N, M - 1,
                        Edge1(SliceAlongAxis(work, axis, M - 1, M), SliceAlongAxis(work, axis, M - 2, M - 1), dxnv, uniform));
                }
                else
                {
                    // 2nd-order one-sided edges.
                    double a0, b0, c0, a1, b1, c1;
                    if (uniform)
                    {
                        double s = (double)ax_dx;
                        a0 = -1.5 / s; b0 = 2.0 / s; c0 = -0.5 / s;
                        a1 = 0.5 / s; b1 = -2.0 / s; c1 = 1.5 / s;
                    }
                    else
                    {
                        NDArray axd = (NDArray)ax_dx;
                        double d1 = axd.GetDouble(0), d2 = axd.GetDouble(1);
                        a0 = -(2.0 * d1 + d2) / (d1 * (d1 + d2));
                        b0 = (d1 + d2) / (d1 * d2);
                        c0 = -d1 / (d2 * (d1 + d2));
                        long L = axd.size;
                        double e1 = axd.GetDouble(L - 2), e2 = axd.GetDouble(L - 1);
                        a1 = e2 / (e1 * (e1 + e2));
                        b1 = -(e2 + e1) / (e1 * e2);
                        c1 = (2.0 * e2 + e1) / (e2 * (e1 + e2));
                    }
                    // front: out[0] = a0*f[0] + b0*f[1] + c0*f[2]
                    WriteEdge(outp, work, axis, N, 0,
                        Edge2(SliceAlongAxis(work, axis, 0, 1), SliceAlongAxis(work, axis, 1, 2),
                              SliceAlongAxis(work, axis, 2, 3), a0, b0, c0, uniform));
                    // back: out[-1] = a1*f[-3] + b1*f[-2] + c1*f[-1]
                    WriteEdge(outp, work, axis, N, M - 1,
                        Edge2(SliceAlongAxis(work, axis, M - 3, M - 2), SliceAlongAxis(work, axis, M - 2, M - 1),
                              SliceAlongAxis(work, axis, M - 1, M), a1, b1, c1, uniform));
                }

                outvals[ai] = outp;
            }

            return new GradientResult(outvals, single: len_axes == 1);
        }

        // out[0]/out[-1] = (hi - lo) / d, weak (uniform) or strong (non-uniform) divisor.
        private static NDArray Edge1(NDArray hi, NDArray lo, double d, bool uniform)
        {
            NDArray num = hi - lo;
            NDArray res = uniform ? num / d : num / NDArray.Scalar(d);   // strong 0-d scalar off the uniform path
            num.Dispose(); hi.Dispose(); lo.Dispose();
            return res;
        }

        // a*f0 + b*f1 + c*f2, coefficients weak (uniform) or strong 0-d scalars (non-uniform).
        private static NDArray Edge2(NDArray f0, NDArray f1, NDArray f2, double a, double b, double c, bool uniform)
        {
            NDArray res;
            if (uniform)
            {
                NDArray t0 = f0 * a, t1 = f1 * b, t2 = f2 * c;
                NDArray s01 = t0 + t1; res = s01 + t2;
                t0.Dispose(); t1.Dispose(); t2.Dispose(); s01.Dispose();
            }
            else
            {
                NDArray na = NDArray.Scalar(a), nb = NDArray.Scalar(b), nc = NDArray.Scalar(c);
                NDArray t0 = na * f0, t1 = nb * f1, t2 = nc * f2;
                NDArray s01 = t0 + t1; res = s01 + t2;
                na.Dispose(); nb.Dispose(); nc.Dispose();
                t0.Dispose(); t1.Dispose(); t2.Dispose(); s01.Dispose();
            }
            f0.Dispose(); f1.Dispose(); f2.Dispose();
            return res;
        }

        // Assign a length-1 edge slice at index `at` along `axis` into `outp`, then dispose the value.
        private static void WriteEdge(NDArray outp, NDArray work, int axis, int N, long at, NDArray value)
        {
            outp[BuildAxisSlices(N, axis, at, at + 1)] = value;
            value.Dispose();
        }

        // Reshape a 1-D coefficient array to broadcast along `axis`: shape is all-1 except `axis`.
        private static NDArray ReshapeAlong(NDArray a, int axis, int N)
        {
            long[] shp = new long[N];
            for (int i = 0; i < N; i++) shp[i] = 1;
            shp[axis] = a.shape[0];
            var r = a.reshape(new Shape(shp));
            a.Dispose();
            return r;
        }

        // Slice spec [:, ..., start:stop, ..., :] with a length-(stop-start) range on `axis`.
        private static Slice[] BuildAxisSlices(int N, int axis, long start, long stop)
        {
            var slices = new Slice[N];
            for (int i = 0; i < N; i++)
                slices[i] = i == axis ? new Slice(start, stop) : Slice.All;
            return slices;
        }

        // ndim of a spacing argument: 0 for a C# scalar, else the NDArray's ndim.
        private static int GradSpacingNdim(object v)
        {
            if (v is null) throw new ArgumentNullException(nameof(v));
            if (v is NDArray nd) return nd.ndim;
            if (v is Array arr) return arr.Rank;    // a C# array is a coordinate list (ndim ≥ 1)
            return 0;                               // a scalar
        }

        // Convert a scalar spacing to double.
        private static double GradToDouble(object v)
        {
            if (v is NDArray nd) return nd.GetDouble(0);
            return Convert.ToDouble(v);
        }

        private static bool IsIntegerLike(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.UInt16
            or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
            or NPTypeCode.Char => true,
            _ => false,
        };

        // True when every element equals the first (NumPy's `(diffx == diffx[0]).all()`).
        private static bool IsConstant(NDArray diffx)
        {
            long n = diffx.size;
            if (n <= 1) return true;
            double first = diffx.GetDouble(0);
            for (long i = 1; i < n; i++)
                if (diffx.GetDouble(i) != first) return false;
            return true;
        }
    }
}
