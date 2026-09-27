using System;
using System.Numerics;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    /// <summary>
    ///     NumPy 2.4.2's Python layer of the <c>numpy.polynomial</c> evaluation family — <c>{p}val</c>,
    ///     <c>polyutils._valnd</c> and <c>polyutils._gridnd</c> — over the Tier-3A IL kernels of
    ///     <c>ILKernelGenerator.Polynomial*.cs</c>. Every basis goes through the same three methods; the basis
    ///     is data (<see cref="PolyBasis"/>), so there is no per-basis and no per-dtype code here either.
    /// </summary>
    /// <remarks>
    ///     <para>What NumPy does and this reproduces, line for line:</para>
    ///     <list type="number">
    ///       <item><c>c = np.array(c, ndmin=1)</c>; an integer or bool series becomes float64
    ///             (<c>c.dtype.char in '?bBhHiIlLqQpP'</c>).</item>
    ///       <item>A tuple/list x becomes an array; a Python scalar x STAYS a Python scalar (a NEP 50 weak
    ///             value — a C# primitive here), so <c>chebval(2.0, float32_c)</c> is float32.</item>
    ///       <item>An array x with <c>tensor=True</c> reshapes c to <c>c.shape + (1,)*x.ndim</c>: every series
    ///             is evaluated at every point, result shape <c>c.shape[1:] + x.shape</c>; with
    ///             <c>tensor=False</c> x broadcasts over the columns of c.</item>
    ///       <item>An empty series raises <c>IndexError</c> at NumPy's first coefficient read
    ///             (<c>c[-1]</c> for polyval, <c>c[-2]</c> for the Clenshaw bases).</item>
    ///     </list>
    /// </remarks>
    internal static unsafe class NDPolyEval
    {
        /// <summary>
        ///     <c>{p}val(x, c, tensor)</c> for any basis (see the class remarks for the NumPy semantics).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="x">Points: an <see cref="NDArray"/>, a C# array (converted like NumPy's list), or a C#
        ///     primitive (a Python scalar: <c>bool</c>, the integer types, <c>float</c>/<c>double</c>,
        ///     <see cref="Complex"/>). <c>char</c>/<see cref="Half"/>/<c>decimal</c> have no Python literal and
        ///     become strong 0-d arrays, the house convention.</param>
        /// <param name="c">Coefficients, the series along axis 0 (low degree first).</param>
        /// <param name="tensor">NumPy's <c>tensor</c> flag.</param>
        /// <returns>The values; a 0-d array for a scalar x with a 1-D series.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> or <paramref name="c"/> is null.</exception>
        /// <exception cref="IndexError">The series is empty.</exception>
        /// <exception cref="IncorrectShapeException"><c>tensor=False</c> and <c>c.shape[1:]</c> does not broadcast
        ///     with <c>x.shape</c> (NumPy's ValueError text, house exception type).</exception>
        /// <exception cref="OverflowException">A Python int in the recurrence does not fit an integer x dtype
        ///     (e.g. lagval with 70 coefficients on int8 x).</exception>
        internal static NDArray Val(PolyBasis basis, object x, NDArray c, bool tensor)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (c is null) throw new ArgumentNullException(nameof(c));

            NDArray expanded = null, converted = null;
            try
            {
                // c = np.array(c, ndmin=1); integer/bool series -> float64. A view keeps pointing into the
                // caller's buffer (the kernel only reads it); only the float64 conversion allocates.
                NDArray cc = c.ndim == 0 ? (expanded = np.expand_dims(c, 0)) : c;
                if (PolyTyping.IsIntLike(cc.typecode))
                    cc = converted = cc.astype(NPTypeCode.Double);

                NDArray xa = ClassifyX(x, out var xw, out bool weak);
                return Evaluate(basis, xa, xw, weak, cc, tensor);
            }
            finally
            {
                converted?.Dispose();
                expanded?.Dispose();
            }
        }

        /// <summary>
        ///     <c>polyutils._valnd(val_f, c, *args)</c>: every ordinate is <c>np.asanyarray</c>'d (so a scalar is a
        ///     STRONG 0-d array here, unlike <c>{p}val</c>), all shapes must be equal, then the first ordinate is
        ///     evaluated with <c>tensor=True</c> and every later one with <c>tensor=False</c> over the previous
        ///     result — NumPy's exact two-pass intermediate.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">Coefficients; axis i is ordinate i's degree.</param>
        /// <param name="args">The ordinates (x, y[, z]).</param>
        /// <returns>The values, shape of the ordinates (plus <c>c.shape[len(args):]</c>).</returns>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="IndexError"><paramref name="args"/> is empty — NumPy's <c>args[0]</c> on an empty list.</exception>
        /// <exception cref="ValueError">The ordinates differ in shape: <c>x, y are incompatible</c>,
        ///     <c>x, y, z are incompatible</c> or (any other count) <c>ordinates are incompatible</c>.</exception>
        internal static NDArray ValNd(PolyBasis basis, NDArray c, NDArray[] args)
        {
            if (c is null) throw new ArgumentNullException(nameof(c));
            if (args is null) throw new ArgumentNullException(nameof(args));
            if (args.Length == 0) throw new IndexError("list index out of range");
            for (int i = 0; i < args.Length; i++)
                if (args[i] is null) throw new ArgumentNullException(nameof(args));

            var shape0 = args[0].shape;
            for (int i = 1; i < args.Length; i++)
            {
                if (!SameShape(args[i].shape, shape0))
                    throw new ValueError(args.Length switch
                    {
                        3 => "x, y, z are incompatible",
                        2 => "x, y are incompatible",
                        _ => "ordinates are incompatible",
                    });
            }

            // use tensor on only the first
            NDArray r = Val(basis, args[0], c, tensor: true);
            for (int i = 1; i < args.Length; i++)
            {
                var next = Val(basis, args[i], r, tensor: false);
                r.Dispose();   // each intermediate is ours; the caller's c is never disposed
                r = next;
            }
            return r;
        }

        /// <summary>
        ///     <c>polyutils._gridnd(val_f, c, *args)</c>: <c>c = val_f(xi, c)</c> for each ordinate, with
        ///     <c>tensor=True</c> and WITHOUT converting the ordinates — a Python-scalar ordinate stays weak.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">Coefficients.</param>
        /// <param name="args">The ordinates, each anything <see cref="Val"/> accepts.</param>
        /// <returns>The grid, shape <c>x.shape + y.shape[ + z.shape]</c> (plus trailing coefficient axes).</returns>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        internal static NDArray GridNd(PolyBasis basis, NDArray c, object[] args)
        {
            if (c is null) throw new ArgumentNullException(nameof(c));
            if (args is null) throw new ArgumentNullException(nameof(args));
            NDArray r = c;
            for (int i = 0; i < args.Length; i++)
            {
                var next = Val(basis, args[i], r, tensor: true);
                if (!ReferenceEquals(r, c)) r.Dispose();
                r = next;
            }
            // _gridnd with no ordinates returns c itself (NumPy hands back the same object).
            return r;
        }

        /// <summary>
        ///     Maps a C# argument onto NumPy's x forms: an <see cref="NDArray"/> stays an array; the C# primitives
        ///     that ARE Python literals become weak Python scalars (the <c>np.r_</c> house convention: bool,
        ///     the integer types, float/double and <see cref="Complex"/>); everything else goes through
        ///     <c>np.asanyarray</c> (C# arrays ≙ Python lists; char/Half/decimal become strong 0-d arrays).
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <param name="weakValue">The Python value when weak.</param>
        /// <param name="isWeak">Whether x is a Python scalar.</param>
        /// <returns>The array form, or null when weak.</returns>
        private static NDArray ClassifyX(object x, out PyScalar weakValue, out bool isWeak)
        {
            weakValue = default;
            isWeak = true;
            switch (x)
            {
                case NDArray nd: isWeak = false; return nd;
                // A Python bool in the step tables only ever meets Python-int arithmetic (2*x, x*0, 1 - x)
                // or an inexact series (c1*x), where it behaves exactly as the int 0/1 (PyKind remarks).
                case bool b: weakValue = PyScalar.Int(b ? 1 : 0); return null;
                case sbyte v: weakValue = PyScalar.Int(v); return null;
                case byte v: weakValue = PyScalar.Int(v); return null;
                case short v: weakValue = PyScalar.Int(v); return null;
                case ushort v: weakValue = PyScalar.Int(v); return null;
                case int v: weakValue = PyScalar.Int(v); return null;
                case uint v: weakValue = PyScalar.Int(v); return null;
                case long v: weakValue = PyScalar.Int(v); return null;
                case ulong v: weakValue = PyScalar.Int(new BigInteger(v)); return null;
                case float v: weakValue = PyScalar.Float(v); return null;   // the float's exact value as a Python float
                case double v: weakValue = PyScalar.Float(v); return null;
                case Complex v: weakValue = PyScalar.Cplx(v); return null;
                default:
                    isWeak = false;
                    return np.asanyarray(x);
            }
        }

        /// <summary>The evaluation proper (x classified, c normalized).</summary>
        /// <param name="basis">The basis.</param>
        /// <param name="xa">x as an array (null when weak).</param>
        /// <param name="xw">x as a Python value (when weak).</param>
        /// <param name="weak">x is a Python scalar.</param>
        /// <param name="c">Coefficients, at least 1-D, never int-like.</param>
        /// <param name="tensor">NumPy's tensor flag.</param>
        /// <returns>The values.</returns>
        /// <exception cref="IndexError">Empty series.</exception>
        /// <exception cref="IncorrectShapeException">tensor=False broadcast mismatch.</exception>
        /// <exception cref="OverflowException">A Python int does not fit an integer x dtype.</exception>
        private static NDArray Evaluate(PolyBasis basis, NDArray xa, PyScalar xw, bool weak, NDArray c, bool tensor)
        {
            long nc = c.shape[0];
            if (nc == 0)
                throw new IndexError($"index {(basis == PolyBasis.Power ? -1 : -2)} is out of bounds for axis 0 with size 0");

            bool coefOperand = c.ndim > 1;
            var xMode = weak ? PolyXMode.Weak : xa.ndim == 0 ? PolyXMode.Shared : PolyXMode.PerPoint;
            NPTypeCode tx = weak ? NPTypeCode.Empty : xa.typecode;
            bool scalarMath = !coefOperand && xMode != PolyXMode.PerPoint;   // every value is a NumPy scalar
            var unit = UnitBroadcast(c, xa, xMode, tx, tensor);             // NumPy's one-element NpyIter products
            var k = ILKernelGenerator.GetPolyEvalKernel(basis, xMode, tx, xw.Kind, c.typecode, nc, coefOperand, scalarMath, unit);

            IntPtr scratch = IntPtr.Zero;
            NDArray preconv = null, c0 = null, y = null;
            try
            {
                long* aux = stackalloc long[7];
                int csize = c.dtypesize;
                aux[0] = c.Shape.strides[0] * csize;
                aux[1] = nc;
                // Python ints that do not fit an integer x raise here — before any shape check, as in NumPy,
                // where lagval's `(2*nd - 1) - x` runs before its first c1*(...) broadcast.
                aux[2] = k.Pool.Prepare(nc, xw, out scratch);
                aux[3] = (long)((byte*)c.Storage.Address + c.Shape.offset * csize);
                if (k.Preconv)
                {
                    // Steady-loop copy in the loop dtype: nc elements, NumPy's own widening (exact).
                    preconv = c.astype(k.Tl);
                    aux[4] = (long)((byte*)preconv.Storage.Address + preconv.Shape.offset * preconv.dtypesize);
                    aux[5] = preconv.Shape.strides[0] * preconv.dtypesize;
                }
                if (xMode == PolyXMode.Shared)
                    aux[6] = (long)((byte*)xa.Storage.Address + xa.Shape.offset * xa.dtypesize);

                if (!coefOperand)
                {
                    if (xMode != PolyXMode.PerPoint)
                    {
                        // 0-d result: one point, no iterator — the kernel's only operand is y.
                        y = new NDArray(k.Tl, Shape.NewScalar(), false);
                        void** ptrs = stackalloc void*[1];
                        long* strides = stackalloc long[1];
                        ptrs[0] = (byte*)y.Storage.Address;
                        strides[0] = 0;
                        k.Fn(ptrs, strides, 1, aux);
                        return Detach(ref y);
                    }

                    // 1-D series at an array x: result has x's shape, allocated in x's memory order (NumPy's
                    // ufunc order='K' output) so the iterator walks both contiguously.
                    y = new NDArray(k.Tl, LikeKeepOrder(xa.Shape), false);
                    if (y.size == 0) return Detach(ref y);
                    var gflags = NDIterGlobalFlags.EXTERNAL_LOOP;
                    var xflags = NDIterPerOpFlags.READONLY;
                    bool buffered = !IsElementContiguousInOrder(xa, y);
                    if (buffered)
                    {
                        // A strided x is gathered into contiguous chunks so the vector chains run (2.7-3.0x
                        // over strided scalar chains); the coefficients live in aux, so buffering is safe.
                        gflags |= NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.GROWINNER;
                        xflags |= NDIterPerOpFlags.CONTIG;
                    }
                    using (var iter = NDIterRef.MultiNew(2, new[] { xa, y }, gflags, NPY_ORDER.NPY_KEEPORDER,
                               NPY_CASTING.NPY_SAFE_CASTING, new[] { xflags, NDIterPerOpFlags.WRITEONLY },
                               buffered ? new[] { xa.typecode, k.Tl } : null))
                        iter.ForEach(k.Fn, aux);
                    return Detach(ref y);
                }

                // N-D series: c0 is a view of c[0]; the kernel reads c[k] at c0 + k*kstride.
                var cOuter = c.Shape.dimensions.AsSpan(1).ToArray();
                if (xMode == PolyXMode.PerPoint)
                {
                    long[] outDims;
                    if (tensor)
                    {
                        // c.reshape(c.shape + (1,)*x.ndim): c0 gains x.ndim trailing unit axes (a view — adding
                        // unit axes never needs a copy), so it broadcasts against x's axes.
                        var axes = new int[xa.ndim];
                        for (int d = 0; d < axes.Length; d++) axes[d] = cOuter.Length + d;
                        using (var row0 = c[0])
                            c0 = np.expand_dims(row0, axes);   // the expanded view holds its own reference
                        outDims = new long[cOuter.Length + xa.ndim];
                        cOuter.CopyTo(outDims, 0);
                        xa.Shape.dimensions.CopyTo(outDims, cOuter.Length);
                    }
                    else
                    {
                        c0 = c[0];
                        outDims = BroadcastDims(c0.Shape, xa.Shape);
                    }
                    y = new NDArray(k.Tl, new Shape(outDims), false);   // dims only: never a view's Shape
                    if (y.size == 0) return Detach(ref y);
                    // NEVER buffered: the kernel reads beyond c0 (see the kernel file header).
                    using (var iter = NDIterRef.MultiNew(3, new[] { xa, c0, y }, NDIterGlobalFlags.EXTERNAL_LOOP,
                               NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                               new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, null))
                        iter.ForEach(k.Fn, aux);
                    return Detach(ref y);
                }

                // Scalar x (weak or 0-d) over an N-D series: the points are c's columns, shape c.shape[1:].
                c0 = c[0];
                y = new NDArray(k.Tl, new Shape(cOuter), false);
                if (y.size == 0) return Detach(ref y);
                using (var iter = NDIterRef.MultiNew(2, new[] { c0, y }, NDIterGlobalFlags.EXTERNAL_LOOP,
                           NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                           new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, null))
                    iter.ForEach(k.Fn, aux);
                return Detach(ref y);
            }
            finally
            {
                if (scratch != IntPtr.Zero) NativeMemory.Free((void*)scratch);
                preconv?.Dispose();
                c0?.Dispose();
                y?.Dispose();   // non-null only when an exception escaped before Detach
            }
        }

        /// <summary>
        ///     Whether this call is NumPy's single-element broadcast (<see cref="PolyUnitBroadcast"/>), and which side
        ///     is deeper. It is when four things hold: the series is N-D, x is a per-point array, the loop is
        ///     complex, and the result has ONE element while <c>c[k]</c> and x differ in ndim. NumPy then runs each
        ///     complex product of a coefficient-derived and an x-derived array on NpyIter's one-element loop, i.e.
        ///     <c>CDOUBLE_multiply</c>'s contracted fallback, not <c>simd_cmul</c>.
        /// </summary>
        /// <param name="c">Coefficients, at least 1-D.</param>
        /// <param name="xa">x as an array (null when weak — never read then).</param>
        /// <param name="xMode">How x arrives.</param>
        /// <param name="tx">x's dtype.</param>
        /// <param name="tensor">NumPy's tensor flag: a tensor evaluation appends <c>x.ndim</c> unit axes to every
        ///     <c>c[k]</c> (<c>c.reshape(c.shape + (1,)*x.ndim)</c>), so the series side is then always deeper.</param>
        /// <returns>The mode; <see cref="PolyUnitBroadcast.None"/> for every other call, so the common calls keep their
        ///     kernels.</returns>
        /// <remarks>
        ///     Only complex products have more than one NumPy form (a real product is one IEEE multiply on every
        ///     path), and a {p}val loop is complex exactly when x or the series is — the step tables hold no complex
        ///     constants and a weak x is never per point. The result has one element exactly when every axis of
        ///     <c>c.shape[1:]</c> and of x is 1: with <c>tensor=True</c> the result shape is their concatenation,
        ///     with <c>tensor=False</c> their broadcast.
        /// </remarks>
        private static PolyUnitBroadcast UnitBroadcast(NDArray c, NDArray xa, PolyXMode xMode, NPTypeCode tx, bool tensor)
        {
            if (c.ndim < 2 || xMode != PolyXMode.PerPoint) return PolyUnitBroadcast.None;
            if (tx != NPTypeCode.Complex && c.typecode != NPTypeCode.Complex) return PolyUnitBroadcast.None;
            var cd = c.Shape.dimensions; var xd = xa.Shape.dimensions;
            for (int i = 1; i < cd.Length; i++) if (cd[i] != 1) return PolyUnitBroadcast.None;
            for (int i = 0; i < xd.Length; i++) if (xd[i] != 1) return PolyUnitBroadcast.None;
            int dc = c.ndim - 1 + (tensor ? xa.ndim : 0), dx = xa.ndim;
            return dc == dx ? PolyUnitBroadcast.None : dc > dx ? PolyUnitBroadcast.CoefDeeper : PolyUnitBroadcast.PointsDeeper;
        }

        /// <summary>Hands the result out of the <c>finally</c>'s cleanup: clears the local, returns the array.</summary>
        /// <param name="y">The result local.</param><returns>The result.</returns>
        private static NDArray Detach(ref NDArray y)
        {
            var r = y;
            y = null;
            return r;
        }

        /// <summary>
        ///     The shape NumPy's ufunc allocates for an <c>order='K'</c> output mirroring <paramref name="s"/>
        ///     (<c>PyArray_NewLikeArray</c>): C- or F-contiguous sources keep that order, a neither-contiguous one
        ///     its sorted stride permutation (<see cref="Shape.KeepOrder"/>), all dense with offset 0.
        /// </summary>
        /// <param name="s">The source shape.</param><returns>A fresh owned-allocation shape.</returns>
        private static Shape LikeKeepOrder(Shape s)
        {
            if (s.NDim == 0) return Shape.NewScalar();
            if (!s.IsContiguous && !s.IsFContiguous) return s.KeepOrder();
            return new Shape((long[])s.dimensions.Clone(), OrderResolver.Resolve('K', s));
        }

        /// <summary>
        ///     Whether x is dense with the SAME memory order as the freshly allocated y — then the iterator
        ///     coalesces both into contiguous chunks and buffering would only add a copy (measured ~20%).
        /// </summary>
        /// <param name="x">Points.</param><param name="y">The result (allocated by <see cref="LikeKeepOrder"/>).</param>
        /// <returns>True when x needs no gathering.</returns>
        private static bool IsElementContiguousInOrder(NDArray x, NDArray y)
        {
            var xs = x.Shape; var ys = y.Shape;
            if (xs.IsContiguous && ys.IsContiguous) return true;
            if (xs.IsFContiguous && ys.IsFContiguous) return true;
            // Same element strides as the dense y (a neither-C-nor-F permutation KeepOrder mirrored exactly):
            // x is then dense in the same order. A unit axis's stride is irrelevant to the walk.
            var a = xs.strides; var b = ys.strides;
            if (a.Length != b.Length) return false;
            for (int d = 0; d < a.Length; d++)
                if (a[d] != b[d] && xs.dimensions[d] != 1) return false;
            return true;
        }

        /// <summary>NumPy's right-aligned broadcast of the <c>tensor=False</c> operands, with NumPy's ValueError text.</summary>
        /// <param name="c0">The column shape <c>c.shape[1:]</c>.</param><param name="x">x's shape.</param>
        /// <returns>The broadcast dims.</returns>
        /// <exception cref="IncorrectShapeException">Incompatible shapes: "operands could not be broadcast together
        ///     with shapes (2,) (3,) " — the column shape first, as NumPy's <c>c1*x</c> reports it.</exception>
        private static long[] BroadcastDims(Shape c0, Shape x)
        {
            var a = c0.dimensions; var b = x.dimensions;
            int n = Math.Max(a.Length, b.Length);
            var r = new long[n];
            for (int d = 0; d < n; d++)
            {
                long da = d < n - a.Length ? 1 : a[d - (n - a.Length)];
                long db = d < n - b.Length ? 1 : b[d - (n - b.Length)];
                if (da != db && da != 1 && db != 1)
                    throw new IncorrectShapeException(
                        $"operands could not be broadcast together with shapes {c0.ToPythonTuple()} {x.ToPythonTuple()} ");
                r[d] = da == 1 ? db : da;
            }
            return r;
        }

        /// <summary>Element-wise shape equality (NumPy's tuple <c>==</c>).</summary>
        /// <param name="a">First.</param><param name="b">Second.</param><returns>True when equal.</returns>
        private static bool SameShape(long[] a, long[] b)
        {
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i]) return false;
            return true;
        }
    }
}
