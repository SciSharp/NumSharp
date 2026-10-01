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
        ///     <c>{p}val(x, c, tensor)</c> with an array_like <paramref name="c"/>: NumPy's <c>np.array(c, ndmin=1)</c> of
        ///     any argument kind, done FIRST — before x is looked at, NumPy's statement order, so a ragged c is reported
        ///     before a ragged x — then the evaluation of <see cref="Val(PolyBasis, object, NDArray, bool)"/>.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="x">Points (see <see cref="Val(PolyBasis, object, NDArray, bool)"/>).</param>
        /// <param name="c">Coefficients, anything <c>np.array</c> accepts under the house mapping: an <see cref="NDArray"/>
        ///     (used as is), a typed C# array or <c>Memory&lt;T&gt;</c> of a dtype (an ndarray), a Python tuple / list (a
        ///     ValueTuple, <c>object[]</c>, a jagged or <c>NDArray[]</c> array, any enumerable — coerced by
        ///     <see cref="PolySequence"/>, nested to any depth), or a scalar (a Python scalar keeps NumPy's discovered
        ///     dtype: an int is int64, and integer / bool series are evaluated as float64 either way).</param>
        /// <param name="tensor">NumPy's <c>tensor</c> flag.</param>
        /// <returns>The values; a 0-d array for a scalar x with a 1-D series.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError">A ragged tuple / list c or x (np.array's inhomogeneous-shape text; c's first).</exception>
        /// <exception cref="NotSupportedException">A null or str c, or a sequence holding one (NumPy builds an object /
        ///     str array, dtypes NumSharp does not have).</exception>
        /// <exception cref="IndexError">The series is empty.</exception>
        /// <exception cref="IncorrectShapeException"><c>tensor=False</c> and the shapes do not broadcast.</exception>
        /// <exception cref="OverflowException">A Python int in the recurrence does not fit an integer x dtype.</exception>
        internal static NDArray Val(PolyBasis basis, object x, object c, bool tensor)
        {
            if (c is NDArray nd)
                return Val(basis, x, nd, tensor);
            // The converted series is this call's intermediate: the evaluation always returns a fresh array — never c, x
            // or a view of either — so it is released however the call ends.
            NDArray cc = ArrayOf(c);
            try
            {
                return Val(basis, x, cc, tensor);
            }
            finally
            {
                cc.Dispose();
            }
        }

        /// <summary>
        ///     <c>np.array(a, ndmin=1)</c> of an argument that is not an <see cref="NDArray"/> — a series — under the house
        ///     mapping (<see cref="NDPolySeries.AsCoefficientArray"/>): a typed C# array is an ndarray of its dtype, a Python
        ///     tuple / list goes through np.array's coercion (<see cref="PolySequence"/>), and a Python scalar becomes a
        ///     one-element 1-D array of NumPy's discovered dtype (int64 / float64 / complex128 / bool), a NumPy scalar (Half,
        ///     char, decimal) one of its own. An ordinate, which keeps a scalar 0-D, goes through <see cref="OrdinateArray"/>.
        /// </summary>
        /// <param name="a">The argument (not an NDArray).</param>
        /// <returns>A NEW array the caller owns and must dispose.</returns>
        /// <exception cref="ValueError">A ragged tuple / list.</exception>
        /// <exception cref="NotSupportedException">null (NumPy's object array) or a str (NumPy's str array), or a sequence
        ///     holding one.</exception>
        private static NDArray ArrayOf(object a)
        {
            var v = NDPolySeries.AsCoefficientArray(a);
            // A str / object array is refused as soon as it exists: evaluation computes with it right away (a str series
            // is not rejected by any check first). The conversion's own refusal names the item that made it.
            if (v.NonNumeric)
                throw v.Refusal ?? new NotSupportedException("a str argument makes NumPy build a str array, a dtype NumSharp does not have");
            return v.Source;
        }

        /// <summary>
        ///     <c>np.asanyarray(a)</c> of an ordinate that is not an <see cref="NDArray"/> (<c>_valnd</c>'s conversion): a
        ///     Python or NumPy scalar becomes a 0-D array of the dtype np.array discovers for it (a Python int is int64 —
        ///     uint64 when it only fits there —, a float float64, a complex complex128, a bool bool; Half / char / decimal
        ///     keep theirs), so it is a STRONG operand; anything else converts as <see cref="ArrayOf"/> does.
        /// </summary>
        /// <param name="a">The ordinate (not null, not an NDArray).</param>
        /// <returns>A NEW array the caller owns and must dispose.</returns>
        /// <exception cref="ValueError">A ragged tuple / list.</exception>
        /// <exception cref="NotSupportedException">A str, a sequence holding a str / null, or a Python int past uint64
        ///     (NumPy's str / object arrays).</exception>
        private static NDArray OrdinateArray(object a)
        {
            if (!PolyNumber.IsScalarValue(a))
                return ArrayOf(a);
            // Not ArrayOf: that is np.array(a, ndmin=1), which would make the scalar a one-element 1-D array.
            var num = PolyNumber.FromObject(a);
            return num.AsArrayOf(num.DiscoveredDtype());
        }

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

            NDArray expanded = null, converted = null, convertedX = null;
            try
            {
                // c = np.array(c, ndmin=1); integer/bool series -> float64. A view keeps pointing into the
                // caller's buffer (the kernel only reads it); only the float64 conversion allocates.
                NDArray cc = c.ndim == 0 ? (expanded = np.expand_dims(c, 0)) : c;
                bool intLike = PolyTyping.IsIntLike(cc.typecode);
                // The layout NumPy's recurrence reads c[k] through — taken BEFORE NumSharp's own float64 conversion,
                // whose layout is NumSharp's business; the result's layout is decided from NumPy's (see SeriesLayout).
                Shape cLayout = SeriesLayout(basis, cc.Shape, intLike);
                if (intLike)
                    cc = converted = cc.astype(NPTypeCode.Double);

                NDArray xa = ClassifyX(x, out var xw, out bool weak);
                // An x array built HERE (a Python sequence, a typed C# array, a Half / char / decimal scalar) is this
                // call's intermediate, never the caller's: Evaluate always returns a fresh result, not x or a view of
                // it, so the conversion is released with the others. A caller's NDArray comes back as itself.
                if (xa is not null && !ReferenceEquals(xa, x))
                    convertedX = xa;
                return Evaluate(basis, xa, xw, weak, cc, cLayout, tensor);
            }
            finally
            {
                convertedX?.Dispose();
                converted?.Dispose();
                expanded?.Dispose();
            }
        }

        /// <summary>
        ///     <c>polyutils._valnd(val_f, c, *args)</c>: every ordinate is <c>np.asanyarray</c>'d first (so a scalar is a
        ///     STRONG 0-d array here, unlike <c>{p}val</c>'s x, and a tuple / list goes through np.array's coercion), all
        ///     shapes must be equal, then the first ordinate is evaluated with <c>tensor=True</c> — which is where c is
        ///     converted — and every later one with <c>tensor=False</c> over the previous result: NumPy's exact two-pass
        ///     intermediate, and its error order (a ragged ordinate, then the shape check, then c).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">Coefficients, any array_like (see <see cref="Val(PolyBasis, object, object, bool)"/>); axis i is
        ///     ordinate i's degree.</param>
        /// <param name="args">The ordinates (x, y[, z]): each an <see cref="NDArray"/> (used as is), a typed C# array, a
        ///     Python tuple / list, or a scalar (a Python int becomes an int64 0-d array, NumPy's discovered dtype).</param>
        /// <returns>The values, shape of the ordinates (plus <c>c.shape[len(args):]</c>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="args"/> or an ordinate is null.</exception>
        /// <exception cref="IndexError"><paramref name="args"/> is empty — NumPy's <c>args[0]</c> on an empty list.</exception>
        /// <exception cref="ValueError">A ragged tuple / list ordinate or c (np.array's inhomogeneous-shape text), or
        ///     ordinates that differ in shape: <c>x, y are incompatible</c>, <c>x, y, z are incompatible</c> or (any other
        ///     count) <c>ordinates are incompatible</c>.</exception>
        /// <exception cref="NotSupportedException">A null / str c, or a str ordinate (NumPy's object / str arrays).</exception>
        internal static NDArray ValNd(PolyBasis basis, object c, object[] args)
        {
            if (args is null) throw new ArgumentNullException(nameof(args));
            if (args.Length == 0) throw new IndexError("list index out of range");
            for (int i = 0; i < args.Length; i++)
                if (args[i] is null) throw new ArgumentNullException(nameof(args));

            // args = [np.asanyarray(a) for a in args]: all converted before anything else is looked at, in order, so a
            // ragged ordinate raises here — before the shape check and before c is read. A conversion is this call's
            // intermediate (released below); a caller's NDArray is used as it is.
            var arrays = new NDArray[args.Length];
            var converted = new NDArray[args.Length];
            try
            {
                for (int i = 0; i < args.Length; i++)
                    arrays[i] = args[i] as NDArray ?? (converted[i] = OrdinateArray(args[i]));

                var shape0 = arrays[0].shape;
                for (int i = 1; i < arrays.Length; i++)
                {
                    if (!SameShape(arrays[i].shape, shape0))
                        throw new ValueError(arrays.Length switch
                        {
                            3 => "x, y, z are incompatible",
                            2 => "x, y are incompatible",
                            _ => "ordinates are incompatible",
                        });
                }

                // use tensor on only the first (its {p}val converts c, after the ordinates — NumPy's order)
                NDArray r = Val(basis, arrays[0], c, tensor: true);
                try
                {
                    for (int i = 1; i < arrays.Length; i++)
                    {
                        var next = Val(basis, arrays[i], r, tensor: false);
                        r.Dispose();   // each intermediate is ours; the caller's c is never disposed
                        r = next;
                    }
                    return r;
                }
                catch
                {
                    // A later pass failed (a broadcast mismatch, an OverflowError): the partial result is ours.
                    r.Dispose();
                    throw;
                }
            }
            finally
            {
                // The results are fresh arrays, never views of an ordinate, so the conversions can always go.
                foreach (var a in converted)
                    a?.Dispose();
            }
        }

        /// <summary>
        ///     <c>polyutils._gridnd(val_f, c, *args)</c>: <c>c = val_f(xi, c)</c> for each ordinate, with
        ///     <c>tensor=True</c> and WITHOUT converting the ordinates — a Python-scalar ordinate stays weak. The first
        ///     call converts an array_like c, before its own x (<c>{p}val</c>'s statement order).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="c">Coefficients, any array_like (see <see cref="Val(PolyBasis, object, object, bool)"/>).</param>
        /// <param name="args">The ordinates, each anything <see cref="Val(PolyBasis, object, NDArray, bool)"/> accepts.</param>
        /// <returns>The grid, shape <c>x.shape + y.shape[ + z.shape]</c> (plus trailing coefficient axes).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="args"/> or an ordinate is null.</exception>
        /// <exception cref="ValueError">A ragged tuple / list c or ordinate (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="NotSupportedException">A null / str c (NumPy's object / str arrays).</exception>
        internal static NDArray GridNd(PolyBasis basis, object c, object[] args)
        {
            if (args is null) throw new ArgumentNullException(nameof(args));
            // _gridnd with no ordinates returns c itself (NumPy hands back the same object): the caller's array, or np.array
            // of any other value — C# has no untyped return here. Unreachable from the fixed-arity grid2d / grid3d facades.
            if (args.Length == 0)
                return c as NDArray ?? ArrayOf(c);
            NDArray r = Val(basis, args[0], c, tensor: true);
            try
            {
                for (int i = 1; i < args.Length; i++)
                {
                    var next = Val(basis, args[i], r, tensor: true);
                    r.Dispose();   // each intermediate is ours; the caller's c is never disposed
                    r = next;
                }
                return r;
            }
            catch
            {
                // A later ordinate failed (a ragged tuple / list, an OverflowError): the partial grid is ours.
                r.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Maps a C# argument onto NumPy's x forms: an <see cref="NDArray"/> stays an array; the C# primitives
        ///     that ARE Python literals become weak Python scalars (the <c>np.r_</c> house convention: bool,
        ///     the integer types, <see cref="BigInteger"/> — a Python int of any size — float/double and
        ///     <see cref="Complex"/>); a Python tuple / list (<see cref="PolySequence.IsSequence"/>: a ValueTuple,
        ///     object[], a jagged or NDArray[] array, any other enumerable) is NumPy's <c>np.asarray(x)</c> of it, nested
        ///     to any depth; everything else goes through <c>np.asanyarray</c> (a typed C# array is an ndarray of its
        ///     dtype; char/Half/decimal become strong 0-d arrays).
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <param name="weakValue">The Python value when weak.</param>
        /// <param name="isWeak">Whether x is a Python scalar.</param>
        /// <returns>The array form, or null when weak.</returns>
        /// <exception cref="ValueError">A ragged Python sequence (NumPy's inhomogeneous-shape text).</exception>
        /// <exception cref="NotSupportedException">A sequence holding a str / None item, or a value np.asanyarray does not
        ///     understand.</exception>
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
                // A Python int past uint64 (2**70): the x-only subtrees run CPython's exact int arithmetic and the
                // constant pool converts it with NumPy's rounding / OverflowError, like any other Python int.
                case BigInteger v: weakValue = PyScalar.Int(v); return null;
                case float v: weakValue = PyScalar.Float(v); return null;   // the float's exact value as a Python float
                case double v: weakValue = PyScalar.Float(v); return null;
                case Complex v: weakValue = PyScalar.Cplx(v); return null;
                default:
                    isWeak = false;
                    return PolySequence.IsSequence(x) ? PolySequence.ToArray(x) : np.asanyarray(x);
            }
        }

        /// <summary>The evaluation proper (x classified, c normalized).</summary>
        /// <param name="basis">The basis.</param>
        /// <param name="xa">x as an array (null when weak).</param>
        /// <param name="xw">x as a Python value (when weak).</param>
        /// <param name="weak">x is a Python scalar.</param>
        /// <param name="c">Coefficients, at least 1-D, never int-like.</param>
        /// <param name="cLayout">The layout NumPy's own recurrence reads the series through (<see cref="SeriesLayout"/>) —
        ///     same dims as <paramref name="c"/>; decides the result's strides, never its values.</param>
        /// <param name="tensor">NumPy's tensor flag.</param>
        /// <returns>The values, laid out as NumPy's result is (<see cref="ResultShape"/>).</returns>
        /// <exception cref="IndexError">Empty series.</exception>
        /// <exception cref="IncorrectShapeException">tensor=False broadcast mismatch.</exception>
        /// <exception cref="OverflowException">A Python int does not fit an integer x dtype.</exception>
        private static NDArray Evaluate(PolyBasis basis, NDArray xa, PyScalar xw, bool weak, NDArray c, Shape cLayout, bool tensor)
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

                    // 1-D series at an array x: result has x's shape, laid out as NumPy's ufunc chain lays it out —
                    // x's memory order, a broadcast axis abstaining (ResultShape) — so the iterator walks both
                    // contiguously wherever x is dense.
                    y = new NDArray(k.Tl, ResultShape(basis, cLayout, xa, tensor, (long[])xa.Shape.dimensions.Clone()), false);
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
                    // A fresh dense layout built from the dims (never a view's Shape, which would overrun the
                    // buffer): NumPy's, which follows the series' and x's memory orders (ResultShape).
                    y = new NDArray(k.Tl, ResultShape(basis, cLayout, xa, tensor, outDims), false);
                    if (y.size == 0) return Detach(ref y);
                    // NEVER buffered: the kernel reads beyond c0 (see the kernel file header).
                    using (var iter = NDIterRef.MultiNew(3, new[] { xa, c0, y }, NDIterGlobalFlags.EXTERNAL_LOOP,
                               NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                               new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, null))
                        iter.ForEach(k.Fn, aux);
                    return Detach(ref y);
                }

                // Scalar x (weak or 0-d) over an N-D series: the points are c's columns, shape c.shape[1:], laid out
                // in the series' memory order as NumPy's c[k]-only ufunc chain lays them out (ResultShape).
                c0 = c[0];
                y = new NDArray(k.Tl, ResultShape(basis, cLayout, null, tensor, cOuter), false);
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

        /// <summary>
        ///     The layout through which NumPy's <c>{p}val</c> reads its series: <c>np.array(c, ndmin=1, copy=…)</c> then, for
        ///     an integer / bool series, its float64 conversion — taken from the caller's <paramref name="c"/>, never from
        ///     NumSharp's own conversion. It only decides the RESULT's strides (<see cref="ResultShape"/>).
        /// </summary>
        /// <param name="basis">The basis — the six sources differ here.</param>
        /// <param name="c">The caller's series layout (at least 1-D).</param>
        /// <param name="intLike">The series is integer / bool (NumPy converts it to float64 first).</param>
        /// <returns>The layout NumPy's <c>c[k]</c> views are taken from.</returns>
        /// <remarks>
        ///     NumPy 2.4.2's sources, per basis: <c>chebval</c> copies (<c>copy=True</c>, order 'K' — a C- or F-contiguous
        ///     series keeps its order, any other its stride order, a broadcast axis sorting innermost:
        ///     <see cref="Shape.KeepOrder"/>), and its int conversion <c>astype(np.double)</c> keeps that layout. The
        ///     other Clenshaw bases do not copy (<c>copy=None</c>: the caller's own strides, a broadcast series included)
        ///     unless the series is integer, where <c>astype(np.double)</c> copies with order 'K'. <c>polyval</c> does not
        ///     copy either, and converts an integer series with <c>c + 0.0</c> — a ufunc, laid out by NpyIter
        ///     (<see cref="Shape.NpyIterOutputShape"/>), which differs from 'K' on a broadcast axis.
        /// </remarks>
        internal static Shape SeriesLayout(PolyBasis basis, Shape c, bool intLike)
        {
            // A C- or F-contiguous (or ≤ 1-D) series comes out of every one of those copies in its own order, so the
            // common calls skip the copy layout's allocation (a 1-D series' c[k] never votes on the result anyway).
            if (c.NDim <= 1 || c.IsContiguous || c.IsFContiguous)
                return c;
            if (basis == PolyBasis.Power)
                return intLike ? Shape.NpyIterOutputShape((long[])c.dimensions.Clone(), new ReadOnlySpan<Shape>(in c)) : c;
            return basis == PolyBasis.Chebyshev || intLike ? c.KeepOrder() : c;
        }

        /// <summary>
        ///     The layout of NumPy's <c>{p}val</c> result — the strides its chain of ufunc calls leaves the last value
        ///     with — for a result of <paramref name="outDims"/>. The kernel computes every element independently, so
        ///     this decides only where each value is stored: NumPy's flags and strides, the values untouched.
        /// </summary>
        /// <param name="basis">The basis (its step table is the chain replayed).</param>
        /// <param name="cLayout">The series layout NumPy's recurrence indexes (<see cref="SeriesLayout"/>).</param>
        /// <param name="xa">x as an array, or null for a Python scalar (a 0-d array reads as a scalar too).</param>
        /// <param name="tensor">NumPy's tensor flag (with an array x, every <c>c[k]</c> gains x's rank of unit axes).</param>
        /// <param name="outDims">The result dims the caller already computed (and validated). The returned shape ADOPTS
        ///     this array — pass a fresh one (the callers clone x's dims or build the array themselves).</param>
        /// <returns>A fresh dense, offset-0 shape of <paramref name="outDims"/>.</returns>
        /// <remarks>
        ///     <para><b>The rule.</b> NumPy's result is the output of the recurrence's LAST ufunc call, and every call's
        ///     output is laid out by NpyIter from the layouts of its operands (<see cref="Shape.NpyIterOutputShape"/>) —
        ///     <c>c[k]</c> views of the series (stride 0 on x's axes under tensor), x, Python scalars (no vote), and the
        ///     earlier outputs. Which operands can vote on which axes settles every case but one in closed form:</para>
        ///     <list type="bullet">
        ///         <item>a C-contiguous series with a C-contiguous (or scalar) x, or a result of rank ≤ 1: C — every
        ///         vote is C-ordered (the common calls pay nothing);</item>
        ///         <item>at most a 2-D series with an x that is C-contiguous, at most 1-D, or a scalar: C without any vote
        ///         — the series block (one axis) and x's block are each C, and so is their concatenation;</item>
        ///         <item>a scalar x (Python, or a 0-d array): only <c>c[k]</c> ever votes — NpyIter's layout of
        ///         <c>c[0]</c>;</item>
        ///         <item>a 1-D series: <c>c[k]</c> is a scalar or all unit axes and never votes — NpyIter's layout of x
        ///         (an output laid out from x votes exactly as x does, so the chain never moves off it);</item>
        ///         <item>an N-D series under tensor: every pair of a series axis and an x axis is seen by no operand
        ///         until the first mixed value, whose sort cannot move a series axis inside an x axis (ambiguous pairs
        ///         never shift the insertion point), and every later value votes that order — the series block
        ///         (NpyIter's layout of <c>c[0]</c>) outside, x's block (NpyIter's layout of x) inside.</item>
        ///     </list>
        ///     <para>The remaining case — an N-D series broadcast against a per-point x (tensor=False), where the two
        ///     share axes and the votes interleave with NumPy's "C order wins conflicts" — replays the basis's step
        ///     table over layouts (<see cref="ReplayResultShape"/>, 0.5–2.4 µs), except when one operand settles it: a
        ///     C-contiguous x, or a C-contiguous series, whose extents ARE the result's. Such an operand votes C on every
        ///     pair of non-unit axes, so every value computed from it is C, and the conflict rule makes every op that
        ///     meets a C value C too. Every basis's last op meets x (<c>c0 + c1*x</c> and its variants; Horner's
        ///     <c>c[-i] + c0*x</c>), and a C series makes every <c>c[k]</c>-derived value C. So the result is C without
        ///     a replay. Every closed form was checked against NumPy 2.4.2 (32,740 random layouts, 0 differences) and
        ///     the replay itself (27,595, 0); a unit test cross-checks them, the replay shortcut included, in C#.</para>
        ///     <para>Extent-1 axes are not modelled: NumPy's own strides there depend on which loop path each call took;
        ///     no flag reads them.</para>
        /// </remarks>
        internal static Shape ResultShape(PolyBasis basis, Shape cLayout, NDArray xa, bool tensor, long[] outDims)
        {
            // C: a rank ≤ 1 result, or C-ordered voters only (a 1-D series never votes — its c[k] is a scalar or all
            // unit axes — so any 1-D series counts as C here).
            if (outDims.Length <= 1 || ((cLayout.NDim <= 1 || cLayout.IsContiguous) && (xa is null || xa.Shape.IsContiguous)))
                return new Shape(outDims);

            bool scalarX = xa is null || xa.ndim == 0;
            if (scalarX || cLayout.NDim == 1 || tensor)
            {
                // Both blocks C without a vote: c[0]'s block has at most one axis (a 2-D series, or none for a 1-D
                // one) and x's block is C (a scalar, at most 1-D, or C-contiguous) — the common non-C-series calls
                // (an F-ordered (n, k) series at a vector of points) skip every allocation below.
                if (cLayout.NDim <= 2 && (scalarX || xa.ndim <= 1 || xa.Shape.IsContiguous))
                    return new Shape(outDims);
                var xDims = scalarX ? Array.Empty<long>() : xa.Shape.dimensions;
                Shape xs = scalarX ? default : xa.Shape;
                var xBlock = scalarX ? default : Shape.NpyIterOutputShape((long[])xDims.Clone(), new ReadOnlySpan<Shape>(in xs));
                if (cLayout.NDim == 1 && !scalarX)
                    return xBlock;   // a 1-D series at an array x (a scalar x with a 1-D series is a 0-d result, above)
                int ncd = cLayout.NDim - 1;
                var cDims = new long[ncd];
                var cStrides = new long[ncd];
                Array.Copy(cLayout.dimensions, 1, cDims, 0, ncd);
                Array.Copy(cLayout.strides, 1, cStrides, 0, ncd);
                // The operand shape and the block share cDims: neither is ever written again.
                var cOperand = new Shape(cDims, cStrides);
                var cBlock = Shape.NpyIterOutputShape(cDims, new ReadOnlySpan<Shape>(in cOperand));
                if (scalarX)
                    return cBlock;
                // tensor: the series block outside, x's block inside (x.size elements per series point).
                long xSize = xa.size;
                var strides = new long[ncd + xDims.Length];
                for (int d = 0; d < ncd; d++)
                    strides[d] = cBlock.strides[d] * xSize;
                for (int d = 0; d < xDims.Length; d++)
                    strides[ncd + d] = xBlock.strides[d];
                return new Shape(outDims, strides);
            }

            // tensor=False with an N-D series: a C-contiguous operand spanning the whole result makes it C (remarks).
            if ((xa.Shape.IsContiguous && SameDims(xa.Shape.dimensions, 0, outDims))
                || (cLayout.IsContiguous && SameDims(cLayout.dimensions, 1, outDims)))
                return new Shape(outDims);
            return ReplayResultShape(basis, cLayout, xa, tensor, outDims);
        }

        /// <summary>Whether <c>dims[from..]</c> equals <paramref name="outDims"/> extent for extent (no broadcasting).</summary>
        /// <param name="dims">An operand's dims.</param>
        /// <param name="from">The first operand axis compared (1 drops a series' coefficient axis).</param>
        /// <param name="outDims">The result dims.</param>
        /// <returns>True when the operand spans the result exactly.</returns>
        private static bool SameDims(long[] dims, int from, long[] outDims)
        {
            if (dims.Length - from != outDims.Length)
                return false;
            for (int d = 0; d < outDims.Length; d++)
                if (dims[from + d] != outDims[d])
                    return false;
            return true;
        }

        /// <summary>
        ///     <see cref="ResultShape"/> by brute force: NumPy's statements replayed over layouts — the basis's step table
        ///     (<see cref="PolySteps.Eval"/>, the same table the kernel is emitted from) walked with every value's layout
        ///     in place of its data, each array op laid out by <see cref="Shape.NpyIterOutputShape"/> from its two operands.
        ///     The general path for an N-D series broadcast against a per-point x (tensor=False); internal so a test can
        ///     check the closed forms of <see cref="ResultShape"/> against it.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="cLayout">The series layout NumPy's recurrence indexes (<see cref="SeriesLayout"/>).</param>
        /// <param name="xa">x as an array, or null for a Python scalar (a 0-d array reads as a scalar too).</param>
        /// <param name="tensor">NumPy's tensor flag.</param>
        /// <param name="outDims">The result dims.</param>
        /// <returns>A fresh dense, offset-0 shape of <paramref name="outDims"/> (C if the replay's broadcast ever
        ///     disagreed with them — a defensive fallback, never reached when the caller validated the shapes).</returns>
        /// <remarks>The recurrence's loop stops at its fixpoint: a step maps the accumulators' layouts to the next ones
        ///     by a fixed rule, so the first repeat ends it — in practice after one or two steps, whatever the series
        ///     length (0.5–2.4 µs a call, measured; the closed forms cost ~0.1 µs).</remarks>
        internal static Shape ReplayResultShape(PolyBasis basis, Shape cLayout, NDArray xa, bool tensor, long[] outDims)
        {
            // c[k]: axis 0 dropped; c.reshape(c.shape + (1,)*x.ndim) under tensor with an array x (unit axes abstain).
            int extra = xa is not null && tensor ? xa.ndim : 0;
            int nck = cLayout.NDim - 1 + extra;
            var ckDims = new long[nck];
            var ckStrides = new long[nck];
            for (int d = 1; d < cLayout.NDim; d++)
            {
                ckDims[d - 1] = cLayout.dimensions[d];
                ckStrides[d - 1] = cLayout.strides[d];
            }
            for (int d = cLayout.NDim - 1; d < nck; d++)
                ckDims[d] = 1;
            // Every slot starts as a scalar layout: a default(Shape) has null dims, and a slot a basis never assigns
            // before reading would otherwise crash the replay instead of abstaining.
            var env = new LayoutEnv
            {
                Ck = nck == 0 ? Shape.NewScalar() : new Shape(ckDims, ckStrides),
                X = xa is null || xa.ndim == 0 ? Shape.NewScalar() : xa.Shape,
                X2 = Shape.NewScalar(),
                C0 = Shape.NewScalar(),
                C1 = Shape.NewScalar(),
                Tmp = Shape.NewScalar(),
            };
            var prog = PolySteps.Eval(basis);
            env.X2 = prog.Pre is null ? Shape.NewScalar() : LayoutOf(prog.Pre, env);
            long nc = cLayout.dimensions[0];

            Shape r;
            if (prog.Horner)
            {
                // c0 = c[-1] + x*0 ; c0 = c[-i] + c0*x
                env.C0 = LayoutOf(prog.HornerInit, env);
                for (long i = 2; i <= nc; i++)
                {
                    var next = LayoutOf(prog.HornerStep, env);
                    if (SameLayout(next, env.C0))
                        break;   // a fixpoint: every later step maps it to itself
                    env.C0 = next;
                }
                r = env.C0;
            }
            else if (nc == 1)
            {
                env.C0 = env.Ck;   // c0 = c[0], c1 = 0 (a Python int, folded into FinalLen1)
                r = LayoutOf(prog.FinalLen1, env);
            }
            else
            {
                env.C0 = env.Ck;
                env.C1 = env.Ck;
                for (long i = 3; i <= nc; i++)
                {
                    // tmp = c0 ; c0 = <StepC0 over c[-i], c1> ; c1 = <StepC1 over tmp, c1, x, x2> — both from the old c1.
                    env.Tmp = env.C0;
                    var n0 = LayoutOf(prog.StepC0, env);
                    var n1 = LayoutOf(prog.StepC1, env);
                    if (SameLayout(n0, env.C0) && SameLayout(n1, env.C1))
                        break;
                    env.C0 = n0;
                    env.C1 = n1;
                }
                r = LayoutOf(prog.Final, env);
            }

            // Defensive: the replay's broadcast must reproduce the caller's dims; anything else falls back to C.
            if (r.NDim != outDims.Length)
                return new Shape((long[])outDims.Clone());
            for (int d = 0; d < outDims.Length; d++)
                if (r.dimensions[d] != outDims[d])
                    return new Shape((long[])outDims.Clone());
            return r;
        }

        /// <summary>The layouts of the step table's leaves during a <see cref="ReplayResultShape"/> replay.</summary>
        private struct LayoutEnv
        {
            /// <summary><c>x</c> (0-d for a scalar x).</summary>
            public Shape X;
            /// <summary><c>x2</c> (the x-only pre-op's output; 0-d when the basis has none).</summary>
            public Shape X2;
            /// <summary>The current <c>c0</c>.</summary>
            public Shape C0;
            /// <summary>The current <c>c1</c>.</summary>
            public Shape C1;
            /// <summary>The <c>tmp</c> of a Clenshaw step (the previous <c>c0</c>).</summary>
            public Shape Tmp;
            /// <summary>Every <c>c[k]</c> view (they share one layout).</summary>
            public Shape Ck;
        }

        /// <summary>
        ///     The layout of one step-table expression: a leaf reads <paramref name="env"/>, a Python value is 0-d (it never
        ///     votes), and an array op is a ufunc call whose output NpyIter lays out from its two operands.
        /// </summary>
        /// <param name="e">The expression.</param><param name="env">The leaves' layouts.</param>
        /// <returns>The layout of the expression's value.</returns>
        /// <exception cref="ArgumentOutOfRangeException">An expression node of an unknown kind.</exception>
        private static Shape LayoutOf(PolyExpr e, in LayoutEnv env)
        {
            switch (e)
            {
                case PolyLeaf l:
                    return l.S switch
                    {
                        PolySym.X => env.X,
                        PolySym.X2 => env.X2,
                        PolySym.C0 => env.C0,
                        PolySym.C1 => env.C1,
                        PolySym.Tmp => env.Tmp,
                        _ => env.Ck,
                    };
                case PolyWeak:
                    return Shape.NewScalar();
                case PolyBin b:
                {
                    Shape a = LayoutOf(b.A, env), c = LayoutOf(b.B, env);
                    if (a.NDim == 0 && c.NDim == 0)
                        return Shape.NewScalar();
                    // The operands broadcast (the caller validated the whole evaluation's shapes): right-aligned, an
                    // extent of 1 yielding to the other.
                    int nd = Math.Max(a.NDim, c.NDim);
                    var dims = new long[nd];
                    for (int d = 0; d < nd; d++)
                    {
                        int ia = d - (nd - a.NDim), ic = d - (nd - c.NDim);
                        long da = ia < 0 ? 1 : a.dimensions[ia], dc = ic < 0 ? 1 : c.dimensions[ic];
                        dims[d] = da == 1 ? dc : da;
                    }
                    return Shape.NpyIterOutputShape(dims, new[] { a, c });
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(e), e?.GetType().Name, "unknown step-table node");
            }
        }

        /// <summary>Whether two layouts are the same dims and strides (a replay fixpoint test).</summary>
        /// <param name="a">One layout.</param><param name="b">The other.</param>
        /// <returns>True when identical.</returns>
        private static bool SameLayout(in Shape a, in Shape b)
        {
            if (a.NDim != b.NDim) return false;
            for (int d = 0; d < a.NDim; d++)
                if (a.dimensions[d] != b.dimensions[d] || a.strides[d] != b.strides[d])
                    return false;
            return true;
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
        /// <remarks>Also the layout of NumPy's <c>np.array(c, copy=True)</c> (order='K') — the copy the calculus family
        ///     (<c>NDPolyCalc</c>) returns for <c>m=0</c> and builds its degenerate results from.</remarks>
        internal static Shape LikeKeepOrder(Shape s)
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
