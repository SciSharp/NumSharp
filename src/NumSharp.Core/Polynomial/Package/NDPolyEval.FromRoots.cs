using System;
using System.Numerics;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

// =============================================================================
// NDPolyEval.FromRoots.cs — numpy.polynomial.polynomial.polyvalfromroots
// =============================================================================
//
// NUMPY 2.4.2 (numpy/polynomial/polynomial.py), the statements this file reproduces in order:
//
//   r = np.array(r, ndmin=1, copy=None)
//   if r.dtype.char in '?bBhHiIlLqQpP':
//       r = r.astype(np.double)
//   if isinstance(x, (tuple, list)):
//       x = np.asarray(x)
//   if isinstance(x, np.ndarray):
//       if tensor:
//           r = r.reshape(r.shape + (1,) * x.ndim)
//       elif x.ndim >= r.ndim:
//           raise ValueError("x.ndim must be < r.ndim when tensor == False")
//   return np.prod(x - r, axis=0)
//
// ONE PASS, NO TEMPORARY
// ----------------------
// NumPy materializes x - r (nr times the result's size) and reduces it. The kernel is U3's step-table engine with its
// third skeleton, the ROOT PRODUCT (ILKernelGenerator.Polynomial.Eval.cs, EmitPolyRootProduct): per point it starts at
// the multiply identity (PyUFunc_ReduceWrapper assigns it before its loop) and multiplies x - r[k] for k = 0, 1, ...
// — every root in order, read once, the result written once. The operand forms are {p}val's: a 1-D r in auxdata, an
// N-D r through the (x, r[0], y) NDIter form, the tensor reshape as trailing unit axes of r[0].
//
// WHAT DECIDES THE BITS
// ---------------------
//   * The dtype T of x - r: NEP 50 (a Python-scalar x is weak), U3's typing pass. np.prod keeps T.
//   * The product form (PolyRootForm), for complex128 and float16 only — a float32 / float64 / decimal product rounds
//     once per step in every form. NumPy's reduction iterator sorts the axes of the x - r temporary by its strides;
//     when the roots axis is the innermost non-unit axis (one output element, or an F-ordered r under tensor=False),
//     the inner loop walks the ROOTS with the output's stride 0 and the multiply loops take their reduce branches:
//     CDOUBLE_multiply's MSVC-contracted fallback loop instead of simd_cmul, HALF_multiply's float32 accumulation
//     narrowed once per call instead of a narrow per root. One call holds every root (no cast is needed, so nothing
//     is buffered — probed up to 20,000 roots under every np.setbufsize). The temporary's layout is NpyIter's
//     (Shape.NpyIterOutputShape over x and the reshaped r).
//   * The result's layout: the reduction output of that temporary (Shape.ReductionOutputShape), i.e. x's memory order
//     under tensor, the series' columns' otherwise — written by the kernel in place, no relayout.
//
// ARGUMENTS NUMPY COMPUTES WITH AS PYTHON OBJECTS
// -----------------------------------------------
// None, a str and a sequence holding them (or holding a Python int past uint64) make NumPy build a str or object array.
//   * A str operand (no object one): np.subtract has no loop — the house string ufuncs raise NumPy's UFuncTypeError text,
//     resolved before any broadcast, as NumPy resolves it.
//   * A bare None (x or r): np.subtract runs its object loop, broadcasting first, then Python's `-` on the first
//     element pair raises CPython's TypeError, naming each operand's element type as the object cast makes it ('float'
//     for every float dtype, 'int' for every integer, 'str' for a str array). With no element pair (an empty operand),
//     np.prod of the empty object array is its identity, the Python int 1, for a 0-d result — and an object array
//     (refused: NumSharp has no object dtype) otherwise.
//   * An object array from a sequence: after NumPy's shape checks, refused — except a 0-d result (a scalar x and a 1-D
//     r) of number items, which NumPy's object loop and object product compute as Python numbers (PolyNumber), the
//     first element copied (a non-empty object reduction uses no identity). A None / str item there is the documented
//     object-arithmetic divergence (NotSupportedException where CPython raises its TypeError).
//
// =============================================================================

namespace NumSharp
{
    internal static unsafe partial class NDPolyEval
    {
        /// <summary>What one argument of <c>polyvalfromroots</c> is after NumPy's conversion of it.</summary>
        private enum RootArgKind : byte
        {
            /// <summary>A numeric array (any NumSharp dtype).</summary>
            Numeric,
            /// <summary>A string array (U / S / StringDType), or NumPy's str array NumSharp could not build (no array then).</summary>
            Str,
            /// <summary>NumPy's object array from a sequence or an object scalar (no memory: dims and the refusal only).</summary>
            Object,
            /// <summary>A bare None: <c>np.array(None, ndmin=1)</c> for r (an object <c>[None]</c>), the None object for x.</summary>
            None,
            /// <summary>A Python scalar x (a NEP 50 weak value).</summary>
            Weak,
        }

        /// <summary>
        ///     One argument of <c>polyvalfromroots</c> after NumPy's conversion: its kind, the array NumSharp holds for it
        ///     (when there is one), NumPy's dims (for the shape checks that precede any computation), and for x whether NumPy
        ///     sees an ndarray (<c>isinstance(x, np.ndarray)</c> after the tuple / list conversion).
        /// </summary>
        private readonly struct RootArg
        {
            /// <summary>The kind.</summary>
            public readonly RootArgKind Kind;
            /// <summary>The numeric or string array; null for an object array, a bare None, a weak scalar, or a str sequence the
            ///     house could not build.</summary>
            public readonly NDArray Array;
            /// <summary>NumPy's dims (empty for a 0-d value, a weak scalar and a bare-None x; <c>(1,)</c> for a bare-None r).</summary>
            public readonly long[] Dims;
            /// <summary>x only: NumPy's <c>isinstance(x, np.ndarray)</c> — an NDArray, a converted tuple / list or a typed C#
            ///     array. A str, None, a Python or a NumPy scalar is not (the reshape and the ndim check skip it).</summary>
            public readonly bool IsNdarray;
            /// <summary>The Python value of a weak x.</summary>
            public readonly PyScalar Weak;
            /// <summary>What NumSharp raises where NumPy computes with an object / str array it cannot represent.</summary>
            public readonly NotSupportedException Refusal;

            /// <summary>Describes one argument.</summary>
            /// <param name="kind">The kind.</param>
            /// <param name="array">The array (or null).</param>
            /// <param name="dims">NumPy's dims.</param>
            /// <param name="isNdarray">x only: NumPy sees an ndarray.</param>
            /// <param name="weak">The Python value of a weak x.</param>
            /// <param name="refusal">The deferred refusal, or null.</param>
            public RootArg(RootArgKind kind, NDArray array, long[] dims, bool isNdarray, PyScalar weak = default,
                NotSupportedException refusal = null)
            {
                Kind = kind; Array = array; Dims = dims; IsNdarray = isNdarray; Weak = weak; Refusal = refusal;
            }

            /// <summary>An array argument (numeric or string, by its dtype); NumPy's dims are the array's.</summary>
            /// <param name="a">The array.</param>
            /// <param name="isNdarray">NumPy sees an ndarray.</param>
            /// <returns>The argument.</returns>
            public static RootArg Of(NDArray a, bool isNdarray)
                => new RootArg(a.typecode == NPTypeCode.String ? RootArgKind.Str : RootArgKind.Numeric, a, a.Shape.dimensions, isNdarray);
        }

        /// <summary>
        ///     <c>polyvalfromroots(x, r, tensor)</c>: the value at <paramref name="x"/> of the monic polynomial with roots
        ///     <paramref name="r"/>, <c>prod(x - r[k])</c>, as NumPy 2.4.2 computes it (see the file header).
        /// </summary>
        /// <param name="x">Points: an <see cref="NDArray"/> (any layout), a Python tuple / list (a ValueTuple, <c>object[]</c>,
        ///     a jagged / <c>NDArray[]</c> array, any enumerable — NumPy's <c>np.asarray(x)</c>), a typed C# array (an ndarray), a
        ///     Python scalar (<c>bool</c>, an integer type, <see cref="BigInteger"/>, <c>float</c>, <c>double</c>,
        ///     <see cref="Complex"/>: weak), a NumPy scalar (<see cref="Half"/>, <c>char</c>, <c>decimal</c>: strong 0-d), a str,
        ///     or null (Python's None).</param>
        /// <param name="r">Roots, anything <c>np.array</c> accepts (an <see cref="NDArray"/> used as is, a sequence, a typed C#
        ///     array, a scalar, a str, null). Axis 0 runs over the roots of one polynomial; further axes index polynomials.</param>
        /// <param name="tensor">NumPy's tensor flag: with an array x, every polynomial is evaluated at every point (result
        ///     shape <c>r.shape[1:] + x.shape</c>); when false, x broadcasts over the polynomials (result shape
        ///     <c>broadcast(x.shape, r.shape[1:])</c>).</param>
        /// <returns>A fresh, owning array of NumPy's dtype and layout (a 0-d array for a scalar x with a 1-D r).</returns>
        /// <exception cref="ValueError">A ragged tuple / list (r's first, NumPy's order), or <c>tensor=False</c> with
        ///     <c>x.ndim &gt;= r.ndim</c>.</exception>
        /// <exception cref="IncorrectShapeException"><c>tensor=False</c> and the shapes do not broadcast (NumPy's ValueError
        ///     text, x's shape first).</exception>
        /// <exception cref="TypeError">A str operand (NumPy's no-loop text), or a bare None meeting an element (CPython's
        ///     <c>unsupported operand type(s) for -</c> text).</exception>
        /// <exception cref="OverflowException">A Python int x too large for the loop's float dtype (CPython's
        ///     <c>int too large to convert to float</c>).</exception>
        /// <exception cref="NotSupportedException">NumPy's result is an object array, or its object arithmetic meets an
        ///     item NumSharp cannot read as a number (see the file header).</exception>
        internal static NDArray ValFromRoots(object x, object r, bool tensor)
        {
            // Every array made here — r's conversion and its 0-d expansion, r's float64 copy, x's conversion — is this
            // call's intermediate: the result is always a fresh array, never r, x or a view of either.
            NDArray rConverted = null, rExpanded = null, rFloat = null, xConverted = null;
            try
            {
                // r = np.array(r, ndmin=1, copy=None) — FIRST: a ragged r is reported before a ragged x.
                var ro = RootsArgument(r, out rConverted, out rExpanded);
                NDArray ra = ro.Array;
                // The layout NumPy's x - r reads r through: the caller's (copy=None), or for an integer / bool r its float64
                // copy's — astype(np.double) keeps the K order. It decides the temporary's layout, never a value.
                Shape rLayout = default;
                if (ro.Kind == RootArgKind.Numeric)
                {
                    rLayout = ra.Shape;
                    if (PolyTyping.IsIntLike(ra.typecode))
                    {
                        rLayout = LikeKeepOrder(ra.Shape);
                        ra = rFloat = ra.astype(NPTypeCode.Double);
                        // Every later statement sees the float64 roots — the object loop too, which names their elements
                        // 'float' (polyvalfromroots(None, [1, 2]) is 'NoneType' and 'float').
                        ro = RootArg.Of(ra, isNdarray: true);
                    }
                }

                // if isinstance(x, (tuple, list)): x = np.asarray(x)
                var xo = PointsArgument(x, out xConverted);

                // if isinstance(x, np.ndarray): the tensor reshape, or the ndim check (before x - r, so before any loop
                // resolution: a str or object operand still meets this ValueError first).
                if (xo.IsNdarray && !tensor && xo.Dims.Length >= ro.Dims.Length)
                    throw new ValueError("x.ndim must be < r.ndim when tensor == False");
                bool expand = xo.IsNdarray && tensor;

                // x - r: an object operand runs np.subtract's object loop (a str partner is cast to object), a str one has
                // no loop at all, everything else is the numeric root product.
                if (xo.Kind is RootArgKind.Object or RootArgKind.None || ro.Kind is RootArgKind.Object or RootArgKind.None)
                    return ObjectFromRoots(x, xo, r, ro, expand);
                if (xo.Kind == RootArgKind.Str || ro.Kind == RootArgKind.Str)
                    throw StrFromRoots(x, xo, ro, ra);
                return NumericFromRoots(xo, ra, rLayout, expand);
            }
            finally
            {
                xConverted?.Dispose();
                rFloat?.Dispose();
                rExpanded?.Dispose();
                rConverted?.Dispose();
            }
        }

        /// <summary>
        ///     <c>np.array(r, ndmin=1, copy=None)</c>: an <see cref="NDArray"/> as is (a 0-d one expanded to one element, a view),
        ///     a str a <c>(1,)</c> str array, a sequence through np.array's coercion (<see cref="Backends.UnmanagedStorage.FromSequence"/>), a scalar or
        ///     typed C# array as <see cref="NDPolySeries.AsCoefficientArray"/> converts it, None an object <c>[None]</c>.
        /// </summary>
        /// <param name="r">The caller's roots.</param>
        /// <param name="converted">An array this call created (the caller disposes it), or null.</param>
        /// <param name="expanded">The one-element view of a 0-d array (the caller disposes it), or null.</param>
        /// <returns>The converted argument.</returns>
        /// <exception cref="ValueError">A ragged sequence (np.array's inhomogeneous-shape text).</exception>
        private static RootArg RootsArgument(object r, out NDArray converted, out NDArray expanded)
        {
            converted = expanded = null;
            switch (r)
            {
                case null:
                    // np.array(None, ndmin=1): a one-element object array holding None.
                    return new RootArg(RootArgKind.None, null, new long[] { 1 }, isNdarray: true);
                case NDArray nd:
                {
                    var a = nd.ndim == 0 ? (expanded = np.expand_dims(nd, 0)) : nd;
                    return RootArg.Of(a, isNdarray: true);
                }
                case string s:
                    // np.array('ab', ndmin=1): a (1,) str array of the string's width.
                    converted = np.array(new[] { s });
                    return RootArg.Of(converted, isNdarray: true);
            }
            if (Backends.UnmanagedStorage.IsSequence(r))
                return SequenceArgument(r, out converted, isNdarray: true);
            // A Python / NumPy scalar or a typed C# array. What AsCoefficientArray calls non-numeric here is an OBJECT
            // array (a str was handled above): a Python int past uint64, an object np.asanyarray cannot read.
            var v = NDPolySeries.AsCoefficientArray(r);
            if (v.NonNumeric)
                return new RootArg(RootArgKind.Object, null, new long[] { 1 }, isNdarray: true, refusal: v.Refusal);
            converted = v.Source;   // always a new array for a non-NDArray argument
            var b = converted.ndim == 0 ? (expanded = np.expand_dims(converted, 0)) : converted;
            return RootArg.Of(b, isNdarray: true);
        }

        /// <summary>
        ///     The x conversions NumPy makes or skips: a tuple / list becomes an ndarray (<c>np.asarray</c>), an NDArray and a
        ///     typed C# array are ndarrays, a Python scalar stays a weak Python value, a NumPy scalar (Half / char / decimal)
        ///     a strong 0-d value, a str a str scalar, null Python's None.
        /// </summary>
        /// <param name="x">The caller's points.</param>
        /// <param name="converted">An array this call created (the caller disposes it), or null.</param>
        /// <returns>The converted argument.</returns>
        /// <exception cref="ValueError">A ragged sequence.</exception>
        /// <exception cref="NotSupportedException">An object np.asanyarray cannot read.</exception>
        private static RootArg PointsArgument(object x, out NDArray converted)
        {
            converted = null;
            switch (x)
            {
                case null:
                    return new RootArg(RootArgKind.None, null, System.Array.Empty<long>(), isNdarray: false);
                case NDArray nd:
                    return RootArg.Of(nd, isNdarray: true);
                case string s:
                    // A Python str is not an ndarray: np.subtract converts it to a 0-d str array.
                    converted = np.asanyarray(s);
                    return RootArg.Of(converted, isNdarray: false);
            }
            if (Backends.UnmanagedStorage.IsSequence(x))
                return SequenceArgument(x, out converted, isNdarray: true);
            // A Python scalar is weak; anything else goes through np.asanyarray — a typed C# array is an ndarray, a NumPy
            // scalar a strong 0-d array (not an ndarray in NumPy's eyes, but of ndim 0 either way: neither the reshape nor
            // the ndim check can tell them apart).
            var xa = ClassifyX(x, out var xw, out bool weak);
            if (weak)
                return new RootArg(RootArgKind.Weak, null, System.Array.Empty<long>(), isNdarray: false, weak: xw);
            converted = xa;   // ClassifyX built it (a new array)
            return RootArg.Of(xa, isNdarray: Backends.UnmanagedStorage.IsArrayLike(x));
        }

        /// <summary>
        ///     <c>np.array</c> / <c>np.asarray</c> of a Python sequence: a numeric array, or — where NumPy builds a str array —
        ///     the house's str array (so np.subtract can raise NumPy's own text), or NumPy's object array as its dims and refusal.
        /// </summary>
        /// <param name="seq">The sequence.</param>
        /// <param name="converted">The array created (the caller disposes it), or null.</param>
        /// <param name="isNdarray">NumPy sees an ndarray (always true: both conversions return one).</param>
        /// <returns>The converted argument.</returns>
        /// <exception cref="ValueError">A ragged sequence.</exception>
        private static RootArg SequenceArgument(object seq, out NDArray converted, bool isNdarray)
        {
            converted = Backends.UnmanagedStorage.FromSequenceOrNonNumeric(seq, out var nn);
            if (converted is not null)
                return RootArg.Of(converted, isNdarray);
            if (nn.IsObject)
                return new RootArg(RootArgKind.Object, null, nn.Dims, isNdarray, refusal: nn.Refusal);
            // A str array: the house builds the string dtypes, with NumPy's discovered width (np.array(['a', 1.0]) is
            // <U32). A sequence form it does not read keeps NumPy's dims and the refusal.
            try
            {
                converted = np.asanyarray(seq);
            }
            catch (NotSupportedException)
            {
                converted = null;
            }
            if (converted is not null && converted.typecode == NPTypeCode.String)
                return RootArg.Of(converted, isNdarray);
            converted?.Dispose();
            converted = null;
            return new RootArg(RootArgKind.Str, null, nn.Dims, isNdarray, refusal: nn.Refusal);
        }

        /// <summary>
        ///     <c>x - r</c> through np.subtract's OBJECT loop, then np.prod (see the file header): the broadcast check, the
        ///     empty-operand results, CPython's TypeError at a bare None, and the 0-d object product of number items.
        /// </summary>
        /// <param name="x">The caller's points (their own C# type names a weak scalar's Python type).</param>
        /// <param name="xo">x converted.</param>
        /// <param name="r">The caller's roots (an object sequence's items).</param>
        /// <param name="ro">r converted.</param>
        /// <param name="expand">r was reshaped with x's rank of trailing unit axes (tensor with an ndarray x).</param>
        /// <returns>The Python int 1 as a 0-d int64 array (an empty 1-D r at a bare None), or a 0-d object product.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast (NumPy's ValueError text).</exception>
        /// <exception cref="TypeError">CPython's text where a bare None meets an element.</exception>
        /// <exception cref="NotSupportedException">NumPy's result is an object array, or the arithmetic meets an operand
        ///     NumSharp cannot name or read as a number.</exception>
        /// <exception cref="OverflowException">A Python int item too large for a float meets a float (CPython's text).</exception>
        private static NDArray ObjectFromRoots(object x, in RootArg xo, object r, in RootArg ro, bool expand)
        {
            // np.subtract broadcasts its operands before the object loop runs a single element.
            long[] temp = expand ? ConcatDims(ro.Dims, xo.Dims) : BroadcastRootDims(xo.Dims, ro.Dims);
            long size = 1;
            foreach (var d in temp)
                size *= d;
            if (size == 0)
            {
                // No element pair ever meets Python's `-`. np.prod of the empty object array along axis 0 is its identity —
                // the Python int 1 — for a 0-d result, and an object array of ones (or an empty one) otherwise.
                if (temp.Length == 1 && temp[0] == 0)
                    return NDArray.Scalar(1L);
                throw new NotSupportedException(
                    "NumPy's result here is an object array (np.prod over an object x - r), a dtype NumSharp does not have");
            }
            if (xo.Kind == RootArgKind.None || ro.Kind == RootArgKind.None)
            {
                // Python's `-` on the first element pair: None has no `-` with anything, so it raises there, naming the two
                // operands as the object cast hands them over.
                string xn = ObjectLoopTypeName(x, xo), rn = ObjectLoopTypeName(r, ro);
                if (xn is null || rn is null)
                    throw xo.Refusal ?? ro.Refusal ?? new NotSupportedException(
                        "NumPy's object loop meets an operand NumSharp has no Python value for (an object array's items or a decimal)");
                throw new TypeError($"unsupported operand type(s) for -: '{xn}' and '{rn}'");
            }
            // An object array of NUMBERS: NumPy computes with Python objects. A 0-d result (a scalar x and a 1-D r) is one
            // Python number, which NumSharp can compute; every other result is an object array.
            if (temp.Length == 1 && ro.Kind == RootArgKind.Object && xo.Kind is RootArgKind.Weak or RootArgKind.Numeric)
                return ObjectRootProduct(x, xo, r);
            throw ro.Refusal ?? xo.Refusal ?? new NotSupportedException(
                "NumPy computes this x - r with Python objects into an object array, a dtype NumSharp does not have");
        }

        /// <summary>
        ///     The 0-d object product: <c>x - r</c> over the items of a 1-D object r (each meeting x through Python's own
        ///     operator, x first cast to the object dtype), then np.prod of that object array — its FIRST element copied and
        ///     the others multiplied in order (a non-empty object reduction uses no identity) — returned as
        ///     <c>np.array(result)</c> would hold it.
        /// </summary>
        /// <param name="x">The caller's points (a weak Python scalar, or a NumPy scalar / 0-d array).</param>
        /// <param name="xo">x converted.</param>
        /// <param name="r">The caller's roots: an object sequence, or an object scalar (np.array of it holds it alone).</param>
        /// <returns>A fresh 0-d array of the result's discovered dtype.</returns>
        /// <exception cref="NotSupportedException">An item NumSharp cannot read as a number (None, a str — CPython's TypeError
        ///     in NumPy), or a Python int result past uint64 (an object scalar).</exception>
        /// <exception cref="OverflowException">CPython's float conversion of a too-large Python int.</exception>
        private static NDArray ObjectRootProduct(object x, in RootArg xo, object r)
        {
            // x as the object loop receives it: a Python value as is; a NumPy scalar or 0-d array cast to the object dtype,
            // i.e. its Python value (a float16 a Python float, a char a Python int).
            PolyNumber xv = xo.Kind == RootArgKind.Weak ? PolyNumber.FromObject(x) : NDPolyAlgebra.ObjectLoopValue(xo.Array);
            object[] items = Backends.UnmanagedStorage.IsSequence(r) ? Backends.UnmanagedStorage.SequenceItems(r) : new[] { r };
            // x - r is a whole array op: every subtraction runs (and may raise) before the product starts.
            var diff = new PolyNumber[items.Length];
            for (int k = 0; k < items.Length; k++)
                diff[k] = PolyNumber.Binary(BinaryOp.Subtract, xv, PolyNumber.FromObject(items[k]));
            var acc = diff[0];
            for (int k = 1; k < diff.Length; k++)
                acc = PolyNumber.Binary(BinaryOp.Multiply, acc, diff[k]);
            if (!acc.TryDiscoveredDtype(out var t))
                throw acc.ObjectArrayRefusal();
            var y = new NDArray(t, Shape.NewScalar(), false);
            acc.WriteElement(t, (byte*)y.Storage.Address);
            return y;
        }

        /// <summary>
        ///     The Python type name an operand's element has in np.subtract's object loop (the object cast's: every float dtype
        ///     a Python float, every integer a Python int, a str array's elements str), or null where NumSharp cannot name it.
        /// </summary>
        /// <param name="arg">The caller's argument (a weak scalar's own C# type gives its Python type).</param>
        /// <param name="a">The converted argument.</param>
        /// <returns>The name, or null for an object array's items and for a decimal (NumSharp's dtype with no Python value).</returns>
        private static string ObjectLoopTypeName(object arg, in RootArg a)
        {
            switch (a.Kind)
            {
                case RootArgKind.None:
                    return "NoneType";
                case RootArgKind.Weak:
                    return arg switch
                    {
                        bool => "bool",
                        float or double => "float",
                        Complex => "complex",
                        _ => "int",   // the integer types and BigInteger: Python ints
                    };
                case RootArgKind.Numeric:
                case RootArgKind.Str:
                    if (a.Array is null)
                        return null;
                    return a.Array.typecode switch
                    {
                        NPTypeCode.Boolean => "bool",
                        NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double => "float",
                        NPTypeCode.Complex => "complex",
                        NPTypeCode.String => a.Array.dtype.kind == 'S' ? "bytes" : "str",
                        NPTypeCode.Decimal => null,
                        _ => "int",   // every integer dtype and char (NumPy's uint16)
                    };
                default:
                    return null;
            }
        }

        /// <summary>
        ///     NumPy's answer for a <c>x - r</c> with a string operand and no object one: np.subtract's no-loop TypeError, which
        ///     NumPy resolves before it broadcasts (so it wins over a shape mismatch), naming each operand's descriptor — a weak
        ///     Python scalar's as np.array discovers it (a bool <c>bool</c>, an int <c>int64</c>, a float <c>float64</c>).
        /// </summary>
        /// <param name="x">The caller's points.</param>
        /// <param name="xo">x converted.</param>
        /// <param name="ro">r converted.</param>
        /// <param name="ra">r as NumPy subtracts it (an integer r already float64).</param>
        /// <returns>The exception to throw.</returns>
        private static Exception StrFromRoots(object x, in RootArg xo, in RootArg ro, NDArray ra)
        {
            if (ra is null || (xo.Kind != RootArgKind.Weak && xo.Array is null))
                return xo.Refusal ?? ro.Refusal ?? new NotSupportedException("NumPy builds a str array here that NumSharp cannot");
            NDArray xs = xo.Array, owned = null;
            try
            {
                if (xo.Kind == RootArgKind.Weak)
                    xs = owned = NDPolySeries.PolyUfuncOperand(x);   // the WEAK 0-d array of the discovered dtype a Python scalar stands for (a Python int names int64 in the legacy text, _PyLongDType in the class-level one)
                return NDStringUfuncs.BinaryError("subtract", xs, ra, null, null);
            }
            finally
            {
                owned?.Dispose();
            }
        }

        /// <summary>
        ///     The numeric <c>np.prod(x - r, axis=0)</c>: NumPy's temporary's dims and NpyIter layout (from which the result's
        ///     layout and, for complex128 / float16, the product form follow), then the root-product kernel.
        /// </summary>
        /// <param name="xo">x converted (numeric or weak).</param>
        /// <param name="ra">r, at least 1-D, numeric, not integer / bool.</param>
        /// <param name="rLayout">The layout NumPy's subtraction reads r through (same dims as <paramref name="ra"/>).</param>
        /// <param name="expand">r gains x's rank of trailing unit axes (tensor with an ndarray x).</param>
        /// <returns>The fresh result.</returns>
        /// <exception cref="IncorrectShapeException"><c>tensor=False</c> and the shapes do not broadcast.</exception>
        /// <exception cref="OverflowException">A weak Python int x does not convert to the loop's float dtype.</exception>
        private static NDArray NumericFromRoots(in RootArg xo, NDArray ra, Shape rLayout, bool expand)
        {
            bool weak = xo.Kind == RootArgKind.Weak;
            NDArray xa = weak ? null : xo.Array;
            var xMode = weak ? PolyXMode.Weak : xa.ndim == 0 ? PolyXMode.Shared : PolyXMode.PerPoint;
            bool scalarX = xMode != PolyXMode.PerPoint;
            long[] rDims = ra.Shape.dimensions;
            // The dims of NumPy's x - r temporary: (r.shape + x.shape) under tensor, the right-aligned broadcast otherwise.
            long[] tempDims = expand ? ConcatDims(rDims, xa.Shape.dimensions)
                            : scalarX ? (long[])rDims.Clone()
                            : BroadcastRootDims(xa.Shape.dimensions, rDims);
            long nr = rDims[0];
            int nOut = tempDims.Length - 1;
            var outDims = new long[nOut];
            System.Array.Copy(tempDims, 1, outDims, 0, nOut);
            long outSize = 1;
            foreach (var d in outDims)
                outSize *= d;

            // The loop dtype (the root product keeps x - r's), needed before the product form can be chosen.
            NPTypeCode tx = weak ? NPTypeCode.Empty : xa.typecode;
            var prog = ILKernelGenerator.PolyProgram(PolyBasis.Power, xMode, xo.Weak.Kind, PolyRootForm.Slab);
            NPTypeCode t = PolyTyping.ResultType(prog, tx, ra.typecode, 1, 0);
            bool twoForms = t is NPTypeCode.Complex or NPTypeCode.Half;

            Shape yLayout;
            bool rootsInnermost;
            if ((scalarX || xa.Shape.IsContiguous) && (ra.ndim == 1 || rLayout.IsContiguous))
            {
                // Every voter is C-ordered (a 1-D r votes on the roots axis alone), so the temporary is C: the roots axis is
                // outermost, the innermost non-unit axis only when it is the only one (one output element).
                yLayout = nOut == 0 ? Shape.NewScalar() : Shape.Adopt(outDims);
                rootsInnermost = nr > 1 && outSize == 1;
            }
            else
            {
                // NpyIter's vote over the subtraction's operands: x (none for a scalar) and r with its trailing unit axes.
                Shape rp = expand ? AppendUnitAxes(rLayout, xa.ndim) : rLayout;
                Shape tempLayout = scalarX
                    ? Shape.NpyIterOutputShape((long[])tempDims.Clone(), new ReadOnlySpan<Shape>(in rp))
                    : Shape.NpyIterOutputShape((long[])tempDims.Clone(), new[] { xa.Shape, rp });
                yLayout = nOut == 0 ? Shape.NewScalar() : Shape.ReductionOutputShape(outDims, tempLayout, 0);
                rootsInnermost = RootsAxisInnermost(tempLayout);
            }
            var form = twoForms && rootsInnermost ? PolyRootForm.Reduce : PolyRootForm.Slab;
            return EvaluateRoots(xa, xo.Weak, xMode, ra, expand, yLayout, form);
        }

        /// <summary>
        ///     Runs the root-product kernel: the operand forms of <c>{p}val</c>'s <see cref="Evaluate"/> (a 1-D r in auxdata; an
        ///     N-D r as the <c>r[0]</c> operand read at a stride), into a fresh result laid out as <paramref name="yLayout"/>. An
        ///     empty r of any rank is np.prod's identity, filled without the kernel (after the weak x's conversion ran).
        /// </summary>
        /// <param name="xa">x as an array (null when weak).</param>
        /// <param name="xw">x as a Python value (when weak).</param>
        /// <param name="xMode">How x arrives.</param>
        /// <param name="rr">r: at least 1-D, numeric, not integer / bool.</param>
        /// <param name="expand">tensor with an ndarray x: <c>r[0]</c> gains x's rank of trailing unit axes.</param>
        /// <param name="yLayout">The result layout (a fresh dense shape, 0-d for a scalar x with a 1-D r).</param>
        /// <param name="form">The reduction loop NumPy's np.prod runs (complex128 / float16 only).</param>
        /// <returns>The result.</returns>
        /// <exception cref="OverflowException">A weak Python int x does not convert to the loop dtype.</exception>
        private static NDArray EvaluateRoots(NDArray xa, PyScalar xw, PolyXMode xMode, NDArray rr, bool expand, Shape yLayout,
            PolyRootForm form)
        {
            long nr = rr.Shape.dimensions[0];
            // An empty r reads no root (the result is a fill below); its kernel still comes from the auxdata form, so the
            // weak x's conversion runs whatever r's rank.
            bool coefOperand = rr.ndim > 1 && nr > 0;
            NPTypeCode tx = xMode == PolyXMode.Weak ? NPTypeCode.Empty : xa.typecode;
            var k = ILKernelGenerator.GetPolyEvalKernel(PolyBasis.Power, xMode, tx, xw.Kind, rr.typecode, nr, coefOperand,
                scalarMath: false, PolyUnitBroadcast.None, form);

            IntPtr scratch = IntPtr.Zero;
            NDArray preconv = null, c0 = null, y = null;
            try
            {
                long* aux = stackalloc long[7];
                int rsize = rr.dtypesize;
                aux[0] = rr.Shape.strides[0] * rsize;
                aux[1] = nr;
                // A weak x that does not convert to the loop dtype raises here, before anything is allocated — NumPy converts
                // it while it resolves the subtraction.
                aux[2] = k.Pool.Prepare(nr, xw, out scratch);
                aux[3] = (long)((byte*)rr.Storage.Address + rr.Shape.offset * rsize);
                if (k.Preconv)
                {
                    // The roots in the loop dtype: NumPy's subtraction casts r exactly so (a widening), once here.
                    preconv = rr.astype(k.Tl);
                    aux[4] = (long)((byte*)preconv.Storage.Address + preconv.Shape.offset * preconv.dtypesize);
                    aux[5] = preconv.Shape.strides[0] * preconv.dtypesize;
                }
                if (xMode == PolyXMode.Shared)
                    aux[6] = (long)((byte*)xa.Storage.Address + xa.Shape.offset * xa.dtypesize);

                y = new NDArray(k.Tl, yLayout, false);
                if (y.size == 0)
                    return Detach(ref y);
                if (nr == 0)
                {
                    // No root: np.prod's identity at every point. The kernel would write it too, but it walks x (and the
                    // iterator) only to discard every value: the vectorized fill writes the result alone (100K float64
                    // points: 25.8 us through the kernel, below NumPy's 22.8 us identity fill). The weak x has already been
                    // converted (Prepare), so a Python int too large for the dtype still raised.
                    y.fill(1L);
                    return Detach(ref y);
                }

                if (!coefOperand)
                {
                    if (xMode != PolyXMode.PerPoint)
                    {
                        // A scalar x with a 1-D r (an empty N-D r returned above): one point, no iterator — the kernel's
                        // only operand is y.
                        void** ptrs = stackalloc void*[1];
                        long* strides = stackalloc long[1];
                        ptrs[0] = (byte*)y.Storage.Address;
                        strides[0] = 0;
                        k.Fn(ptrs, strides, 1, aux);
                        return Detach(ref y);
                    }
                    // Roots in auxdata at an array x: with a 1-D r the result has x's dims, and a strided x is gathered into
                    // contiguous chunks so the vector chains run ({p}val's measured choice).
                    var gflags = NDIterGlobalFlags.EXTERNAL_LOOP;
                    var xflags = NDIterPerOpFlags.READONLY;
                    bool buffered = !IsElementContiguousInOrder(xa, y);
                    if (buffered)
                    {
                        gflags |= NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.GROWINNER;
                        xflags |= NDIterPerOpFlags.CONTIG;
                    }
                    using (var it = NDIterRef.MultiNew(2, new[] { xa, y }, gflags, NPY_ORDER.NPY_KEEPORDER,
                               NPY_CASTING.NPY_SAFE_CASTING, new[] { xflags, NDIterPerOpFlags.WRITEONLY },
                               buffered ? new[] { xa.typecode, k.Tl } : null))
                        it.ForEach(k.Fn, aux);
                    return Detach(ref y);
                }

                // An N-D r: r[0] is the operand, the kernel reads r[k] at r[0] + k*kstride — so never buffered (a buffer would
                // hold r[0] alone).
                if (xMode == PolyXMode.PerPoint)
                {
                    if (expand)
                    {
                        // r.reshape(r.shape + (1,)*x.ndim): r[0] gains x.ndim trailing unit axes (a view).
                        var axes = new int[xa.ndim];
                        for (int d = 0; d < axes.Length; d++)
                            axes[d] = rr.ndim - 1 + d;
                        using (var row0 = rr[0])
                            c0 = np.expand_dims(row0, axes);
                    }
                    else
                        c0 = rr[0];
                    using (var it = NDIterRef.MultiNew(3, new[] { xa, c0, y }, NDIterGlobalFlags.EXTERNAL_LOOP,
                               NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                               new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, null))
                        it.ForEach(k.Fn, aux);
                    return Detach(ref y);
                }
                // A scalar x over an N-D r: the points are r's columns.
                c0 = rr[0];
                using (var it = NDIterRef.MultiNew(2, new[] { c0, y }, NDIterGlobalFlags.EXTERNAL_LOOP,
                           NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                           new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, null))
                    it.ForEach(k.Fn, aux);
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
        ///     Whether the roots axis (axis 0) is the innermost non-unit axis of NumPy's <c>x - r</c> temporary — the axis its
        ///     reduction iterator walks in the inner loop, with the output's stride 0.
        /// </summary>
        /// <param name="temp">The temporary's dense layout (NpyIter's), positive strides.</param>
        /// <returns>True when the roots axis has the smallest stride among the axes longer than one.</returns>
        private static bool RootsAxisInnermost(in Shape temp)
        {
            var d = temp.dimensions;
            var s = temp.strides;
            if (d[0] <= 1)
                return false;
            for (int i = 1; i < d.Length; i++)
                if (d[i] > 1 && s[i] < s[0])
                    return false;
            return true;
        }

        /// <summary>
        ///     NumPy's broadcast of <c>x - r</c> with <c>tensor=False</c> (or no tensor reshape): right-aligned, an extent of 1
        ///     yielding.
        /// </summary>
        /// <param name="x">x's dims.</param>
        /// <param name="r">r's dims.</param>
        /// <returns>The broadcast dims (a new array).</returns>
        /// <exception cref="IncorrectShapeException">Incompatible shapes: NumPy's text, x's shape first —
        ///     <c>operands could not be broadcast together with shapes (3,) (4,5) </c>.</exception>
        private static long[] BroadcastRootDims(long[] x, long[] r)
        {
            int n = Math.Max(x.Length, r.Length);
            var b = new long[n];
            for (int d = 0; d < n; d++)
            {
                long dx = d < n - x.Length ? 1 : x[d - (n - x.Length)];
                long dr = d < n - r.Length ? 1 : r[d - (n - r.Length)];
                if (dx != dr && dx != 1 && dr != 1)
                    throw new IncorrectShapeException(
                        $"operands could not be broadcast together with shapes {new Shape(x).ToPythonTuple()} {new Shape(r).ToPythonTuple()} ");
                b[d] = dx == 1 ? dr : dx;
            }
            return b;
        }

        /// <summary>The concatenation <c>a + b</c> of two dims tuples (a new array).</summary>
        /// <param name="a">The leading dims.</param>
        /// <param name="b">The trailing dims.</param>
        /// <returns>The joined dims.</returns>
        private static long[] ConcatDims(long[] a, long[] b)
        {
            var c = new long[a.Length + b.Length];
            a.CopyTo(c, 0);
            b.CopyTo(c, a.Length);
            return c;
        }

        /// <summary>
        ///     A layout with <paramref name="n"/> trailing unit axes appended (stride 0: a unit axis abstains from NpyIter's
        ///     vote, so its stride is never read) — <c>r.reshape(r.shape + (1,)*n)</c> as the subtraction sees it.
        /// </summary>
        /// <param name="s">The layout.</param>
        /// <param name="n">How many unit axes.</param>
        /// <returns>A new shape (never used to allocate or index memory).</returns>
        private static Shape AppendUnitAxes(in Shape s, int n)
        {
            int nd = s.NDim;
            var dims = new long[nd + n];
            var strides = new long[nd + n];
            System.Array.Copy(s.dimensions, dims, nd);
            System.Array.Copy(s.strides, strides, nd);
            for (int d = nd; d < nd + n; d++)
                dims[d] = 1;
            return Shape.Adopt(dims, strides);
        }
    }
}
