using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyWeight.cs — {p}weight: the weight function of each orthogonal basis
// =============================================================================
//
// NumPy 2.4.2 writes each weight as ONE Python statement over whatever the caller passed:
//
//   chebweight(x)  = 1.0/(np.sqrt(1.0 + x)*np.sqrt(1.0 - x))      (chebyshev.py)
//   legweight(x)   = x*0.0 + 1.0                                   (legendre.py)
//   lagweight(x)   = np.exp(-x)                                    (laguerre.py)
//   hermweight(x)  = np.exp(-x**2)                                 (hermite.py)
//   hermeweight(x) = np.exp(-.5*x**2)                              (hermite_e.py)
//
// x is never converted first, so which arithmetic runs each operator depends on what x IS:
//
//   * A Python scalar (a C# bool / integer / float / double / Complex) is Python arithmetic until the first np.*
//     call: legweight(0.5) is a Python float, chebweight(2) is nan (1.0 - 2 is -1.0, np.sqrt(-1.0) is nan), and
//     hermweight(2**70) raises NumPy's object-loop TypeError (-(2**70)**2 fits no integer dtype, so np.exp gets an
//     object array). PolyNumber replays each operator with PyScalar (CPython 3.12's arithmetic).
//   * `x**2` of an ndarray is NOT np.power(x, 2): ndarray.__pow__ calls fast_scalar_power (number.c), which maps
//     an exact Python int 2 to np.square for EVERY dtype. So a bool array squares to int8 and hermweight of a bool
//     array is float16, where np.power would give int64 and a float64 result (plan F1).
//   * A Python sequence (a C# list, object[], tuple), None or a str meets Python's operator itself, which raises
//     before NumPy sees the value — `1.0 + [0.5]` is "unsupported operand type(s) for +: 'float' and 'list'".
//   * A 0-d array or a NumPy scalar (C#'s Half / char / decimal) makes every operator a ufunc or scalarmath call
//     whose result is a NumPy scalar; the complex product of two of those is scalarmath's naive one.
//
// ROUTES. An ndarray of rank >= 1 is computed in ONE fused np.evaluate pass over NumPy's expression, node for node
// (NDExpr types every node as NumPy does, the literals stay weak, `x**2` is the Square node), byte-identical to
// the statement-by-statement composition and to NumPy for every real dtype (plan §3.2: 2.4-9.7x NumPy at 1M, the
// fused pass saves the intermediate arrays NumPy materializes). Two dtypes go through the statements on the house
// ufuncs instead:
//   * float16, because the house float16 kernels (widen, compute, narrow in SIMD) are 2-5x faster than NDExpr's
//     float16 path (plan §3.2);
//   * decimal (no NumPy dtype), because NDExpr computes a decimal transcendental through double while np.sqrt /
//     np.exp use the house decimal routines — the statements over the house ufuncs ARE NumSharp's definition of a
//     weight with no NumPy counterpart, so the decimal answer must not depend on the route.
// String arrays go the same way, so the house ufunc raises NumPy's no-loop error for the first statement.
// Three shortcuts keep the array routes fast without changing a bit:
//   * a BROADCAST x is computed once on its un-broadcast core and copied into NumPy's layout (each weight is
//     elementwise and deterministic, so a repeated value has a repeated weight);
//   * an INTEGER or bool x is cast once into the first float loop — NumPy's own buffered input cast — so the float
//     part runs on one dtype and stays vectorized (a fused pass over a mixed-dtype tree converts element by
//     element and runs scalar); the integer part (-x, x**2) still runs in x's dtype, where it wraps;
//   * an N-D x that is not C-contiguous is computed over its C-ordered axis permutation and the C result is
//     relabelled with NumPy's layout (mapdomain's route).
// Everything scalar-sized is the same statement replay over PolyNumber, whose ufunc calls run the house kernel on
// one stack element (PolyNumber.FloatUfunc): no 0-d arrays but the result.
//
// SPEED (benchmark/polynomial/polyweight_results.md, 189 cells, all bit-exact): geomean 1.78x NumPy. The cells below
// 1.5x are floors both libraries share or that NumSharp cannot cross here: a float64 exp is the C runtime's on both
// sides (about 1.0x up to 100K points; above 1.5x at 1M, where NumPy pays for its temporaries); a float32 exp is the
// same AVX2 algorithm on both sides (1.1-1.35x); a float16 exp is the house np.exp kernel, itself 0.77x NumPy's HALF
// loop; and a scalar call returns a 0-d NDArray, whose allocation alone (0.34 us) costs more than NumPy's whole
// Python-float legweight (0.11 us).
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     The weight functions of the five orthogonal bases (<c>chebweight</c>, <c>legweight</c>, <c>lagweight</c>,
    ///     <c>hermweight</c>, <c>hermeweight</c>): NumPy 2.4.2's one statement each, run on whatever the caller passed
    ///     with the arithmetic Python would use (see the file header).
    /// </summary>
    internal static class NDPolyWeight
    {
        /// <summary>
        ///     <c>{p}weight(x)</c> for the basis <paramref name="basis"/>.
        /// </summary>
        /// <param name="basis">Chebyshev, Legendre, Laguerre, Hermite or HermiteE (the power basis has no weight).</param>
        /// <param name="x">
        ///     The points: an <see cref="NDArray"/> of any layout and dtype, a typed C# array (an ndarray), a Python
        ///     scalar (a C# <c>bool</c>, integer, <see cref="System.Numerics.BigInteger"/>, <c>float</c>, <c>double</c> or
        ///     <see cref="System.Numerics.Complex"/>) or a NumPy scalar (<see cref="Half"/>, <c>char</c>, <c>decimal</c>).
        /// </param>
        /// <returns>
        ///     The weights. An ndarray argument of rank ≥ 1 gives an array of x's shape in NumPy's result dtype and layout;
        ///     a scalar-sized argument gives a 0-d array (NumPy returns a Python float for <c>legweight</c> of a Python
        ///     scalar, and a NumPy scalar otherwise).
        /// </returns>
        /// <exception cref="TypeError">
        ///     <paramref name="x"/> is a Python sequence (a list, <c>object[]</c> or tuple), null or a string: Python's own
        ///     operator error for the basis's first operator (<see cref="FirstOperatorTypeError"/>). Or a bool array for
        ///     <c>lagweight</c> (NumPy's boolean-negative error), or a Python int whose first NumPy operand fits neither
        ///     int64 nor uint64 (NumPy's object-loop error).
        /// </exception>
        /// <exception cref="OverflowException">
        ///     CPython's own overflow on a Python-scalar x: a huge int meeting a float (<c>int too large to convert to
        ///     float</c>), or the square of a float or complex past the float range (<see cref="PyScalar.PowTwo"/>).
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="basis"/> is the power basis.</exception>
        internal static NDArray Weight(PolyBasis basis, object x)
        {
            // A sequence, None or a str is not converted by NumPy's statement: the first Python operator meets it
            // and raises. Checked before anything is built, so nothing is allocated for the error.
            string pythonType = OperandPythonType(x);
            if (pythonType != null)
                throw FirstOperatorTypeError(basis, pythonType);

            PolyNumber v = PolyNumber.FromObject(x);
            if (!v.IsNdArray)
                return Statements(basis, v).ToNDArray();

            NDArray xa = v.Array;
            // The package computes with plain arrays: a numpy.matrix x is its plain array here (NumPy's own answer would
            // apply the matrix operators — `x**2` the matrix power, `sqrt(1+x)*sqrt(1-x)` a matrix product — which no
            // weight's definition intends; mapdomain makes the same choice).
            using var plain = NDMatrixBoundary.PlainFor(xa);
            return WeightOfArray(basis, xa);
        }

        /// <summary>
        ///     The weight of an ndarray of rank ≥ 1, laid out as NumPy lays it out. Every statement is a ufunc whose only
        ///     array operand is x (or an intermediate laid out like x), so the result is laid out as NpyIter lays out an
        ///     output over x alone (<see cref="Shape.NpyIterOutputShape"/>): an F-ordered or permuted x keeps its memory
        ///     order, a broadcast axis abstains.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="xa">The argument (rank ≥ 1, any layout).</param>
        /// <returns>A new array of x's shape in NumPy's result dtype and layout.</returns>
        /// <exception cref="TypeError">A bool array for <c>lagweight</c>, or a string array (the first statement's ufunc).</exception>
        /// <remarks>
        ///     <para><b>A broadcast x</b> (a stride-0 axis longer than 1) repeats its values, and each weight is an
        ///     elementwise, deterministic function: the weight of a repeated value is the weight of the value. So the weight
        ///     runs once over the un-broadcast core — every stride-0 axis cut to one element — and is copied into NumPy's
        ///     layout. NumPy computes every repeat; the bits are the same.</para>
        ///     <para><b>Any other N-D x that is not C-contiguous</b> makes NumPy's layout something other than C only when
        ///     its memory order says so; then the weight runs over <c>np.transpose(x, perm)</c>, perm = that layout's axes
        ///     in memory order, and its C result is RELABELLED with NumPy's layout (<see cref="NDArray.AdoptLayout"/>): no
        ///     element moves (mapdomain's route; an iterator writing the permuted layout directly walks it in logical order,
        ///     gathering, 2.7× slower).</para>
        /// </remarks>
        private static NDArray WeightOfArray(PolyBasis basis, NDArray xa)
        {
            if (xa.size > 0 && !xa.Shape.IsContiguous)
            {
                Shape xs = xa.Shape;
                if (xs.IsBroadcasted)
                {
                    // The core: every stride-0 axis of extent > 1 cut to its first element (a view; the other axes intact).
                    var cut = new Slice[xa.ndim];
                    for (int d = 0; d < cut.Length; d++)
                        cut[d] = xs.strides[d] == 0 && xs.dimensions[d] > 1 ? new Slice(0, 1) : Slice.All;
                    NDArray core = xa[cut];
                    NDArray wc = WeightOfArray(basis, core);
                    try
                    {
                        Shape outLayout = Shape.NpyIterOutputShape((long[])xs.dimensions.Clone(), new ReadOnlySpan<Shape>(in xs));
                        var r = new NDArray(wc.typecode, outLayout, false);
                        using var spread = np.broadcast_to(wc, Shape.Adopt((long[])xs.dimensions.Clone()));
                        np.copyto(r, spread);
                        return r;
                    }
                    finally
                    {
                        wc.Dispose();
                        core.Dispose();   // the view's own counted reference; the caller's array is untouched
                    }
                }
                if (xa.ndim >= 2)
                {
                    Shape layout = Shape.NpyIterOutputShape((long[])xs.dimensions.Clone(), new ReadOnlySpan<Shape>(in xs));
                    if (!layout.IsContiguous)
                    {
                        NDArray xv = np.transpose(xa, layout.DenseAxisOrder());
                        try
                        {
                            return NDArray.AdoptLayout(Compute(basis, xv), layout);
                        }
                        finally
                        {
                            xv.Dispose();   // the view's own counted reference; the caller's array is untouched
                        }
                    }
                }
            }
            return Compute(basis, xa);
        }

        /// <summary>
        ///     The weight of an ndarray of rank ≥ 1, by its route: the statements on the house ufuncs for float16, decimal and
        ///     the string dtypes (<see cref="Fuses"/>); a cast into the first float loop and then one single-dtype pass for an
        ///     integer or bool x (<see cref="CastFirst"/>); otherwise one fused
        ///     <see cref="np.evaluate(NDExpr, NDArray, NDArray, DType, string, char)"/> pass over the basis's tree.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="x">The argument (rank ≥ 1; a view of the caller's x when NumPy's layout is not C).</param>
        /// <returns>A new array of x's shape (C-ordered for a C-ordered x, which <see cref="NDArray.AdoptLayout"/> checks
        ///     rather than trusts).</returns>
        /// <exception cref="TypeError">A bool array for <c>lagweight</c>, or a string array (the first statement's ufunc).</exception>
        private static NDArray Compute(PolyBasis basis, NDArray x)
        {
            if (!Fuses(x))
                return Statements(basis, PolyNumber.FromArray(x)).Array;
            if (x.size > 0 && PolyTyping.IsIntLike(x.typecode) && CastFirst(basis, x) is { } cast)
                return cast;
            return np.evaluate(Tree(basis, x));
        }

        /// <summary>
        ///     The weight of an integer or bool x through NumPy's own cast: NumPy's ufuncs cast such an x into each
        ///     statement's float loop as they run (buffered). Here the integer part of the statement — <c>-x</c>,
        ///     <c>-x**2</c>, <c>x**2</c>, in x's dtype, where it wraps — runs first, then ONE cast into the float loop, then
        ///     the float part on that single dtype, which keeps it vectorized: a fused pass over a mixed-dtype tree converts
        ///     element by element and runs scalar. The bits are NumPy's: its input cast, then its loop.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="x">An integer or bool array (rank ≥ 1, non-empty).</param>
        /// <returns>The weight, or null when the fused tree is the faster route: an <c>exp</c> loop of float64, where the C
        ///     runtime's scalar <c>exp</c> bounds both routes and the fused pass saves the intermediate arrays.</returns>
        /// <exception cref="TypeError">A bool array for <c>lagweight</c> (NumPy's boolean-negative error).</exception>
        /// <remarks>
        ///     Measured, 100K points (NumPy in brackets): chebweight int16 531 → 322 µs (509), uint8 511 → 284 µs (494), bool
        ///     498 → 282 µs (496); lagweight int16 427 → 128 µs (154); hermweight int16 465 → 142 µs (199), bool 4,439 →
        ///     1,328 µs (1,481); hermeweight int16 620 → 559 µs (634).
        /// </remarks>
        private static NDArray CastFirst(PolyBasis basis, NDArray x)
        {
            switch (basis)
            {
                case PolyBasis.Chebyshev:
                case PolyBasis.Legendre:
                {
                    // `1.0 + x` and `x*0.0`: a Python float against an integer or bool array is the float64 loop.
                    using var xf = x.astype(NPTypeCode.Double);
                    return np.evaluate(Tree(basis, xf));
                }
                case PolyBasis.Laguerre:
                case PolyBasis.Hermite:
                {
                    // -x / -x**2 in x's dtype, then np.exp's float tier of that dtype. np.square of a bool is int8, whose
                    // tier (float16) is the bool's own, so x's dtype decides the loop either way.
                    NPTypeCode loop = PolyNumber.FloatLoop(x.typecode);
                    if (loop == NPTypeCode.Double)
                        return null;
                    var xn = PolyNumber.FromArray(x);
                    using NDArray prefix = (basis == PolyBasis.Laguerre ? PolyNumber.Negate(xn) : PolyNumber.Negate(PolyNumber.PowTwo(xn))).Array;
                    using NDArray cast = prefix.astype(loop);
                    return np.exp(cast);
                }
                case PolyBasis.HermiteE:
                {
                    // x**2 in x's dtype, then the Python float -0.5 makes the float64 loop: exp(-0.5 * float64(x**2)).
                    using NDArray square = PolyNumber.PowTwo(PolyNumber.FromArray(x)).Array;
                    using NDArray sf = square.astype(NPTypeCode.Double);
                    return np.evaluate(NDExpr.Exp(-0.5 * (NDExpr)sf));
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(basis), basis, "the power basis has no weight function");
            }
        }

        /// <summary>
        ///     Whether an ndarray argument takes the fused pass: every dtype but float16 (the house float16 ufuncs are
        ///     faster), decimal (no NumPy dtype: the house ufuncs define its answer) and the string dtypes U / S /
        ///     StringDType (no numeric loop: the first statement's house ufunc raises NumPy's error for the pair).
        /// </summary>
        /// <param name="x">The argument (rank ≥ 1).</param>
        /// <returns>True when <see cref="Tree"/> computes it.</returns>
        private static bool Fuses(NDArray x)
            => x.typecode is not (NPTypeCode.Half or NPTypeCode.Decimal or NPTypeCode.String);

        /// <summary>
        ///     The basis's statement as an <see cref="NDExpr"/> tree over <paramref name="x"/>, operator for operator in
        ///     NumPy's order. The C# <c>double</c> literals are NEP 50 weak (they adopt x's dtype where NumPy's Python
        ///     floats do), and <c>x**2</c> is the Square node (<c>np.square</c>, NumPy's <c>fast_scalar_power</c>).
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="x">The argument.</param>
        /// <returns>The tree; every call with the same basis and dtype hits one cached kernel.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The power basis.</exception>
        internal static NDExpr Tree(PolyBasis basis, NDArray x)
        {
            NDExpr X = x;
            return basis switch
            {
                PolyBasis.Chebyshev => 1.0 / (NDExpr.Sqrt(1.0 + X) * NDExpr.Sqrt(1.0 - X)),
                PolyBasis.Legendre => X * 0.0 + 1.0,
                PolyBasis.Laguerre => NDExpr.Exp(-X),
                PolyBasis.Hermite => NDExpr.Exp(-NDExpr.Square(X)),
                PolyBasis.HermiteE => NDExpr.Exp(-0.5 * NDExpr.Square(X)),
                _ => throw new ArgumentOutOfRangeException(nameof(basis), basis, "the power basis has no weight function"),
            };
        }

        /// <summary>
        ///     The basis's statement replayed operator by operator over <see cref="PolyNumber"/>, in Python's evaluation
        ///     order (left operand first, so the first error is NumPy's): CPython arithmetic between Python values,
        ///     scalarmath between NumPy scalars, the house ufuncs as soon as an ndarray is involved.
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="x">The argument: any <see cref="PolyNumber"/>.</param>
        /// <returns>The weight, of NumPy's kind: a Python float (<c>legweight</c> of a Python scalar), a NumPy scalar,
        ///     or an ndarray.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The power basis.</exception>
        internal static PolyNumber Statements(PolyBasis basis, in PolyNumber x)
        {
            var one = PolyNumber.FromPython(PyScalar.Float(1.0));
            switch (basis)
            {
                case PolyBasis.Chebyshev:
                {
                    // 1.0/(np.sqrt(1.0 + x)*np.sqrt(1.0 - x)): the left sqrt is evaluated before the right one.
                    var left = PolyNumber.FloatUfunc(UnaryOp.Sqrt, "sqrt", s_sqrt, PolyNumber.Binary(BinaryOp.Add, one, x));
                    var right = PolyNumber.FloatUfunc(UnaryOp.Sqrt, "sqrt", s_sqrt, PolyNumber.Binary(BinaryOp.Subtract, one, x));
                    return PolyNumber.Binary(BinaryOp.Divide, one, PolyNumber.Binary(BinaryOp.Multiply, left, right));
                }
                case PolyBasis.Legendre:
                {
                    // x*0.0 + 1.0
                    var zero = PolyNumber.FromPython(PyScalar.Float(0.0));
                    return PolyNumber.Binary(BinaryOp.Add, PolyNumber.Binary(BinaryOp.Multiply, x, zero), one);
                }
                case PolyBasis.Laguerre:
                    // np.exp(-x)
                    return PolyNumber.FloatUfunc(UnaryOp.Exp, "exp", s_exp, PolyNumber.Negate(x));
                case PolyBasis.Hermite:
                    // np.exp(-x**2): ** binds tighter than unary minus.
                    return PolyNumber.FloatUfunc(UnaryOp.Exp, "exp", s_exp, PolyNumber.Negate(PolyNumber.PowTwo(x)));
                case PolyBasis.HermiteE:
                {
                    // np.exp(-.5*x**2): ** first, then the Python float -0.5 times it.
                    var minusHalf = PolyNumber.FromPython(PyScalar.Float(-0.5));
                    return PolyNumber.FloatUfunc(UnaryOp.Exp, "exp", s_exp, PolyNumber.Binary(BinaryOp.Multiply, minusHalf, PolyNumber.PowTwo(x)));
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(basis), basis, "the power basis has no weight function");
            }
        }

        /// <summary><c>np.sqrt</c> as a delegate, built once (a static lambda: no closure).</summary>
        private static readonly Func<NDArray, NDArray> s_sqrt = static a => np.sqrt(a);

        /// <summary><c>np.exp</c> as a delegate, built once.</summary>
        private static readonly Func<NDArray, NDArray> s_exp = static a => np.exp(a);

        /// <summary>
        ///     The Python type name of an argument Python's operators refuse before NumPy sees it, or null for an
        ///     argument NumPy computes with: <c>NoneType</c> for null, <c>str</c> for a string, <c>tuple</c> for an
        ///     <see cref="System.Runtime.CompilerServices.ITuple"/>, <c>list</c> for every other Python sequence of the
        ///     house mapping (an <c>object[]</c>, a non-dtype C# array, any other enumerable — see
        ///     <see cref="Backends.UnmanagedStorage.IsSequence"/>). A typed dtype array is an ndarray and is computed with.
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <returns>The type name, or null.</returns>
        private static string OperandPythonType(object x) => x switch
        {
            null => "NoneType",
            string => "str",
            System.Runtime.CompilerServices.ITuple => "tuple",
            _ => Backends.UnmanagedStorage.IsSequence(x) ? "list" : null,
        };

        /// <summary>
        ///     Python's TypeError for the basis's first operator applied to a value of Python type
        ///     <paramref name="pythonType"/> (probed, NumPy 2.4.2 on CPython 3.12):
        ///     <list type="bullet">
        ///         <item><description><c>chebweight</c>: <c>1.0 + x</c> — <c>unsupported operand type(s) for +: 'float' and 'list'</c>;</description></item>
        ///         <item><description><c>legweight</c>: <c>x*0.0</c> — a list, tuple or str is a sequence, whose repetition needs an int:
        ///         <c>can't multiply sequence by non-int of type 'float'</c>; None: <c>unsupported operand type(s) for *: 'NoneType' and 'float'</c>;</description></item>
        ///         <item><description><c>lagweight</c>: <c>-x</c> — <c>bad operand type for unary -: 'list'</c>;</description></item>
        ///         <item><description><c>hermweight</c> / <c>hermeweight</c>: <c>x**2</c> (evaluated first) —
        ///         <c>unsupported operand type(s) for ** or pow(): 'list' and 'int'</c>.</description></item>
        ///     </list>
        /// </summary>
        /// <param name="basis">The basis.</param>
        /// <param name="pythonType">The argument's Python type name.</param>
        /// <returns>The exception to throw.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The power basis.</exception>
        private static TypeError FirstOperatorTypeError(PolyBasis basis, string pythonType) => basis switch
        {
            PolyBasis.Chebyshev => new TypeError($"unsupported operand type(s) for +: 'float' and '{pythonType}'"),
            PolyBasis.Legendre => pythonType == "NoneType"
                ? new TypeError("unsupported operand type(s) for *: 'NoneType' and 'float'")
                : new TypeError("can't multiply sequence by non-int of type 'float'"),
            PolyBasis.Laguerre => new TypeError($"bad operand type for unary -: '{pythonType}'"),
            PolyBasis.Hermite or PolyBasis.HermiteE => new TypeError($"unsupported operand type(s) for ** or pow(): '{pythonType}' and 'int'"),
            _ => throw new ArgumentOutOfRangeException(nameof(basis), basis, "the power basis has no weight function"),
        };
    }
}
