using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyBase.Algebra.cs — evaluation AT a series, and the class methods' shared cores
// =============================================================================
//
// convert / cast change basis by evaluating the series AT the target class's identity series:
// `self(kind.identity(domain, window=window, symbol=self.symbol))`. NumPy's __call__ then runs
// pu.mapdomain on the series — np.asanyarray(series) is a 0-d OBJECT array, so `scl * x` is the object
// loop (scl's Python value times the series: series.__rmul__) and `off + ...` meets the NumPy scalar
// offset (series.__radd__) — and {p}val's recurrence with x a series: every statement of the U3 step
// trees (PolySteps.Eval, NumPy's source expressions) is a Python operator whose operands are NumPy
// scalars (the coefficients c[k]), Python ints / floats (the recurrence's literals) or series, so it
// is either NumPy scalarmath (two numbers) or the series protocol (PyBinary, NDPolyBase.Protocol.cs).
// SeriesVal interprets the trees exactly that way, statement for statement, like NDPolyEval.ValAtScalar
// does for a NumPy-scalar x.
//
// The class methods (identity, basis, fromroots, fit, cast, Chebyshev.interpolate) are static members of
// each concrete class (C# cannot inherit a static that returns the derived type); their bodies are the
// cores here, keyed by the class record.
//
// =============================================================================

namespace NumSharp
{
    public abstract partial class NDPolyBase
    {
        // ---------------------------------------------------------------------------------------------
        //  Evaluation at a series
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>self(arg)</c> for a series <paramref name="arg"/>: <c>pu.mapdomain(arg, self.domain, self.window)</c>
        ///     — the object loop's <c>scl * arg</c> with scl's Python value, then the NumPy-scalar offset plus it — and
        ///     <c>{p}val(mapped, self.coef)</c> through the series protocol.
        /// </summary>
        /// <param name="arg">The series to evaluate at.</param>
        /// <returns>A new series of <paramref name="arg"/>'s class.</returns>
        /// <exception cref="TypeError">The protocol's errors (e.g. <c>Domains differ</c> for a NaN domain).</exception>
        private protected NDPolyBase EvaluateAtSeries(NDPolyBase arg)
        {
            var (off, scl) = MapParmsOf(domain, window);
            // The object-dtype multiply converts scl to the Python value NumPy's getitem makes (a float for every real
            // float dtype, a complex for complex) and calls Python's `*` with the series: series.__rmul__.
            var sclPy = NDPolyAlgebra.ObjectLoopValue(scl);
            var scaled = (NDPolyBase)PyBinary(PyBinOp.Mul, OfNumber(sclPy), Operand(arg));
            NDPolyBase x;
            try
            {
                // `off + <series>`: the NumPy scalar defers to the series (__array_ufunc__ = None): series.__radd__(off).
                x = (NDPolyBase)PyBinary(PyBinOp.Add, OfNumber(off), Operand(scaled));
            }
            finally
            {
                scaled.Dispose();
            }
            try
            {
                return SeriesVal(Basis, x, Coef);
            }
            finally
            {
                x.Dispose();
            }
        }

        /// <summary>One value of the series recurrence: a number (Python or NumPy scalar) or a series.</summary>
        private readonly struct SVal
        {
            /// <summary>The number (when <see cref="Series"/> is null).</summary>
            public readonly PolyNumber Num;
            /// <summary>The series, or null.</summary>
            public readonly NDPolyBase Series;

            /// <summary>A number.</summary><param name="num">The number.</param>
            public SVal(in PolyNumber num)
            {
                Num = num;
                Series = null;
            }

            /// <summary>A series.</summary><param name="series">The series.</param>
            public SVal(NDPolyBase series)
            {
                Num = default;
                Series = series;
            }

            /// <summary>The value as a protocol operand.</summary>
            public PyOperand Operand => Series is null ? OfNumber(Num) : NDPolyBase.Operand(Series);
        }

        /// <summary>
        ///     <c>{p}val(x, c)</c> with a series x: NumPy's recurrence (the strong step tree of the basis) interpreted with
        ///     NumPy scalarmath between numbers and the series protocol whenever a series is involved. Every intermediate series
        ///     is disposed; the final one is returned.
        /// </summary>
        /// <param name="basis">The basis of <paramref name="c"/>.</param>
        /// <param name="x">The series argument (not consumed).</param>
        /// <param name="c">The coefficients (1-D; an integer or bool array is converted to float64 first, as {p}val does).</param>
        /// <returns>A new series of <paramref name="x"/>'s class.</returns>
        /// <exception cref="NotSupportedException">An N-D coefficient array (assigned): NumPy's rows meet the series as
        ///     arrays, a path NumSharp does not follow.</exception>
        /// <exception cref="IndexError">An empty coefficient array.</exception>
        /// <exception cref="InvalidOperationException">The recurrence ended in a number (impossible for the step tables:
        ///     every final expression reads x; a guard against a malformed table).</exception>
        private static NDPolyBase SeriesVal(PolyBasis basis, NDPolyBase x, NDArray c)
        {
            if (c.ndim != 1)
                throw new NotSupportedException("evaluating a series at a series needs 1-D coefficients: NumPy's rows of an N-D array meet the series as arrays, a path NumSharp does not follow");
            using var scope = NDScope.Open();   // the float64 conversion of an integer series
            NDArray cc = PolyTyping.IsIntLike(c.typecode) ? c.astype(NPTypeCode.Double) : c;
            var prog = PolySteps.Eval(basis);
            long n = cc.size;
            if (n == 0)
                throw new IndexError($"index {(prog.Horner ? -1 : -2)} is out of bounds for axis 0 with size 0");
            var temps = new List<NDPolyBase>();
            NDPolyBase result = null;
            try
            {
                unsafe
                {
                    int size = cc.dtypesize;
                    byte* p = (byte*)cc.Storage.Address + cc.Shape.offset * size;
                    long stride = n > 1 ? cc.Shape.strides[0] * size : size;
                    NPTypeCode t = cc.typecode;
                    SVal C(long i) => new SVal(PolyNumber.FromScalar(t, p + i * stride));
                    var xv = new SVal(x);

                    SVal final;
                    if (prog.Horner)
                    {
                        var h = Run(prog.HornerInit, xv, default, C(n - 1), default, default, default, 0, temps);
                        for (long i = 2; i <= n; i++)
                            h = Run(prog.HornerStep, xv, default, C(n - i), h, default, default, 0, temps);
                        final = h;
                    }
                    else
                    {
                        SVal x2 = prog.Pre is not null && (prog.PreAlways || n >= 3)
                            ? Run(prog.Pre, xv, default, default, default, default, default, 0, temps)
                            : default;
                        if (n == 1)
                            final = Run(prog.FinalLen1, xv, x2, default, C(0), default, default, 0, temps);
                        else if (n == 2)
                            final = Run(prog.Final, xv, x2, default, C(0), C(1), default, 0, temps);
                        else
                        {
                            SVal a0 = C(n - 2), a1 = C(n - 1);
                            long nd = n;
                            for (long i = 3; i <= n; i++)
                            {
                                var tmp = a0;
                                nd--;
                                a0 = Run(prog.StepC0, xv, x2, C(n - i), default, a1, default, nd, temps);
                                a1 = Run(prog.StepC1, xv, x2, default, default, a1, tmp, nd, temps);
                            }
                            final = Run(prog.Final, xv, x2, default, a0, a1, default, 0, temps);
                        }
                    }
                    result = final.Series ?? throw new InvalidOperationException("a series recurrence ended in a number");
                }
                return result;
            }
            finally
            {
                // Every series the recurrence made but the result (x itself is the caller's).
                foreach (var s in temps)
                    if (!ReferenceEquals(s, result))
                        s.Dispose();
            }
        }

        /// <summary>
        ///     Evaluates one step expression of the series recurrence: leaves read the current values, a weak constant is its
        ///     row's Python value, and a node is NumPy scalarmath for two numbers or the series protocol otherwise (the new
        ///     series recorded in <paramref name="temps"/>).
        /// </summary>
        /// <param name="e">The expression.</param>
        /// <param name="x">x.</param><param name="x2">x2.</param><param name="ck">The coefficient read.</param>
        /// <param name="c0">c0.</param><param name="c1">c1.</param><param name="tmp">tmp.</param>
        /// <param name="row">The row counter <c>nd</c>.</param>
        /// <param name="temps">Collects every series made.</param>
        /// <returns>The value.</returns>
        /// <exception cref="InvalidOperationException">A node kind the tables never contain.</exception>
        private static SVal Run(PolyExpr e, in SVal x, in SVal x2, in SVal ck, in SVal c0, in SVal c1, in SVal tmp, long row,
            List<NDPolyBase> temps)
        {
            switch (e)
            {
                case PolyLeaf l:
                    return l.S switch
                    {
                        PolySym.X => x,
                        PolySym.X2 => x2,
                        PolySym.Ck => ck,
                        PolySym.C0 => c0,
                        PolySym.C1 => c1,
                        PolySym.Tmp => tmp,
                        _ => throw new InvalidOperationException($"symbol {l.S} has no value in a series evaluation"),
                    };
                case PolyWeak w:
                    return new SVal(PolyNumber.FromPython(w.W.Value(row, default)));
                case PolyBin b:
                {
                    var a = Run(b.A, x, x2, ck, c0, c1, tmp, row, temps);
                    var c = Run(b.B, x, x2, ck, c0, c1, tmp, row, temps);
                    if (a.Series is null && c.Series is null)
                        return new SVal(PolyNumber.Binary(b.Op, a.Num, c.Num));
                    var op = b.Op switch
                    {
                        BinaryOp.Add => PyBinOp.Add,
                        BinaryOp.Subtract => PyBinOp.Sub,
                        BinaryOp.Multiply => PyBinOp.Mul,
                        _ => PyBinOp.TrueDiv,
                    };
                    var r = (NDPolyBase)PyBinary(op, a.Operand, c.Operand);
                    temps.Add(r);
                    return new SVal(r);
                }
                default:
                    throw new InvalidOperationException("unknown step-tree node");
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  Class-method cores
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>cls.identity(domain=None, window=None, symbol='x')</c>: <c>off, scl = pu.mapparms(window, domain)</c>,
        ///     <c>coef = cls._line(off, scl)</c>, <c>cls(coef, domain, window, symbol)</c> — the series whose value at x is x.
        /// </summary>
        /// <param name="kind">The class.</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The identity series.</returns>
        /// <exception cref="ValueError">The constructor's errors.</exception>
        /// <exception cref="IndexError">A domain or window shorter than two (mapparms' indexing).</exception>
        private protected static NDPolyBase Identity(NDPolyKind kind, object domain, object window, string symbol)
        {
            using var scope = NDScope.Open();
            object d = domain ?? kind.DefaultDomain();
            object w = window ?? kind.DefaultWindow();
            var (off, scl) = NDPolySeries.MapParms(w, d);
            NDArray coef = NDPolySeries.Line(kind.Basis, off, scl);
            var r = kind.Blank();
            r.Initialize(coef, d, w, symbol, coefIsFresh: true);
            return r;
        }

        /// <summary>
        ///     NumPy's <c>cls.basis(deg, domain=None, window=None, symbol='x')</c>: <c>ideg = int(deg)</c>; <c>ideg != deg or
        ///     ideg &lt; 0</c> raises; then <c>cls([0]*ideg + [1], domain, window, symbol)</c> — the basis polynomial of degree
        ///     <c>deg</c> (float64 coefficients: the Python list of ints as_series converts).
        /// </summary>
        /// <param name="kind">The class.</param>
        /// <param name="deg">The degree.</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The series.</returns>
        /// <exception cref="ValueError"><c>deg must be non-negative integer</c>; int()'s ValueErrors.</exception>
        /// <exception cref="TypeError">int() refuses <paramref name="deg"/>.</exception>
        /// <exception cref="OverflowException">A degree no list can have (<c>cannot fit 'int' into an index-sized integer</c>).</exception>
        private protected static NDPolyBase BasisOf(NDPolyKind kind, object deg, object domain, object window, string symbol)
        {
            using var scope = NDScope.Open();
            object d = domain ?? kind.DefaultDomain();
            object w = window ?? kind.DefaultWindow();
            BigInteger ideg = PyIntOf(deg, out bool differs);
            if (differs || ideg.Sign < 0)
                throw new ValueError("deg must be non-negative integer");
            if (ideg >= long.MaxValue)
                throw new OverflowException("cannot fit 'int' into an index-sized integer");
            long n = (long)ideg + 1;
            var coef = np.zeros(new Shape(n), NPTypeCode.Double);
            unsafe
            {
                ((double*)coef.Storage.Address)[n - 1] = 1.0;
            }
            var r = kind.Blank();
            r.Initialize(coef, d, w, symbol, coefIsFresh: true);
            return r;
        }

        /// <summary>
        ///     NumPy's <c>cls.fromroots(roots, domain=[], window=None, symbol='x')</c>: <c>[roots] = as_series([roots])</c>;
        ///     the domain (None: <c>getdomain(roots)</c>; <c>[]</c>: the class default); <c>rnew = off + scl*roots</c> with
        ///     <c>off, scl = mapparms(domain, window)</c>; <c>coef = cls._fromroots(rnew) / scl**deg</c>; the new series.
        /// </summary>
        /// <param name="kind">The class.</param>
        /// <param name="roots">The roots (1-D).</param>
        /// <param name="domain"><see cref="System.Reflection.Missing"/> or an empty list for the class default (NumPy's
        ///     <c>[]</c>), null for the roots' own interval (NumPy's None), or a domain.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The series with those roots.</returns>
        /// <exception cref="ValueError">as_series' errors for the roots; the constructor's errors.</exception>
        /// <exception cref="OverflowException">A Python-float scale whose power overflows (<c>(34, 'Numerical result out of
        ///     range')</c>, CPython's float power).</exception>
        private protected static NDPolyBase FromRootsOf(NDPolyKind kind, object roots, object domain, object window, string symbol)
        {
            using var scope = NDScope.Open();
            NDArray r = AsSeries(roots, fresh: false);
            object d;
            if (domain is null)
                d = NDPolySeries.GetDomain(r);
            else if (domain is System.Reflection.Missing || IsEmptyPythonList(domain))
                d = kind.DefaultDomain();
            else
                d = domain;
            object w = window ?? kind.DefaultWindow();
            long deg = r.size;
            var (off, scl) = NDPolySeries.MapParms(d, w);
            var rnew = PolyNumber.Binary(BinaryOp.Add, off, PolyNumber.Binary(BinaryOp.Multiply, scl, PolyNumber.FromArray(r)));
            NDArray fr = NDPolyAlgebra.FromRoots(kind.Basis, rnew.Array);
            var coef = PolyNumber.Binary(BinaryOp.Divide, PolyNumber.FromArray(fr), PowInt(scl, deg));
            var res = kind.Blank();
            res.Initialize(coef.Array, d, w, symbol, coefIsFresh: true);
            return res;
        }

        /// <summary>
        ///     NumPy's <c>cls.fit(x, y, deg, domain=None, rcond=None, full=False, w=None, window=None, symbol='x')</c>: the domain
        ///     (None: <c>getdomain(x)</c>, widened by 1 each way when degenerate; <c>[]</c>: the class default), x mapped into the
        ///     window, <c>cls._fit(xnew, y, deg, w=w, rcond=rcond, full=full)</c>, and the series.
        /// </summary>
        /// <param name="kind">The class.</param>
        /// <param name="x">The sample points.</param><param name="y">The values.</param><param name="deg">The degree(s).</param>
        /// <param name="domain">The domain: null (the points' interval), an empty list (the class default) or a domain.</param>
        /// <param name="rcond">The relative singular-value cutoff, or null.</param>
        /// <param name="full">Attach lstsq's statistics (the concrete class's Deconstruct reads them).</param>
        /// <param name="w">Weights, or null.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <param name="symbol">The symbol.</param>
        /// <returns>The fitted series.</returns>
        /// <exception cref="ValueError">The fit's and the constructor's errors.</exception>
        /// <exception cref="TypeError">The fit's argument errors.</exception>
        /// <exception cref="NotSupportedException">No LAPACK backend (lstsq needs <c>NumSharp.Interop.OpenBLAS</c>).</exception>
        private protected static NDPolyBase FitOf(NDPolyKind kind, object x, object y, object deg, object domain, object rcond,
            bool full, object w, object window, string symbol)
        {
            using var scope = NDScope.Open();
            object d;
            if (domain is null)
            {
                NDArray gd = NDPolySeries.GetDomain(x);
                WidenIfDegenerate(gd);
                d = gd;
            }
            else if (IsEmptyPythonList(domain))
                d = kind.DefaultDomain();
            else
                d = domain;
            object win = window ?? kind.DefaultWindow();
            object xnew = np.polynomial.polyutils.mapdomain(x, d, win);
            NDPolyFitResult res = NDPolyFit.Fit(kind.Basis, xnew, y, deg, rcond, full, w);
            var r = kind.Blank();
            r.Initialize(res, d, win, symbol, coefIsFresh: false);
            if (full)
                r.AttachFit(new FitStatus(res.residuals, res.rank, res.singular_values, res.rcond));
            return r;
        }

        /// <summary>
        ///     fit's <c>if domain[0] == domain[1]: domain[0] -= 1; domain[1] += 1</c> on getdomain's array, in place: NumPy
        ///     scalar comparisons and scalarmath with the Python int 1 in the domain's dtype.
        /// </summary>
        /// <param name="gd">getdomain's two-element array (float16 / float32 / float64 / complex128 / decimal).</param>
        private static unsafe void WidenIfDegenerate(NDArray gd)
        {
            NPTypeCode t = gd.typecode;
            int size = gd.dtypesize;
            byte* p0 = (byte*)gd.Storage.Address + gd.Shape.offset * size;
            byte* p1 = p0 + gd.Shape.strides[0] * size;
            var a = PolyNumber.FromScalar(t, p0);
            var b = PolyNumber.FromScalar(t, p1);
            if (!ScalarEquals(a, b))
                return;
            var one = PolyNumber.FromPython(PyScalar.Int(1));
            PolyNumber.Binary(BinaryOp.Subtract, a, one).WriteAs(t, p0);
            PolyNumber.Binary(BinaryOp.Add, b, one).WriteAs(t, p1);
        }

        /// <summary>
        ///     NumPy's <c>cls.cast(series, domain=None, window=None)</c>: <c>series.convert(domain, cls, window)</c> with the
        ///     class defaults for a missing domain / window.
        /// </summary>
        /// <param name="kind">The target class.</param>
        /// <param name="series">The series to convert.</param>
        /// <param name="domain">The domain, or null for the class default.</param>
        /// <param name="window">The window, or null for the class default.</param>
        /// <returns>The converted series.</returns>
        /// <exception cref="AttributeError"><paramref name="series"/> is null (<c>'NoneType' object has no attribute 'convert'</c>).</exception>
        private protected static NDPolyBase CastOf(NDPolyKind kind, NDPolyBase series, object domain, object window)
        {
            if (series is null)
                throw new AttributeError("'NoneType' object has no attribute 'convert'");
            object d = domain ?? kind.DefaultDomain();
            object w = window ?? kind.DefaultWindow();
            return series.convert(d, kind.ClrType, w);
        }

        /// <summary>
        ///     NumPy's <c>Chebyshev.interpolate(func, deg, domain=None, args=())</c>: <c>xfunc = lambda x: func(pu.mapdomain(x,
        ///     cls.window, domain), *args)</c>, <c>coef = chebinterpolate(xfunc, deg)</c>, <c>cls(coef, domain=domain)</c> (the
        ///     class window, the symbol 'x').
        /// </summary>
        /// <param name="func">The function, called once with the mapped Chebyshev points and the unpacked
        ///     <paramref name="args"/>.</param>
        /// <param name="deg">The degree.</param>
        /// <param name="domain">The interval to interpolate over, or null for the class default.</param>
        /// <param name="args">Extra arguments (any Python iterable), or null for none.</param>
        /// <returns>The interpolating series.</returns>
        /// <exception cref="TypeError"><c>deg must be an int</c>; the args unpacking's errors; <c>'NoneType' object is not
        ///     callable</c>.</exception>
        /// <exception cref="ValueError"><c>expected deg &gt;= 0</c>; the constructor's errors (they come after the
        ///     interpolation, as in NumPy).</exception>
        internal static NDPolyBase InterpolateOf(Func<NDArray, object[], object> func, object deg, object domain, object args)
        {
            using var scope = NDScope.Open();
            var kind = NDChebyshev.Kind;
            object d = domain ?? kind.DefaultDomain();
            NDArray clsWindow = kind.DefaultWindow();
            // NumPy's lambda: the mapdomain argument is evaluated before the *args unpacking, and both before the call.
            object XFunc(NDArray x, object[] _)
            {
                NDArray mapped = NDPolySeries.MapDomain(PolyNumber.FromArray(x), clsWindow, d).ToNDArray();
                object[] unpacked = NDPolyInterpolate.StarArgs(args);
                if (func is null)
                    throw new TypeError("'NoneType' object is not callable");
                return func(mapped, unpacked);
            }
            NDArray coef = NDPolyInterpolate.ChebInterpolate(XFunc, deg, null);
            var r = kind.Blank();
            r.Initialize(coef, d, null, "x", coefIsFresh: true);
            return r;
        }

        /// <summary>
        ///     Python's <c>isinstance(domain, list) and len(domain) == 0</c> under the house map, where a list is an
        ///     <c>object[]</c> (a jagged array too) or any other enumerable that is not a typed array, a tuple, a str or an
        ///     array.
        /// </summary>
        /// <param name="o">The argument.</param>
        /// <returns>Whether it is an empty Python list.</returns>
        private static bool IsEmptyPythonList(object o)
        {
            switch (o)
            {
                case object[] arr:
                    return arr.Length == 0;
                case null:
                case Array:
                case string:
                case NDArray:
                case ITuple:
                    return false;
                case ICollection c:
                    return c.Count == 0;
                case IEnumerable e:
                {
                    var en = e.GetEnumerator();
                    try
                    {
                        return !en.MoveNext();
                    }
                    finally
                    {
                        (en as IDisposable)?.Dispose();
                    }
                }
                default:
                    return false;
            }
        }

        /// <summary>Python's <c>a == b</c> of two NumPy scalars (IEEE: NaN unequal, -0.0 equal to 0.0; complex by both parts).</summary>
        /// <param name="a">One scalar.</param><param name="b">The other.</param>
        /// <returns>Whether they are equal.</returns>
        private static bool ScalarEquals(in PolyNumber a, in PolyNumber b) => (a.ToObject(), b.ToObject()) switch
        {
            (double x, double y) => x == y,
            (float x, float y) => x == y,
            (Half x, Half y) => x == y,
            (Complex x, Complex y) => x.Real == y.Real && x.Imaginary == y.Imaginary,
            (decimal x, decimal y) => x == y,
            var (x, y) => Equals(x, y),
        };

        /// <summary>
        ///     <c>scl**deg</c> for fromroots: CPython's power for a Python float / complex scale (a list domain), NumPy scalarmath
        ///     for a NumPy scalar one — <c>npy_pow</c> (the C runtime's <c>pow</c>) for float64, <c>npy_powf</c> rounded once
        ///     for float32 and float16, <c>npy_cpow</c> for complex128.
        /// </summary>
        /// <param name="scl">The scale.</param>
        /// <param name="deg">The exponent (≥ 1, the root count).</param>
        /// <returns>The power (a NumPy scalar for a NumPy scale, a Python value for a Python one).</returns>
        /// <exception cref="OverflowException">CPython's float overflow (<c>(34, 'Numerical result out of range')</c>) or
        ///     complex overflow (<c>complex exponentiation</c>).</exception>
        /// <exception cref="NotSupportedException">A scale of another dtype (an integer, decimal or complex64 scale, which no
        ///     domain as_series makes).</exception>
        private static unsafe PolyNumber PowInt(in PolyNumber scl, long deg)
        {
            if (scl.IsPython)
            {
                switch (scl.Py.Kind)
                {
                    case PyKind.Float:
                        return PolyNumber.FromPython(PyScalar.Float(PyFloatPow(scl.Py.F, deg)));
                    case PyKind.Complex:
                        return PolyNumber.FromPython(PyScalar.Cplx(PyComplexPow(scl.Py.C, deg)));
                    default:
                        return PolyNumber.FromPython(PyScalar.Int(BigInteger.Pow(scl.Py.I, (int)Math.Min(deg, int.MaxValue))));
                }
            }
            if (scl.Kind == PolyNumberKind.Scalar && !scl.IsComplex64)
            {
                switch (scl.Dtype)
                {
                    case NPTypeCode.Double:
                    {
                        double r = Math.Pow((double)scl.ToObject(), deg);
                        return PolyNumber.FromScalar(NPTypeCode.Double, &r);
                    }
                    case NPTypeCode.Single:
                    {
                        float r = MathF.Pow((float)scl.ToObject(), deg);
                        return PolyNumber.FromScalar(NPTypeCode.Single, &r);
                    }
                    case NPTypeCode.Half:
                    {
                        // half_ctype_power: npy_powf of the widened operands, one rounding back to float16.
                        float f = (float)(Half)scl.ToObject();
                        Half r = (Half)MathF.Pow(f, deg);
                        return PolyNumber.FromScalar(NPTypeCode.Half, &r);
                    }
                    case NPTypeCode.Complex:
                    {
                        Complex r = DirectILKernelGenerator.ComplexPowNumPy((Complex)scl.ToObject(), new Complex(deg, 0));
                        return PolyNumber.FromScalar(NPTypeCode.Complex, &r);
                    }
                }
            }
            throw new NotSupportedException($"scl**deg for a {(scl.IsComplex64 ? "complex64" : scl.Dtype.AsNumpyDtypeName())} scale has no NumSharp scalarmath");
        }

        /// <summary>
        ///     CPython 3.12's <c>float ** int</c> (<c>float_pow</c>) for an exponent ≥ 1: the special cases, then the C
        ///     runtime's <c>pow</c>; an overflowing finite result raises OverflowError (errno ERANGE).
        /// </summary>
        /// <param name="iv">The base.</param><param name="n">The exponent (≥ 1).</param>
        /// <returns>The power.</returns>
        /// <exception cref="OverflowException"><c>(34, 'Numerical result out of range')</c>.</exception>
        private static double PyFloatPow(double iv, long n)
        {
            double iw = n;
            bool odd = (n & 1) == 1;
            if (double.IsNaN(iv))
                return iv;
            if (double.IsInfinity(iv))
                return odd ? iv : Math.Abs(iv);
            if (iv == 0.0)
                return odd ? iv : 0.0;
            bool negate = false;
            if (iv < 0.0)
            {
                iv = -iv;
                negate = odd;
            }
            if (iv == 1.0)
                return negate ? -1.0 : 1.0;
            double ix = Math.Pow(iv, iw);
            if (double.IsInfinity(ix))
                throw new OverflowException("(34, 'Numerical result out of range')");
            return negate ? -ix : ix;
        }

        /// <summary>
        ///     CPython 3.12's <c>complex ** int</c> (<c>complex_pow</c>): an exponent up to 100 by binary exponentiation with
        ///     the naive product (<c>c_powu</c>), a larger one through the polar form (<c>_Py_c_pow</c>); an infinite part
        ///     raises OverflowError.
        /// </summary>
        /// <param name="a">The base.</param><param name="n">The exponent (≥ 1).</param>
        /// <returns>The power.</returns>
        /// <exception cref="OverflowException"><c>complex exponentiation</c>.</exception>
        private static Complex PyComplexPow(Complex a, long n)
        {
            Complex r;
            if (n <= 100)
            {
                // c_powu: r = 1; p = a; for each set bit: r = r*p; p = p*p (the mask stops past n).
                r = new Complex(1.0, 0.0);
                Complex p = a;
                long mask = 1;
                while (mask > 0 && n >= mask)
                {
                    if ((n & mask) != 0)
                        r = PyScalar.ComplexProd(r, p);
                    mask <<= 1;
                    p = PyScalar.ComplexProd(p, p);
                }
            }
            else if (a.Real == 0.0 && a.Imaginary == 0.0)
                r = new Complex(0.0, 0.0);
            else
            {
                double vabs = Hypot(a.Real, a.Imaginary);
                double len = Math.Pow(vabs, n);
                double at = Math.Atan2(a.Imaginary, a.Real);
                double phase = at * n;
                r = new Complex(len * Math.Cos(phase), len * Math.Sin(phase));
            }
            if (double.IsInfinity(r.Real) || double.IsInfinity(r.Imaginary))
                throw new OverflowException("complex exponentiation");
            return r;
        }

        /// <summary>The C runtime's <c>hypot</c> (overflow-safe, as CPython calls it).</summary>
        /// <param name="x">One leg.</param><param name="y">The other.</param>
        /// <returns><c>sqrt(x² + y²)</c>.</returns>
        private static double Hypot(double x, double y) => double.Hypot(x, y);

        // ---------------------------------------------------------------------------------------------
        //  linspace
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>np.linspace(start, stop, num)</c> (endpoint, no dtype), statement for statement: the dtype is the
        ///     endpoints' result type made inexact; <c>delta = stop - start</c> in it; <c>y = arange(0, num)</c>;
        ///     <c>y *= delta/div</c> — or <c>y /= div; y *= delta</c> when that step underflows to zero —; <c>y += start</c>;
        ///     <c>y[-1] = stop</c>.
        /// </summary>
        /// <param name="start">domain[0] (a NumPy scalar of an array domain, or a Python number of a list).</param>
        /// <param name="stop">domain[1].</param>
        /// <param name="num">The number of points.</param>
        /// <returns>The points.</returns>
        /// <exception cref="ValueError"><c>Number of samples, -1, must be non-negative.</c></exception>
        private static unsafe NDArray Linspace(in PolyNumber start, in PolyNumber stop, long num)
        {
            if (num < 0)
                throw new ValueError($"Number of samples, {num}, must be non-negative.");
            // _array_converter keeps two NumPy scalars as 0-d arrays and converts two Python numbers to arrays of their
            // discovered dtypes (int64 / float64 / complex128); the result type is then made inexact.
            PolyNumber s = start.IsPython ? PolyNumber.FromArray(start.ToNDArray()) : start;
            PolyNumber e = stop.IsPython ? PolyNumber.FromArray(stop.ToNDArray()) : stop;
            NPTypeCode dt = PolyTyping.Promote(s.Dtype, e.Dtype);
            if (PolyTyping.IsIntLike(dt))
                dt = NPTypeCode.Double;
            long div = num - 1;

            // delta = np.subtract(stop, start, dtype=dt): both cast to dt, one subtraction there.
            PolyRaw2 raw;
            s.WriteAs(dt, &raw.A);
            e.WriteAs(dt, &raw.B);
            var sd = PolyNumber.FromScalar(dt, &raw.A);
            var ed = PolyNumber.FromScalar(dt, &raw.B);
            var delta = PolyNumber.Binary(BinaryOp.Subtract, ed, sd);

            NDArray y = np.arange(num).astype(dt);
            var yv = PolyNumber.FromArray(y);
            if (div > 0)
            {
                var step = PolyNumber.Binary(BinaryOp.Divide, delta, PolyNumber.FromPython(PyScalar.Int(div)));
                if (!step.NotZero())
                {
                    // Denormal handling (gh-5437): divide first, then scale.
                    yv = PolyNumber.Binary(BinaryOp.Divide, yv, PolyNumber.FromPython(PyScalar.Int(div)));
                    yv = PolyNumber.Binary(BinaryOp.Multiply, yv, delta);
                }
                else
                    yv = PolyNumber.Binary(BinaryOp.Multiply, yv, step);
            }
            else
                yv = PolyNumber.Binary(BinaryOp.Multiply, yv, delta);
            yv = PolyNumber.Binary(BinaryOp.Add, yv, s.Kind == PolyNumberKind.Scalar ? PolyNumber.FromArray(s.ToNDArray()) : s);
            NDArray r = yv.Array;
            if (num > 1)
            {
                // y[-1] = stop: setitem casts the endpoint into dt.
                byte* last = (byte*)r.Storage.Address + (r.Shape.offset + (num - 1) * r.Shape.strides[0]) * r.dtypesize;
                e.WriteAs(dt, last);
            }
            return r;
        }

        /// <summary>Two 16-byte scalar slots on the stack.</summary>
        private struct PolyRaw2
        {
            /// <summary>The first slot.</summary>
            public PolyRaw16 A;
            /// <summary>The second slot.</summary>
            public PolyRaw16 B;
        }
    }
}
