using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    /// <summary>
    ///     <c>numpy.polynomial.polyutils</c> — the helpers the six basis modules are built on, reachable as
    ///     <c>np.polynomial.polyutils</c>: series normalization (<see cref="as_series"/>, <see cref="trimseq"/>,
    ///     <see cref="trimcoef"/>) and the linear domain maps (<see cref="getdomain"/>, <see cref="mapparms"/>,
    ///     <see cref="mapdomain"/>).
    /// </summary>
    /// <remarks>
    ///     <para>Every function reproduces NumPy 2.4.2 bit for bit, including which ARITHMETIC runs: NumPy never
    ///     converts a domain, it indexes it, so a domain given as Python numbers is computed by CPython (exact
    ///     ints; a zero-length domain raises <see cref="DivideByZeroException"/>, Python's ZeroDivisionError), and
    ///     one given as an array by NumPy's scalar math (a zero-length domain yields inf/nan). In C#: an
    ///     <see cref="NDArray"/> or a TYPED array (<c>double[]</c>, <c>float[]</c>, <c>Complex[]</c>, …) is an
    ///     ndarray; an <c>object[]</c> / <see cref="System.Collections.IList"/> is a Python list and a
    ///     <see cref="ValueTuple"/> a Python tuple of Python numbers (a C# <c>bool</c> / integer / <c>float</c> /
    ///     <c>double</c> / <see cref="Complex"/> is a Python number; <see cref="Half"/>, <c>char</c> and
    ///     <c>decimal</c> are NumPy scalars of their dtype).</para>
    ///     <para><c>format_float</c> belongs to the printing unit and is not here yet.</para>
    ///     <para>https://numpy.org/doc/stable/reference/routines.polynomials.polyutils.html</para>
    /// </remarks>
    [ModuleName("np.polynomial.polyutils")]
    public class PolyUtilsModule
    {
        /// <summary>Constructs the module; there is one shared instance behind <c>np.polynomial.polyutils</c>.</summary>
        internal PolyUtilsModule() { }

        /// <summary>
        ///     Returns every item of <paramref name="alist"/> as a 1-D coefficient array, all of one dtype —
        ///     <c>np.common_type</c> of the items (every integer becomes float64; float16/float32/float64/complex128
        ///     are kept) — each a FRESH copy, with trailing zeros removed first when <paramref name="trim"/>.
        /// </summary>
        /// <param name="alist">The coefficient sequences: an <see cref="NDArray"/> (a 1-D array yields one
        ///     single-coefficient series per element, a 2-D array one series per row), or any C# collection of
        ///     coefficient arrays / numbers.</param>
        /// <param name="trim">Remove trailing zeros from each series (NumPy's default: true).</param>
        /// <returns>The series, in order (an empty collection yields an empty array).</returns>
        /// <exception cref="TypeError"><paramref name="alist"/> is not iterable (null, a number, a 0-d array).</exception>
        /// <exception cref="ValueError">A series is empty (<c>Coefficient array is empty</c>) or not 1-d
        ///     (<c>Coefficient array is not 1-d</c>), or the series have no common type — bool or str series
        ///     (<c>Coefficient arrays have no common type</c>; NumPy's object-array fallback does not exist here).</exception>
        /// <exception cref="NotSupportedException">A null item or a Python int beyond uint64 (NumPy would build an
        ///     object array).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.as_series.html</remarks>
        [NDScoped]
        public NDArray[] as_series(object alist, bool trim = true) => NDPolySeries.AsSeries(alist, trim);

        /// <summary>
        ///     Removes trailing zeros from <paramref name="seq"/>, keeping at least one element: <paramref name="seq"/>
        ///     ITSELF when it is empty or ends in a nonzero (NaN counts as nonzero, -0.0 as zero), otherwise the VIEW
        ///     <c>seq[:i+1]</c> ending at the last nonzero — writes through it reach <paramref name="seq"/>.
        /// </summary>
        /// <param name="seq">The sequence (rank ≥ 1; the rows of a 2-D+ array must hold exactly one element).</param>
        /// <returns>The same instance, or a view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="seq"/> is null.</exception>
        /// <exception cref="TypeError">A 0-d array (<c>len() of unsized object</c>).</exception>
        /// <exception cref="ValueError">Rows of several elements or of none (NumPy's truth-value errors).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.trimseq.html</remarks>
        [NDScoped]
        public NDArray trimseq(NDArray seq) => NDPolySeries.TrimSeq(seq);

        /// <summary>
        ///     <see cref="trimseq(NDArray)"/> for a Python LIST — an <c>object[]</c>, or by array covariance a jagged or
        ///     <c>NDArray[]</c> array: <paramref name="seq"/> ITSELF when it is empty or its last item is nonzero, otherwise
        ///     a NEW array of the items up to the last nonzero one (at least one), NumPy's list slice <c>seq[:i+1]</c> (of the
        ///     same element type). An item is zero exactly when Python's <c>item != 0</c> is false: a number by value (NaN is
        ///     nonzero, -0.0 zero, a complex zero only when both parts are), an array item by its truth value, and anything
        ///     else — a str, a nested list, null — never equals 0.
        /// </summary>
        /// <param name="seq">The list.</param>
        /// <returns>The same instance, or the trimmed copy.</returns>
        /// <exception cref="TypeError"><paramref name="seq"/> is null (<c>object of type 'NoneType' has no len()</c>).</exception>
        /// <exception cref="ValueError">A tested array item holds several elements or none (NumPy's truth-value
        ///     errors).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.trimseq.html</remarks>
        public object[] trimseq(object[] seq) => NDPolySeries.TrimSeqList(seq);

        /// <summary>
        ///     <see cref="trimseq(NDArray)"/> for any other argument — NumPy returns the KIND it was given: a Python tuple
        ///     (a <see cref="ValueTuple"/> / <see cref="Tuple"/>) itself or a new ValueTuple of the kept items; any other
        ///     Python sequence (a <see cref="System.Collections.IEnumerable"/>) itself or an <c>object[]</c> of the kept
        ///     items; a str itself; an <see cref="NDArray"/>, a typed C# array or a <c>Memory&lt;T&gt;</c> (ndarrays) itself
        ///     or a view, as <see cref="trimseq(NDArray)"/> does.
        /// </summary>
        /// <param name="seq">The sequence.</param>
        /// <returns>The same instance, or the trimmed sequence of the same kind.</returns>
        /// <exception cref="TypeError">A value without a length: null or a number (<c>object of type 'int' has no
        ///     len()</c>), a 0-d array (<c>len() of unsized object</c>).</exception>
        /// <exception cref="ValueError">A tested array item or row holds several elements or none.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.trimseq.html</remarks>
        public object trimseq(object seq) => NDPolySeries.TrimSeqAny(seq);

        /// <summary>
        ///     Removes the trailing coefficients whose magnitude does not exceed <c>tol = 0</c> — i.e. the trailing
        ///     exact zeros — and returns the rest as a fresh copy of the coefficient dtype; an all-zero series
        ///     becomes <c>c[:1]*0</c>. See <see cref="trimcoef(object, object)"/>.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <returns>The trimmed copy.</returns>
        /// <exception cref="ValueError">An empty or non-1-d series, or no common type (bool).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.trimcoef.html</remarks>
        [NDScoped]
        public NDArray trimcoef(object c) => NDPolySeries.TrimCoef(c, PolyNumber.FromPython(PyScalar.Int(0)));

        /// <summary>
        ///     Removes the trailing coefficients whose magnitude is at most <paramref name="tol"/>
        ///     (<c>np.abs(c) &gt; tol</c> decides what stays) and returns the rest as a fresh copy of the coefficient
        ///     dtype; when nothing exceeds the tolerance the result is <c>c[:1]*0</c> (a zero that keeps -0.0 and
        ///     NaN the way NumPy's multiply does). A complex series is judged by its float64 magnitude.
        /// </summary>
        /// <param name="c">The series (1-D).</param>
        /// <param name="tol">The tolerance: a Python number (C# primitive — adopts the coefficients' precision, so a
        ///     float16 series compares against the float16-rounded tolerance) or a NumPy scalar / 0-d array (strong:
        ///     the comparison runs in the promoted dtype). A NaN tolerance trims everything.</param>
        /// <returns>The trimmed copy.</returns>
        /// <exception cref="ValueError"><c>tol must be non-negative</c>; an empty or non-1-d series; no common type;
        ///     a tolerance array of several elements (NumPy's truth-value error).</exception>
        /// <exception cref="TypeError">A Python complex tolerance (<c>'&lt;' not supported between instances of
        ///     'complex' and 'int'</c>), or a null one.</exception>
        /// <exception cref="OverflowException">A Python int tolerance too large to convert (<c>int too large to convert
        ///     to float</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.trimcoef.html</remarks>
        [NDScoped]
        public NDArray trimcoef(object c, object tol) => NDPolySeries.TrimCoef(c, Tolerance(tol));

        /// <summary>
        ///     The smallest domain containing the points: <c>[x.min(), x.max()]</c> of the points converted to their
        ///     common type (integers become float64); for complex points the corners
        ///     <c>[min re + min im·j, max re + max im·j]</c> of the bounding rectangle.
        /// </summary>
        /// <param name="x">The 1-D points.</param>
        /// <returns>A new 2-element array (float16/float32/float64/complex128).</returns>
        /// <exception cref="ValueError">Empty or non-1-d points, or no common type (bool).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.getdomain.html</remarks>
        [NDScoped]
        public NDArray getdomain(object x) => NDPolySeries.GetDomain(x);

        /// <summary>
        ///     The offset and scale of the linear map <c>L(x) = off + scl*x</c> sending <c>old[0] → new[0]</c> and
        ///     <c>old[1] → new[1]</c>, computed exactly as NumPy writes it:
        ///     <c>off = (old[1]*new[0] - old[0]*new[1]) / (old[1] - old[0])</c>, <c>scl = (new[1] - new[0]) / (old[1] - old[0])</c>.
        /// </summary>
        /// <param name="old">The source domain (indexed at 0 and 1, never converted — see the class remarks for what
        ///     a C# argument means).</param>
        /// <param name="new">The target domain.</param>
        /// <returns><c>(off, scl)</c>: each a boxed C# number — <c>double</c> / <see cref="Complex"/> for a Python result,
        ///     the dtype's C# type (<c>double</c>, <c>float</c>, <see cref="Half"/>, <see cref="Complex"/>, …) for a NumPy
        ///     scalar — or an <see cref="NDArray"/> when a domain is 2-D (its items are then arrays).</returns>
        /// <exception cref="IndexError">A domain shorter than 2 (NumPy's / CPython's text), a 0-d array.</exception>
        /// <exception cref="TypeError">A domain that is not subscriptable (a number, null), or NumPy's bool subtraction.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers (ZeroDivisionError).</exception>
        /// <exception cref="OverflowException">A Python int too large (<c>integer division result too large for a
        ///     float</c>, or out of a NumPy integer dtype's range).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapparms.html</remarks>
        public (object off, object scl) mapparms(object old, object @new)
        {
            // Python lists/tuples of Python numbers: the machine-number lane allocates nothing, so no scope is needed.
            if (NDPolySeries.TryMapParmsFast(old, @new, out var laneOff, out var laneScl))
                return (laneOff.ToObject(), laneScl.ToObject());
            using var scope = NDScope.Open();
            var (off, scl) = NDPolySeries.MapParms(old, @new);
            (object off, object scl) r = (off.ToObject(), scl.ToObject());
            scope.Returns((ITuple)r);   // an array component (2-D domain) outlives the scope
            return r;
        }

        /// <summary>
        ///     <see cref="mapparms(object, object)"/> for domains written as C# 2-tuples — Python tuples, NumPy's own
        ///     docstring spelling (<c>mapparms((-1, 1), (0, 2))</c>) — read without boxing. When every element is a
        ///     Python int, float or complex (a C# integer, bool, <c>float</c>, <c>double</c> or <see cref="Complex"/>)
        ///     CPython's arithmetic runs on machine numbers (exact ints; an int intermediate beyond <c>long</c>
        ///     switches to exact big integers, so the answer never changes); any other element (<see cref="Half"/>,
        ///     an <see cref="NDArray"/>, a <see cref="BigInteger"/>, …) takes the general path with the same meaning it
        ///     has in the <see cref="object"/> overload.
        /// </summary>
        /// <typeparam name="T0">Type of <c>old[0]</c>.</typeparam>
        /// <typeparam name="T1">Type of <c>old[1]</c>.</typeparam>
        /// <typeparam name="T2">Type of <c>new[0]</c>.</typeparam>
        /// <typeparam name="T3">Type of <c>new[1]</c>.</typeparam>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns><c>(off, scl)</c> — a boxed <c>double</c> or <see cref="Complex"/> each for Python-number domains
        ///     (true division never yields a Python int), otherwise as <see cref="mapparms(object, object)"/> returns
        ///     them.</returns>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers (<c>division by zero</c> for
        ///     ints, <c>float division by zero</c> once a float is involved — CPython 3.12's texts).</exception>
        /// <exception cref="TypeError">NumPy's bool subtraction (NumPy bool elements).</exception>
        /// <exception cref="OverflowException">A Python int too large (see <see cref="mapparms(object, object)"/>).</exception>
        /// <exception cref="NotSupportedException">A null or string element.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapparms.html</remarks>
        public (object off, object scl) mapparms<T0, T1, T2, T3>((T0, T1) old, (T2, T3) @new)
        {
            if (NDPolySeries.TryMapParmsFast(old, @new, out var off, out var scl))
                return (off.ToObject(), scl.ToObject());
            using var scope = NDScope.Open();   // array elements leave ufunc intermediates behind
            var (o, s) = NDPolySeries.MapParmsSlow(old, @new);
            (object off, object scl) r = (o.ToObject(), s.ToObject());
            scope.Returns((ITuple)r);   // an array component outlives the scope
            return r;
        }

        /// <summary>
        ///     <see cref="mapparms(object, object)"/> for float64 domains — the arrays a <c>double[]</c> spells — as plain
        ///     IEEE arithmetic (NumPy's float64 scalar math: a zero-length domain gives inf/nan, never raises).
        /// </summary>
        /// <param name="old">The source domain (≥ 2 elements).</param>
        /// <param name="new">The target domain (≥ 2 elements).</param>
        /// <returns><c>(off, scl)</c>.</returns>
        /// <exception cref="ArgumentNullException">A domain is null.</exception>
        /// <exception cref="IndexError">A domain shorter than 2 (<c>index 1 is out of bounds for axis 0 with size N</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapparms.html</remarks>
        public (double off, double scl) mapparms(double[] old, double[] @new)
        {
            CheckDomain(old, nameof(old));
            CheckDomain(@new, nameof(@new));
            double oldlen = old[1] - old[0];
            double newlen = @new[1] - @new[0];
            double off = (old[1] * @new[0] - old[0] * @new[1]) / oldlen;
            double scl = newlen / oldlen;
            return (off, scl);
        }

        /// <summary>
        ///     <see cref="mapparms(object, object)"/> for complex128 domains — the arrays a <see cref="Complex"/>[] spells —
        ///     with NumPy's complex SCALAR math: the naive product (no FMA) and NumPy's Smith division (inf/nan, never
        ///     a raise, on a zero-length domain).
        /// </summary>
        /// <param name="old">The source domain (≥ 2 elements).</param>
        /// <param name="new">The target domain (≥ 2 elements).</param>
        /// <returns><c>(off, scl)</c>.</returns>
        /// <exception cref="ArgumentNullException">A domain is null.</exception>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapparms.html</remarks>
        public (Complex off, Complex scl) mapparms(Complex[] old, Complex[] @new)
        {
            CheckDomain(old, nameof(old));
            CheckDomain(@new, nameof(@new));
            using var scope = NDScope.Open();   // reclaims the two domain arrays
            var (off, scl) = NDPolySeries.MapParms(np.array(old), np.array(@new));
            return ((Complex)off.ToObject(), (Complex)scl.ToObject());
        }

        /// <summary>
        ///     Maps the points <paramref name="x"/> from the domain <paramref name="old"/> to <paramref name="new"/>:
        ///     <c>off + scl*x</c> with <c>(off, scl) = mapparms(old, new)</c> — one fused pass over the array, with
        ///     NumPy's result dtype (a Python-number domain keeps a float32/float16 x's precision; an array domain's
        ///     dtype promotes with x's).
        /// </summary>
        /// <param name="x">The points (any shape; a TYPED C# array converts to an ndarray of its dtype — a Python list,
        ///     an <c>object[]</c> or a jagged / <c>NDArray[]</c> array, binds <see cref="mapdomain(object[], object, object)"/>).</param>
        /// <param name="old">The source domain (see the class remarks).</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The mapped points (a new array; 0-d for a 0-d x).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <exception cref="TypeError">A domain that is not subscriptable.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        [NDScoped]
        public NDArray mapdomain(NDArray x, object old, object @new)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            var r = NDPolySeries.MapDomain(PolyNumber.FromArray(x), old, @new);
            return r.ToNDArray();
        }

        /// <summary>
        ///     <see cref="mapdomain(NDArray, object, object)"/> for a Python float <paramref name="x"/> (C# integer
        ///     and float literals bind here too — a Python int meets a float scale as its correctly rounded float, so
        ///     nothing changes): NumPy leaves a Python scalar unconverted, so the arithmetic is CPython's for a
        ///     Python-number domain and NumPy's scalar math for an array domain.
        /// </summary>
        /// <param name="x">The point.</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The mapped point as a boxed C# number (<c>double</c> for a Python float or float64 scalar, the
        ///     dtype's C# type for another NumPy scalar), or an <see cref="NDArray"/> for a 2-D domain.</returns>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <exception cref="TypeError">A domain that is not subscriptable.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        public object mapdomain(double x, object old, object @new) => MapDomainBoxed(PolyNumber.FromPython(PyScalar.Float(x)), old, @new);

        /// <summary>
        ///     <see cref="mapdomain(NDArray, object, object)"/> for a Python complex <paramref name="x"/> (CPython's
        ///     complex arithmetic against a Python-number domain).
        /// </summary>
        /// <param name="x">The point.</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The mapped point as a boxed C# number, or an <see cref="NDArray"/> for a 2-D domain.</returns>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <exception cref="TypeError">A domain that is not subscriptable.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        public object mapdomain(Complex x, object old, object @new) => MapDomainBoxed(PolyNumber.FromPython(PyScalar.Cplx(x)), old, @new);

        /// <summary>
        ///     <see cref="mapdomain(NDArray, object, object)"/> for any <paramref name="x"/>: a Python number (C#
        ///     bool excepted — NumPy converts a bool with np.asanyarray) or a NumPy scalar (<see cref="Half"/>,
        ///     <c>char</c>, <c>decimal</c>) is mapped unconverted; anything else is converted to an array first — a
        ///     typed C# array as an ndarray of its dtype, a Python tuple / list (a ValueTuple, <c>object[]</c>, a jagged or
        ///     <c>NDArray[]</c> array, any enumerable) by np.array's coercion, nested to any depth.
        /// </summary>
        /// <param name="x">The point(s).</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>An <see cref="NDArray"/> for array points, a boxed C# number for a scalar point.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <exception cref="TypeError">A domain that is not subscriptable.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <exception cref="ValueError">A ragged tuple / list of points (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="NotSupportedException">A str point, or a list holding one (NumPy's str arrays).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        public object mapdomain(object x, object old, object @new)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            PolyNumber px;
            NDArray converted = null;   // an array built here from x: this call's intermediate
            switch (x)
            {
                case bool b:
                    // type(True) is bool, not in (int, float, complex): NumPy's np.asanyarray makes it a 0-d array.
                    px = PolyNumber.FromArray(converted = NDArray.Scalar(b));
                    break;
                case NDArray nd:
                    px = PolyNumber.FromArray(nd);
                    break;
                default:
                    // `if type(x) not in (int, float, complex) and not isinstance(x, np.generic): x = np.asanyarray(x)`:
                    // FromObject keeps a Python number or NumPy scalar a scalar and converts a typed C# array (whole) or a
                    // Python tuple / list (np.array's nested coercion) to an ndarray — the only way it yields an array
                    // here, since a caller's NDArray took the case above.
                    px = PolyNumber.FromObject(x);
                    if (px.Kind == PolyNumberKind.Array)
                        converted = px.Array;
                    break;
            }
            try
            {
                return MapDomainBoxed(px, old, @new);
            }
            finally
            {
                // off + scl*x is always a fresh array (never x or a view of it), so the conversion can always go.
                converted?.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="mapdomain(object, object, object)"/> for a Python LIST of points — an <c>object[]</c>, or by array
        ///     covariance a jagged or <c>NDArray[]</c> array. Without it a C# array argument would bind the
        ///     <see cref="NDArray"/> overload through the implicit array conversion, which only understands TYPED arrays; here
        ///     the list goes through np.array's coercion (nested to any depth, the dtype discovered over every item, a
        ///     ragged list raising NumPy's text) before the domains are read, as NumPy's <c>np.asanyarray(x)</c> does.
        /// </summary>
        /// <param name="x">The points.</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The mapped points (a new array).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError">A ragged list (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="NotSupportedException">A list holding a str or null (NumPy's str / object arrays).</exception>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <exception cref="TypeError">A domain that is not subscriptable.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        public NDArray mapdomain(object[] x, object old, object @new)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            // The conversion is this call's intermediate; the NDArray overload always returns a fresh array.
            using var xa = PolySequence.ToArray(x);
            return mapdomain(xa, old, @new);
        }

        /// <summary>
        ///     <see cref="mapdomain(NDArray, object, object)"/> for a Python float point and float64 domains: plain
        ///     IEEE arithmetic (<c>off + scl*x</c> in float64; NumPy's float64 scalar math — a zero-length domain gives
        ///     inf/nan, never raises).
        /// </summary>
        /// <param name="x">The point.</param>
        /// <param name="old">The source domain (≥ 2 elements).</param>
        /// <param name="new">The target domain (≥ 2 elements).</param>
        /// <returns>The mapped point.</returns>
        /// <exception cref="ArgumentNullException">A domain is null.</exception>
        /// <exception cref="IndexError">A domain shorter than 2.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        public double mapdomain(double x, double[] old, double[] @new)
        {
            var (off, scl) = mapparms(old, @new);
            return off + scl * x;
        }

        /// <summary>
        ///     <see cref="mapdomain(NDArray, object, object)"/> with the domains written as C# 2-tuples (see
        ///     <see cref="mapparms{T0,T1,T2,T3}"/>): one fused pass over <paramref name="x"/>, whose offset/scale come
        ///     from CPython's arithmetic on machine numbers when the domains are Python ints/floats/complexes.
        /// </summary>
        /// <typeparam name="T0">Type of <c>old[0]</c>.</typeparam>
        /// <typeparam name="T1">Type of <c>old[1]</c>.</typeparam>
        /// <typeparam name="T2">Type of <c>new[0]</c>.</typeparam>
        /// <typeparam name="T3">Type of <c>new[1]</c>.</typeparam>
        /// <param name="x">The points (any shape; a typed C# array converts to an ndarray — a Python list binds the
        ///     <c>object[]</c> overload).</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The mapped points (a new array; 0-d for a 0-d x).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="TypeError">NumPy's bool subtraction (NumPy bool elements).</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <exception cref="NotSupportedException">A null or string domain element.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        public NDArray mapdomain<T0, T1, T2, T3>(NDArray x, (T0, T1) old, (T2, T3) @new)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            using var scope = NDScope.Open();   // array domain elements leave ufunc intermediates behind
            var (off, scl) = NDPolySeries.MapParms(old, @new);
            return scope.Returns(NDPolySeries.MapDomainWith(PolyNumber.FromArray(x), off, scl).ToNDArray());
        }

        /// <summary>
        ///     <see cref="mapdomain{T0,T1,T2,T3}(NDArray, ValueTuple{T0,T1}, ValueTuple{T2,T3})"/> for a Python LIST of
        ///     points (an <c>object[]</c>, a jagged or <c>NDArray[]</c> array) — see
        ///     <see cref="mapdomain(object[], object, object)"/> for why a list needs its own overload: np.array's coercion of
        ///     the list first, then the tuple-domain map.
        /// </summary>
        /// <typeparam name="T0">Type of <c>old[0]</c>.</typeparam>
        /// <typeparam name="T1">Type of <c>old[1]</c>.</typeparam>
        /// <typeparam name="T2">Type of <c>new[0]</c>.</typeparam>
        /// <typeparam name="T3">Type of <c>new[1]</c>.</typeparam>
        /// <param name="x">The points.</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The mapped points (a new array).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError">A ragged list (np.array's inhomogeneous-shape text).</exception>
        /// <exception cref="NotSupportedException">A list holding a str or null, or a null / string domain element.</exception>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="TypeError">NumPy's bool subtraction (NumPy bool elements).</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        public NDArray mapdomain<T0, T1, T2, T3>(object[] x, (T0, T1) old, (T2, T3) @new)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            // np.asanyarray(x) happens before mapparms reads the domains (NumPy's statement order); the conversion is
            // this call's intermediate and the tuple-domain overload always returns a fresh array.
            using var xa = PolySequence.ToArray(x);
            return mapdomain(xa, old, @new);
        }

        /// <summary>
        ///     <see cref="mapdomain(double, object, object)"/> with the domains written as C# 2-tuples: a domain of
        ///     Python numbers maps the Python float <paramref name="x"/> with CPython's float/complex arithmetic on
        ///     machine numbers and no allocation beyond the boxed result.
        /// </summary>
        /// <typeparam name="T0">Type of <c>old[0]</c>.</typeparam>
        /// <typeparam name="T1">Type of <c>old[1]</c>.</typeparam>
        /// <typeparam name="T2">Type of <c>new[0]</c>.</typeparam>
        /// <typeparam name="T3">Type of <c>new[1]</c>.</typeparam>
        /// <param name="x">The point (a Python float; C# integer literals bind here too, as in the object-domain overload).</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The mapped point as a boxed C# number (<c>double</c> for a real Python domain, <see cref="Complex"/>
        ///     for a complex one), or an <see cref="NDArray"/> for a domain of arrays.</returns>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="TypeError">NumPy's bool subtraction.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <exception cref="NotSupportedException">A null or string domain element.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        public object mapdomain<T0, T1, T2, T3>(double x, (T0, T1) old, (T2, T3) @new)
        {
            // Python numbers throughout: CPython's float/complex operations — one IEEE multiply and one IEEE add
            // (never fused), each with CPython's NaN operand priority (PyScalar's helpers, not C# operators).
            if (NDPolySeries.TryMapParmsFast(old, @new, out var off, out var scl)
                && NDPolySeries.TryMapDomainLane(NDPolySeries.PyNum.Float(x), off, scl, out var r))
                return r.ToObject();
            return MapDomainBoxed(PolyNumber.FromPython(PyScalar.Float(x)), old, @new);
        }

        /// <summary>
        ///     <see cref="mapdomain(Complex, object, object)"/> with the domains written as C# 2-tuples: CPython's
        ///     complex arithmetic (a float operand joins as <c>(f, 0.0)</c>, CPython 3.12's rule), without
        ///     allocation for a Python-number domain.
        /// </summary>
        /// <typeparam name="T0">Type of <c>old[0]</c>.</typeparam>
        /// <typeparam name="T1">Type of <c>old[1]</c>.</typeparam>
        /// <typeparam name="T2">Type of <c>new[0]</c>.</typeparam>
        /// <typeparam name="T3">Type of <c>new[1]</c>.</typeparam>
        /// <param name="x">The point (a Python complex).</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The mapped point as a boxed C# number (<see cref="Complex"/> for a Python domain), or an
        ///     <see cref="NDArray"/> for a domain of arrays.</returns>
        /// <exception cref="DivideByZeroException">A zero-length domain of Python numbers.</exception>
        /// <exception cref="TypeError">NumPy's bool subtraction.</exception>
        /// <exception cref="OverflowException">Python int overflow.</exception>
        /// <exception cref="NotSupportedException">A null or string domain element.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polyutils.mapdomain.html</remarks>
        public object mapdomain<T0, T1, T2, T3>(Complex x, (T0, T1) old, (T2, T3) @new)
        {
            if (NDPolySeries.TryMapParmsFast(old, @new, out var off, out var scl)
                && NDPolySeries.TryMapDomainLane(NDPolySeries.PyNum.Cplx(x), off, scl, out var r))
                return r.ToObject();
            return MapDomainBoxed(PolyNumber.FromPython(PyScalar.Cplx(x)), old, @new);
        }

        /// <summary>Runs <see cref="NDPolySeries.MapDomain"/> under a scope and boxes a scalar result.</summary>
        /// <param name="x">The classified point(s).</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The boxed scalar, or the array (yielded out of the scope).</returns>
        private static object MapDomainBoxed(in PolyNumber x, object old, object @new)
        {
            // A Python point and Python-number list/tuple domains: the machine-number lane, allocation-free (mapparms
            // raises the same ZeroDivisionError first either way — Python has already left x unconverted).
            if (NDPolySeries.TryPyNum(x, out var px) && NDPolySeries.TryMapParmsFast(old, @new, out var laneOff, out var laneScl)
                && NDPolySeries.TryMapDomainLane(px, laneOff, laneScl, out var laneR))
                return laneR.ToObject();
            using var scope = NDScope.Open();
            var r = NDPolySeries.MapDomain(x, old, @new);
            // A 0-d ufunc result is a NumPy scalar in NumPy: box its value like any other scalar.
            object o = r.IsZeroDimArray ? PolyNumber.ScalarOf(r.Array).ToObject() : r.ToObject();
            return o is NDArray nd ? scope.Returns(nd) : o;
        }

        /// <summary><see cref="MapDomainBoxed(in PolyNumber, object, object)"/> for tuple domains (the general lane of the
        ///     generic overloads, once the machine-number lane declined).</summary>
        /// <typeparam name="T0">Type of <c>old[0]</c>.</typeparam>
        /// <typeparam name="T1">Type of <c>old[1]</c>.</typeparam>
        /// <typeparam name="T2">Type of <c>new[0]</c>.</typeparam>
        /// <typeparam name="T3">Type of <c>new[1]</c>.</typeparam>
        /// <param name="x">The classified point.</param>
        /// <param name="old">The source domain.</param>
        /// <param name="new">The target domain.</param>
        /// <returns>The boxed scalar, or the array (yielded out of the scope).</returns>
        private static object MapDomainBoxed<T0, T1, T2, T3>(in PolyNumber x, in (T0, T1) old, in (T2, T3) @new)
        {
            using var scope = NDScope.Open();
            var (off, scl) = NDPolySeries.MapParmsSlow(old, @new);
            var r = NDPolySeries.MapDomainWith(x, off, scl);
            object o = r.IsZeroDimArray ? PolyNumber.ScalarOf(r.Array).ToObject() : r.ToObject();
            return o is NDArray nd ? scope.Returns(nd) : o;
        }

        /// <summary>A tolerance argument as NumPy sees it (a Python number, a NumPy scalar, an array).</summary>
        /// <param name="tol">The C# value.</param>
        /// <returns>The value.</returns>
        /// <exception cref="TypeError">null (<c>'&lt;' not supported between instances of 'NoneType' and 'int'</c>).</exception>
        internal static PolyNumber Tolerance(object tol)
        {
            if (tol is null)
                throw new TypeError("'<' not supported between instances of 'NoneType' and 'int'");
            return PolyNumber.FromObject(tol);
        }

        /// <summary>The length check NumPy's <c>old[1]</c> performs on a float64/complex128 domain array.</summary>
        /// <param name="d">The domain.</param>
        /// <param name="name">Parameter name for the null check.</param>
        /// <exception cref="ArgumentNullException"><paramref name="d"/> is null.</exception>
        /// <exception cref="IndexError">Fewer than 2 elements.</exception>
        private static void CheckDomain(Array d, string name)
        {
            if (d is null) throw new ArgumentNullException(name);
            if (d.Length < 2)
                throw new IndexError($"index 1 is out of bounds for axis 0 with size {d.Length}");
        }
    }
}
