using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyPowerArgument.cs — the Python semantics of {p}pow's `pow` and `maxpower` arguments
// =============================================================================
//
// Every {p}pow (polyutils._pow, and chebpow's own copy of it) reads its two scalar arguments with three Python
// statements, after as_series has checked the series:
//
//     power = int(pow)
//     if power != pow or power < 0:
//         raise ValueError("Power must be a non-negative integer.")
//     elif maxpower is not None and power > maxpower:
//         raise ValueError("Power is too large")
//
// so WHICH values pass, and which error a bad one raises, is CPython's int() and its rich comparisons — plus NumPy's
// comparison ufuncs when an argument is a NumPy scalar or an ndarray. This file ports exactly those three statements
// for every C# value the house boundary maps (NDPolyNumber's header): all probed against NumPy 2.4.2.
//
// int(pow)
//   * a Python int / bool / BigInteger is itself; a Python float (C# double or float) truncates toward zero, NaN
//     raising ValueError "cannot convert float NaN to integer" and ±inf OverflowError "cannot convert float infinity
//     to integer"; Half (np.float16) the same; char (a NumPy integer scalar) its code; decimal truncates;
//   * a str parses with Python's int(str) grammar ("3", " 3 ", "3_0" parse; "x" and "" raise "invalid literal for int()
//     with base 10: 'x'") — and a parsed str then fails `power != pow` (an int never equals a str);
//   * a Python complex raises TypeError "int() argument must be a string, a bytes-like object or a real number, not
//     'complex'", None "... not 'NoneType'", a list "... not 'list'", a tuple "... not 'tuple'";
//   * an ndarray of one or more dims (an NDArray, a typed C# array, a Memory<T>) raises TypeError "only
//     0-dimensional arrays can be converted to Python scalars" — whatever its size; a 0-d NDArray converts its element
//     (a complex one raises the 'complex' TypeError).
// power != pow
//   an integer argument always equals its int(); a float one (Python float, np.float16, a 0-d float array — NumPy
//   converts the weak int to the float dtype, and trunc(v) is representable there) differs exactly when it had a
//   fractional part; a decimal likewise; a str always differs.
// power > maxpower (only when maxpower is not None; power is ≥ 0 by then)
//   * a Python int / bool / BigInteger / char / decimal compares exactly;
//   * a Python float compares EXACTLY (CPython's int-vs-float): NaN and +inf are never exceeded, -inf always is;
//   * np.float16 and a 0-d / one-element float or complex ndarray compare in THAT dtype: NEP 50 converts the weak int
//     to it — to a double first (OverflowError "int too large to convert to float" at 2**1024 after rounding), then
//     rounded to float16 / float32 — so 2049 > np.float16(2048) is False; complex compares lexicographically (NumPy's
//     CGT); a bool ndarray compares in int64 (OverflowError "int too big to convert" past int64); an integer ndarray
//     compares exactly (NumPy 2's out-of-range comparison rule);
//   * an ndarray's comparison is then tested for truth: one element (any ndim) is its value; none raises ValueError
//     "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.";
//     more than one "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()" —
//     after the comparison's own conversion error, which the ufunc raises first;
//   * a Python complex / str / list / tuple raises TypeError "'>' not supported between instances of 'int' and
//     'complex'" (resp. 'str' / 'list' / 'tuple').
//
// None of this loops over data: an ndarray argument either has one element (read directly) or fails by its size alone.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     Python's <c>int(pow)</c>, <c>power != pow</c> and <c>power &gt; maxpower</c> for the <c>{p}pow</c> family (see the
    ///     file header): which power and limit values NumPy accepts, and the exact error each rejected one raises.
    /// </summary>
    internal static class PolyPowerArgument
    {
        /// <summary>NumPy's text for an ndarray of one or more dims handed to <c>int()</c>.</summary>
        private const string OnlyZeroDim = "only 0-dimensional arrays can be converted to Python scalars";

        /// <summary>
        ///     Python's <c>int(pow)</c> under the house mapping of C# values (see the file header).
        /// </summary>
        /// <param name="pow">The power argument.</param>
        /// <returns>The integer power (any size).</returns>
        /// <exception cref="ValueError">A NaN (<c>cannot convert float NaN to integer</c>) or a str that is not an integer
        ///     literal (<c>invalid literal for int() with base 10: '…'</c>).</exception>
        /// <exception cref="OverflowException">An infinity (<c>cannot convert float infinity to integer</c>).</exception>
        /// <exception cref="TypeError">A complex, None, a list or tuple (<c>int() argument must be a string, a bytes-like
        ///     object or a real number, not '…'</c>), or an ndarray of one or more dims (<c>only 0-dimensional arrays can be
        ///     converted to Python scalars</c>).</exception>
        public static BigInteger Int(object pow)
        {
            switch (pow)
            {
                case NDArray nd:
                    // int(ndarray) needs a 0-d array; its element then converts like the matching scalar (float16 / float32 NaN
                    // and inf raise the float texts, a complex element the 'complex' TypeError).
                    if (nd.ndim != 0)
                        throw new TypeError(OnlyZeroDim);
                    return PythonInt.From(nd.GetAtIndex(0));
                case ITuple:
                    throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'tuple'");
            }
            // A typed C# array / Memory<T> of a dtype is an ndarray of at least one dim: int() refuses it before reading it.
            if (PolySequence.IsArrayLike(pow))
                throw new TypeError(OnlyZeroDim);
            // Any other list-like (object[], List<T>, a jagged array, LINQ) is a Python list.
            if (PolySequence.IsSequence(pow))
                throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'list'");
            // Scalars (and None / str / an unknown object) are PythonInt.From's: exactly CPython's int() of them.
            return PythonInt.From(pow);
        }

        /// <summary>
        ///     Python's <c>power != pow</c> after <c>power = int(pow)</c> succeeded: whether <paramref name="pow"/> had a
        ///     fractional part (or is a str, which never equals an int).
        /// </summary>
        /// <param name="power"><c>int(pow)</c>.</param>
        /// <param name="pow">The power argument (<see cref="Int"/> accepted it).</param>
        /// <returns>True when NumPy raises <c>Power must be a non-negative integer.</c> for it.</returns>
        public static bool DiffersFromInt(BigInteger power, object pow)
        {
            // A 0-d ndarray compares through NumPy's ufunc with the weak int converted to its dtype; a truncated float is
            // exactly representable in its own format, so the comparison is "was it already integral" — as for a scalar.
            if (pow is NDArray nd)
                pow = nd.GetAtIndex(0);
            return pow switch
            {
                double d => Math.Truncate(d) != d,
                float f => MathF.Truncate(f) != f,
                Half h => Math.Truncate((double)h) != (double)h,
                decimal m => decimal.Truncate(m) != m,
                string => true,
                _ => false,   // an integer value (Python int / bool / BigInteger / char / an integer 0-d array) is int()-exact
            };
        }

        /// <summary>
        ///     Python's <c>power &gt; maxpower</c>, tested for truth, for a <paramref name="maxpower"/> that is not None (see
        ///     the file header for every kind).
        /// </summary>
        /// <param name="power">The integer power (≥ 0).</param>
        /// <param name="maxpower">The limit (not null).</param>
        /// <returns>True when NumPy raises <c>Power is too large</c>.</returns>
        /// <exception cref="TypeError">A complex, str, list, tuple or other non-comparable limit (<c>'&gt;' not supported
        ///     between instances of 'int' and '…'</c>).</exception>
        /// <exception cref="OverflowException">A power NumPy cannot convert into the limit's comparison dtype (<c>int too
        ///     large to convert to float</c> for a float16 / float / complex ndarray or np.float16, <c>int too big to
        ///     convert</c> for a bool ndarray).</exception>
        /// <exception cref="ValueError">An ndarray limit with no element or more than one (NumPy's truth-value texts).</exception>
        public static bool Exceeds(BigInteger power, object maxpower)
        {
            switch (maxpower)
            {
                case bool b: return power > (b ? BigInteger.One : BigInteger.Zero);
                case sbyte v: return power > v;
                case byte v: return power > v;
                case short v: return power > v;
                case ushort v: return power > v;
                case int v: return power > v;
                case uint v: return power > v;
                case long v: return power > v;
                case ulong v: return power > v;
                case BigInteger v: return power > v;
                case char v: return power > (int)v;                        // a NumPy integer scalar: exact
                case decimal v: return DecimalLess(v, power);              // NumSharp's decimal scalar: exact
                case double v: return IntExceedsFloat(power, v);           // a Python float: CPython's exact comparison
                case float v: return IntExceedsFloat(power, v);            // a C# float is a Python float (house rule)
                case Half v: return HalfLess(v, ToDoubleOrThrow(power));   // np.float16: NEP 50 converts the int to float16
                case Complex:
                    throw new TypeError("'>' not supported between instances of 'int' and 'complex'");
                case string:
                    throw new TypeError("'>' not supported between instances of 'int' and 'str'");
                case NDArray nd:
                    return ArrayExceeds(power, nd);
                case ITuple:
                    throw new TypeError("'>' not supported between instances of 'int' and 'tuple'");
            }
            // A typed C# array / Memory<T> is an ndarray (the comparison is a ufunc, then its truth value is tested).
            if (PolySequence.IsArrayLike(maxpower))
                return ArrayExceeds(power, np.asanyarray(maxpower));
            if (PolySequence.IsSequence(maxpower))
                throw new TypeError("'>' not supported between instances of 'int' and 'list'");
            throw new TypeError($"'>' not supported between instances of 'int' and '{NDPolySeries.PythonTypeName(maxpower)}'");
        }

        /// <summary>
        ///     <c>bool(power &gt; nd)</c>: NumPy's comparison ufunc with the weak int converted to <paramref name="nd"/>'s
        ///     comparison dtype (the conversion's OverflowError first), then the truth value of the result array.
        /// </summary>
        /// <param name="power">The integer power (≥ 0).</param>
        /// <param name="nd">The limit array (any layout, any rank).</param>
        /// <returns>The single element's comparison.</returns>
        /// <exception cref="OverflowException">The power does not convert into the comparison dtype.</exception>
        /// <exception cref="ValueError">No element, or more than one.</exception>
        private static bool ArrayExceeds(BigInteger power, NDArray nd)
        {
            NPTypeCode t = nd.typecode;
            // The ufunc converts the weak int before it compares anything, so its error does not depend on the size.
            double p = 0;
            if (t is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex)
                p = ToDoubleOrThrow(power);
            else if (t == NPTypeCode.Boolean && (power > long.MaxValue || power < long.MinValue))
                throw new OverflowException("int too big to convert");   // bool with a weak int is an int64 loop

            // `if power > maxpower:` tests the result array for truth: only a one-element result has one.
            if (nd.size == 0)
                throw new ValueError("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
            if (nd.size > 1)
                throw new ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");

            object e = nd.GetAtIndex(0);
            switch (t)
            {
                case NPTypeCode.Half: return HalfLess((Half)e, p);
                case NPTypeCode.Single: return (float)p > (float)e;             // RTNE double -> float, then compared
                case NPTypeCode.Double: return p > (double)e;
                case NPTypeCode.Complex:
                {
                    // NumPy's CGT with x = (p, 0): (xr > yr && !isnan(xi) && !isnan(yi)) || (xr == yr && xi > yi).
                    var z = (Complex)e;
                    return (p > z.Real && !double.IsNaN(z.Imaginary)) || (p == z.Real && 0.0 > z.Imaginary);
                }
                case NPTypeCode.Boolean: return power > ((bool)e ? BigInteger.One : BigInteger.Zero);
                case NPTypeCode.Decimal: return DecimalLess((decimal)e, power);
                case NPTypeCode.Char: return power > (int)(char)e;
                default:
                    // The eight integer widths: NumPy 2 compares a Python int with an integer array exactly, even out of range.
                    return power > PythonInt.From(e);
            }
        }

        /// <summary>
        ///     CPython's exact <c>int &gt; float</c>: NaN and +inf are never exceeded, -inf always is, and a finite float is
        ///     exceeded by an integer exactly when the integer exceeds its floor.
        /// </summary>
        /// <param name="power">The integer.</param>
        /// <param name="d">The float.</param>
        /// <returns><c>power &gt; d</c>.</returns>
        private static bool IntExceedsFloat(BigInteger power, double d)
        {
            if (double.IsNaN(d) || double.IsPositiveInfinity(d))
                return false;
            if (double.IsNegativeInfinity(d))
                return true;
            // BigInteger(double) of an integral double is exact at any magnitude.
            return power > new BigInteger(Math.Floor(d));
        }

        /// <summary>
        ///     NumPy's float16 comparison <c>h_power &gt; h</c>, where <c>h_power</c> is the weak int converted to float16 the
        ///     way NumPy converts it — through the double <paramref name="p"/>, rounded to nearest even (the house cast kernel,
        ///     npy_double_to_half: overflow to inf).
        /// </summary>
        /// <param name="h">The limit.</param>
        /// <param name="p">The power as a double (already range-checked).</param>
        /// <returns>The comparison (false when either side is NaN).</returns>
        private static unsafe bool HalfLess(Half h, double p)
        {
            Half hp;
            DirectILKernelGenerator.GetPolyCastKernel(NPTypeCode.Double, NPTypeCode.Half)((byte*)&p, 1, sizeof(double), (byte*)&hp);
            // Two float16 values compare exactly as their (exact) double values.
            return (double)hp > (double)h;
        }

        /// <summary>
        ///     <c>power &gt; m</c> exactly, for NumSharp's decimal limit (no NumPy analog: the exact comparison a Python
        ///     <c>decimal.Decimal</c> gives).
        /// </summary>
        /// <param name="m">The limit.</param>
        /// <param name="power">The integer power.</param>
        /// <returns><c>power &gt; m</c>.</returns>
        private static bool DecimalLess(decimal m, BigInteger power)
        {
            // The integer exceeds m exactly when it exceeds m's floor (an integer, exact in BigInteger at every decimal size).
            return power > new BigInteger(decimal.Floor(m));
        }

        /// <summary>
        ///     CPython's <c>float(int)</c> as NumPy's conversion of a weak int into a float loop performs it: correctly
        ///     rounded, and an OverflowError when the rounded value leaves the double range.
        /// </summary>
        /// <param name="power">The integer.</param>
        /// <returns>The double.</returns>
        /// <exception cref="OverflowException"><c>int too large to convert to float</c>.</exception>
        private static double ToDoubleOrThrow(BigInteger power)
        {
            double d = PythonInt.ToDouble(power);
            if (double.IsInfinity(d))
                throw new OverflowException("int too large to convert to float");
            return d;
        }
    }
}
