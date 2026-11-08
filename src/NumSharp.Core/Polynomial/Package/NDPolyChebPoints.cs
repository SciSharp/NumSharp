using System;
using System.Numerics;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyChebPoints.cs — chebpts1, chebpts2, chebgauss: Chebyshev points and Gauss-Chebyshev quadrature
// =============================================================================
//
// NumPy 2.4.2 (numpy/polynomial/chebyshev.py), statement for statement:
//
//   def chebpts1(npts):                                 def chebpts2(npts):
//       _npts = int(npts)                                   _npts = int(npts)
//       if _npts != npts:                                   if _npts != npts:
//           raise ValueError("npts must be integer")            raise ValueError("npts must be integer")
//       if _npts < 1:                                       if _npts < 2:
//           raise ValueError("npts must be >= 1")               raise ValueError("npts must be >= 2")
//       x = 0.5 * np.pi / _npts * np.arange(-_npts + 1, _npts + 1, 2)
//       return np.sin(x)                                    x = np.linspace(-np.pi, 0, _npts)
//                                                           return np.cos(x)
//   def chebgauss(deg):
//       ideg = pu._as_int(deg, "deg")
//       if ideg <= 0:
//           raise ValueError("deg must be a positive integer")
//       x = np.cos(np.pi * np.arange(1, 2 * ideg, 2) / (2.0 * ideg))
//       w = np.ones(ideg) * (np.pi / ideg)
//       return x, w
//
// ARGUMENTS. npts goes through Python's int() and the `_npts != npts` test (PolyPowerArgument, shared with {p}pow): a
// float with a fraction, a str (int('3') succeeds, then '3' != 3) and a 0-d array holding 3.5 are "npts must be integer";
// None, a complex, a list and a sized ndarray are int()'s TypeErrors; NaN / inf are its ValueError / OverflowError.
// deg goes through operator.index (PolyIndexArgument): only an integer — a Python int or bool, a char, a 0-d integer
// array — is accepted; a float, a str or a 0-d bool array is "deg must be an integer, received {deg}".
//
// VALUES. Each node is NumPy's statement chain on an exact integer ramp value, computed by one emitted fill
// (DirectILKernelGenerator.GetPolyChebNodesKernel) that writes the result and nothing else:
//   chebpts1: sin(c * (1 - n + 2k)) with c = (0.5*pi)/float(n), CPython's float / int;
//   chebpts2: cos(k * (pi/(n - 1)) + (-pi)) for k < n - 1 and cos(0.0) for the endpoint (linspace's y[-1] = stop);
//   chebgauss: cos((pi * (2k + 1)) / (2.0*deg)), and w = pi/deg in every slot (np.ones(deg) * c is c exactly).
// sin / cos are Math.Sin / Math.Cos, the C runtime functions NumPy's float64 loops call on win-amd64, so the results
// are bit-identical there (the oracle tier is host-pinned for that reason). chebpts1's ramp is antisymmetric, so its fill
// computes the lower half and stores each sine's negation in the mirrored slot: the C runtime's sin is odd bit for bit
// (measured; DirectILKernelGenerator.PolyChebPoints.cs), so that is the same bytes as one sin per element, at half the cost.
//
// SIZES. np.arange computes its length in DOUBLE arithmetic (PyArray_ArangeObj: _calc_length's `(stop - start) / step`
// as a Python int true division, ceil, a range check against (double)NPY_MAX_INTP = 2**63, then a C cast), and any
// OverflowError on the way becomes ValueError("Maximum allowed size exceeded"). ArangeLength ports that, so the huge
// counts give NumPy's answers, all probed:
//   * a length below 2**60 allocates (NumPy's MemoryError, NumSharp's OutOfMemoryException, past the machine);
//   * 2**60 up to the band below: "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum
//     possible size." (8-byte elements);
//   * a length whose double rounds to EXACTLY 2**63 passes the range check and casts to INT64_MIN on x86 (cvttsd2si):
//     a negative length, so np.arange returns an EMPTY array and NumPy goes on — chebpts1 returns an empty float64
//     array (n in [2**63 - 512, 2**63 + 1]) or an empty OBJECT array (n in [2**63 + 2, 2**63 + 1024], where the
//     ramp's start fits no integer dtype: NotSupportedException here), chebpts2 fails on linspace's `y[-1] = stop`
//     (IndexError "index -1 is out of bounds for axis 0 with size 0"), and chebgauss reaches np.ones(deg), whose shape
//     conversion or byte count fails ("Maximum allowed dimension exceeded" past int64, else "array is too big");
//   * past the band: "Maximum allowed size exceeded" — except chebpts1, whose `0.5*np.pi / n` converts n to a float
//     BEFORE np.arange runs and so raises CPython's OverflowError "int too large to convert to float" from n ~ 2**1024.
//
// SPEED (benchmark/polynomial/polypts_results.md, 35 cells, all bit-exact): geomean 3.37x NumPy. The trigonometric call
// dominates every node (about 4.5-5.5 ns, the same C runtime function on both sides), so the gain is what NumPy spends
// around it — its ramp, its intermediates and its Python statements — plus chebpts1's halved sin count (2.26-4.83x). The
// two cells under 1.5x are chebpts2 and chebgauss at 10K points (1.15x): NumPy's other passes are cache-resident there,
// and NumSharp's managed cos call is ~10% slower per element than NumPy's C loop.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     <c>chebpts1</c>, <c>chebpts2</c> and <c>chebgauss</c>: NumPy 2.4.2's statements (see the file header), each node
    ///     written by one emitted fill.
    /// </summary>
    internal static class NDPolyChebPoints
    {
        /// <summary>NumPy's text for an np.arange length that does not fit npy_intp (an OverflowError re-raised).</summary>
        private const string MaxSize = "Maximum allowed size exceeded";

        /// <summary><c>(double)NPY_MAX_INTP</c>: <c>2**63 - 1</c> rounded to a double, i.e. <c>2**63</c>.</summary>
        private const double MaxIntpAsDouble = 9223372036854775808.0;

        /// <summary>
        ///     <c>chebpts1(npts)</c>: the Chebyshev points of the first kind, <c>sin(0.5*pi/n * (1 - n + 2k))</c> for
        ///     <c>k</c> in <c>[0, n)</c> — the roots of <c>T_n</c>, in increasing order.
        /// </summary>
        /// <param name="npts">The number of points: anything Python's <c>int()</c> accepts that equals its integer.</param>
        /// <returns>A new float64 array of <c>n</c> points (empty only in the <c>2**63</c> band, see the file header).</returns>
        /// <exception cref="ValueError"><c>npts must be integer</c>, <c>npts must be &gt;= 1</c>, int()'s ValueErrors (NaN, a
        ///     non-integer str), <c>array is too big; …</c> or <c>Maximum allowed size exceeded</c>.</exception>
        /// <exception cref="TypeError">int()'s TypeErrors (None, a complex, a list or tuple, an ndarray of one or more dims).</exception>
        /// <exception cref="OverflowException">int() of an infinity, or an <c>npts</c> too large for a float (CPython's
        ///     <c>int too large to convert to float</c>, raised by <c>0.5*np.pi / npts</c>).</exception>
        /// <exception cref="NotSupportedException">An <c>npts</c> in <c>[2**63 + 2, 2**63 + 1024]</c>, for which NumPy returns an
        ///     empty object array.</exception>
        /// <exception cref="OutOfMemoryException">A count below the byte limit that the machine cannot allocate (NumPy's MemoryError).</exception>
        public static unsafe NDArray Pts1(object npts)
        {
            BigInteger n = PointCount(npts, 1, "npts must be >= 1");
            // `0.5 * np.pi / _npts`: CPython float / int, the int converted by PyLong_AsDouble — evaluated BEFORE np.arange,
            // so an n past the float range raises here and never reaches the arange length check.
            double c = 0.5 * Math.PI / IntToFloat(n);
            long len = ArangeLength(1 - n, n + 1, 2);
            if (len == 0)
            {
                // The 2**63 band (any other n >= 1 gives n elements): NumPy's arange is empty, of float64 when the ramp's
                // start fits int64 (its stop then fits uint64), of object otherwise — and np.sin keeps the dtype.
                if (!FitsInt64OrUInt64(1 - n))
                    throw new NotSupportedException(
                        "chebpts1: NumPy returns an empty object array for this npts (np.arange's start fits no integer dtype), a dtype NumSharp does not have");
                return new NDArray(NPTypeCode.Double, Shape.Vector(0), false);
            }
            // The result has the int64 ramp's element size, so its allocation guard reports the arange's "array is too big".
            var x = new NDArray(NPTypeCode.Double, Shape.Vector(len), false);
            DirectILKernelGenerator.GetPolyChebNodesKernel(PolyChebNodes.Pts1)((double*)x.Unsafe.Address, len, (double)(long)(1 - n), 2.0, c);
            return x;
        }

        /// <summary>
        ///     <c>chebpts2(npts)</c>: the Chebyshev points of the second kind, <c>cos(linspace(-pi, 0, n))</c> — the extrema
        ///     of <c>T_(n-1)</c> with both endpoints, in increasing order.
        /// </summary>
        /// <param name="npts">The number of points: anything Python's <c>int()</c> accepts that equals its integer.</param>
        /// <returns>A new float64 array of <c>n</c> points.</returns>
        /// <exception cref="ValueError"><c>npts must be integer</c>, <c>npts must be &gt;= 2</c>, int()'s ValueErrors,
        ///     <c>array is too big; …</c> or <c>Maximum allowed size exceeded</c>.</exception>
        /// <exception cref="TypeError">int()'s TypeErrors.</exception>
        /// <exception cref="OverflowException">int() of an infinity.</exception>
        /// <exception cref="IndexError">An <c>npts</c> in the <c>2**63</c> band, whose linspace ramp is empty (NumPy's
        ///     <c>index -1 is out of bounds for axis 0 with size 0</c> from <c>y[-1, ...] = stop</c>).</exception>
        /// <exception cref="OutOfMemoryException">A count below the byte limit that the machine cannot allocate.</exception>
        public static unsafe NDArray Pts2(object npts)
        {
            BigInteger n = PointCount(npts, 2, "npts must be >= 2");
            // linspace(-pi, 0, n): its ramp is np.arange(0, n, dtype=float64), sized by the same length rule.
            long len = ArangeLength(0, n, 1);
            if (len == 0)
                throw new IndexError("index -1 is out of bounds for axis 0 with size 0");
            var x = new NDArray(NPTypeCode.Double, Shape.Vector(len), false);
            var dst = (double*)x.Unsafe.Address;
            // step = delta / div: delta = 0.0 - (-pi) = pi exactly, div = n - 1 converted to a double (exact below 2**53).
            double step = Math.PI / (double)(len - 1);
            DirectILKernelGenerator.GetPolyChebNodesKernel(PolyChebNodes.Pts2)(dst, len - 1, -Math.PI, step, 0.0);
            // linspace's `y[-1, ...] = stop` (the int 0 stored as +0.0), then np.cos of it.
            dst[len - 1] = Math.Cos(0.0);
            return x;
        }

        /// <summary>
        ///     <c>chebgauss(deg)</c>: Gauss-Chebyshev quadrature — the nodes <c>cos(pi*(2k + 1)/(2*deg))</c> and the weights
        ///     <c>pi/deg</c>, exact for polynomials up to degree <c>2*deg - 1</c> under the weight <c>1/sqrt(1 - x**2)</c>.
        /// </summary>
        /// <param name="deg">The number of nodes: an integer (Python's <c>operator.index</c>).</param>
        /// <returns>(x, w): two new float64 arrays of <c>deg</c> elements, the nodes in DECREASING order.</returns>
        /// <exception cref="TypeError"><c>deg must be an integer, received {deg}</c> (a float, a str, a 0-d bool array, …).</exception>
        /// <exception cref="ValueError"><c>deg must be a positive integer</c>, <c>array is too big; …</c>,
        ///     <c>Maximum allowed dimension exceeded</c> or <c>Maximum allowed size exceeded</c> (see the file header).</exception>
        /// <exception cref="OutOfMemoryException">A count below the byte limit that the machine cannot allocate.</exception>
        public static unsafe (NDArray x, NDArray w) Gauss(object deg)
        {
            BigInteger ideg = PolyIndexArgument.AsInt(deg, "deg");
            if (ideg.Sign <= 0)
                throw new ValueError("deg must be a positive integer");
            // x's np.arange(1, 2 * ideg, 2) runs first.
            long len = ArangeLength(1, 2 * ideg, 2);
            if (len == 0)
            {
                // The 2**63 band: the arange is empty, NumPy goes on to w = np.ones(ideg), and its shape conversion (past
                // int64) or byte count (8 * ideg, past int64 for every ideg in the band) fails.
                if (ideg > long.MaxValue)
                    throw new ValueError("Maximum allowed dimension exceeded");
                AllocationGuard.CheckElementCount((long)ideg, sizeof(double));
            }
            // `2.0 * ideg` and `np.pi / ideg`: CPython float arithmetic on the int converted to a double (exact here).
            double twoDeg = 2.0 * IntToFloat(ideg);
            double w0 = Math.PI / IntToFloat(ideg);
            var x = new NDArray(NPTypeCode.Double, Shape.Vector(len), false);
            DirectILKernelGenerator.GetPolyChebNodesKernel(PolyChebNodes.Gauss)((double*)x.Unsafe.Address, len, 1.0, 2.0, twoDeg);
            // np.ones(ideg) * c: 1.0 * c is c exactly (c is finite and positive), so the weights are c in every slot.
            var w = np.full(Shape.Vector(len), w0);
            return (x, w);
        }

        /// <summary>
        ///     <c>_npts = int(npts)</c>, <c>if _npts != npts</c> and the lower bound, in NumPy's order.
        /// </summary>
        /// <param name="npts">The argument.</param>
        /// <param name="minimum">The smallest accepted count (1 for chebpts1, 2 for chebpts2).</param>
        /// <param name="tooFew">NumPy's text for a count below <paramref name="minimum"/>.</param>
        /// <returns>The count.</returns>
        /// <exception cref="ValueError"><c>npts must be integer</c>, <paramref name="tooFew"/>, or int()'s ValueErrors.</exception>
        /// <exception cref="TypeError">int()'s TypeErrors.</exception>
        /// <exception cref="OverflowException">int() of an infinity.</exception>
        private static BigInteger PointCount(object npts, int minimum, string tooFew)
        {
            BigInteger n = PolyPowerArgument.Int(npts);
            if (PolyPowerArgument.DiffersFromInt(n, npts))
                throw new ValueError("npts must be integer");
            if (n < minimum)
                throw new ValueError(tooFew);
            return n;
        }

        /// <summary>
        ///     CPython's int-to-float conversion inside float arithmetic (<c>PyLong_AsDouble</c>): correctly rounded, raising
        ///     past the float range.
        /// </summary>
        /// <param name="v">The integer.</param>
        /// <returns>The nearest double.</returns>
        /// <exception cref="OverflowException">CPython's <c>int too large to convert to float</c>.</exception>
        private static double IntToFloat(BigInteger v)
        {
            double d = PythonInt.ToDouble(v);
            if (double.IsInfinity(d))
                throw new OverflowException("int too large to convert to float");
            return d;
        }

        /// <summary>
        ///     The length <c>np.arange(start, stop, step)</c> gives Python-int arguments — <c>PyArray_ArangeObj</c>'s
        ///     <c>_calc_length</c> and <c>_arange_safe_ceil_to_intp</c>, including what they do at the int64 edge on x86.
        /// </summary>
        /// <param name="start">The start (a Python int).</param>
        /// <param name="stop">The stop.</param>
        /// <param name="step">The step (nonzero).</param>
        /// <returns>The element count: 0 for an empty range and for the band whose double rounds to exactly <c>2**63</c>.</returns>
        /// <exception cref="ValueError"><c>Maximum allowed size exceeded</c>: the quotient does not fit a float, or its
        ///     ceiling does not fit npy_intp (NumPy re-raises both OverflowErrors as this ValueError).</exception>
        /// <remarks>
        ///     NumPy forms <c>(stop - start) / step</c> as a Python int true division (correctly rounded, OverflowError past
        ///     the float range), takes its ceiling, and range-checks it against <c>(double)NPY_MAX_INTP</c>, which is
        ///     <c>2**63</c> after rounding. A ceiling of exactly <c>2**63</c> passes and its <c>(npy_intp)</c> cast is
        ///     <c>INT64_MIN</c> under x86's cvttsd2si (the C cast is undefined there; .NET 9+ would saturate instead, so the
        ///     case is decided explicitly): a negative length, which np.arange turns into an empty array. The same rule as
        ///     <c>np.random.permutation</c>'s length (its <c>ArangeLength</c>), for any start / stop / step.
        /// </remarks>
        internal static long ArangeLength(BigInteger start, BigInteger stop, BigInteger step)
        {
            BigInteger next = stop - start;
            double value;
            try
            {
                value = PyScalar.IntTrueDivide(next, step);
            }
            catch (OverflowException)
            {
                throw new ValueError(MaxSize);
            }
            // NumPy's "underflow and divide-by-inf" check: a zero quotient of a NONZERO difference is an underflow — one
            // element for a positive zero, none for a negative one.
            if (value == 0.0 && !next.IsZero)
                return double.IsNegative(value) ? 0 : 1;
            double ivalue = Math.Ceiling(value);
            if (!(-MaxIntpAsDouble <= ivalue && ivalue <= MaxIntpAsDouble))
                throw new ValueError(MaxSize);
            if (ivalue == MaxIntpAsDouble)
                return 0;
            long len = (long)ivalue;
            return len <= 0 ? 0 : len;
        }

        /// <summary>
        ///     Whether NumPy's dtype discovery gives a Python int an integer dtype (int64, else uint64) rather than object.
        /// </summary>
        /// <param name="v">The integer.</param>
        /// <returns>True when <paramref name="v"/> fits int64 or uint64.</returns>
        private static bool FitsInt64OrUInt64(BigInteger v) => v >= long.MinValue && v <= ulong.MaxValue;
    }
}
