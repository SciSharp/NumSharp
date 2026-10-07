using System;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     One-dimensional linear interpolation for monotonically increasing sample points. <br></br>
        ///     Returns the piecewise-linear interpolant to the discrete data points (xp, fp), evaluated
        ///     at each coordinate in x. Port of NumPy's <c>numpy.interp</c>
        ///     (<c>compiled_base.c::arr_interp</c> + the <c>_function_base_impl.py</c> wrapper).
        /// </summary>
        /// <param name="x">The x-coordinates at which to evaluate the interpolated values (any shape; the result has the same shape).</param>
        /// <param name="xp">The x-coordinates of the data points — 1-D, must be increasing unless <paramref name="period"/> is given.</param>
        /// <param name="fp">The y-coordinates of the data points, same length as xp (float or complex).</param>
        /// <param name="left">Value returned for <c>x &lt; xp[0]</c>; default is <c>fp[0]</c>. Ignored when <paramref name="period"/> is given.</param>
        /// <param name="right">Value returned for <c>x &gt; xp[-1]</c>; default is <c>fp[-1]</c>. Ignored when <paramref name="period"/> is given.</param>
        /// <param name="period">A period for the x-coordinates, allowing proper interpolation of angular x-coordinates. Must be non-zero.</param>
        /// <returns>The interpolated values, same shape as x (float64, or complex128 when fp is complex). A scalar (0-d) if x is a scalar.</returns>
        /// <exception cref="ValueError"><paramref name="fp"/> or <paramref name="xp"/> is not 1-D (<c>object too deep for desired
        ///     array</c> / <c>object of too small depth for desired array</c>), xp is empty, the lengths differ, or
        ///     <paramref name="period"/> is zero (and, with a period, <c>Data points must be 1-D sequences</c>).</exception>
        /// <exception cref="InvalidCastException">Without a period, an operand does not cast 'safely' to the loop's dtype
        ///     (text, or a complex x / xp — NumPy's TypeError <c>Cannot cast array data from dtype('&lt;U2') to dtype('float64')
        ///     according to the rule 'safe'</c>, <c>Cannot cast scalar …</c> for a 0-d x).</exception>
        /// <exception cref="StringComputationNotSupportedException">With a period, a non-empty text operand: NumPy parses the
        ///     text to float64 (the computational phase of the string dtypes).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.interp.html
        ///     <para>
        ///     <b>Check order.</b> NumPy picks the loop from <paramref name="fp"/> alone (<c>np.iscomplexobj(fp)</c>). Without a
        ///     period, the compiled loop converts fp, then xp, then x (<c>PyArray_ContiguousFromAny</c>), each by checking the
        ///     operand's rank FIRST (fp and xp must be 1-D; x may have any rank) and then its 'safe' cast — so a 2-D text xp is
        ///     the rank's ValueError, not the cast's TypeError, and fp's error wins over xp's. With a period, NumPy instead
        ///     converts x, xp and fp with <c>np.asarray(…, dtype=…)</c>, which PARSES text (an empty text array converts to an
        ///     empty float array without parsing).
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray interp(NDArray x, NDArray xp, NDArray fp,
            double? left = null, double? right = null, double? period = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(x, xp, fp);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy dispatches on fp's dtype alone; the operand checks run inside each loop in NumPy's order (fp, xp, x).
            bool isComplex = fp.typecode == NPTypeCode.Complex;
            Complex? cleft = left.HasValue ? new Complex(left.Value, 0) : (Complex?)null;
            Complex? cright = right.HasValue ? new Complex(right.Value, 0) : (Complex?)null;
            return isComplex
                ? InterpComplex(x, xp, fp, cleft, cright, period)
                : InterpReal(x, xp, fp, left, right, period, complexFill: false);
        }

        /// <summary>
        ///     Complex-fill overload of <see cref="interp(NDArray,NDArray,NDArray,double?,double?,double?)"/>: complex
        ///     <paramref name="left"/> / <paramref name="right"/> fill values for a complex <paramref name="fp"/>.
        /// </summary>
        /// <param name="x">The x-coordinates at which to evaluate the interpolated values.</param>
        /// <param name="xp">The x-coordinates of the data points (1-D).</param>
        /// <param name="fp">The y-coordinates of the data points (1-D, same length as xp).</param>
        /// <param name="left">Value returned for <c>x &lt; xp[0]</c>; null means <c>fp[0]</c>.</param>
        /// <param name="right">Value returned for <c>x &gt; xp[-1]</c>; null means <c>fp[-1]</c>.</param>
        /// <param name="period">A period for the x-coordinates; when given, <paramref name="left"/> and <paramref name="right"/>
        ///     are ignored.</param>
        /// <returns>The interpolated values, same shape as x: complex128 for a complex fp, else float64.</returns>
        /// <exception cref="ValueError">As <see cref="interp(NDArray,NDArray,NDArray,double?,double?,double?)"/>.</exception>
        /// <exception cref="InvalidCastException">As <see cref="interp(NDArray,NDArray,NDArray,double?,double?,double?)"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">As
        ///     <see cref="interp(NDArray,NDArray,NDArray,double?,double?,double?)"/>.</exception>
        /// <exception cref="TypeError"><paramref name="fp"/> is not complex, no period is given, and <paramref name="left"/> or
        ///     <paramref name="right"/> is given: <c>must be real number, not complex</c>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.interp.html
        ///     <para>
        ///     NumPy picks the loop from <paramref name="fp"/> only. A real fp runs the real loop, which converts the fill values
        ///     with <c>PyFloat_AsDouble</c> AFTER its operand and length checks — and that refuses a Python complex even with a
        ///     zero imaginary part. (This overload used to compute in complex for a real fp, and so returned complex128 where
        ///     NumPy raises or returns float64.)
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray interp(NDArray x, NDArray xp, NDArray fp,
            Complex? left, Complex? right = null, double? period = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(x, xp, fp);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (fp.typecode == NPTypeCode.Complex)
                return InterpComplex(x, xp, fp, left, right, period);
            // The real loop: a period discards the fill values before the loop reads them (NumPy sets them to None).
            return InterpReal(x, xp, fp, null, null, period,
                complexFill: !period.HasValue && (left.HasValue || right.HasValue));
        }

        // =====================================================================
        // Real (float64) path
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>arr_interp</c> (the float64 loop), after the optional period preprocessing.
        /// </summary>
        /// <param name="x">The query points (any rank).</param>
        /// <param name="xp">The sample points (must be 1-D).</param>
        /// <param name="fp">The sample values (must be 1-D, real).</param>
        /// <param name="left">The fill below <c>xp[0]</c>, or null for <c>fp[0]</c>.</param>
        /// <param name="right">The fill above <c>xp[-1]</c>, or null for <c>fp[-1]</c>.</param>
        /// <param name="period">The period, or null.</param>
        /// <param name="complexFill">True when the caller passed a complex fill value without a period: NumPy's
        ///     <c>PyFloat_AsDouble</c> refuses it once the operand and length checks have passed.</param>
        /// <returns>The float64 result, x's shape.</returns>
        /// <exception cref="ValueError">A rank, emptiness, length or period error (see the public overload).</exception>
        /// <exception cref="InvalidCastException">An operand that does not cast 'safely' to float64.</exception>
        /// <exception cref="StringComputationNotSupportedException">A period with a non-empty text operand.</exception>
        /// <exception cref="TypeError"><paramref name="complexFill"/> is true.</exception>
        private static unsafe NDArray InterpReal(NDArray x, NDArray xp, NDArray fp,
            double? left, double? right, double? period, bool complexFill)
        {
            if (period.HasValue)
                (x, xp, fp, left, right) = ApplyPeriodReal(x, xp, fp, period.Value);

            // afp/axp are 1-D float64; ax is float64 of any shape, read in C-order. NumPy converts fp, then xp, then x —
            // each one's rank before its cast — so the first failing operand in that order reports.
            var afp = ToContiguous1DDouble(fp, "fp");
            var axp = ToContiguous1DDouble(xp, "xp");
            ThrowIfNotSafeInterpCast(x, NPTypeCode.Double);
            var xc = x.astype(NPTypeCode.Double, copy: true, order: 'C');

            long lenxp = axp.size;
            if (lenxp == 0)
                throw new ValueError("array of sample points is empty");
            if (afp.size != lenxp)
                throw new ValueError("fp and xp are not of the same length.");
            // PyFloat_AsDouble(left/right) runs after the length checks; a Python complex fill is refused there.
            if (complexFill)
                throw new TypeError("must be real number, not complex");

            var result = new NDArray(NPTypeCode.Double, Shape.Adopt(x.Shape.dimensions), fillZeros: false);
            long lenx = xc.size;

            double* dx = (double*)axp.Address;
            double* dy = (double*)afp.Address;
            double* dz = (double*)xc.Address;
            double* dres = (double*)result.Address;

            double lval = left ?? dy[0];
            double rval = right ?? dy[lenxp - 1];

            if (lenxp == 1)
            {
                double xp0 = dx[0], fp0 = dy[0];
                for (long i = 0; i < lenx; i++)
                {
                    double xv = dz[i];
                    dres[i] = xv < xp0 ? lval : (xv > xp0 ? rval : fp0);
                }
            }
            else
            {
                // Precompute slopes only when there are relatively few of them (NumPy's heuristic).
                double[] slopes = null;
                if (lenxp <= lenx)
                {
                    slopes = new double[lenxp - 1];
                    for (long i = 0; i < lenxp - 1; i++)
                        slopes[i] = (dy[i + 1] - dy[i]) / (dx[i + 1] - dx[i]);
                }

                fixed (double* slopesPtr = slopes)
                {
                    long j = 0;
                    for (long i = 0; i < lenx; i++)
                    {
                        double xv = dz[i];
                        if (double.IsNaN(xv)) { dres[i] = xv; continue; }

                        j = BinarySearchWithGuess(xv, dx, lenxp, j);
                        if (j == -1) dres[i] = lval;
                        else if (j == lenxp) dres[i] = rval;
                        else if (j == lenxp - 1) dres[i] = dy[j];
                        else if (dx[j] == xv) dres[i] = dy[j];   // avoid non-finite interpolation
                        else
                        {
                            double slope = slopes != null
                                ? slopesPtr[j]
                                : (dy[j + 1] - dy[j]) / (dx[j + 1] - dx[j]);

                            double res = slope * (xv - dx[j]) + dy[j];
                            if (double.IsNaN(res))
                            {
                                res = slope * (xv - dx[j + 1]) + dy[j + 1];
                                if (double.IsNaN(res) && dy[j] == dy[j + 1]) res = dy[j];
                            }
                            dres[i] = res;
                        }
                    }
                }
            }

            return result;
        }

        // =====================================================================
        // Complex (complex128) path — as arr_interp but fp/lval/rval are complex.
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>arr_interp_complex</c> (the complex128 loop for a complex fp), after the optional period preprocessing.
        /// </summary>
        /// <param name="x">The query points (any rank).</param>
        /// <param name="xp">The sample points (must be 1-D).</param>
        /// <param name="fp">The complex sample values (must be 1-D).</param>
        /// <param name="left">The fill below <c>xp[0]</c>, or null for <c>fp[0]</c>.</param>
        /// <param name="right">The fill above <c>xp[-1]</c>, or null for <c>fp[-1]</c>.</param>
        /// <param name="period">The period, or null.</param>
        /// <returns>The complex128 result, x's shape.</returns>
        /// <exception cref="ValueError">A rank, emptiness, length or period error (see the public overload).</exception>
        /// <exception cref="InvalidCastException">xp or x does not cast 'safely' to float64.</exception>
        /// <exception cref="StringComputationNotSupportedException">A period with a non-empty text x or xp.</exception>
        private static unsafe NDArray InterpComplex(NDArray x, NDArray xp, NDArray fp,
            Complex? left, Complex? right, double? period)
        {
            if (period.HasValue)
                (x, xp, fp) = ApplyPeriodComplex(x, xp, fp, period.Value);

            // NumPy's order: fp, xp, x — each one's rank before its cast.
            var afp = ToContiguous1DComplex(fp);
            var axp = ToContiguous1DDouble(xp, "xp");
            ThrowIfNotSafeInterpCast(x, NPTypeCode.Double);
            var xc = x.astype(NPTypeCode.Double, copy: true, order: 'C');

            long lenxp = axp.size;
            if (lenxp == 0)
                throw new ValueError("array of sample points is empty");
            if (afp.size != lenxp)
                throw new ValueError("fp and xp are not of the same length.");

            var result = new NDArray(NPTypeCode.Complex, Shape.Adopt(x.Shape.dimensions), fillZeros: false);
            long lenx = xc.size;

            double* dx = (double*)axp.Address;
            Complex* dy = (Complex*)afp.Address;
            double* dz = (double*)xc.Address;
            Complex* dres = (Complex*)result.Address;

            Complex lval = left ?? dy[0];
            Complex rval = right ?? dy[lenxp - 1];

            if (lenxp == 1)
            {
                double xp0 = dx[0];
                Complex fp0 = dy[0];
                for (long i = 0; i < lenx; i++)
                {
                    double xv = dz[i];
                    dres[i] = xv < xp0 ? lval : (xv > xp0 ? rval : fp0);
                }
            }
            else
            {
                Complex[] slopes = null;
                if (lenxp <= lenx)
                {
                    slopes = new Complex[lenxp - 1];
                    for (long i = 0; i < lenxp - 1; i++)
                    {
                        double inv = 1.0 / (dx[i + 1] - dx[i]);
                        slopes[i] = new Complex(
                            (dy[i + 1].Real - dy[i].Real) * inv,
                            (dy[i + 1].Imaginary - dy[i].Imaginary) * inv);
                    }
                }

                long j = 0;
                for (long i = 0; i < lenx; i++)
                {
                    double xv = dz[i];
                    if (double.IsNaN(xv)) { dres[i] = new Complex(xv, xv); continue; }

                    j = BinarySearchWithGuess(xv, dx, lenxp, j);
                    if (j == -1) dres[i] = lval;
                    else if (j == lenxp) dres[i] = rval;
                    else if (j == lenxp - 1) dres[i] = dy[j];
                    else if (dx[j] == xv) dres[i] = dy[j];
                    else
                    {
                        Complex slope;
                        if (slopes != null) slope = slopes[j];
                        else
                        {
                            double inv = 1.0 / (dx[j + 1] - dx[j]);
                            slope = new Complex(
                                (dy[j + 1].Real - dy[j].Real) * inv,
                                (dy[j + 1].Imaginary - dy[j].Imaginary) * inv);
                        }

                        // Per component, with NumPy's "try the other direction" NaN fixup.
                        double re = slope.Real * (xv - dx[j]) + dy[j].Real;
                        if (double.IsNaN(re))
                        {
                            re = slope.Real * (xv - dx[j + 1]) + dy[j + 1].Real;
                            if (double.IsNaN(re) && dy[j].Real == dy[j + 1].Real) re = dy[j].Real;
                        }
                        double im = slope.Imaginary * (xv - dx[j]) + dy[j].Imaginary;
                        if (double.IsNaN(im))
                        {
                            im = slope.Imaginary * (xv - dx[j + 1]) + dy[j + 1].Imaginary;
                            if (double.IsNaN(im) && dy[j].Imaginary == dy[j + 1].Imaginary) im = dy[j].Imaginary;
                        }
                        dres[i] = new Complex(re, im);
                    }
                }
            }

            return result;
        }

        // =====================================================================
        // binary_search_with_guess — port of compiled_base.c.
        //   key < arr[0]       -> -1
        //   key == arr[len-1]  -> len-1
        //   key > arr[len-1]   -> len
        //   else               -> i such that arr[i] <= key < arr[i+1]
        // =====================================================================

        private const long LIKELY_IN_CACHE_SIZE = 8;

        private static unsafe long BinarySearchWithGuess(double key, double* arr, long len, long guess)
        {
            long imin = 0, imax = len;

            if (key > arr[len - 1]) return len;
            if (key < arr[0]) return -1;

            if (len <= 4)
            {
                // linear search from index 1 (we already know key >= arr[0]).
                long i;
                for (i = 1; i < len && key >= arr[i]; i++) { }
                return i - 1;
            }

            if (guess > len - 3) guess = len - 3;
            if (guess < 1) guess = 1;

            if (key < arr[guess])
            {
                if (key < arr[guess - 1])
                {
                    imax = guess - 1;
                    if (guess > LIKELY_IN_CACHE_SIZE && key >= arr[guess - LIKELY_IN_CACHE_SIZE])
                        imin = guess - LIKELY_IN_CACHE_SIZE;
                }
                else
                {
                    return guess - 1;
                }
            }
            else
            {
                if (key < arr[guess + 1]) return guess;

                if (key < arr[guess + 2]) return guess + 1;

                imin = guess + 2;
                if (guess < len - LIKELY_IN_CACHE_SIZE - 1 && key < arr[guess + LIKELY_IN_CACHE_SIZE])
                    imax = guess + LIKELY_IN_CACHE_SIZE;
            }

            while (imin < imax)
            {
                long imid = imin + ((imax - imin) >> 1);
                if (key >= arr[imid]) imin = imid + 1;
                else imax = imid;
            }
            return imin - 1;
        }

        // =====================================================================
        // helpers
        // =====================================================================

        /// <summary>
        ///     NumPy's <c>PyArray_ContiguousFromAny(a, NPY_DOUBLE, 1, 1)</c>: a contiguous 1-D float64 copy of a sample array.
        /// </summary>
        /// <param name="a">xp or fp.</param>
        /// <param name="name">The operand's name (kept for call-site readability; NumPy's messages do not name it).</param>
        /// <returns>The 1-D float64 copy.</returns>
        /// <exception cref="ValueError">The array is not 1-D (<c>object too deep for desired array</c> above, <c>object of too
        ///     small depth for desired array</c> for 0-d).</exception>
        /// <exception cref="InvalidCastException">The dtype does not cast 'safely' to float64 (text, complex).</exception>
        /// <remarks>The rank is checked before the dtype, as <c>PyArray_FromAny</c> does.</remarks>
        private static NDArray ToContiguous1DDouble(NDArray a, string name)
        {
            if (a.ndim > 1)
                throw new ValueError("object too deep for desired array");
            if (a.ndim < 1)
                throw new ValueError("object of too small depth for desired array");
            ThrowIfNotSafeInterpCast(a, NPTypeCode.Double);
            return a.astype(NPTypeCode.Double, copy: true, order: 'C');
        }

        /// <summary>
        ///     NumPy's <c>PyArray_ContiguousFromAny(fp, NPY_CDOUBLE, 1, 1)</c> for the complex loop: a contiguous 1-D complex128
        ///     copy of fp.
        /// </summary>
        /// <param name="a">The complex fp.</param>
        /// <returns>The 1-D complex128 copy.</returns>
        /// <exception cref="ValueError">The array is not 1-D.</exception>
        private static NDArray ToContiguous1DComplex(NDArray a)
        {
            if (a.ndim > 1)
                throw new ValueError("object too deep for desired array");
            if (a.ndim < 1)
                throw new ValueError("object of too small depth for desired array");
            return a.astype(NPTypeCode.Complex, copy: true, order: 'C');
        }

        /// <summary>
        ///     The 'safe' cast check of NumPy's interp operand conversion (<c>PyArray_FromArray</c> without FORCECAST): text
        ///     never casts safely to a number, and complex never casts safely to float64. NumSharp's other dtypes (bool, the
        ///     integers, the floats and decimal) convert as they always have.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="target">The loop dtype the operand is converted to (float64 or complex128).</param>
        /// <exception cref="InvalidCastException">NumPy's TypeError text: <c>Cannot cast array data from dtype('&lt;U2') to
        ///     dtype('float64') according to the rule 'safe'</c>, or <c>Cannot cast scalar from …</c> for a 0-d operand (its
        ///     <c>npy_set_invalid_cast_error</c> wording).</exception>
        private static void ThrowIfNotSafeInterpCast(NDArray a, NPTypeCode target)
        {
            bool refused = Backends.Strings.NDStringUfuncs.IsString(a)
                           || (target == NPTypeCode.Double && a.typecode == NPTypeCode.Complex);
            if (refused)
                throw new InvalidCastException(
                    $"Cannot cast {(a.ndim == 0 ? "scalar" : "array data")} from {a.dtype.ToString(true)} to " +
                    $"dtype('{target.AsNumpyDtypeName()}') according to the rule 'safe'");
        }

        // period preprocessing (numpy _function_base_impl.py): normalize periodic boundaries,
        // sort by xp, and wrap-extend the endpoints. left/right are discarded (set null).
        private static (NDArray x, NDArray xp, NDArray fp, double? left, double? right)
            ApplyPeriodReal(NDArray x, NDArray xp, NDArray fp, double period)
        {
            var (nx, nxp, nfp) = PreparePeriodic(x, xp, fp, period, NPTypeCode.Double);
            return (nx, nxp, nfp, null, null);
        }

        private static (NDArray x, NDArray xp, NDArray fp)
            ApplyPeriodComplex(NDArray x, NDArray xp, NDArray fp, double period)
            => PreparePeriodic(x, xp, fp, period, NPTypeCode.Complex);

        /// <summary>
        ///     NumPy's period preprocessing (<c>_function_base_impl.py</c>): x and xp wrapped into <c>[0, period)</c>, the
        ///     samples sorted by xp and extended by one wrapped sample at each end.
        /// </summary>
        /// <param name="x">The query points.</param>
        /// <param name="xp">The sample points.</param>
        /// <param name="fp">The sample values.</param>
        /// <param name="period">The period (non-zero; its absolute value is used).</param>
        /// <param name="fpType">The loop dtype of fp (float64 or complex128).</param>
        /// <returns>The prepared x, xp and fp.</returns>
        /// <exception cref="ValueError">A zero period (checked first), a sample array that is not 1-D, or lengths that differ.</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text operand: <c>np.asarray(…, dtype=…)</c> parses
        ///     text into numbers (the computational phase). An EMPTY text operand converts without parsing, as in NumPy.</exception>
        private static (NDArray x, NDArray xp, NDArray fp) PreparePeriodic(
            NDArray x, NDArray xp, NDArray fp, double period, NPTypeCode fpType)
        {
            if (period == 0)
                throw new ValueError("period must be a non-zero value");
            period = Math.Abs(period);

            // np.asarray(x / xp / fp, dtype=…) in that order: a FORCED (unsafe) conversion, so a complex x drops its imaginary
            // part (NumPy warns) and text is parsed — astype refuses a non-empty text parse as the computational phase.
            var xd = x.astype(NPTypeCode.Double, copy: true, order: 'C');
            var xpd = xp.astype(NPTypeCode.Double, copy: true, order: 'C');
            var fpd = fp.astype(fpType, copy: true, order: 'C');

            if (xpd.ndim != 1 || fpd.ndim != 1)
                throw new ValueError("Data points must be 1-D sequences");
            if (xpd.Shape.dimensions[0] != fpd.Shape.dimensions[0])
                throw new ValueError("fp and xp are not of the same length");

            // normalize periodic boundaries: x %= period; xp %= period
            xd = np.mod(xd, (NDArray)period);
            xpd = np.mod(xpd, (NDArray)period);

            // sort by xp, reorder fp likewise
            var order = np.argsort(xpd);
            xpd = xpd[order];
            fpd = fpd[order];

            // wrap-extend endpoints: xp = [xp[-1]-period, xp..., xp[0]+period], fp = [fp[-1], fp..., fp[0]]
            var xpFirst = xpd["0:1"];
            var xpLast = xpd[(xpd.size - 1).ToString() + ":"];
            // concatenate copies the wrap temps' values out, leaving them dead the moment it
            // returns — reclaim them here eagerly rather than holding them to the entry scope's
            // close (the [NDScoped] interp entries would sweep them anyway; Dispose is idempotent).
            using NDArray xpLow = xpLast - period, xpHigh = xpFirst + period;
            xpd = np.concatenate(new[] { xpLow, xpd, xpHigh });
            var fpFirst = fpd["0:1"];
            var fpLast = fpd[(fpd.size - 1).ToString() + ":"];
            fpd = np.concatenate(new[] { fpLast, fpd, fpFirst });

            return (xd, xpd, fpd);
        }
    }
}
