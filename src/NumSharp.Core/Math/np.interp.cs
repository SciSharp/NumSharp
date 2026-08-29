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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.interp.html</remarks>
        public static NDArray interp(NDArray x, NDArray xp, NDArray fp,
            double? left = null, double? right = null, double? period = null)
        {
            bool isComplex = fp.typecode == NPTypeCode.Complex;
            Complex? cleft = left.HasValue ? new Complex(left.Value, 0) : (Complex?)null;
            Complex? cright = right.HasValue ? new Complex(right.Value, 0) : (Complex?)null;
            return isComplex
                ? InterpComplex(x, xp, fp, cleft, cright, period)
                : InterpReal(x, xp, fp, left, right, period);
        }

        /// <summary>
        ///     Complex-fp overload of <see cref="interp(NDArray,NDArray,NDArray,double?,double?,double?)"/>
        ///     accepting complex <paramref name="left"/> / <paramref name="right"/> fill values.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.interp.html</remarks>
        public static NDArray interp(NDArray x, NDArray xp, NDArray fp,
            Complex? left, Complex? right = null, double? period = null)
            => InterpComplex(x, xp, fp, left, right, period);

        // =====================================================================
        // Real (float64) path
        // =====================================================================

        private static unsafe NDArray InterpReal(NDArray x, NDArray xp, NDArray fp,
            double? left, double? right, double? period)
        {
            if (period.HasValue)
                (x, xp, fp, left, right) = ApplyPeriodReal(x, xp, fp, period.Value);

            // afp/axp are 1-D float64; ax is float64 of any shape, read in C-order.
            var afp = ToContiguous1DDouble(fp, "fp");
            var axp = ToContiguous1DDouble(xp, "xp");
            var xc = x.astype(NPTypeCode.Double, copy: true, order: 'C');

            long lenxp = axp.size;
            if (lenxp == 0)
                throw new ArgumentException("array of sample points is empty");
            if (afp.size != lenxp)
                throw new ArgumentException("fp and xp are not of the same length.");

            var result = new NDArray(NPTypeCode.Double, new Shape(x.shape), fillZeros: false);
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

        private static unsafe NDArray InterpComplex(NDArray x, NDArray xp, NDArray fp,
            Complex? left, Complex? right, double? period)
        {
            if (period.HasValue)
                (x, xp, fp) = ApplyPeriodComplex(x, xp, fp, period.Value);

            var afp = ToContiguous1DComplex(fp);
            var axp = ToContiguous1DDouble(xp, "xp");
            var xc = x.astype(NPTypeCode.Double, copy: true, order: 'C');

            long lenxp = axp.size;
            if (lenxp == 0)
                throw new ArgumentException("array of sample points is empty");
            if (afp.size != lenxp)
                throw new ArgumentException("fp and xp are not of the same length.");

            var result = new NDArray(NPTypeCode.Complex, new Shape(x.shape), fillZeros: false);
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

        private static NDArray ToContiguous1DDouble(NDArray a, string name)
        {
            if (a.ndim > 1)
                throw new ArgumentException("object too deep for desired array");
            if (a.ndim < 1)
                throw new ArgumentException("object of too small depth for desired array");
            return a.astype(NPTypeCode.Double, copy: true, order: 'C');
        }

        private static NDArray ToContiguous1DComplex(NDArray a)
        {
            if (a.ndim > 1)
                throw new ArgumentException("object too deep for desired array");
            if (a.ndim < 1)
                throw new ArgumentException("object of too small depth for desired array");
            return a.astype(NPTypeCode.Complex, copy: true, order: 'C');
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

        private static (NDArray x, NDArray xp, NDArray fp) PreparePeriodic(
            NDArray x, NDArray xp, NDArray fp, double period, NPTypeCode fpType)
        {
            if (period == 0)
                throw new ArgumentException("period must be a non-zero value");
            period = Math.Abs(period);

            var xd = x.astype(NPTypeCode.Double, copy: true, order: 'C');
            var xpd = xp.astype(NPTypeCode.Double, copy: true, order: 'C');
            var fpd = fp.astype(fpType, copy: true, order: 'C');

            if (xpd.ndim != 1 || fpd.ndim != 1)
                throw new ArgumentException("Data points must be 1-D sequences");
            if (xpd.shape[0] != fpd.shape[0])
                throw new ArgumentException("fp and xp are not of the same length");

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
            xpd = np.concatenate(new[] { xpLast - period, xpd, xpFirst + period });
            var fpFirst = fpd["0:1"];
            var fpLast = fpd[(fpd.size - 1).ToString() + ":"];
            fpd = np.concatenate(new[] { fpLast, fpd, fpFirst });

            return (xd, xpd, fpd);
        }
    }
}
