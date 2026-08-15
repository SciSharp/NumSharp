using System;
using System.Numerics;
using NumSharp.Fourier;
using NumSharp.Backends.Iteration;

// =============================================================================
// The umath driver (port of numpy fft/_pocketfft_umath.cpp's fft_loop /
// rfft_impl / irfft_loop). For each 1-D lane along the transform axis it does
// copy_input (gather min(nin,n) strided elements, zero-pad) -> plan.exec(buffer,
// fct, dir) -> copy_output (scatter), including the exact FFTPACK half-complex
// packing. The all-but-axis walk mirrors AxisSort.DriveAllButAxis but is done
// with an explicit stride odometer so every layout (contiguous / strided /
// transposed / broadcast-read / sliced-offset) is handled directly.
//
// On win-amd64 numpy always takes this scalar per-lane path (POCKETFFT_NO_VECTORS
// is defined for MSVC), so this driver + the scalar engine == numpy bit-for-bit.
// =============================================================================

namespace NumSharp
{
    public static unsafe class PocketFFTDriver
    {
        /// <summary>
        /// Run a 1-D pocketfft transform along <paramref name="axis"/> of <paramref name="a"/>.
        /// </summary>
        /// <param name="a">Input array. Coerced to complex128 (c2c / irfft) or double (rfft).</param>
        /// <param name="n">The transform length: c2c/irfft output length, rfft input (npts).</param>
        /// <param name="axis">Transform axis (negative allowed).</param>
        /// <param name="isReal">true = real transform (rfft when forward, irfft when inverse).</param>
        /// <param name="isForward">true = forward (fft/rfft), false = inverse (ifft/irfft).</param>
        /// <param name="fct">Normalisation factor applied inside the transform.</param>
        /// <param name="out">Optional preallocated output (shape = a.shape with axis-&gt;n_out).</param>
        public static NDArray Execute(NDArray a, int n, int axis, bool isReal, bool isForward, double fct, NDArray @out = null)
        {
            int ndim = a.ndim;
            if (ndim == 0)
                throw new ArgumentException("FFT requires an array with at least one dimension.");
            int ax = axis < 0 ? axis + ndim : axis;
            if (ax < 0 || ax >= ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), $"axis {axis} is out of bounds for array of dimension {ndim}");
            if (n < 1)
                throw new ArgumentException($"Invalid number of FFT data points ({n}) specified.");

            // Resolve n_out and the input/output dtypes per transform kind.
            long nout;
            NPTypeCode inDtype, outDtype;
            if (!isReal)
            {
                nout = n; inDtype = NPTypeCode.Complex; outDtype = NPTypeCode.Complex;
            }
            else if (isForward)
            {
                nout = (long)n / 2 + 1; inDtype = NPTypeCode.Double; outDtype = NPTypeCode.Complex; // rfft
            }
            else
            {
                nout = n; inDtype = NPTypeCode.Complex; outDtype = NPTypeCode.Double; // irfft
            }

            // Coerce the input to the dtype the engine reads.
            NDArray src = a.GetTypeCode == inDtype ? a : a.astype(inDtype, copy: false);

            // Build the output shape = input shape with axis -> nout.
            var outDims = (long[])src.Shape.dimensions.Clone();
            outDims[ax] = nout;
            var outShape = new Shape(outDims);

            NDArray result;
            NDArray castTarget = null;   // set when out's dtype is a same_kind cast OF the loop dtype
            if (!(@out is null))
            {
                if (@out.ndim != ndim)
                    throw new ArgumentException("output array has wrong shape.");
                for (int d = 0; d < ndim; d++)
                    if (@out.shape[d] != outDims[d])
                        throw new ArgumentException("output array has wrong shape.");

                NPTypeCode outTc = @out.GetTypeCode;
                if (outTc == outDtype)
                {
                    // Exact loop dtype: the engine writes straight into out (in place, any layout).
                    result = @out;
                }
                else if (NDIterCasting.CanCast(outDtype, outTc, NPY_CASTING.NPY_SAME_KIND_CASTING))
                {
                    // NumPy's ufunc out= accepts any same_kind cast FROM the loop output dtype — e.g.
                    // irfft's float64 into a complex128 (imag=0) or float32 out. Compute in the loop
                    // dtype, then cast into out at the end. (fft/rfft's loop is complex128, whose only
                    // same_kind target here is complex128, so those still require a complex out.)
                    result = new NDArray(outDtype, outShape, false);
                    castTarget = @out;
                }
                else
                {
                    // Verbatim NumPy ufunc-cast rejection (house ArgumentException, exactly as
                    // DefaultEngine.UfuncOut.ValidateOutCast renders it — no trailing period).
                    throw new ArgumentException(
                        $"Cannot cast ufunc '{UfuncName(isReal, isForward, n)}' output from " +
                        $"dtype('{outDtype.AsNumpyDtypeName()}') to " +
                        $"dtype('{outTc.AsNumpyDtypeName()}') with casting rule 'same_kind'");
                }
            }
            else
            {
                result = new NDArray(outDtype, outShape, false);
            }

            long nin = src.shape[ax];
            long inAxisStride = src.Shape.strides[ax];
            long outAxisStride = result.Shape.strides[ax];
            long[] inStrides = src.Shape.strides;
            long[] outStrides = result.Shape.strides;
            long inBase0 = src.Shape.offset;
            long outBase0 = result.Shape.offset;

            // Number of independent 1-D transforms (product of all-but-axis dims).
            long lanes = 1;
            for (int d = 0; d < ndim; d++) if (d != ax) lanes *= src.shape[d];
            if (lanes == 0) return castTarget ?? result; // an empty non-axis dimension: nothing to do

            if (!isReal)
                RunC2C(src, result, ax, ndim, n, nin, inAxisStride, outAxisStride, inStrides, outStrides, inBase0, outBase0, lanes, isForward, fct);
            else if (isForward)
                RunR2C(src, result, ax, ndim, n, nin, inAxisStride, outAxisStride, inStrides, outStrides, inBase0, outBase0, lanes, fct);
            else
                RunC2R(src, result, ax, ndim, n, nin, inAxisStride, outAxisStride, inStrides, outStrides, inBase0, outBase0, lanes, fct);

            if (castTarget is not null)   // NB: NDArray overloads != element-wise; use `is not null`.
            {
                // Cast the loop-dtype result into the requested out (float64->complex128 sets imag=0;
                // float64->float32/float16 narrows), writing through out's own strides. Returns out.
                np.copyto(castTarget, result);
                return castTarget;
            }
            return result;
        }

        // NumPy's ufunc name for the out= cast-error message (matches the fft ufunc registration:
        // fft/ifft, rfft_n_even/rfft_n_odd by input parity, irfft).
        private static string UfuncName(bool isReal, bool isForward, long n)
        {
            if (!isReal) return isForward ? "fft" : "ifft";
            if (isForward) return (n % 2 == 0) ? "rfft_n_even" : "rfft_n_odd";
            return "irfft";
        }

        // Advance the all-but-axis odometer (rightmost non-axis dim fastest), updating both bases.
        private static void Advance(long[] idx, int ndim, int ax, NDArray src,
            long[] inStrides, long[] outStrides, ref long inBase, ref long outBase)
        {
            for (int d = ndim - 1; d >= 0; d--)
            {
                if (d == ax) continue;
                idx[d]++;
                inBase += inStrides[d];
                outBase += outStrides[d];
                if (idx[d] < src.shape[d]) return;
                idx[d] = 0;
                inBase -= src.shape[d] * inStrides[d];
                outBase -= src.shape[d] * outStrides[d];
            }
        }

        private static void RunC2C(NDArray src, NDArray result, int ax, int ndim, long n, long nin,
            long inAxisStride, long outAxisStride, long[] inStrides, long[] outStrides,
            long inBase0, long outBase0, long lanes, bool fwd, double fct)
        {
            var plan = Fourier.PocketFFTPlanCache.GetComplex(n);
            Complex* ipAddr = (Complex*)src.Address;
            Complex* opAddr = (Complex*)result.Address;
            var bufArr = new Cmplx[n];
            var idx = new long[ndim];
            long inBase = inBase0, outBase = outBase0;
            long ncopy = nin <= n ? nin : n;
            fixed (Cmplx* buf = bufArr)
            {
                for (long lane = 0; lane < lanes; lane++)
                {
                    for (long k = 0; k < ncopy; k++)
                    {
                        Complex v = ipAddr[inBase + k * inAxisStride];
                        buf[k] = new Cmplx(v.Real, v.Imaginary);
                    }
                    for (long k = ncopy; k < n; k++) buf[k] = new Cmplx(0.0, 0.0);

                    plan.Exec(buf, fct, fwd);

                    for (long k = 0; k < n; k++)
                    {
                        Cmplx v = buf[k];
                        opAddr[outBase + k * outAxisStride] = new Complex(v.r, v.i);
                    }
                    if (lane + 1 < lanes) Advance(idx, ndim, ax, src, inStrides, outStrides, ref inBase, ref outBase);
                }
            }
        }

        private static void RunR2C(NDArray src, NDArray result, int ax, int ndim, long npts, long nin,
            long inAxisStride, long outAxisStride, long[] inStrides, long[] outStrides,
            long inBase0, long outBase0, long lanes, double fct)
        {
            var plan = Fourier.PocketFFTPlanCache.GetReal(npts);
            double* ipAddr = (double*)src.Address;
            Complex* opAddr = (Complex*)result.Address;
            var bufArr = new double[npts];
            var idx = new long[ndim];
            long inBase = inBase0, outBase = outBase0;
            long ncopy = nin <= npts ? nin : npts;
            long half = (npts - 1) / 2;
            fixed (double* buf = bufArr)
            {
                for (long lane = 0; lane < lanes; lane++)
                {
                    for (long k = 0; k < ncopy; k++) buf[k] = ipAddr[inBase + k * inAxisStride];
                    for (long k = ncopy; k < npts; k++) buf[k] = 0.0;

                    plan.Exec(buf, fct, true);

                    // pack FFTPACK half-complex R0,R1,I1,... into complex128
                    opAddr[outBase] = new Complex(buf[0], 0.0);
                    for (long kk = 1; kk <= half; kk++)
                        opAddr[outBase + kk * outAxisStride] = new Complex(buf[2 * kk - 1], buf[2 * kk]);
                    if ((npts & 1) == 0)
                        opAddr[outBase + (npts / 2) * outAxisStride] = new Complex(buf[npts - 1], 0.0);

                    if (lane + 1 < lanes) Advance(idx, ndim, ax, src, inStrides, outStrides, ref inBase, ref outBase);
                }
            }
        }

        private static void RunC2R(NDArray src, NDArray result, int ax, int ndim, long nout, long nin,
            long inAxisStride, long outAxisStride, long[] inStrides, long[] outStrides,
            long inBase0, long outBase0, long lanes, double fct)
        {
            var plan = Fourier.PocketFFTPlanCache.GetReal(nout);
            Complex* ipAddr = (Complex*)src.Address;
            double* opAddr = (double*)result.Address;
            var bufArr = new double[nout];
            var idx = new long[ndim];
            long inBase = inBase0, outBase = outBase0;
            long half = (nout - 1) / 2;
            long ncopy = nin - 1; if (ncopy > half) ncopy = half; if (ncopy < 0) ncopy = 0;
            fixed (double* buf = bufArr)
            {
                for (long lane = 0; lane < lanes; lane++)
                {
                    // build FFTPACK half-complex buffer from the complex half-spectrum
                    buf[0] = nin >= 1 ? ipAddr[inBase].Real : 0.0;
                    if (nout > 1)
                    {
                        for (long kk = 1; kk <= ncopy; kk++)
                        {
                            Complex v = ipAddr[inBase + kk * inAxisStride];
                            buf[2 * kk - 1] = v.Real;
                            buf[2 * kk] = v.Imaginary;
                        }
                        for (long kk = ncopy + 1; kk <= half; kk++)
                        {
                            buf[2 * kk - 1] = 0.0;
                            buf[2 * kk] = 0.0;
                        }
                        if ((nout & 1) == 0)
                            buf[nout - 1] = (nout / 2 >= nin) ? 0.0 : ipAddr[inBase + (nout / 2) * inAxisStride].Real;
                    }

                    plan.Exec(buf, fct, false);

                    for (long k = 0; k < nout; k++)
                        opAddr[outBase + k * outAxisStride] = buf[k];

                    if (lane + 1 < lanes) Advance(idx, ndim, ax, src, inStrides, outStrides, ref inBase, ref outBase);
                }
            }
        }
    }
}
