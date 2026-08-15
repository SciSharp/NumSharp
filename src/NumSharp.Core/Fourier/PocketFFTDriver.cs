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
        /// <param name="floatPrec">
        ///     The value being transformed is float32/float16 precision (numpy returns complex64/float32/
        ///     float16). NumSharp keeps the result dtype complex128/float64 (no complex64 — issue #569) but
        ///     reproduces numpy's VALUES. numpy computes almost every one of these in DOUBLE and rounds the
        ///     output to the numpy result precision — so here the double engine runs and the result is
        ///     rounded element-wise (complex outputs → float32 components; irfft/hfft real output → float32,
        ///     or float16 when the input is a float16 real array). The ONE exception is <c>rfft</c> of a
        ///     float32 real input, which numpy runs through its single-precision <c>ff-&gt;F</c> loop; that
        ///     case (and only that) takes the single-precision <see cref="RfftpF"/> engine and needs no
        ///     rounding (it already produces float values). The <paramref name="fct"/> arrives already
        ///     computed in the right real_dtype (see <c>RawFft</c>).
        /// </param>
        public static NDArray Execute(NDArray a, int n, int axis, bool isReal, bool isForward, double fct, NDArray @out = null, bool floatPrec = false, bool effNormUnity = true)
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

            // Compute mode for a float32/float16-precision transform (numpy parity — see the docs on
            // `floatPrec`). numpy runs a SINGLE-precision loop when the operand already matches one WITHOUT
            // a real->complex128 promotion — i.e. a complex64-precision operand (fft/ifft/irfft of an N-D
            // intermediate: a complex operand under floatPrec) OR rfft of a float32 REAL input. Every other
            // float32/float16 transform promotes real->complex128 and is DOUBLE-computed then ROUNDED to
            // numpy's result precision (fft/ifft/irfft of a real float32/float16 first leaf; rfft(float16)).
            //   useSingle : run the single engine (no rounding — it already produces float values).
            //   roundTo   : else, round the double result — complex output -> Single (complex64 components);
            //               irfft/hfft real output -> Single (float32), or Half when the input is float16.
            bool useSingle = floatPrec
                && !effNormUnity                                   // ortho/forward (float fct) -> single loop; backward/None (int fct=1) -> double
                && (a.typecode == NPTypeCode.Complex               // fft/ifft/irfft on a complex64-precision operand
                    || (isReal && isForward && a.typecode == NPTypeCode.Single)); // rfft(float32) ff->F loop
            NPTypeCode? roundTo = null;
            if (floatPrec && !useSingle)
                roundTo = (isReal && !isForward && a.typecode == NPTypeCode.Half) ? NPTypeCode.Half : NPTypeCode.Single;

            if (!isReal)
            {
                if (useSingle) RunC2CF(src, result, ax, ndim, n, nin, inAxisStride, outAxisStride, inStrides, outStrides, inBase0, outBase0, lanes, isForward, (float)fct);
                else RunC2C(src, result, ax, ndim, n, nin, inAxisStride, outAxisStride, inStrides, outStrides, inBase0, outBase0, lanes, isForward, fct);
            }
            else if (isForward)
            {
                if (useSingle) RunR2CF(src, result, ax, ndim, n, nin, inAxisStride, outAxisStride, inStrides, outStrides, inBase0, outBase0, lanes, (float)fct);
                else RunR2C(src, result, ax, ndim, n, nin, inAxisStride, outAxisStride, inStrides, outStrides, inBase0, outBase0, lanes, fct);
            }
            else
            {
                if (useSingle) RunC2RF(src, result, ax, ndim, n, nin, inAxisStride, outAxisStride, inStrides, outStrides, inBase0, outBase0, lanes, (float)fct);
                else RunC2R(src, result, ax, ndim, n, nin, inAxisStride, outAxisStride, inStrides, outStrides, inBase0, outBase0, lanes, fct);
            }

            if (roundTo.HasValue)
                RoundInPlace(result, roundTo.Value == NPTypeCode.Half);

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

        // =====================================================================================
        // SINGLE-PRECISION lane runners — numpy's single fft loops (Ff->F / ff->F / Ff->f). These are
        // reached when the operand already matches a single loop WITHOUT a real->complex128 promotion:
        //   * fft/ifft/irfft of a COMPLEX64-precision value (a complex operand under floatPrec — the
        //     N-D intermediates: numpy's fft(complex64) is single, NOT round(double)); and
        //   * rfft of a float32 REAL input (numpy's ff->F loop).
        // (fft/ifft/irfft of a float32/float16 REAL input instead promote real->complex128 and run
        // DOUBLE + round — RoundInPlace — because numpy's own loop selection does; likewise rfft(float16).)
        // The operand is still the double/complex128 up-cast (lossless), narrowed to float on read; the
        // transform + fct run in the single engine; each result is up-cast float->double on write, so the
        // stored bytes are numpy's complex64/float32 result up-cast into complex128/float64.
        // =====================================================================================

        private static void RunC2CF(NDArray src, NDArray result, int ax, int ndim, long n, long nin,
            long inAxisStride, long outAxisStride, long[] inStrides, long[] outStrides,
            long inBase0, long outBase0, long lanes, bool fwd, float fct)
        {
            var plan = Fourier.PocketFFTPlanCacheF.GetComplex(n);
            Complex* ipAddr = (Complex*)src.Address;
            Complex* opAddr = (Complex*)result.Address;
            var bufArr = new CmplxF[n];
            var idx = new long[ndim];
            long inBase = inBase0, outBase = outBase0;
            long ncopy = nin <= n ? nin : n;
            fixed (CmplxF* buf = bufArr)
            {
                for (long lane = 0; lane < lanes; lane++)
                {
                    for (long k = 0; k < ncopy; k++)
                    {
                        Complex v = ipAddr[inBase + k * inAxisStride];
                        buf[k] = new CmplxF((float)v.Real, (float)v.Imaginary);
                    }
                    for (long k = ncopy; k < n; k++) buf[k] = new CmplxF(0f, 0f);

                    plan.Exec(buf, fct, fwd);

                    for (long k = 0; k < n; k++)
                    {
                        CmplxF v = buf[k];
                        opAddr[outBase + k * outAxisStride] = new Complex(v.r, v.i);
                    }
                    if (lane + 1 < lanes) Advance(idx, ndim, ax, src, inStrides, outStrides, ref inBase, ref outBase);
                }
            }
        }

        private static void RunR2CF(NDArray src, NDArray result, int ax, int ndim, long npts, long nin,
            long inAxisStride, long outAxisStride, long[] inStrides, long[] outStrides,
            long inBase0, long outBase0, long lanes, float fct)
        {
            var plan = Fourier.PocketFFTPlanCacheF.GetReal(npts);
            double* ipAddr = (double*)src.Address;
            Complex* opAddr = (Complex*)result.Address;
            var bufArr = new float[npts];
            var idx = new long[ndim];
            long inBase = inBase0, outBase = outBase0;
            long ncopy = nin <= npts ? nin : npts;
            long half = (npts - 1) / 2;
            fixed (float* buf = bufArr)
            {
                for (long lane = 0; lane < lanes; lane++)
                {
                    for (long k = 0; k < ncopy; k++) buf[k] = (float)ipAddr[inBase + k * inAxisStride];
                    for (long k = ncopy; k < npts; k++) buf[k] = 0f;

                    plan.Exec(buf, fct, true);

                    // pack FFTPACK half-complex R0,R1,I1,... into complex128 (float->double on write)
                    opAddr[outBase] = new Complex(buf[0], 0.0);
                    for (long kk = 1; kk <= half; kk++)
                        opAddr[outBase + kk * outAxisStride] = new Complex(buf[2 * kk - 1], buf[2 * kk]);
                    if ((npts & 1) == 0)
                        opAddr[outBase + (npts / 2) * outAxisStride] = new Complex(buf[npts - 1], 0.0);

                    if (lane + 1 < lanes) Advance(idx, ndim, ax, src, inStrides, outStrides, ref inBase, ref outBase);
                }
            }
        }

        private static void RunC2RF(NDArray src, NDArray result, int ax, int ndim, long nout, long nin,
            long inAxisStride, long outAxisStride, long[] inStrides, long[] outStrides,
            long inBase0, long outBase0, long lanes, float fct)
        {
            var plan = Fourier.PocketFFTPlanCacheF.GetReal(nout);
            Complex* ipAddr = (Complex*)src.Address;
            double* opAddr = (double*)result.Address;
            var bufArr = new float[nout];
            var idx = new long[ndim];
            long inBase = inBase0, outBase = outBase0;
            long half = (nout - 1) / 2;
            long ncopy = nin - 1; if (ncopy > half) ncopy = half; if (ncopy < 0) ncopy = 0;
            fixed (float* buf = bufArr)
            {
                for (long lane = 0; lane < lanes; lane++)
                {
                    // build FFTPACK half-complex buffer from the complex half-spectrum (double->float on read)
                    buf[0] = nin >= 1 ? (float)ipAddr[inBase].Real : 0f;
                    if (nout > 1)
                    {
                        for (long kk = 1; kk <= ncopy; kk++)
                        {
                            Complex v = ipAddr[inBase + kk * inAxisStride];
                            buf[2 * kk - 1] = (float)v.Real;
                            buf[2 * kk] = (float)v.Imaginary;
                        }
                        for (long kk = ncopy + 1; kk <= half; kk++)
                        {
                            buf[2 * kk - 1] = 0f;
                            buf[2 * kk] = 0f;
                        }
                        if ((nout & 1) == 0)
                            buf[nout - 1] = (nout / 2 >= nin) ? 0f : (float)ipAddr[inBase + (nout / 2) * inAxisStride].Real;
                    }

                    plan.Exec(buf, fct, false);

                    for (long k = 0; k < nout; k++)
                        opAddr[outBase + k * outAxisStride] = buf[k];   // float -> double on write

                    if (lane + 1 < lanes) Advance(idx, ndim, ax, src, inStrides, outStrides, ref inBase, ref outBase);
                }
            }
        }

        // Round every element of a freshly-computed DOUBLE result down to numpy's float32/float16 result
        // precision, stored back as double/complex128 — numpy computes float32/float16 fft in double and
        // casts the OUTPUT to complex64/float32/float16. `half`: round reals to float16 (irfft/hfft of a
        // float16 real input); otherwise float32 (all complex outputs, and float32 real outputs). Walks the
        // result through its own offset/strides (contiguous fast path + a full odometer for a strided out=).
        private static void RoundInPlace(NDArray result, bool half)
        {
            long size = result.size;
            if (size == 0) return;
            int ndim = result.ndim;
            long[] dims = result.Shape.dimensions;
            long[] strides = result.Shape.strides;
            long off = result.Shape.offset;
            bool contig = result.Shape.IsContiguous;

            if (result.typecode == NPTypeCode.Complex)
            {
                Complex* p = (Complex*)result.Address;
                if (contig)
                    for (long i = 0; i < size; i++)
                    {
                        Complex z = p[off + i];
                        p[off + i] = half ? new Complex((double)(Half)z.Real, (double)(Half)z.Imaginary)
                                          : new Complex((double)(float)z.Real, (double)(float)z.Imaginary);
                    }
                else
                {
                    var idx = new long[ndim];
                    long b = off;
                    for (long i = 0; i < size; i++)
                    {
                        Complex z = p[b];
                        p[b] = half ? new Complex((double)(Half)z.Real, (double)(Half)z.Imaginary)
                                    : new Complex((double)(float)z.Real, (double)(float)z.Imaginary);
                        b = NextOffset(idx, dims, strides, ndim, b);
                    }
                }
            }
            else // Double (irfft/hfft real output)
            {
                double* p = (double*)result.Address;
                if (contig)
                    for (long i = 0; i < size; i++)
                        p[off + i] = half ? (double)(Half)p[off + i] : (double)(float)p[off + i];
                else
                {
                    var idx = new long[ndim];
                    long b = off;
                    for (long i = 0; i < size; i++)
                    {
                        p[b] = half ? (double)(Half)p[b] : (double)(float)p[b];
                        b = NextOffset(idx, dims, strides, ndim, b);
                    }
                }
            }
        }

        // Advance a C-order odometer over all dims (rightmost fastest), returning the next base offset.
        private static long NextOffset(long[] idx, long[] dims, long[] strides, int ndim, long cur)
        {
            for (int d = ndim - 1; d >= 0; d--)
            {
                idx[d]++;
                cur += strides[d];
                if (idx[d] < dims[d]) return cur;
                idx[d] = 0;
                cur -= dims[d] * strides[d];
            }
            return cur; // past the last element; value unused
        }
    }
}
