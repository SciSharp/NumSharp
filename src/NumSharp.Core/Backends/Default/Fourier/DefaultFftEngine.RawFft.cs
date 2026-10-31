using System;

namespace NumSharp.Backends
{
    public partial class DefaultFftEngine
    {
        // =====================================================================================
        // Layer-3 port of numpy/fft/_pocketfft.py: _raw_fft, _raw_fftnd, _cook_nd_args,
        // _swap_direction, plus the axis/shape helpers (normalize_axis_index and the a.shape[axis]
        // tuple-index used before _raw_fft). NumPy 2.4.2 is the source of truth. Until phase PE's E3
        // these were FourierModule's private statements (Fourier/np.fft.RawFft.cs); _raw_fft now ends
        // in this engine's c2c / r2c / c2r kernels instead of calling PocketFFTDriver itself.
        //
        // DTYPE POLICY (NumSharp has ONLY complex128 — no complex64): float64/complex128/int/bool
        // inputs -> complex128 output for the forward/complex transforms; irfft/hfft -> float64 real
        // output. float32/float16 promote to complex128 (compute-in-double) — a deliberate
        // [Misaligned] divergence from NumPy 2.x's native complex64 path (it matches NumPy's own
        // still-published "numpy.fft promotes float32 ... to ... complex128" docstring). The dtype is
        // resolved by the kernel when it allocates the output; these statements resolve the output
        // SHAPE (n_out along axis) for the out= validation.
        // =====================================================================================

        /// <summary>
        ///     Port of <c>_swap_direction</c>. Maps a norm to its opposite direction
        ///     (<c>backward/None ↔ forward</c>, <c>ortho ↔ ortho</c>); an invalid value raises
        ///     NumPy's KeyError-path message (note the space after the first comma — this is the
        ///     message the INVERSE transforms and <c>hfft/ihfft</c> surface for a bad norm).
        /// </summary>
        /// <param name="norm">The caller's normalization mode: null, <c>"backward"</c>, <c>"ortho"</c> or
        ///     <c>"forward"</c>.</param>
        /// <returns>The mode of the opposite direction: <c>"forward"</c> for null / <c>"backward"</c>,
        ///     <c>"ortho"</c> for <c>"ortho"</c>, <c>"backward"</c> for <c>"forward"</c>.</returns>
        /// <exception cref="ValueError">Any other value: <c>Invalid norm value {norm}; should be "backward", "ortho" or
        ///     "forward".</c> (with a space after the first comma, unlike the forward transforms' text).</exception>
        internal static string SwapDirection(string norm)
        {
            if (norm == null || norm == "backward") return "forward";
            if (norm == "ortho") return "ortho";
            if (norm == "forward") return "backward";
            // KeyError path -> ValueError WITH a space after "backward,".
            throw new ValueError($"Invalid norm value {norm}; should be \"backward\", \"ortho\" or \"forward\".");
        }

        /// <summary>
        ///     Port of <c>numpy.lib.array_utils.normalize_axis_index</c> as used inside <c>_raw_fft</c>.
        ///     Wraps a negative axis and range-checks it, raising the house <see cref="AxisError"/>
        ///     (NumPy's <c>AxisError</c>) reporting the ORIGINAL axis. This is the axis error surfaced
        ///     when <c>n</c> is given (when <c>n</c> is <c>None</c> the public method's <c>a.shape[axis]</c>
        ///     raises the IndexError first — see <see cref="ShapeAt"/>).
        /// </summary>
        /// <param name="axis">The caller's axis; negative counts from the end.</param>
        /// <param name="ndim">The operand's number of dimensions.</param>
        /// <returns>The axis in <c>[0, ndim)</c>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> outside <c>[-ndim, ndim)</c> (a 0-d operand has no
        ///     axis).</exception>
        internal static int NormalizeAxisIndex(int axis, int ndim)
        {
            int ax = axis;
            if (ax < 0) ax += ndim;
            if (ax < 0 || ax >= ndim)
                throw new AxisError(axis, ndim);
            return ax;
        }

        /// <summary>
        ///     Python <c>a.shape[axis]</c> (tuple indexing with negative wrap): the length along
        ///     <paramref name="axis"/>. Used by the public 1-D methods to default <c>n</c> when it is
        ///     <c>None</c>; an out-of-range axis raises the IndexError NumPy leaks there
        ///     (<c>"tuple index out of range"</c>), NOT an AxisError.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="axis">The axis; negative counts from the end.</param>
        /// <returns>The operand's length along the axis.</returns>
        /// <exception cref="IndexError"><paramref name="axis"/> out of range: <c>tuple index out of range</c>.</exception>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null (as before the move: the facades pass it
        ///     on unchecked).</exception>
        internal static int ShapeAt(NDArray a, int axis)
        {
            int nd = a.ndim;
            int ax = axis;
            if (ax < 0) ax += nd;
            if (ax < 0 || ax >= nd)
                throw new IndexError("tuple index out of range");
            return (int)a.Shape.dimensions[ax];
        }

        /// <summary>
        ///     NumPy <c>np.take(a.shape, axis, mode='raise')</c> as used by <c>_cook_nd_args</c> when
        ///     <c>s</c> is <c>None</c> but <c>axes</c> is given. Reports the ORIGINAL axis with take's
        ///     message (<c>"index {i} is out of bounds for axis 0 with size {ndim}"</c>) — distinct from
        ///     <see cref="ShapeAt"/>'s "tuple index out of range".
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="axis">The axis; negative counts from the end.</param>
        /// <returns>The operand's length along the axis.</returns>
        /// <exception cref="IndexError">A 0-d operand (<c>cannot do a non-empty take from an empty axes.</c>), or an axis
        ///     out of range (take's text, naming the original axis).</exception>
        internal static int TakeShape(NDArray a, int axis)
        {
            int nd = a.ndim;
            if (nd == 0)
                // np.take from a 0-d shape (fft2/rfft2 of a scalar): NumPy's take raises this, distinct
                // from the per-axis out-of-bounds message below (which presumes a non-empty shape).
                throw new IndexError("cannot do a non-empty take from an empty axes.");
            int ax = axis;
            if (ax < 0) ax += nd;
            if (ax < 0 || ax >= nd)
                throw new IndexError($"index {axis} is out of bounds for axis 0 with size {nd}");
            return (int)a.Shape.dimensions[ax];
        }

        /// <summary>
        ///     Port of <c>_raw_fft</c> (numpy 2.4.2). Validates <c>n</c>, resolves the normalization
        ///     factor <c>fct</c> (inverse swaps the norm first) in the input's <c>real_dtype</c> (whose
        ///     promotion refuses text), resolves the output length along the axis (<c>rfft</c> →
        ///     <c>n//2+1</c>; others → <c>n</c>), normalizes the axis, validates a provided
        ///     <paramref name="out"/>, then hands the transform to the kernel that stands for NumPy's
        ///     <c>_pocketfft_umath</c> ufunc of its kind (<see cref="C2CKernel"/>, <see cref="R2CKernel"/>,
        ///     <see cref="C2RKernel"/>).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="n">Transform length (already defaulted by the caller).</param>
        /// <param name="axis">Transform axis (not yet normalized).</param>
        /// <param name="isReal"><c>true</c> for the rfft/irfft family (real↔complex-half).</param>
        /// <param name="isForward"><c>true</c> for the forward transforms (fft/rfft).</param>
        /// <param name="norm">One of <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated output; its shape is validated.</param>
        /// <param name="name">The public API name (e.g. "fft"), kept for the callers' diagnostics; the statements do not
        ///     read it.</param>
        /// <param name="floatPrec">
        ///     The ORIGINAL user input is float32/float16, so numpy 2.x returns complex64/float32/float16.
        ///     NumSharp has no complex64 (issue #569), so the result dtype stays complex128/float64 but the
        ///     VALUES are made bit-identical to numpy's. numpy computes almost all of these in DOUBLE and
        ///     ROUNDS to complex64/float32/float16 — with ONE exception: <c>rfft</c> of a float32 input,
        ///     which numpy runs through its single-precision <c>ff-&gt;F</c> loop (see
        ///     <see cref="PocketFFTDriver"/>). The normalisation factor is computed in the input's
        ///     <c>real_dtype</c> (float32/float16), matching numpy's <c>reciprocal(sqrt(n, dtype=...))</c>.
        ///     Threaded through the N-D compositions so every 1-D leaf reproduces its own numpy loop.
        /// </param>
        /// <returns>The transform: complex128 for the complex and forward outputs, float64 for <c>irfft</c>
        ///     (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ValueError"><c>n &lt; 1</c>, an unknown <paramref name="norm"/>, or an
        ///     <paramref name="out"/> of the wrong shape or read-only.</exception>
        /// <exception cref="DTypePromotionError">A text input (<c>S</c>, <c>U</c> or <c>StringDType</c>):
        ///     <c>result_type(a.real.dtype, 1.0)</c> has no common DType (NumPy's text).</exception>
        /// <exception cref="TypeError">A complex input to <c>rfft</c> (no complex loop).</exception>
        /// <exception cref="AxisError">An axis outside the input (the caller's <c>a.shape[axis]</c> lookup reports
        ///     NumPy's IndexError first when <c>n</c> was omitted).</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> whose dtype the kernel's result cannot be cast to
        ///     under same_kind (the kernel's refusal, NumPy's UFuncTypeError text).</exception>
        internal NDArray RawFft(NDArray a, int n, int axis, bool isReal, bool isForward, string norm, NDArray @out, string name, bool floatPrec = false)
        {
            if (n < 1)
                throw new ValueError($"Invalid number of FFT data points ({n}) specified.");

            // Inverse transforms swap the norm direction first (may raise the WITH-space message).
            string effNorm = norm;
            if (!isForward)
                effNorm = SwapDirection(norm);

            // NumPy's next statement is real_dtype = result_type(a.real.dtype, 1.0): text (S, U or StringDType: IsText never
            // throws) has no common DType with the weak float, so a text input raises DTypePromotionError with NumPy's text,
            // thrown by the promotion itself (`a.real` of a non-complex array is the array).
            if (Backends.Strings.NDStringUfuncs.IsText(a))
            {
                DTypePromotion.ResultType(a.dtype, 1.0);
                // No string class promotes with the abstract Python float, so the promotion above always raises.
                throw new InvalidOperationException($"result_type({a.dtype.ToString(true)}, float) returned for the FFT of text.");
            }

            // fct = reciprocal(sqrt/n) computed in the input's real_dtype (numpy's _raw_fft:
            // real_dtype = result_type(a.real.dtype, 1.0)). For a float32 input (or a complex64-precision
            // N-D intermediate — a complex operand under floatPrec) this is float32; for a float16 input
            // it is float16; otherwise double. The value is then carried as a double (upcast) — numpy
            // itself upcasts the fct when the transform runs in double.
            int realKind; // 0 double, 1 float32, 2 float16
            if (floatPrec && a.typecode == NPTypeCode.Half) realKind = 2;
            else if (floatPrec && (a.typecode == NPTypeCode.Single || a.typecode == NPTypeCode.Complex)) realKind = 1;
            else realKind = 0;

            double fct;
            if (effNorm == null || effNorm == "backward")
                fct = 1.0;
            else if (effNorm == "ortho")
                fct = FftFct(n, realKind, ortho: true);
            else if (effNorm == "forward")
                fct = FftFct(n, realKind, ortho: false);
            else
                // Only reachable for a FORWARD transform with a bad norm (the inverse path already
                // threw in SwapDirection above). NumPy's forward message has NO space after "backward,".
                throw new ValueError($"Invalid norm value {norm}; should be \"backward\",\"ortho\" or \"forward\".");

            int nOut = n;
            if (isReal && isForward)
                nOut = n / 2 + 1;   // rfft output length (n//2 + 1); irfft/fft/ifft keep n_out = n.

            axis = NormalizeAxisIndex(axis, a.ndim);

            // NB: `@out is not null` — NDArray overloads `!=` element-wise (would yield NDArray<bool>).
            if (@out is not null)
            {
                // NumPy: raise if len(out.shape) != a.ndim or out.shape[axis] != n_out.
                if (@out.ndim != a.ndim || @out.Shape.dimensions[axis] != nOut)
                    throw new ValueError("output array has wrong shape.");

                // A read-only out refuses AFTER the shape check (probed 2.4.2: a wrong-shaped
                // read-only out reports the shape error — _raw_fft's Python-level check fires
                // before the ufunc's writeable validation, whose "output array is read-only" text
                // this reproduces through the house guard, same as the elementwise ufunc out=
                // path). Without it the kernel writes @out's buffer directly and a non-writeable
                // target (a setflags(write=False) array, a broadcast view, an 'r' memmap) is
                // silently corrupted. One guard here covers the whole family — fft/ifft/rfft/
                // irfft and the hfft/ihfft compositions all funnel through RawFft.
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            }

            // rfft (real-forward) has NO complex loop: NumPy's rfft_n_even/rfft_n_odd ufunc refuses a
            // complex input rather than silently drop its imaginary part. Fire here — after n<1, norm,
            // axis and out have all been validated — matching NumPy's ordering (the type error is raised
            // at the ufunc call). The message names the parity-selected ufunc exactly as NumPy does.
            if (isReal && isForward && a.typecode == NPTypeCode.Complex)
                throw new TypeError(
                    $"ufunc '{(n % 2 == 0 ? "rfft_n_even" : "rfft_n_odd")}' not supported for the input types, " +
                    "and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");

            // NumPy's last statement is the _pocketfft_umath ufunc of the transform's kind; here it is
            // that kind's kernel (by default the ported pocketfft engine, PocketFFTDriver.cs). The kernel
            // resolves the output (a.shape with axis replaced by nOut, dtype per the policy above —
            // complex128 for the complex/forward outputs, float64 for irfft), runs the strided
            // all-but-axis 1-D transform, and scales by fct. `name` is retained for callers' diagnostics.
            // numpy's fft ufunc picks the SINGLE loop only when the fct operand is a float — i.e. for
            // ortho/forward norms (fct = reciprocal(...) in real_dtype). For backward/None the Python
            // fct is the INT 1, which nudges numpy's loop resolution to the DOUBLE loop, so even a
            // single-eligible operand (complex64 / rfft-float32) is computed in double + rounded.
            bool effNormUnity = effNorm == null || effNorm == "backward";
            if (!isReal)
                return C2CKernel(a, n, axis, isForward, fct, @out, floatPrec, effNormUnity);
            return isForward
                ? R2CKernel(a, n, axis, fct, @out, floatPrec, effNormUnity)
                : C2RKernel(a, n, axis, fct, @out, floatPrec, effNormUnity);
        }

        /// <summary>
        ///     numpy's <c>reciprocal(sqrt(n, dtype=real_dtype))</c> (ortho) / <c>reciprocal(n, real_dtype)</c>
        ///     (forward), computed in the operand's real_dtype then upcast to double. float16 sqrt/reciprocal go via
        ///     float32 exactly as numpy's half ufuncs.
        /// </summary>
        /// <param name="n">The transform length.</param>
        /// <param name="realKind">The real_dtype: 0 = double, 1 = float32, 2 = float16.</param>
        /// <param name="ortho">True for <c>"ortho"</c> (<c>1/sqrt(n)</c>), false for <c>"forward"</c> (<c>1/n</c>).</param>
        /// <returns>The factor, rounded to the real_dtype and carried as a double.</returns>
        private static double FftFct(int n, int realKind, bool ortho)
        {
            switch (realKind)
            {
                case 1: // float32
                    return ortho ? (double)(1f / MathF.Sqrt((float)n)) : (double)(1f / (float)n);
                case 2: // float16 (numpy half ops round through float32)
                    return ortho
                        ? (double)(Half)(1f / (float)((Half)MathF.Sqrt((float)(Half)n)))
                        : (double)(Half)(1f / (float)(Half)n);
                default: // double
                    return ortho ? 1.0 / Math.Sqrt(n) : 1.0 / n;
            }
        }

        /// <summary>
        ///     True when <paramref name="a"/> is a float32/float16 array — the inputs numpy 2.x returns as
        ///     complex64/float32/float16. NumSharp keeps the result dtype complex128/float64 (no complex64 —
        ///     issue #569) but reproduces numpy's VALUES: it computes in double and rounds to the numpy
        ///     result precision, except <c>rfft(float32)</c> which numpy runs single-precision. Complex128,
        ///     float64, integer and bool inputs are contractual double-precision (return false).
        /// </summary>
        /// <param name="a">The operand of the public call.</param>
        /// <returns>True for a float32 or float16 operand.</returns>
        internal static bool IsSinglePrecision(NDArray a)
            => a.typecode == NPTypeCode.Single || a.typecode == NPTypeCode.Half;

        /// <summary>
        ///     Port of <c>_cook_nd_args</c> (numpy 2.4.2). Resolves the per-axis lengths <c>s</c> and the
        ///     <c>axes</c> for the N-D transforms, reproducing: the <c>s</c>-without-<c>axes</c>
        ///     default-to-last-axes behaviour, the length-mismatch <c>ValueError</c>, the <c>-1</c>
        ///     sentinel (use the full input length), and — for <c>irfftn</c> (<paramref name="invreal"/>)
        ///     — the shapeless last-axis default of <c>2*(m-1)</c>. The NumPy 2.0 deprecation warnings
        ///     (s-without-axes, None-in-s) are intentionally not modelled (NumSharp does not surface
        ///     Python warnings).
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="s">The caller's lengths, or null for the operand's shape along the axes.</param>
        /// <param name="axes">The caller's axes, or null for all axes (or the last <c>len(s)</c>).</param>
        /// <param name="invreal">True for irfftn / irfft2, whose shapeless last length is <c>2 * (m - 1)</c>.</param>
        /// <returns>The resolved lengths (a fresh array; <c>-1</c> replaced) and axes (the caller's array when
        ///     given).</returns>
        /// <exception cref="ValueError"><paramref name="s"/> and <paramref name="axes"/> of different lengths.</exception>
        /// <exception cref="IndexError">An axis out of range (take's or the tuple index's text), or no axis for the
        ///     inverse real transform's default (<c>tuple</c> / <c>list index out of range</c>).</exception>
        internal static (int[] s, int[] axes) CookNdArgs(NDArray a, int[] s, int[] axes, bool invreal)
        {
            bool shapeless;
            int[] sList;
            if (s == null)
            {
                shapeless = true;
                if (axes == null)
                {
                    // s = list(a.shape). NDArray.shape is long[]; FFT lengths are int-range.
                    long[] shp = a.Shape.dimensions;
                    sList = new int[shp.Length];
                    for (int i = 0; i < shp.Length; i++)
                        sList[i] = (int)shp[i];
                }
                else
                {
                    // s = take(a.shape, axes) — np.take mode='raise' error taxonomy.
                    sList = new int[axes.Length];
                    for (int i = 0; i < axes.Length; i++)
                        sList[i] = TakeShape(a, axes[i]);
                }
            }
            else
            {
                shapeless = false;
                sList = (int[])s.Clone();
            }

            int[] axesList;
            if (axes == null)
            {
                // axes = range(-len(s), 0)  (the "last len(s) axes" default).
                axesList = new int[sList.Length];
                for (int i = 0; i < sList.Length; i++)
                    axesList[i] = -sList.Length + i;
            }
            else
            {
                axesList = axes;
            }

            if (sList.Length != axesList.Length)
                throw new ValueError("Shape and axes have different lengths.");

            if (invreal && shapeless)
            {
                // irfftn/irfft2 with no axis to transform: NumPy evaluates `s[-1] = (a.shape[axes[-1]] - 1) * 2`
                // right side first, so `axes[-1]` raises — on the caller's TUPLE when axes was given (an explicit
                // empty `axes=()`: "tuple index out of range"), on the LIST `range(-len(s), 0)` built above when it
                // was not (a 0-d input: "list index out of range"). Reproduce it before the subscript below throws a
                // raw IndexOutOfRangeException.
                if (axesList.Length == 0)
                    throw new IndexError(axes != null ? "tuple index out of range" : "list index out of range");
                sList[sList.Length - 1] = (ShapeAt(a, axesList[axesList.Length - 1]) - 1) * 2;
            }

            // Resolve the -1 sentinel to the full input length along that axis (a.shape[axes[i]]).
            for (int i = 0; i < sList.Length; i++)
                if (sList[i] == -1)
                    sList[i] = ShapeAt(a, axesList[i]);

            return (sList, axesList);
        }

        /// <summary>
        ///     Port of <c>_raw_fftnd</c> (numpy 2.4.2): decompose an N-D transform into a sequence of
        ///     1-D transforms, applied over the axes in REVERSE order. Over invalid <c>s</c>/<c>axes</c> it surfaces
        ///     <see cref="CookNdArgs"/>'s validation error before any transform runs — matching NumPy's ordering.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <param name="s">The caller's lengths, or null.</param>
        /// <param name="axes">The caller's axes, or null.</param>
        /// <param name="function">The 1-D transform of each axis: <c>(operand, n, axis, norm, out)</c> to its result
        ///     (the callers pass this engine's <see cref="RawFft"/> with the original operand's precision, where NumPy
        ///     passes its module function).</param>
        /// <param name="norm">The normalization mode handed to every transform.</param>
        /// <param name="out">The output handed to every transform, or null.</param>
        /// <returns>The last transform's result.</returns>
        /// <exception cref="ValueError"><see cref="CookNdArgs"/>'s length mismatch, or the 1-D transforms'
        ///     refusals.</exception>
        /// <exception cref="IndexError">An axis out of range in <see cref="CookNdArgs"/>.</exception>
        internal static NDArray RawFftNd(NDArray a, int[] s, int[] axes,
            Func<NDArray, int, int, string, NDArray, NDArray> function, string norm, NDArray @out)
        {
            var (ss, aa) = CookNdArgs(a, s, axes, invreal: false);
            for (int ii = aa.Length - 1; ii >= 0; ii--)
                a = function(a, ss[ii], aa[ii], norm, @out);
            return a;
        }
    }
}
