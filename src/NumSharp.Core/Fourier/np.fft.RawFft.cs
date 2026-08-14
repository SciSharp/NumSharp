using System;

namespace NumSharp
{
    public partial class FourierModule
    {
        // =====================================================================================
        // Layer-3 port of numpy/fft/_pocketfft.py: _raw_fft, _raw_fftnd, _cook_nd_args,
        // _swap_direction, plus the axis/shape helpers (normalize_axis_index and the a.shape[axis]
        // tuple-index used before _raw_fft). NumPy 2.4.2 is the source of truth.
        //
        // DTYPE POLICY (NumSharp has ONLY complex128 — no complex64): float64/complex128/int/bool
        // inputs -> complex128 output for the forward/complex transforms; irfft/hfft -> float64 real
        // output. float32/float16 promote to complex128 (compute-in-double) — a deliberate
        // [Misaligned] divergence from NumPy 2.x's native complex64 path (it matches NumPy's own
        // still-published "numpy.fft promotes float32 ... to ... complex128" docstring). The dtype is
        // resolved by the engine seam when it allocates the output; the shell resolves the output
        // SHAPE (n_out along axis) for the out= validation.
        // =====================================================================================

        /// <summary>
        ///     Port of <c>_swap_direction</c>. Maps a norm to its opposite direction
        ///     (<c>backward/None ↔ forward</c>, <c>ortho ↔ ortho</c>); an invalid value raises
        ///     NumPy's KeyError-path message (note the space after the first comma — this is the
        ///     message the INVERSE transforms and <c>hfft/ihfft</c> surface for a bad norm).
        /// </summary>
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
        internal static int ShapeAt(NDArray a, int axis)
        {
            int nd = a.ndim;
            int ax = axis;
            if (ax < 0) ax += nd;
            if (ax < 0 || ax >= nd)
                throw new IndexError("tuple index out of range");
            return (int)a.shape[ax];
        }

        /// <summary>
        ///     NumPy <c>np.take(a.shape, axis, mode='raise')</c> as used by <c>_cook_nd_args</c> when
        ///     <c>s</c> is <c>None</c> but <c>axes</c> is given. Reports the ORIGINAL axis with take's
        ///     message (<c>"index {i} is out of bounds for axis 0 with size {ndim}"</c>) — distinct from
        ///     <see cref="ShapeAt"/>'s "tuple index out of range".
        /// </summary>
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
            return (int)a.shape[ax];
        }

        /// <summary>
        ///     Full port of <c>_raw_fft</c>'s front-matter (numpy 2.4.2). Validates <c>n</c>, resolves
        ///     the normalization factor <c>fct</c> (inverse swaps the norm first), resolves the output
        ///     length along the axis (<c>rfft</c> → <c>n//2+1</c>; others → <c>n</c>), normalizes the
        ///     axis and validates a provided <paramref name="out"/> shape — THEN reaches the compute
        ///     seam and throws (the transform kernel is a separate agent's work).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="n">Transform length (already defaulted by the caller).</param>
        /// <param name="axis">Transform axis (not yet normalized).</param>
        /// <param name="isReal"><c>true</c> for the rfft/irfft family (real↔complex-half).</param>
        /// <param name="isForward"><c>true</c> for the forward transforms (fft/rfft).</param>
        /// <param name="norm">One of <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated output; its shape is validated.</param>
        /// <param name="name">The public API name for the seam message (e.g. "fft").</param>
        internal static NDArray RawFft(NDArray a, int n, int axis, bool isReal, bool isForward, string norm, NDArray @out, string name)
        {
            if (n < 1)
                throw new ValueError($"Invalid number of FFT data points ({n}) specified.");

            // Inverse transforms swap the norm direction first (may raise the WITH-space message).
            string effNorm = norm;
            if (!isForward)
                effNorm = SwapDirection(norm);

            // fct: backward/None -> 1 ; ortho -> 1/sqrt(n) ; forward -> 1/n. Computed in double
            // (the complex128 path's real precision).
            double fct;
            if (effNorm == null || effNorm == "backward")
                fct = 1.0;
            else if (effNorm == "ortho")
                fct = 1.0 / Math.Sqrt(n);
            else if (effNorm == "forward")
                fct = 1.0 / n;
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
                if (@out.ndim != a.ndim || @out.shape[axis] != nOut)
                    throw new ValueError("output array has wrong shape.");
            }

            // rfft (real-forward) has NO complex loop: NumPy's rfft_n_even/rfft_n_odd ufunc refuses a
            // complex input rather than silently drop its imaginary part. Fire here — after n<1, norm,
            // axis and out have all been validated — matching NumPy's ordering (the type error is raised
            // at the ufunc call). The message names the parity-selected ufunc exactly as NumPy does.
            if (isReal && isForward && a.typecode == NPTypeCode.Complex)
                throw new TypeError(
                    $"ufunc '{(n % 2 == 0 ? "rfft_n_even" : "rfft_n_odd")}' not supported for the input types, " +
                    "and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");

            // Wired to the ported pocketfft engine (PocketFFTDriver.cs). The driver resolves the
            // output (a.shape with axis replaced by nOut, dtype per the policy above — complex128 for
            // the complex/forward outputs, float64 for irfft), runs the strided all-but-axis 1-D
            // transform, and scales by fct. `name` is retained for callers' diagnostics.
            return PocketFFTDriver.Execute(a, n, axis, isReal, isForward, fct, @out);
        }

        /// <summary>
        ///     Port of <c>_cook_nd_args</c> (numpy 2.4.2). Resolves the per-axis lengths <c>s</c> and the
        ///     <c>axes</c> for the N-D transforms, reproducing: the <c>s</c>-without-<c>axes</c>
        ///     default-to-last-axes behaviour, the length-mismatch <c>ValueError</c>, the <c>-1</c>
        ///     sentinel (use the full input length), and — for <c>irfftn</c> (<paramref name="invreal"/>)
        ///     — the shapeless last-axis default of <c>2*(m-1)</c>. The NumPy 2.0 deprecation warnings
        ///     (s-without-axes, None-in-s) are intentionally not modelled (NumSharp does not surface
        ///     Python warnings).
        /// </summary>
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
                    long[] shp = a.shape;
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
                // irfftn/irfft2 of a 0-d input (or explicit empty axes): NumPy indexes s[-1]/axes[-1] on
                // an empty list and leaks a plain IndexError. Reproduce it before the subscript below
                // throws a raw IndexOutOfRangeException.
                if (axesList.Length == 0)
                    throw new IndexError("list index out of range");
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
        ///     1-D transforms, applied over the axes in REVERSE order. Because each 1-D leaf currently
        ///     throws at the engine seam, an N-D call over valid input surfaces that
        ///     <see cref="NotImplementedException"/> (from the first leaf); over invalid <c>s</c>/<c>axes</c>
        ///     it surfaces <see cref="CookNdArgs"/>'s validation error first — matching NumPy's ordering.
        /// </summary>
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
