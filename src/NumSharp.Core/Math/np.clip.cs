using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.<br></br>
        ///     Matches NumPy 2.x signature: <c>clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None)</c>. Either or both bounds may be <c>null</c>. The <paramref name="min"/> and <paramref name="max"/> keyword aliases (added in NumPy 2.0) are accepted; mixing <paramref name="a_min"/> with <paramref name="min"/> (or <paramref name="a_max"/> with <paramref name="max"/>) throws.
        /// </summary>
        /// <param name="a">Array containing elements to clip.</param>
        /// <param name="a_min">Minimum value. If null, clipping is not performed on lower interval edge. A C# integer literal is NumPy's Python int: on an integer array a value at or below the dtype's minimum is ignored (it cannot clip anything), and a value that does not fit the dtype on the other side raises.</param>
        /// <param name="a_max">Maximum value. If null, clipping is not performed on upper interval edge. A C# integer literal at or above an integer array's dtype maximum is ignored; one below the dtype's minimum raises.</param>
        /// <param name="out">The results will be placed in this array, which is returned as-is. It may be the input array for in-place clipping. <paramref name="out"/> must have the input's shape, and its dtype must be reachable from the computed dtype by a same_kind cast (NumPy's ufunc rule; e.g. a uint8 clip may write into an int8 or a float64 out).</param>
        /// <param name="dtype">The dtype the computation runs in (NumPy's ufunc <c>dtype=</c>). The inputs are cast to it, and each must reach it by a same_kind cast.</param>
        /// <param name="min">NumPy 2.x keyword alias for <paramref name="a_min"/>. Cannot be combined with <paramref name="a_min"/>.</param>
        /// <param name="max">NumPy 2.x keyword alias for <paramref name="a_max"/>. Cannot be combined with <paramref name="a_max"/>.</param>
        /// <returns>An array with the elements of a, but where values &lt; min are replaced with min, and those &gt; max with max (<paramref name="out"/> when given).</returns>
        /// <exception cref="ArgumentException">Both <paramref name="a_min"/> and <paramref name="min"/> (or both maxima) are given; or a <paramref name="dtype"/> the inputs cannot reach, or an <paramref name="out"/> the result cannot be cast into, under the same_kind rule (NumPy's UFuncTypeError texts); or an <paramref name="out"/> of another shape.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only. NumPy checks this before anything else.</exception>
        /// <exception cref="OverflowException">An integer bound that is kept does not fit the integer dtype the computation runs in, for example <c>np.clip(int8_array, 1000, 2000)</c>: "Python integer 1000 out of bounds for int8" (NumPy's OverflowError). It is raised before any element is read, so an empty array raises too.</exception>
        /// <exception cref="TypeError">Neither bound is given (or both were ignored) and the computation would run in bool. NumPy then calls <c>np.positive</c>, which has no bool loop.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        ///     <para>
        ///     A port of NumPy's <c>numpy/_core/_methods.py::_clip</c>, the function behind both <c>np.clip</c> and
        ///     <c>ndarray.clip</c>. On an integer array it first drops a Python-int bound at or past the dtype's limit:
        ///     <c>np.clip(uint8_array, 0, 256)</c> returns the array unchanged, and <c>np.clip(uint8_array, -1, 100)</c>
        ///     only clips from above. It then routes the call: no bound left → <c>positive</c> (a copy), one bound →
        ///     <c>minimum</c> / <c>maximum</c>, two → the <c>clip</c> ufunc. The routed name is what the cast errors quote.
        ///     A kept integer bound is packed into the computed dtype under NEP 50, so it raises instead of wrapping.
        ///     The drop rule uses the input array's own range, even when <paramref name="dtype"/> changes the computation:
        ///     <c>np.clip(int8_array, 1000, 2000, dtype: np.int16)</c> is legal and returns all 1000.
        ///     </para>
        ///     <para>
        ///     NumSharp cannot tell a C# literal from a 0-d <see cref="NDArray"/> at this boundary, so every 0-d integer
        ///     bound counts as a Python int. NumPy promotes with a 0-d array or a NumPy scalar instead (<c>np.int64(1000)</c>
        ///     makes an int8 clip int64). The values agree; only that result dtype differs.
        ///     </para>
        ///     <para>
        ///     When <c>|value| &gt;= 2^31</c>, NumPy's overflow text depends on the platform's C <c>long</c>. NumSharp
        ///     uses the Linux/macOS (LP64) text: values that fit a 64-bit long say "Python integer ... out of bounds
        ///     for ...", larger ones say "Python int too large to convert to C long".
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray clip(
            NDArray a,
            NDArray a_min = null,
            NDArray a_max = null,
            NDArray @out = null,
            DType dtype = null,
            NDArray min = null,
            NDArray max = null)
        {
            if (a_min is not null && min is not null)
                throw new ArgumentException("clip(): cannot specify both 'a_min' and 'min'.");
            if (a_max is not null && max is not null)
                throw new ArgumentException("clip(): cannot specify both 'a_max' and 'max'.");

            // A read-only out refuses FIRST — probed 2.4.2: NumPy's clip ufunc reports
            // "output array is read-only" ahead of any broadcast/shape error, and even ahead of
            // the no-bounds identity path (clip(a, out=ro) with neither bound still raises).
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");

            var lo = a_min ?? min;
            var hi = a_max ?? max;
            var result = a.TensorEngine.ClipNDArray(a, lo, hi, dtype, @out);

            // A provided out is NumPy's return value itself (reference identity), whatever its layout:
            // relaying an F-contiguous input's result to F order is only for a fresh result. Applied to
            // a C-order out, it returned a copy, so the caller's out was filled but not returned.
            if (@out is not null)
                return result;
            return PreserveFContigFromSource(a, result);
        }

        // Internal helper: after an element-wise op whose output inherits a's layout,
        // relay out to F-contig when the source is strictly F-contig and the result
        // came back as C-contig (current engine default).
        private static NDArray PreserveFContigFromSource(NDArray a, NDArray result)
        {
            // Note: NDArray overloads operator!=, so reference-compare via ReferenceEquals.
            if (!ReferenceEquals(result, null)
                && a.Shape.NDim > 1 && a.size > 1
                && a.Shape.IsFContiguous && !a.Shape.IsContiguous
                && result.Shape.NDim > 1 && !result.Shape.IsFContiguous)
            {
                return result.copy('F');
            }
            return result;
        }
    }
}
