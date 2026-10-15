using System;
using System.Collections.Generic;
using NumSharp.Backends;

namespace NumSharp
{
    // ============================== np.ediff1d ==============================
    // The differences between consecutive elements of a (flattened) array.
    //
    //   ediff1d(ary) == ary.ravel()[1:] - ary.ravel()[:-1]
    //
    // NumPy 2.4.2 reference: numpy/lib/_arraysetops_impl.py::ediff1d
    //
    // Differences from np.diff:
    //   * Always operates on the C-order flattened (1-D) input.
    //   * The result is always 1-D.
    //   * The result dtype is forced to the input dtype; to_begin / to_end are
    //     cast to it under the NumPy `same_kind` rule (else a TypeError).
    //   * Unlike diff, ediff1d uses subtract for every dtype with NO not_equal
    //     special case, so a boolean input raises (matching NumPy, which rejects
    //     boolean subtraction).
    //
    // The subtract is backed by NumSharp's SIMD IL kernel (TensorEngine.Subtract);
    // ediff1d itself contains no per-element loop.
    public static partial class np
    {
        /// <summary>
        ///     The differences between consecutive elements of an array. The
        ///     input is flattened first; the result is always 1-D.
        /// </summary>
        /// <param name="ary">Input array (flattened before differencing).</param>
        /// <param name="to_end">
        ///     Number(s) to append to the end of the returned differences.
        ///     <c>null</c> means none. Cast to <paramref name="ary"/>'s dtype
        ///     under the <c>same_kind</c> casting rule.
        /// </param>
        /// <param name="to_begin">
        ///     Number(s) to prepend to the beginning of the returned differences.
        ///     <c>null</c> means none. Cast to <paramref name="ary"/>'s dtype
        ///     under the <c>same_kind</c> casting rule.
        /// </param>
        /// <returns>
        ///     1-D array of consecutive differences (input dtype), optionally
        ///     bracketed by <paramref name="to_begin"/> and <paramref name="to_end"/>.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="ary"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="to_begin"/> or <paramref name="to_end"/> cannot be
        ///     cast to <paramref name="ary"/>'s dtype under <c>same_kind</c>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="ary"/> is boolean (NumPy forbids boolean
        ///     subtraction).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ediff1d.html
        ///     <para>Scoped by <c>[NDScoped]</c>: the ravel, the prepared <paramref name="to_begin"/> /
        ///     <paramref name="to_end"/> copies, the slice views of the fallback path and the middle differences
        ///     that a concatenation supersedes are released on return, exceptions included, and the result is
        ///     yielded. The body no longer repeats that bookkeeping with a hand-written try/finally.</para>
        /// </remarks>
        [NDScoped]
        public static NDArray ediff1d(NDArray ary, object to_end = null, object to_begin = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(ary);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (ary is null) throw new ArgumentNullException(nameof(ary));

            // ravel() always returns a fresh wrapper (it may share the caller's storage);
            // it is released by this method's scope, which never frees the caller's buffer.
            var flat = np.ravel(ary);
            NPTypeCode dt = flat.GetTypeCode;

            NDArray begin = null, end = null, middle = null;

            // NumPy validates/casts to_begin and to_end before differencing.
            begin = EdiffPrepareEnd(to_begin, dt, "to_begin");
            end = EdiffPrepareEnd(to_end, dt, "to_end");

            // ediff1d differences with subtract for ALL dtypes; NumPy forbids
            // boolean subtraction (np.diff special-cases bool, ediff1d does not).
            if (dt == NPTypeCode.Boolean)
                throw new NotSupportedException(
                    "numpy boolean subtract, the `-` operator, is not supported, " +
                    "use the bitwise_xor, the `^` operator, or the logical_xor function instead.");

            long L = flat.size;
            // Fast path: the flattened input is always C-contiguous 1-D, so the middle
            // differences are the same adjacent-difference stencil np.diff uses — one
            // fused kernel over the source, no slice views, no NDIter. dt is non-boolean
            // here (boolean threw above). Falls back to the slice-subtract for L<2 or an
            // unavailable dtype kernel.
            if (L >= 2)
                middle = DiffAdjacentContiguous(flat, 0);
            if (middle is null)
            {
                long m = L > 0 ? L - 1 : 0;
                var hi = SliceAlongAxis(flat, 0, L - m, L); // flat[1:]
                var lo = SliceAlongAxis(flat, 0, 0, m);     // flat[:-1]
                // Same lean NDIter subtract as np.diff (uninitialised output,
                // no promotion/broadcast re-derivation); `-` operator as fallback.
                middle = DiffSubtractViaNDIter(hi, lo) ?? (hi - lo);
            }

            // Fast path: nothing to bracket — return the middle directly (the scope yields it).
            if (begin is null && end is null)
                return middle;

            // Assemble [to_begin?, middle, to_end?]. All three are already 1-D
            // and of dtype dt, so concatenate keeps the dtype unchanged. The parts are
            // released by this method's scope once the concatenation copied them.
            var parts = new List<NDArray>(3);
            if (begin is not null) parts.Add(begin);
            parts.Add(middle);
            if (end is not null) parts.Add(end);
            return np.concatenate(parts.ToArray(), 0);
        }

        /// <summary>
        ///     Validates and normalises a <c>to_begin</c>/<c>to_end</c> operand:
        ///     converts it to an array, enforces the NumPy <c>same_kind</c> casting
        ///     rule against <paramref name="dt"/>, then returns it flattened and
        ///     cast to <paramref name="dt"/> as a fresh owned 1-D array. Returns
        ///     <c>null</c> when <paramref name="value"/> is <c>null</c>.
        /// </summary>
        /// <param name="value">The <c>to_begin</c>/<c>to_end</c> operand: null, a scalar, a collection or an NDArray.</param>
        /// <param name="dt">The dtype of the flattened input.</param>
        /// <param name="name">The parameter name quoted in the casting error.</param>
        /// <returns>A fresh 1-D array of dtype <paramref name="dt"/>, or <c>null</c>.</returns>
        /// <exception cref="ArgumentException"><paramref name="value"/>'s dtype cannot be cast to
        ///     <paramref name="dt"/> under <c>same_kind</c>.</exception>
        /// <remarks>Covered by <c>[NDScopedCovered]</c>: its only caller is the <c>[NDScoped]</c>
        ///     <see cref="ediff1d"/>, so the conversion of a non-array <paramref name="value"/> and its ravel are
        ///     released by that scope, exceptions included. An <see cref="NDArray"/> <paramref name="value"/> is the
        ///     caller's and is never disposed.</remarks>
        [NDScopedCovered]
        private static NDArray EdiffPrepareEnd(object value, NPTypeCode dt, string name)
        {
            if (value is null) return null;

            NDArray src = value is NDArray nd ? nd : np.asanyarray(value);

            if (!np.can_cast(src.GetTypeCode, dt, "same_kind"))
                throw new ArgumentException(
                    $"dtype of `{name}` must be compatible with input `ary` " +
                    "under the `same_kind` rule.", name);

            var flat = np.ravel(src);
            return flat.astype(dt, copy: true);
        }
    }
}
