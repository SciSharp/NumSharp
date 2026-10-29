using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return selected slices of <paramref name="a"/> along the given
        ///     <paramref name="axis"/> at positions where the 1-D
        ///     <paramref name="condition"/> is truthy.
        /// </summary>
        /// <param name="condition">
        ///     <strong>1-D</strong> array of booleans (or any dtype interpreted as
        ///     truthy). Must be 1-D — a 2-D or 0-D condition raises
        ///     <see cref="ValueError"/>, NumPy's
        ///     <c>ValueError("condition must be a 1-d array")</c>. If
        ///     <c>len(condition) &lt; a.shape[axis]</c>, only the first
        ///     <c>len(condition)</c> positions along <paramref name="axis"/> are
        ///     considered; if longer, any True beyond <c>a.shape[axis]</c> raises
        ///     <see cref="IndexOutOfRangeException"/>.
        /// </param>
        /// <param name="a">Source array.</param>
        /// <param name="axis">
        ///     Axis along which to slice. <c>null</c> (default) flattens
        ///     <paramref name="a"/> first.
        /// </param>
        /// <param name="out">
        ///     Optional destination. When supplied, shape must match the natural
        ///     output and <c>out.dtype</c> must be safely castable to
        ///     <c>a.dtype</c>; values are written via <see cref="np.copyto"/> with
        ///     unsafe casting and the method returns <paramref name="out"/> itself
        ///     (matches NumPy's out= dispatch via PyArray_TakeFrom).
        /// </param>
        /// <returns>
        ///     A copy of <paramref name="a"/> without the slices along
        ///     <paramref name="axis"/> for which <paramref name="condition"/> is
        ///     false. Dtype matches <paramref name="a"/> (or
        ///     <paramref name="out"/>'s dtype when supplied).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="condition"/> or <paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="condition"/> is not 1-D.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>'s rank (rank 1 for a
        ///     0-d <paramref name="a"/>, which NumPy ravels first).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.compress.html
        ///     <para>
        ///     Two execution paths:
        ///     <list type="bullet">
        ///       <item>
        ///         <b>Fast path</b>: bool condition, contig source, no out= or
        ///         out= matches src.dtype, <c>condition.size &lt;= a.shape[axis]</c>.
        ///         Runs the fused mask-driven gather kernel (popcount → alloc →
        ///         SIMD bit-scan + cpblk-per-outer-slab), skipping the
        ///         flatnonzero indices NDArray.
        ///       </item>
        ///       <item>
        ///         <b>Generic path</b>: mirrors NumPy's
        ///         <c>PyArray_Compress</c> chain — flatnonzero(cond) → take(a,
        ///         indices, axis, out, "raise"). Handles non-bool conditions,
        ///         out= with safe dtype cast, and OOB-True via take's RAISE mode.
        ///       </item>
        ///     </list>
        ///     </para>
        ///     <para>
        ///     When <paramref name="condition"/> is 1-D <see cref="np.extract"/> with
        ///     <paramref name="axis"/> = <c>null</c> is equivalent.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray compress(NDArray condition, NDArray a, int? axis = null, NDArray @out = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((condition, a, axis, @out), static s => compress(s.condition, s.a, s.axis, s.@out), @out);
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((condition, a, axis, @out), static s => compress(s.condition, s.a, s.axis, s.@out), @out);
            if (condition is null) throw new ArgumentNullException(nameof(condition));
            if (a is null) throw new ArgumentNullException(nameof(a));

            return UnmanagedStorage.Compress(condition, a, axis, @out);
        }
    }
}
