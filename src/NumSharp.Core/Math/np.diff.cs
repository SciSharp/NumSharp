using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    // ============================== np.diff ==============================
    // Calculate the n-th discrete difference along the given axis.
    //
    //   out[i] = a[i+1] - a[i]   (repeated n times, recursively)
    //
    // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::diff
    //
    // Implementation mirrors NumPy's structure exactly:
    //   1. Optionally concatenate [prepend, a, append] along the axis
    //      (scalar prepend/append broadcast to length 1 along the axis).
    //   2. Repeat n times:  a = op(a[1:], a[:-1])  along the axis,
    //      where op is `!=` (not_equal) for boolean arrays and `-`
    //      (subtract) for everything else.
    //
    // Both `op`s are backed by NumSharp's SIMD IL kernels (TensorEngine
    // .Subtract / .NotEqual), so the element-wise loop runs through the
    // ILKernelGenerator path — diff itself contains no per-element loop.
    // The two operands `a[1:]` and `a[:-1]` are overlapping strided views of
    // the same buffer; the kernel reads them and writes a fresh output, so
    // there is no read/write aliasing hazard.
    public static partial class np
    {
        /// <summary>
        ///     Calculate the n-th discrete difference along the given axis.
        ///     The first difference is <c>out[i] = a[i+1] - a[i]</c>; higher
        ///     differences are computed recursively.
        /// </summary>
        /// <param name="a">Input array (must be at least one dimensional).</param>
        /// <param name="n">
        ///     The number of times values are differenced. If zero, the input
        ///     is returned as-is. Must be non-negative.
        /// </param>
        /// <param name="axis">
        ///     The axis along which the difference is taken; default is the
        ///     last axis. Negative axes count from the end.
        /// </param>
        /// <param name="prepend">
        ///     Value(s) to prepend to <paramref name="a"/> along
        ///     <paramref name="axis"/> prior to differencing. Scalars expand to
        ///     length 1 along the axis. <c>null</c> means "not supplied"
        ///     (NumPy's <c>np._NoValue</c>).
        /// </param>
        /// <param name="append">
        ///     Value(s) to append to <paramref name="a"/> along
        ///     <paramref name="axis"/> prior to differencing. Scalars expand to
        ///     length 1 along the axis. <c>null</c> means "not supplied".
        /// </param>
        /// <returns>
        ///     The n-th differences. The shape matches the (optionally
        ///     prepend/append-extended) input except along <paramref name="axis"/>
        ///     where the size shrinks by <paramref name="n"/>. The dtype is
        ///     preserved (boolean input yields boolean output via not_equal).
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.diff.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="TypeError">A string array (or a string prepend/append) with <paramref name="n"/> above 0:
        ///     NumPy's <c>subtract</c> has no string loop, so its first <c>a[1:] - a[:-1]</c> refuses whatever the size.</exception>
        /// <exception cref="ValueError"><paramref name="a"/> is 0-D ("diff requires input that is at least one
        ///     dimensional").</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>'s rank.</exception>
        [NDScoped]
        public static NDArray diff(NDArray a, int n = 1, int axis = -1,
                                   object prepend = null, object append = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, n, axis, prepend, append), static s => diff(s.a, s.n, s.axis, s.prepend, s.append));
            return np.EngineOf(a).Diff(a, n, axis, prepend, append);
        }

        /// <summary>
        ///     The rank and axis checks <see cref="diff"/> makes before differencing (after its <c>n</c>
        ///     checks), shared with the callers that compose NumPy's <c>diff(a, axis=axis)</c> into a fused
        ///     pass instead of calling it (<see cref="trapezoid"/>), so they reject exactly what diff rejects,
        ///     with the same errors.
        /// </summary>
        /// <param name="a">The array to be differenced.</param>
        /// <param name="axis">The caller's axis; negative counts from the end (NumPy's <c>normalize_axis_index</c>).</param>
        /// <returns>The axis in <c>[0, a.ndim)</c>.</returns>
        /// <exception cref="ValueError"><paramref name="a"/> is 0-D ("diff requires input that is at least one dimensional").</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-a.ndim, a.ndim-1]</c>.</exception>
        internal static int DiffAxis(NDArray a, int axis)
        {
            int nd = a.ndim;
            if (nd == 0)
                throw new ValueError(
                    "diff requires input that is at least one dimensional");

            // normalize_axis_index(axis, nd)
            return Backends.DefaultEngine.check_and_adjust_axis(nd, axis);
        }
    }
}
