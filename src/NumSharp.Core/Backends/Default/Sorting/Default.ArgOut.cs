// NumPy's out= contract of np.argmax / np.argmin (PyArray_ArgMaxWithKeepdims / PyArray_ArgMinWithKeepdims: the out's shape,
// then its safe cast to intp, then an unsafe copy of the indices), which DefaultEngine's ReduceArgMax / ReduceArgMin end
// with, and the checks they run before a text comparison.

using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's argmax/argmin <c>out=</c> contract (probed 2.4.2): the out's shape must EQUAL the
        ///     (keepdims-adjusted) result shape (<c>output array does not match result of np.argmax.</c> —
        ///     nanargmax leaks "argmax", nanargmin "argmin"), then its dtype must be safe-castable TO intp —
        ///     every bool/int dtype through uint32 (uint64 is not: verbatim <c>Cannot cast array data from
        ///     dtype('uint64') to dtype('int64') according to the rule 'safe'</c>, a TypeError) — and the
        ///     int64 indices then cast UNSAFELY into it (an int8 out wraps index 200 to -56; a bool out
        ///     reads index != 0 — probed). Returns the SAME instance.
        ///     Char (no NumPy dtype) is integer-family and safe to int64, so it is allowed (house call).
        ///     <see cref="ReduceArgMax"/> / <see cref="ReduceArgMin"/> end with it, so np.argmax / np.argmin and
        ///     np.nanargmax / np.nanargmin (whose members call them) write their out= this way.
        /// </summary>
        /// <param name="result">The int64 indices.</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="name"><c>argmax</c> or <c>argmin</c> (the nan forms leak the plain name, as NumPy does).</param>
        /// <returns><paramref name="result"/> without an out; otherwise <paramref name="out"/>, the indices written.</returns>
        /// <exception cref="ValueError">The out's shape differs from the result's.</exception>
        /// <exception cref="TypeError">The out's dtype does not cast safely to int64.</exception>
        private static NDArray WriteArgOut(NDArray result, NDArray @out, string name)
        {
            if (@out is null)
                return result;

            ValidateArgOut(@out, result.Shape.dimensions, name);
            // np.copyto(out, result, casting='unsafe'): the array object's assignment at that level.
            UnmanagedStorage.Copyto(@out, result, NPY_CASTING.NPY_UNSAFE_CASTING);
            return @out;
        }

        /// <summary>
        ///     <c>PyArray_ArgMaxWithKeepdims</c>'s two <c>out=</c> checks, in its order: the out must have the result's
        ///     shape exactly (<c>output array does not match result of np.argmax.</c>), then it must reach intp by a SAFE cast
        ///     — bool through uint32 (and NumSharp's char) do; uint64, floats, complex and text do not (NumPy's cast TypeError,
        ///     which words a 0-d out as a <c>scalar</c>).
        /// </summary>
        /// <param name="out">The out array.</param>
        /// <param name="dims">The result's dimensions.</param>
        /// <param name="name"><c>argmax</c> or <c>argmin</c> (the nan forms leak the plain name, as NumPy does).</param>
        /// <exception cref="ValueError">The shape differs from the result's.</exception>
        /// <exception cref="TypeError">The out's dtype does not cast safely to int64.</exception>
        /// <remarks>
        ///     The shape is checked FIRST: NumPy compares the out's dimensions before it converts the out to intp (an out of
        ///     the wrong shape AND the wrong dtype reports the shape). The numeric path used to check the dtype first.
        /// </remarks>
        private static void ValidateArgOut(NDArray @out, long[] dims, string name)
        {
            if (@out.ndim != dims.Length)
                throw new ValueError($"output array does not match result of np.{name}.");
            for (int d = 0; d < dims.Length; d++)
                if (@out.Shape.dimensions[d] != dims[d])
                    throw new ValueError($"output array does not match result of np.{name}.");

            switch (@out.typecode)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                    return;
                default:
                    throw new TypeError(
                        $"Cannot cast {(@out.ndim == 0 ? "scalar" : "array data")} from {@out.dtype.ToString(true)} to " +
                        "dtype('int64') according to the rule 'safe'");
            }
        }

        /// <summary>
        ///     NumPy's checks of <c>np.argmax</c> / <c>np.argmin</c> of text before it compares, once the axis passed: an
        ///     empty reduced length refuses, then <c>out=</c> is checked (<see cref="ValidateArgOut"/>). A text
        ///     comparison can raise (a <c>StringDType</c> None-like NA), and NumPy checks all of these first, so
        ///     <see cref="ReduceArgMax"/> / <see cref="ReduceArgMin"/> run them for a text operand with an out, and
        ///     np.nanargmax / np.nanargmin (which are np.argmax / np.argmin for text: <c>_replace_nan</c> leaves a
        ///     non-inexact array alone) run them for every text operand.
        /// </summary>
        /// <param name="a">The text array.</param>
        /// <param name="axis">The validated axis, or null for the flattened array.</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one (decides the expected out shape).</param>
        /// <param name="name"><c>argmax</c> or <c>argmin</c>.</param>
        /// <exception cref="ValueError">An empty reduced length (<c>attempt to get argmax of an empty sequence</c>), or an
        ///     out of the wrong shape.</exception>
        /// <exception cref="TypeError">An out that does not cast safely to int64.</exception>
        private static void ArgTextChecks(NDArray a, int? axis, NDArray @out, bool keepdims, string name)
        {
            int nd = a.ndim;
            long[] dims;
            long reduced;
            if (axis is not int ax)
            {
                // The flattened array: a 0-d input ravels to one element.
                reduced = nd == 0 ? 1 : a.size;
                dims = new long[keepdims ? nd : 0];
                Array.Fill(dims, 1L);
            }
            else if (nd == 0)
            {
                reduced = 1;
                dims = Array.Empty<long>();
            }
            else
            {
                int norm = ax < 0 ? ax + nd : ax;
                reduced = a.Shape.dimensions[norm];
                dims = new long[keepdims ? nd : nd - 1];
                for (int d = 0, k = 0; d < nd; d++)
                {
                    if (d == norm)
                    {
                        if (keepdims)
                            dims[k++] = 1;
                        continue;
                    }
                    dims[k++] = a.Shape.dimensions[d];
                }
            }
            if (reduced == 0)
                throw new ValueError($"attempt to get {name} of an empty sequence");
            if (@out is not null)
                ValidateArgOut(@out, dims, name);
        }
    }
}
