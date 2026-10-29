// NumPy's out= contract of np.argmax / np.argmin (PyArray_ArgMaxWithKeepdims / PyArray_ArgMinWithKeepdims: the out's shape,
// then its safe cast to intp, then an unsafe copy of the indices), which DefaultEngine's ReduceArgMax / ReduceArgMin end
// with. The file sits beside the facades while it calls np.copyto: ratchet R2 counts np.* calls against every file under
// Backends/ (the np-layer dtype plan, C3); it moves under Backends/Default/ in P6.

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
            np.copyto(@out, result, casting: "unsafe");
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
    }
}
