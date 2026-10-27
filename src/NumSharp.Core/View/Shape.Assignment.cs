using System.Text;

namespace NumSharp
{
    // ───────────────────────────────────────────────────────────────────────────────────────
    //  The broadcast of an assignment's operand — NumPy 2.4.2, array_assign_array.c,
    //  PyArray_AssignArray, and the broadcast_strides it calls (shape.c):
    //
    //      if (PyArray_NDIM(src) > PyArray_NDIM(dst)) {
    //          /* As a special case for backwards compatibility, strip
    //           * away unit dimensions from the left of 'src' */
    //          while (ndim_tmp > PyArray_NDIM(dst) && src_shape_tmp[0] == 1) { ... }
    //          broadcast_strides(dst ndim, dst dims, ndim_tmp, src_shape_tmp, ..., "input array", ...)
    //      }
    //    broadcast_error:
    //      "could not broadcast %s from shape %s into shape %s"
    //
    //  A view assignment (a[1:3] = v, a[:, ::2] = v) and np.copyto run this rule; a where mask runs it without the
    //  stripping. An index-array assignment runs MapIter's own rule instead (NDArray.PrepareAssignedValue: "shape
    //  mismatch: value array of shape ... could not be broadcast to indexing result of shape ...").
    // ───────────────────────────────────────────────────────────────────────────────────────

    public partial struct Shape
    {
        /// <summary>
        ///     NumPy's <c>broadcast_strides</c> for an assignment's operand, as <c>PyArray_AssignArray</c> calls it: the strides
        ///     an operand of <paramref name="dims"/> / <paramref name="strides"/> walks a destination of
        ///     <paramref name="dstDims"/> with (0 on every axis it broadcasts along), after its leading extent-1 axes beyond
        ///     the destination's rank are stripped when <paramref name="stripLeadingUnitAxes"/> is set.
        /// </summary>
        /// <param name="dstDims">The destination's dimensions (the region an assignment writes). Read only.</param>
        /// <param name="dims">The operand's dimensions. Read only.</param>
        /// <param name="strides">The operand's ELEMENT strides (the same length as <paramref name="dims"/>). Read only.</param>
        /// <param name="name">NumPy's operand name in the error: <c>"input array"</c> (the source) or <c>"where mask"</c>.</param>
        /// <param name="stripLeadingUnitAxes">Whether leading extent-1 axes beyond the destination's rank are dropped first:
        ///     <c>PyArray_AssignArray</c> does it for the SOURCE, "as a special case for backwards compatibility" (a
        ///     <c>(1, 1, 3)</c> value fits a <c>(2, 3)</c> region), never for a where mask.</param>
        /// <returns>A fresh array of <paramref name="dstDims"/>.Length element strides.</returns>
        /// <exception cref="ValueError"><c>could not broadcast {name} from shape (…) into shape (…)</c> — the operand's
        ///     shape AFTER the stripping, as NumPy prints it (<c>a[1:2] = [[7, 8]]</c> on a 1-D array: "from shape (2,) into
        ///     shape (1,)").</exception>
        /// <remarks>
        ///     Use it to VALIDATE an assignment before anything is written (the strides may be ignored); the copy iterator
        ///     then needs the operand without the stripped axes (<see cref="Backends.UnmanagedStorage.AliasWithoutLeadingAxes"/>),
        ///     because it broadcasts a source of at most the destination's rank. The check runs for an empty destination
        ///     too, as NumPy's does (a <c>(2,)</c> value cannot be assigned to a <c>(0,)</c> region).
        /// </remarks>
        internal static long[] AssignmentBroadcastStrides(long[] dstDims, long[] dims, long[] strides, string name,
            bool stripLeadingUnitAxes)
        {
            int ndim = dstDims.Length;
            int skip = 0;
            int sndim = dims.Length;
            if (stripLeadingUnitAxes)
            {
                while (sndim - skip > ndim && dims[skip] == 1)
                    skip++;
            }

            int ondim = sndim - skip;
            int start = ndim - ondim;
            var result = new long[ndim];
            if (start < 0)
                throw AssignmentBroadcastError(name, dims, skip, dstDims);

            // Process from the end to the start (NumPy's loop); a missing leading axis broadcasts with stride 0.
            for (int d = ndim - 1; d >= start; d--)
            {
                long extent = dims[skip + d - start];
                if (extent == 1)
                    result[d] = 0;
                else if (extent != dstDims[d])
                    throw AssignmentBroadcastError(name, dims, skip, dstDims);
                else
                    result[d] = strides[skip + d - start];
            }

            return result;
        }

        /// <summary>NumPy's broadcast error text of an assignment (<c>convert_shape_to_string</c>: <c>(4,)</c>, <c>(2,3)</c>,
        ///     <c>()</c>).</summary>
        /// <param name="name">"input array" or "where mask".</param>
        /// <param name="dims">The operand's dimensions.</param>
        /// <param name="skip">How many leading axes were stripped (they are not printed).</param>
        /// <param name="dstDims">The destination's dimensions.</param>
        /// <returns>The ValueError to throw; this method does not throw it.</returns>
        private static ValueError AssignmentBroadcastError(string name, long[] dims, int skip, long[] dstDims)
            => new ValueError($"could not broadcast {name} from shape {PythonTuple(dims, skip)} into shape {PythonTuple(dstDims, 0)}");

        /// <summary>A Python shape tuple without spaces: <c>()</c>, <c>(4,)</c>, <c>(2,3)</c>.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <param name="skip">Leading entries to leave out.</param>
        /// <returns>The tuple text.</returns>
        private static string PythonTuple(long[] dims, int skip)
        {
            int n = dims.Length - skip;
            if (n == 0)
                return "()";
            var sb = new StringBuilder("(");
            for (int d = skip; d < dims.Length; d++)
            {
                if (d > skip)
                    sb.Append(',');
                sb.Append(dims[d]);
            }
            if (n == 1)
                sb.Append(',');
            return sb.Append(')').ToString();
        }
    }
}
