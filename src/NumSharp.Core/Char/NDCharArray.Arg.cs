using NumSharp.Backends.Strings;

// =============================================================================
// NDCharArray.Arg.cs — the members NumPy refuses on a chararray (argmax / argmin / argpartition) and the STRING
// partition that hides ndarray.partition
// =============================================================================
//
// argmax / argmin / argpartition create their int64 result as the operand's SUBCLASS (PyArray_ArgMax / PyArray_ArgMin /
// PyArray_ArgPartition: PyArray_NewFromDescr(Py_TYPE(ap), ...)), so chararray's __array_finalize__ raises
// "Can only create a chararray from string data." — AFTER NumPy's argument checks (the axis, the empty-sequence error,
// the kth bounds and the kind), which come first. With out= NumPy writes into the caller's array and nothing is created
// as the subclass: the call computes. argsort, sort and searchsorted are not refused (argsort works on the plain view,
// sort is in place, searchsorted allocates a plain array): the inherited NDArray members already answer as NumPy does.
//
// chararray.partition(sep) is the string partition (np.char.partition), not ndarray's in-place sort partition:
// c.partition(1) partitions the strings around the text '1'. It computes through np.strings.partition (strings phase 2,
// P2.5), the three parts stacked on a new last axis and returned as a chararray; the forms that pass axis= meet NumPy's
// TypeError.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        #region argmax / argmin

        /// <summary>
        ///     NumPy's <c>chararray.argmax()</c>: always the finalize error, because NumPy creates the index as a chararray
        ///     — after its own checks (an empty array raises <c>attempt to get argmax of an empty sequence</c> first).
        /// </summary>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError">An empty chararray (<c>attempt to get argmax of an empty sequence</c>); otherwise
        ///     always <c>Can only create a chararray from string data.</c></exception>
        /// <remarks>Through an <see cref="NDArray"/> reference the same object computes the plain index (the static rule).</remarks>
        public new long argmax()
        {
            // The plain computation raises NumPy's earlier errors in NumPy's order; its index is never returned.
            base.argmax(null, null, false).Dispose();
            throw NDCharArrayBoundary.FinalizeError();
        }

        /// <summary>
        ///     NumPy's <c>chararray.argmax(axis, keepdims=...)</c>: the finalize error, after NumPy's axis and empty-sequence
        ///     checks.
        /// </summary>
        /// <param name="axis">The axis (checked as NumPy checks it).</param>
        /// <param name="keepdims">NumPy's <c>keepdims</c>.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the array's rank (rank 1 for a 0-d chararray).</exception>
        /// <exception cref="ValueError">The reduced length is 0 (<c>attempt to get argmax of an empty sequence</c>); else
        ///     <c>Can only create a chararray from string data.</c></exception>
        public new NDArray argmax(int axis, bool keepdims = false) => argmax((int?)axis, null, keepdims);

        /// <summary>
        ///     NumPy's <c>chararray.argmax(axis=None, out=None, *, keepdims=False)</c>: with <paramref name="out"/> the
        ///     indices are written into it and it is returned (nothing is created as the subclass); without it, the
        ///     finalize error after NumPy's checks.
        /// </summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">The output array, or null.</param>
        /// <param name="keepdims">NumPy's <c>keepdims</c>.</param>
        /// <returns><paramref name="out"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the array's rank.</exception>
        /// <exception cref="ValueError">The reduced length is 0, <paramref name="out"/> has another shape (<c>output array
        ///     does not match result of np.argmax.</c>), or — without <paramref name="out"/> — always <c>Can only create a
        ///     chararray from string data.</c></exception>
        /// <exception cref="TypeError"><paramref name="out"/>'s dtype does not cast safely to int64.</exception>
        public new NDArray argmax(int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (@out is not null)
                return base.argmax(axis, @out, keepdims);
            base.argmax(axis, null, keepdims).Dispose();
            throw NDCharArrayBoundary.FinalizeError();
        }

        /// <summary>NumPy's <c>chararray.argmin()</c>: always the finalize error, after the empty-sequence check.</summary>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError">An empty chararray (<c>attempt to get argmin of an empty sequence</c>); otherwise
        ///     always <c>Can only create a chararray from string data.</c></exception>
        public new long argmin()
        {
            base.argmin(null, null, false).Dispose();
            throw NDCharArrayBoundary.FinalizeError();
        }

        /// <summary>NumPy's <c>chararray.argmin(axis, keepdims=...)</c>: the finalize error after NumPy's checks.</summary>
        /// <param name="axis">The axis.</param>
        /// <param name="keepdims">NumPy's <c>keepdims</c>.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the array's rank.</exception>
        /// <exception cref="ValueError">The reduced length is 0; else <c>Can only create a chararray from string data.</c></exception>
        public new NDArray argmin(int axis, bool keepdims = false) => argmin((int?)axis, null, keepdims);

        /// <summary>
        ///     NumPy's <c>chararray.argmin(axis=None, out=None, *, keepdims=False)</c>: computes into <paramref name="out"/>;
        ///     without it, the finalize error after NumPy's checks.
        /// </summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">The output array, or null.</param>
        /// <param name="keepdims">NumPy's <c>keepdims</c>.</param>
        /// <returns><paramref name="out"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the array's rank.</exception>
        /// <exception cref="ValueError">The reduced length is 0, <paramref name="out"/> has another shape, or — without
        ///     <paramref name="out"/> — always <c>Can only create a chararray from string data.</c></exception>
        /// <exception cref="TypeError"><paramref name="out"/>'s dtype does not cast safely to int64.</exception>
        public new NDArray argmin(int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (@out is not null)
                return base.argmin(axis, @out, keepdims);
            base.argmin(axis, null, keepdims).Dispose();
            throw NDCharArrayBoundary.FinalizeError();
        }

        #endregion

        #region argpartition

        /// <summary>
        ///     NumPy's <c>chararray.argpartition(kth, axis=-1, kind='introselect', order=None)</c>: always the finalize
        ///     error, after NumPy's checks of the kind, the order, the axis and the kth bounds (skipped for an empty array).
        /// </summary>
        /// <param name="kth">The element index to partition by.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The selection algorithm (only <c>'introselect'</c>).</param>
        /// <param name="order">Must be null.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError">A bad kind or kth (<c>kth(=9) out of bounds (4)</c>); otherwise always
        ///     <c>Can only create a chararray from string data.</c></exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the array's rank.</exception>
        public new NDArray argpartition(int kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            base.argpartition(kth, axis, kind, order).Dispose();
            throw NDCharArrayBoundary.FinalizeError();
        }

        /// <summary>NumPy's <c>chararray.argpartition</c> with several kth indices: the finalize error after NumPy's checks.</summary>
        /// <param name="kth">The element indices to partition by.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The selection algorithm.</param>
        /// <param name="order">Must be null.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError">A bad kind or kth; otherwise always <c>Can only create a chararray from string
        ///     data.</c></exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the array's rank.</exception>
        public new NDArray argpartition(int[] kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            base.argpartition(kth, axis, kind, order).Dispose();
            throw NDCharArrayBoundary.FinalizeError();
        }

        /// <summary>NumPy's <c>chararray.argpartition</c> with the kth indices as an array: the finalize error after NumPy's
        ///     checks.</summary>
        /// <param name="kth">The kth indices (an integer array).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="kind">The selection algorithm.</param>
        /// <param name="order">Must be null.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError">A bad kind or kth; otherwise always <c>Can only create a chararray from string
        ///     data.</c></exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the array's rank.</exception>
        /// <exception cref="TypeError">A non-integer kth array.</exception>
        public new NDArray argpartition(NDArray kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            base.argpartition(kth, axis, kind, order).Dispose();
            throw NDCharArrayBoundary.FinalizeError();
        }

        #endregion

        #region partition (the string partition)

        /// <summary>
        ///     NumPy's <c>chararray.partition(sep)</c>: the STRING partition (<c>np.char.partition</c>: each element split
        ///     around the first <paramref name="sep"/> into three parts), which hides ndarray's in-place sort partition.
        ///     The result adds a final axis of length three for the before, separator and after parts.
        /// </summary>
        /// <param name="sep">The separator, broadcast against the array's elements: a C# string (NumPy's Python <c>str</c>), a
        ///     <c>byte[]</c> (Python <c>bytes</c>), or an array; it is cast to this array's dtype first.</param>
        /// <returns>A chararray containing the three parts of each element on its last axis, at the widest part's width, laid
        ///     out as NumPy's <c>np.stack</c> lays it out: a VIEW of the stacked parts for bytes, a COPY for str (NumPy's
        ///     <c>np.char.asarray</c> compares a str array's width in characters with its itemsize in bytes and copies).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sep"/> is null.</exception>
        /// <exception cref="ValueError">An empty separator (<c>empty separator</c>); an empty array (<c>zero-size array to
        ///     reduction operation maximum which has no identity</c>, NumPy's sizing).</exception>
        /// <exception cref="TypeError">No <c>find</c> loop for the classes (a <c>str</c> separator beside a bytes chararray is
        ///     cast, a number is not text).</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII separator for a bytes chararray.</exception>
        /// <exception cref="IncorrectShapeException">The separator does not broadcast.</exception>
        /// <remarks>
        ///     NumPy's <c>chararray.partition</c> is <c>asarray(partition(self, sep))</c> with numpy.char's
        ///     <c>partition = np.stack(np.strings.partition(a, sep), axis=-1)</c>: the three parts (chararray views in NumPy,
        ///     as its ufunc hands them back) stacked into a plain array (<c>np.concatenate</c> keeps no subclass of equal
        ///     priority), seen as a chararray by <c>np.char.asarray</c>. <see cref="CharModule.partition"/> computes that
        ///     stacked array (plain parts: their class never reaches the stack's answer), and the array, private to the call,
        ///     takes its <c>asarray</c> form without a copy (<see cref="AsArrayOfFresh"/>).
        /// </remarks>
        [NDScoped]
        public NDCharArray partition(object sep) => AsArrayOfFresh(np.@char.partition(this, sep));

        /// <summary>
        ///     NumPy's <c>chararray.rpartition(sep)</c>: each element split around the LAST <paramref name="sep"/> into the
        ///     part before it, the separator and the part after it (without a match: two empty strings and the element). The
        ///     result adds a final axis of length three for the three parts.
        /// </summary>
        /// <param name="sep">The separator, broadcast against the array's elements: a C# string (NumPy's Python <c>str</c>), a
        ///     <c>byte[]</c> (Python <c>bytes</c>), or an array; it is cast to this array's dtype first (a number is its
        ///     text).</param>
        /// <returns>A chararray holding the three parts of each element on its last axis, at the widest part's width: a VIEW
        ///     of the stacked parts for bytes, a COPY for str (NumPy's <c>np.char.asarray</c> compares a str array's width
        ///     in characters with its itemsize in bytes and copies).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sep"/> is null.</exception>
        /// <exception cref="ValueError">An empty separator (<c>empty separator</c>); an empty array (<c>zero-size array to
        ///     reduction operation maximum which has no identity</c>, NumPy's sizing).</exception>
        /// <exception cref="TypeError">No loop for the classes.</exception>
        /// <exception cref="UnicodeEncodeError">A non-ASCII separator for a bytes chararray.</exception>
        /// <exception cref="IncorrectShapeException">The separator does not broadcast.</exception>
        /// <remarks>NumPy's <c>chararray.rpartition</c> is <c>asarray(rpartition(self, sep))</c>, numpy.char's
        ///     <c>rpartition</c> being <c>np.stack(np.strings.rpartition(a, sep), axis=-1)</c> (a plain array, see
        ///     <see cref="CharModule.rpartition"/>), seen as a chararray by <c>np.char.asarray</c>: the stacked array is
        ///     private to the call, so its <c>U</c> form takes it over instead of copying it
        ///     (<see cref="AsArrayOfFresh"/>).</remarks>
        [NDScoped]
        public NDCharArray rpartition(object sep) => AsArrayOfFresh(np.@char.rpartition(this, sep));

        /// <summary>
        ///     <c>c.partition(kth, axis, ...)</c>, the call <c>np.partition</c> makes: NumPy's TypeError, because
        ///     <c>chararray.partition</c> takes only the separator.
        /// </summary>
        /// <param name="kth">The kth index of a sort partition.</param>
        /// <param name="axis">The axis (its presence is what NumPy refuses).</param>
        /// <param name="kind">The selection algorithm.</param>
        /// <param name="order">Must be null.</param>
        /// <exception cref="TypeError">Always: <c>chararray.partition() got an unexpected keyword argument 'axis'</c>.</exception>
        /// <remarks>The axis has no default, so <c>c.partition(1)</c> binds the string partition, as in NumPy.</remarks>
        public new void partition(int kth, int? axis, string kind = "introselect", string order = null)
            => throw NDCharArrayBoundary.PartitionKeywordError();

        /// <summary><see cref="partition(int, int?, string, string)"/> with several kth indices: NumPy's TypeError.</summary>
        /// <param name="kth">The kth indices.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="kind">The selection algorithm.</param>
        /// <param name="order">Must be null.</param>
        /// <exception cref="TypeError">Always: <c>chararray.partition() got an unexpected keyword argument 'axis'</c>.</exception>
        public new void partition(int[] kth, int? axis, string kind = "introselect", string order = null)
            => throw NDCharArrayBoundary.PartitionKeywordError();

        /// <summary><see cref="partition(int, int?, string, string)"/> with the kth indices as an array: NumPy's TypeError.</summary>
        /// <param name="kth">The kth indices.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="kind">The selection algorithm.</param>
        /// <param name="order">Must be null.</param>
        /// <exception cref="TypeError">Always: <c>chararray.partition() got an unexpected keyword argument 'axis'</c>.</exception>
        public new void partition(NDArray kth, int? axis, string kind = "introselect", string order = null)
            => throw NDCharArrayBoundary.PartitionKeywordError();

        #endregion
    }
}
