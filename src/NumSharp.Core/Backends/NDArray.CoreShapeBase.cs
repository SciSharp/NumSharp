using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    // ============================== _core/shape_base.py ==============================
    // NumPy's numpy/_core/shape_base.py statements as members of the array object: atleast_1d's, atleast_2d's and
    // atleast_3d's for one array, the ndmin padding (AtLeastNdView) np.array, np.block's _atleast_nd and np.r_ share, and
    // vstack, hstack, stack, unstack and block. Each np.* facade keeps its class rules (numpy.matrix, numpy.char.chararray)
    // and its null checks and calls the member; engine code (np.poly, np.roots, the polynomial arithmetic, np.cov) calls
    // them directly, inside its public function's boundary. np.atleast_2d(object) and np.atleast_3d(object) / (object[])
    // keep their own statements: an array handed to them as an object passes through np.asanyarray, which keeps a
    // chararray's class, and their np.expand_dims then applies the chararray rule, which these plain statements do not.
    public partial class NDArray
    {
        // ---------------------------- np.atleast_1d ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.atleast_1d(a)</c> for one array after the facade's numpy.char.chararray rule: the array
        ///     itself when it has one dimension or more, else its reshape to <c>(1,)</c>.
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <returns><paramref name="a"/> itself when <c>a.ndim &gt;= 1</c> (so a caller must not dispose the result unless
        ///     it is not <paramref name="a"/>), else a <c>(1,)</c> VIEW of the 0-d array's element: it shares the
        ///     buffer, keeps it alive, and writes through.</returns>
        /// <remarks>No class boundary runs here: an engine caller sits inside its public function's boundary, and the
        ///     facade applies its rules before it calls this.</remarks>
        internal static NDArray atleast_1d(NDArray a)
        {
            switch (a.ndim)
            {
                case 0:
                    return a.reshape(1);
                default:
                    return a;
            }
        }

        // ---------------------------- np.atleast_2d ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.atleast_2d(a)</c> for one array after the facade's numpy.char.chararray rule: the array
        ///     itself when it has two dimensions or more, a <c>(1, 1)</c> reshape of a 0-d array, and a <c>(1, N)</c>
        ///     view of a 1-d one (NumPy's <c>ary[newaxis, :]</c>).
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <returns><paramref name="a"/> itself when <c>a.ndim &gt;= 2</c> (so a caller must not dispose the result
        ///     unless it is not <paramref name="a"/>), else a VIEW of <paramref name="a"/>'s elements with the leading
        ///     axes added: it shares the buffer, keeps it alive, and writes through.</returns>
        /// <remarks>No class boundary runs here: an engine caller sits inside its public function's boundary, and the
        ///     facade applies its rules before it calls this. The 1-d case is np.expand_dims(a, 0)'s statement
        ///     (<see cref="expand_dims(NDArray, int)"/>) without that facade's chararray rule.</remarks>
        internal static NDArray atleast_2d(NDArray a)
        {
            switch (a.ndim)
            {
                case 0:
                    return a.reshape(1, 1);
                case 1:
                    return expand_dims(a, 0);
                default:
                    return a;
            }
        }

        // ---------------------------- ndmin (np.array, np.block, np.r_) ----------------------------

        /// <summary>
        ///     NumPy's <c>array(a, ndmin=ndim, copy=None)</c> over an array: <paramref name="a"/> with ones prepended to its
        ///     shape up to <paramref name="ndim"/> dimensions, always as a VIEW of its buffer (np.array's ndmin,
        ///     np.block's <c>_atleast_nd</c>, np.r_ / np.c_'s directive ndmin).
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <param name="ndim">The minimum number of dimensions; 0 or less, or no more than <paramref name="a"/>'s, adds
        ///     none. It can come from a user's text (np.r_'s directive), so the padding costs O(ndim), never more.</param>
        /// <returns><paramref name="a"/> itself when <c>a.ndim &gt;= ndim</c> (so a caller must not dispose the result unless
        ///     it is not <paramref name="a"/>), else a VIEW of <paramref name="a"/>'s buffer with the leading length-1 axes
        ///     added: it shares the buffer, keeps it alive and writes through. A non-empty <paramref name="a"/> keeps its
        ///     strides, flags and writeability (one alias over <see cref="Shape.PrependDimensions"/>); an empty one is
        ///     reshaped to the padded dimensions.</returns>
        /// <exception cref="System.OutOfMemoryException">An <paramref name="ndim"/> too large to allocate a shape for, the
        ///     answer any other oversized NumSharp request gives.</exception>
        /// <remarks>No class boundary runs here: an engine caller sits inside its public function's boundary, and the
        ///     callers np.array's conversion, np.block (a matrix leaf passes unpadded) and np.r_ / np.c_ apply their
        ///     rules before they call this.</remarks>
        internal static NDArray AtLeastNdView(NDArray a, int ndim)
        {
            if (a.ndim >= ndim)
                return a;

            if (a.size == 0)
            {
                // expand_dims early-returns empty arrays unchanged; reshape an
                // empty directly instead (no data, layout is irrelevant).
                var dims = new long[ndim];
                int pad = ndim - a.ndim;
                for (int i = 0; i < pad; i++)
                    dims[i] = 1;
                for (int i = 0; i < a.ndim; i++)
                    dims[pad + i] = a.Shape.dimensions[i];
                return a.reshape(Shape.Adopt(dims));
            }

            // ONE alias, not one per axis. `ndim` can be user-supplied (np.r_'s ndmin comes
            // from a typed directive string), and expand_dims-per-axis clones dims+strides
            // every step — quadratic, 27.6 s at ndim=100 000. PrependDimensions is the
            // closed form of that loop, so the strides, flags and writeable bit are
            // unchanged; only the cost is.
            return new NDArray(a.Storage.Alias(a.Shape.PrependDimensions(ndim - a.ndim)))
                { TensorEngine = a.TensorEngine };
        }

        // ---------------------------- np.atleast_3d ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.atleast_3d(a)</c> for one array after the facade's class rules: the array itself when it has
        ///     three dimensions or more, a <c>(1, 1, 1)</c> reshape of a 0-d array, a <c>(1, N, 1)</c> view of a 1-d one
        ///     (NumPy's <c>ary[newaxis, :, newaxis]</c>) and an <c>(M, N, 1)</c> view of a 2-d one (<c>ary[:, :, newaxis]</c>).
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <returns><paramref name="a"/> itself when <c>a.ndim &gt;= 3</c> (so a caller must not dispose the result unless it
        ///     is not <paramref name="a"/>), else a VIEW of <paramref name="a"/>'s elements with the axes added: it shares the
        ///     buffer, keeps it alive, and writes through.</returns>
        /// <remarks>No class boundary runs here: the facades apply their rules before they call this. The 1-d case adds the
        ///     trailing axis first, as NumPy's statements do; the inner view it goes through is this call's own and is
        ///     reclaimed (the outer view keeps the buffer alive).</remarks>
        [NDScoped /* reclaims the inner expand_dims alias of the 1-d arm; the ndim >= 3 passthrough is an untracked input */]
        internal static NDArray atleast_3d(NDArray a)
        {
            switch (a.ndim)
            {
                case 0:
                    return a.reshape(1, 1, 1);
                case 1:
                    return expand_dims(expand_dims(a, 1), 0);
                case 2:
                    return expand_dims(a, 2);
                default:
                    return a;
            }
        }

        // ---------------------------- np.vstack / np.hstack ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.vstack(tup)</c> after the facade's class rules and null check: <c>arrs = atleast_2d(*tup);
        ///     return concatenate(arrs, 0)</c> - 1-d arrays become rows, and the arrays are joined along the first axis.
        /// </summary>
        /// <param name="tup">The arrays (the array itself not null; a null entry is NumPy's None, refused).</param>
        /// <returns>A new array: <c>np.concatenate</c>'s result, in its promoted dtype and NumPy's layout vote.</returns>
        /// <exception cref="ValueError">No array: <c>atleast_2d(*tup)</c> of nothing is the empty tuple, and concatenating it
        ///     is np.concatenate's <c>need at least one array to concatenate</c>, the text NumPy's vstack([]) raises.</exception>
        /// <exception cref="NullReferenceException">A null entry (read by atleast_2d).</exception>
        /// <exception cref="IncorrectShapeException">Arrays whose dimensions past the first differ (np.concatenate's
        ///     text).</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to the result dtype (np.concatenate's
        ///     refusal).</exception>
        /// <remarks>No class boundary runs here: the facade applies its rules (a matrix result wrapped, a chararray operand
        ///     computed plain) before it calls this, so the inner atleast_2d and concatenate are the members. The trailing
        ///     underscore is NumPy's convention for a name already taken (D18): <see cref="vstack(NDArray[])"/> is the
        ///     array object's public instance method that stacks the array with others.</remarks>
        internal static NDArray vstack_(NDArray[] tup)
        {
            if (tup.Length == 0)
                throw new ValueError("need at least one array to concatenate");

            var arrs = new NDArray[tup.Length];
            for (int i = 0; i < tup.Length; i++)
                arrs[i] = atleast_2d(tup[i]);
            return concatenate(arrs, 0, null, null, "same_kind");
        }

        /// <summary>
        ///     NumPy's <c>numpy.hstack(tup)</c> after the facade's class rules and null check: <c>arrs = atleast_1d(*tup)</c>,
        ///     then <c>concatenate(arrs, 0)</c> when the first array is 1-d (dimension 0 of 1-d arrays is "horizontal") and
        ///     <c>concatenate(arrs, 1)</c> otherwise.
        /// </summary>
        /// <param name="tup">The arrays (the array itself not null; a null entry is refused).</param>
        /// <returns>A new array: <c>np.concatenate</c>'s result.</returns>
        /// <exception cref="ValueError">No array (np.concatenate's <c>need at least one array to concatenate</c>).</exception>
        /// <exception cref="NullReferenceException">A null entry (read by atleast_1d).</exception>
        /// <exception cref="AxisError">Joining 1-d-or-more arrays along axis 1 when the first is 2-d or more but another is
        ///     1-d is np.concatenate's rank error; a first array of more dimensions joins along its axis 1.</exception>
        /// <exception cref="IncorrectShapeException">Ranks or extents that np.concatenate refuses.</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to the result dtype.</exception>
        /// <remarks>No class boundary runs here: the facade applies its rules before it calls this. The trailing underscore
        ///     is NumPy's convention for a name already taken (D18): <see cref="hstack(NDArray[])"/> is the array object's
        ///     public instance method that stacks the array with others.</remarks>
        internal static NDArray hstack_(NDArray[] tup)
        {
            if (tup.Length == 0)
                throw new ValueError("need at least one array to concatenate");

            var arrs = new NDArray[tup.Length];
            for (int i = 0; i < tup.Length; i++)
                arrs[i] = atleast_1d(tup[i]);   // handles scalars
            return arrs[0].Shape.dimensions.Length == 1
                ? concatenate(arrs, 0, null, null, "same_kind")
                : concatenate(arrs, 1, null, null, "same_kind");
        }

        // ---------------------------- np.stack / np.unstack ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.stack(arrays, axis=0)</c> after the facade's class rules and null check: every array must have
        ///     the same shape, a new axis is inserted at <paramref name="axis"/> of the <c>ndim + 1</c>-d result, and the
        ///     expanded arrays are concatenated along it.
        /// </summary>
        /// <param name="arrays">The arrays (the array itself not null; a null entry is refused).</param>
        /// <param name="axis">The result's new axis (negative counts from the end of the RESULT's rank).</param>
        /// <returns>A new array of <c>(len(arrays),)</c> inserted into the arrays' shape at <paramref name="axis"/>.</returns>
        /// <exception cref="ValueError">No array (<c>need at least one array to stack</c>), or shapes that differ (<c>all
        ///     input arrays must have the same shape</c>, checked for every array before the axis).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> outside the result's rank.</exception>
        /// <exception cref="NullReferenceException">A null entry.</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to the result dtype (np.concatenate's
        ///     refusal).</exception>
        /// <remarks>No class boundary runs here: the facade applies its rules before it calls this, so the inner expand_dims
        ///     and concatenate are the members.</remarks>
        internal static NDArray stack(NDArray[] arrays, int axis)
        {
            // `if not arrays: raise ValueError('need at least one array to stack')`.
            if (arrays.Length == 0)
                throw new ValueError("need at least one array to stack");

            // `shapes = {arr.shape for arr in arrays}; if len(shapes) != 1: raise` — every shape, before the axis.
            var first = arrays[0];
            for (int i = 1; i < arrays.Length; i++)
                if (!arrays[i].Shape.dimensions.AsSpan().SequenceEqual(first.Shape.dimensions))
                    throw new ValueError("all input arrays must have the same shape");

            // `axis = normalize_axis_index(axis, result_ndim)` — NumPy's AxisError against the RESULT's rank, raised once
            // here (the expansion below then sees an axis in range).
            axis = Shape.check_and_adjust_axis(first.ndim + 1, axis);

            // `expanded_arrays = [arr[sl] for arr in arrays]` — the new axis at `axis` of the (ndim + 1)-D result.
            var expanded = new NDArray[arrays.Length];
            for (int i = 0; i < arrays.Length; i++)
                expanded[i] = expand_dims(arrays[i], axis);

            return concatenate(expanded, axis, null, null, "same_kind");
        }

        /// <summary>
        ///     NumPy's <c>numpy.unstack(x, axis=0)</c> after the facade's class rules and null check: <c>tuple(moveaxis(x,
        ///     axis, 0))</c>, one VIEW per index along <paramref name="axis"/>, each of <paramref name="x"/>'s shape without
        ///     that axis.
        /// </summary>
        /// <param name="x">The array (not null).</param>
        /// <param name="axis">The axis to split along (negative counts from the end).</param>
        /// <returns>The views (an empty array for a zero-length <paramref name="axis"/>): each shares <paramref name="x"/>'s
        ///     buffer, keeps it alive and writes through; a read-only <paramref name="x"/> gives read-only views.</returns>
        /// <exception cref="ValueError">A 0-d <paramref name="x"/> (<c>Input array must be at least 1-d.</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range (moveaxis' <c>source: axis ... is out of
        ///     bounds</c>).</exception>
        /// <exception cref="OverflowException">An axis longer than int.MaxValue (the result is a C# array).</exception>
        /// <remarks>No class boundary runs here: the facade applies its rules (a matrix's rows, a chararray's rows or
        ///     elements) before it calls this.</remarks>
        internal static NDArray[] unstack(NDArray x, int axis)
        {
            // NumPy: ValueError("Input array must be at least 1-d.")
            int ndim = x.ndim;
            if (ndim == 0)
                throw new ValueError("Input array must be at least 1-d.");

            // NumPy's unstack is `tuple(moveaxis(x, axis, 0))`, so a bad axis is moveaxis' "source: axis … is out of
            // bounds" AxisError.
            axis = Shape.check_and_adjust_axis(ndim, axis, "source");

            var shape = x.Shape;
            var result = new NDArray[checked((int)shape.dimensions[axis])];
            if (result.Length == 0)
                return result;

            // Every child shares the same dims/strides — x's with `axis` removed.
            // Shape treats the arrays as immutable, so ONE pair serves all views.
            var subDims = new long[ndim - 1];
            var subStrides = new long[ndim - 1];
            for (int j = 0, k = 0; j < ndim; j++)
            {
                if (j == axis) continue;
                subDims[k] = shape.dimensions[j];
                subStrides[k] = shape.strides[j];
                k++;
            }

            // An EMPTY x (another axis is zero-length; a zero `axis` returned above) is unstacked into views too: NumPy's
            // iteration of moveaxis(x, axis, 0) yields views (OWNDATA False) whatever their size. An empty view's offset
            // need not address an element (it may point past the buffer); nothing reads through it.
            long axisStride = shape.strides[axis];
            long baseOffset = shape.offset;
            var storage = x.Storage;
            var engine = x.TensorEngine;
            for (int i = 0; i < result.Length; i++)
            {
                // Alias() inherits writeability, so unstacking a read-only array
                // (broadcast view / 'r' memmap) yields read-only views.
                var sub = new Shape(subDims, subStrides, baseOffset + i * axisStride, shape.bufferSize);
                result[i] = new NDArray(storage.Alias(sub), engine, skipEngineResolve: true);
            }

            return result;
        }

        // ---------------------------- np.block ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.block(arrays)</c> after the facade's matrix rule and null check: the nested lists' depths
        ///     checked (<c>_block_setup</c>), then the blocks assembled by repeated concatenation (innermost lists along the
        ///     last axis, then the second-last, ...) or, past NumPy's crossover of <c>2*512*512</c> elements times the list
        ///     depth, by filling one allocation (<c>_block_slicing</c>).
        /// </summary>
        /// <param name="arrays">The nested lists of blocks (not null), in the C# mapping the facade documents: an NDArray or a
        ///     scalar is a leaf, a 1-D C# array or a non-generic IList a list, a rank-2-or-more C# array a leaf, a tuple
        ///     refused.</param>
        /// <returns>A new array of the greatest of the leaves' ranks and the list depth; a COPY for a bare leaf (depth
        ///     0).</returns>
        /// <exception cref="ValueError">List depths that differ (<c>List depths are mismatched. ...</c>), an empty list
        ///     (<c>List at arrays[...] cannot be empty</c>), or block shapes that do not fit (np.concatenate's texts on the
        ///     concatenation route, <c>Mismatched array shapes in block along axis ...</c> on the slicing route).</exception>
        /// <exception cref="TypeError">A tuple anywhere in the nesting (NumPy refuses the implicit tuple-to-ndarray
        ///     conversion).</exception>
        /// <exception cref="IncorrectShapeException">Blocks whose ranks or extents np.concatenate refuses.</exception>
        /// <exception cref="InvalidCastException">A block that does not cast to the result dtype.</exception>
        /// <remarks>
        ///     <see cref="t_blockTookSlicingRoute"/> records which route ran, for the facade's matrix rule (NumPy's result class
        ///     follows the route: repeated concatenation keeps a matrix leaf's class, the single allocation is an ndarray). A
        ///     matrix leaf is never padded: NumPy's <c>_atleast_nd</c> is <c>array(a, ndmin=ndim, subok=True)</c>, which cannot
        ///     give a numpy.matrix another axis.
        /// </remarks>
        internal static NDArray block(object arrays)
        {
            var (normalized, listNdim, resultNdim, finalSize) = _block_setup(arrays);
            t_blockTookSlicingRoute = (long)listNdim * finalSize > 2 * 512 * 512;

            // NumPy wisdom (numpy/_core/shape_base.py): repeated concatenation
            // copies the data once per nesting level, single-allocation slicing
            // copies once but pays per-leaf assignment setup. The crossover was
            // benchmarked by NumPy at ~2*512*512 elements — replicated verbatim
            // so both libraries always run the same algorithm (and therefore
            // agree on per-path details like the dtype-resolution grouping).
            if ((long)listNdim * finalSize > 2 * 512 * 512)
                return _block_slicing(normalized, listNdim, resultNdim);
            else
                return _block_concatenate(normalized, listNdim, resultNdim);
        }

        /// <summary>
        ///     Which route the last <see cref="block(object)"/> call on this thread took: true for the single-allocation slicing
        ///     route, false for repeated concatenation. Read only by np.block's matrix rule, right after the call it made
        ///     returns; per thread, so concurrent np.block calls on other threads cannot change the answer it reads.
        /// </summary>
        [ThreadStatic] internal static bool t_blockTookSlicingRoute;

        // ------------------------------------------------------------------
        //  Port of numpy/_core/shape_base.py — same helper names, same flow.
        // ------------------------------------------------------------------

        /// <summary>Sentinel for Python's <c>None</c> marking an empty list at the bottom of the nesting.</summary>
        private const int _blockEmptyListMarker = int.MinValue;

        /// <summary>Convert an index chain <c>[0, 1]</c> into <c>"arrays[0][1]"</c> (empty-list markers skipped), the
        ///     position NumPy's errors name.</summary>
        /// <param name="index">The index chain from the outermost list down.</param>
        /// <returns>The Python spelling of the position.</returns>
        private static string _block_format_index(int[] index)
        {
            var sb = new System.Text.StringBuilder("arrays");
            foreach (var i in index)
                if (i != _blockEmptyListMarker)
                    sb.Append('[').Append(i).Append(']');
            return sb.ToString();
        }

        /// <summary>The index chain one level deeper (NumPy's <c>parent_index + [i]</c>), a new array each time because the
        ///     chains of sibling elements must not share storage.</summary>
        /// <param name="index">The parent's chain.</param>
        /// <param name="i">The element's position in its list.</param>
        /// <returns>A new chain ending in <paramref name="i"/>.</returns>
        private static int[] _block_append_index(int[] index, int i)
        {
            var next = new int[index.Length + 1];
            System.Array.Copy(index, next, index.Length);
            next[index.Length] = i;
            return next;
        }

        /// <summary>
        ///     Recursive walk checking that the depths of nested lists in
        ///     <paramref name="arrays"/> all match (ValueError otherwise), while
        ///     classifying each node. Returns the index of an element from the
        ///     bottom of the nesting (empty lists flagged by a trailing marker),
        ///     the max ndim over all leaves, the total element count, and the
        ///     NORMALIZED tree: lists become <c>object[]</c>, leaves become
        ///     <see cref="NDArray"/> — so later passes skip re-classification
        ///     (NumPy re-derives per pass; the C# tree conversion is hoisted).
        /// </summary>
        /// <param name="arrays">The node: a list (a 1-D C# array or a non-generic IList), a leaf, or a tuple (refused).</param>
        /// <param name="parentIndex">The node's position, for the error texts.</param>
        /// <returns>The bottom index, the leaves' maximum rank, the leaves' total size and the normalized node.</returns>
        /// <exception cref="TypeError">A tuple.</exception>
        /// <exception cref="ValueError">Two elements of one list at different depths.</exception>
        private static (int[] index, int maxArrNdim, long size, object normalized)
            _block_check_depths_match(object arrays, int[] parentIndex)
        {
            // NumPy rejects tuples outright: no more than one way to arrange
            // blocks, and no horribly confusing tuple→ndarray implicit
            // conversion. ValueTuple/Tuple map to Python tuples.
            if (arrays is ITuple)
                throw new TypeError(
                    $"{_block_format_index(parentIndex)} is a tuple. " +
                    "Only lists can be used to arrange blocks, and np.block does " +
                    "not allow implicit conversion from tuple to ndarray.");

            // Python-list equivalents: 1-D CLR arrays (object[]/NDArray[]/int[]/
            // jagged) and non-generic IList implementors (List<T>). NDArray is
            // not a System.Array and does not implement IList, so it falls
            // through to the leaf branch below. Rank>=2 rectangular arrays are
            // leaves (Python has no rectangular list).
            IList list = null;
            if (arrays is System.Array clrArray)
            {
                if (clrArray.Rank == 1)
                    list = clrArray;
            }
            else if (arrays is not NDArray && arrays is IList ilist)
                list = ilist;

            if (list is not null && list.Count > 0)
            {
                int[] firstIndex = null;
                int maxArrNdim = 0;
                long finalSize = 0;
                var normalized = new object[list.Count];
                for (int i = 0; i < list.Count; i++)
                {
                    var (index, ndim, size, norm) =
                        _block_check_depths_match(list[i], _block_append_index(parentIndex, i));
                    normalized[i] = norm;

                    if (i == 0)
                    {
                        firstIndex = index;
                        maxArrNdim = ndim;
                        finalSize = size;
                        continue;
                    }

                    finalSize += size;
                    if (ndim > maxArrNdim)
                        maxArrNdim = ndim;
                    if (index.Length != firstIndex.Length)
                        throw new ValueError(
                            "List depths are mismatched. First element was at " +
                            $"depth {firstIndex.Length}, but there is an element at " +
                            $"depth {index.Length} ({_block_format_index(index)})");
                    // propagate the flag that indicates an empty list at the bottom
                    if (index[index.Length - 1] == _blockEmptyListMarker)
                        firstIndex = index;
                }

                return (firstIndex, maxArrNdim, finalSize, normalized);
            }

            if (list is not null) // We've 'bottomed out' on an empty list
                return (_block_append_index(parentIndex, _blockEmptyListMarker), 0, 0, null);

            // We've 'bottomed out' - arrays is either a scalar or an array. A leaf that is not an NDArray (a C# scalar, a
            // rank-2-or-more C# array) is converted by np.asanyarray's body; an NDArray leaf is taken as it is (its class
            // is the facade's business).
            var nd = arrays as NDArray ?? asanyarray(arrays, null, 'K');
            return (parentIndex, nd.ndim, nd.size, nd);
        }

        /// <summary>
        ///     Returns <c>(normalized, list_ndim, result_ndim, final_size)</c> —
        ///     NumPy's <c>_block_setup</c>.
        /// </summary>
        /// <param name="arrays">The facade's argument.</param>
        /// <returns>The normalized tree, the list depth, the result's rank and the total element count.</returns>
        /// <exception cref="ValueError">An empty list at the bottom of the nesting, or mismatched depths.</exception>
        /// <exception cref="TypeError">A tuple in the nesting.</exception>
        private static (object normalized, int listNdim, int resultNdim, long finalSize)
            _block_setup(object arrays)
        {
            var (bottomIndex, arrNdim, finalSize, normalized) =
                _block_check_depths_match(arrays, System.Array.Empty<int>());
            int listNdim = bottomIndex.Length;
            if (listNdim > 0 && bottomIndex[listNdim - 1] == _blockEmptyListMarker)
                throw new ValueError(
                    $"List at {_block_format_index(bottomIndex)} cannot be empty");
            int resultNdim = Math.Max(arrNdim, listNdim);
            return (normalized, listNdim, resultNdim, finalSize);
        }

        /// <summary>
        ///     Internal implementation of block based on repeated concatenation:
        ///     lists at nesting depth <c>d</c> concatenate along axis
        ///     <c>-(maxDepth - d)</c>; leaves get leading 1s to
        ///     <paramref name="resultNdim"/>.
        /// </summary>
        /// <param name="arrays">The normalized node (an <c>object[]</c> above the bottom, an NDArray leaf at it).</param>
        /// <param name="maxDepth">The list depth.</param>
        /// <param name="resultNdim">The result's rank, which the leaves are padded to.</param>
        /// <param name="depth">This node's depth.</param>
        /// <returns>The joined array (a fresh concatenation above the bottom; the leaf or its padded view at it).</returns>
        /// <exception cref="IncorrectShapeException">Blocks np.concatenate cannot join.</exception>
        /// <exception cref="AxisError">A join axis past a leaf's rank (a matrix leaf nested too deep).</exception>
        private static NDArray _block(object arrays, int maxDepth, int resultNdim, int depth)
        {
            if (depth < maxDepth)
            {
                var list = (object[])arrays;
                var arrs = new NDArray[list.Length];
                for (int i = 0; i < list.Length; i++)
                    arrs[i] = _block(list[i], maxDepth, resultNdim, depth + 1);
                // np.concatenate(arrs, axis)'s body: its matrix rule has nothing to see here (the facade's rule decided the
                // class before the call, and inside its region no rule fires).
                var joined = concatenate(arrs, -(maxDepth - depth), null, null, "same_kind");

                // Deeper-level results are fresh concatenate outputs consumed by
                // the join above — release their buffers eagerly instead of
                // queueing multi-KB intermediates on the finalizer queue (the
                // level at depth+1 == maxDepth holds leaves / leaf views, which
                // belong to the caller and must NOT be disposed).
                if (depth + 1 < maxDepth)
                    for (int i = 0; i < arrs.Length; i++)
                        arrs[i].Dispose();

                return joined;
            }

            // We've 'bottomed out' - arrays is a leaf NDArray. NumPy's _atleast_nd is array(a, ndmin=ndim,
            // subok=True), which cannot give a numpy.matrix another axis: a matrix leaf stays two-dimensional, so a
            // list nested deeper than two levels fails in the join above as it does in NumPy (np.block([[[m]]]):
            // "axis -3 is out of bounds for array of dimension 2").
            var leaf = (NDArray)arrays;
            return NDMatrixBoundary.IsMatrix(leaf) ? leaf : AtLeastNdView(leaf, resultNdim);
        }

        /// <summary>NumPy's <c>_block_concatenate</c>: the repeated-concatenation route, with a COPY for a bare leaf.</summary>
        /// <param name="normalized">The normalized tree.</param>
        /// <param name="listNdim">The list depth.</param>
        /// <param name="resultNdim">The result's rank.</param>
        /// <returns>A new array.</returns>
        /// <exception cref="IncorrectShapeException">Blocks np.concatenate cannot join.</exception>
        private static NDArray _block_concatenate(object normalized, int listNdim, int resultNdim)
        {
            var result = _block(normalized, listNdim, resultNdim, 0);
            if (listNdim == 0)
            {
                // Catch an edge case where _block returns a view because
                // `arrays` is a single NDArray and not a list of them: np.block
                // of a bare array/scalar returns a COPY (probed NumPy 2.4.2).
                result = result.copy();
            }

            return result;
        }

        /// <summary>
        ///     Given the shapes being concatenated along <paramref name="axis"/>,
        ///     return the resulting shape and the per-input (start, stop) window
        ///     at that axis — NumPy's <c>_concatenate_shapes</c>.
        /// </summary>
        /// <param name="shapes">The children's shapes.</param>
        /// <param name="axis">The join axis.</param>
        /// <returns>The joined shape and each child's window along the axis.</returns>
        /// <exception cref="ValueError">A child whose extents off the axis differ (<c>Mismatched array shapes in block along
        ///     axis {axis}.</c>).</exception>
        private static (long[] shape, (long start, long stop)[] slicePrefixes)
            _concatenate_shapes(long[][] shapes, int axis)
        {
            var firstShape = shapes[0];
            int ndim = firstShape.Length;

            long axisTotal = 0;
            for (int k = 0; k < shapes.Length; k++)
            {
                var s = shapes[k];
                for (int j = 0; j < ndim; j++)
                {
                    if (j == axis) continue;
                    if (s[j] != firstShape[j])
                        throw new ValueError(
                            $"Mismatched array shapes in block along axis {axis}.");
                }

                axisTotal += s[axis];
            }

            var shape = new long[ndim];
            System.Array.Copy(firstShape, shape, ndim);
            shape[axis] = axisTotal;

            var slicePrefixes = new (long, long)[shapes.Length];
            long offset = 0;
            for (int k = 0; k < shapes.Length; k++)
            {
                slicePrefixes[k] = (offset, offset + shapes[k][axis]);
                offset += shapes[k][axis];
            }

            return (shape, slicePrefixes);
        }

        /// <summary>
        ///     Returns the final shape, along with per-leaf slice windows (one
        ///     (start, stop) per nesting level, covering the LAST
        ///     <c>maxDepth</c> axes) and the leaf arrays, for single-allocation
        ///     assembly — NumPy's <c>_block_info_recursion</c>.
        /// </summary>
        /// <param name="arrays">The normalized node.</param>
        /// <param name="maxDepth">The list depth.</param>
        /// <param name="resultNdim">The result's rank.</param>
        /// <param name="depth">This node's depth.</param>
        /// <returns>The node's shape, its leaves' windows and its leaves (padded views).</returns>
        /// <exception cref="ValueError">Children whose extents off the join axis differ.</exception>
        private static (long[] shape, List<(long start, long stop)[]> slices, List<NDArray> arrays)
            _block_info_recursion(object arrays, int maxDepth, int resultNdim, int depth)
        {
            if (depth < maxDepth)
            {
                var list = (object[])arrays;
                var childShapes = new long[list.Length][];
                var flatSlices = new List<(long, long)[]>();
                var flatArrays = new List<NDArray>();
                var childSliceLists = new List<(long, long)[]>[list.Length];

                for (int i = 0; i < list.Length; i++)
                {
                    var (shape, slices, arrs) =
                        _block_info_recursion(list[i], maxDepth, resultNdim, depth + 1);
                    childShapes[i] = shape;
                    childSliceLists[i] = slices;
                    flatArrays.AddRange(arrs);
                }

                int axis = resultNdim - maxDepth + depth;
                var (combinedShape, slicePrefixes) = _concatenate_shapes(childShapes, axis);

                // Prepend this level's window to every slice tuple beneath it.
                for (int i = 0; i < list.Length; i++)
                {
                    foreach (var innerSlice in childSliceLists[i])
                    {
                        var full = new (long, long)[innerSlice.Length + 1];
                        full[0] = slicePrefixes[i];
                        System.Array.Copy(innerSlice, 0, full, 1, innerSlice.Length);
                        flatSlices.Add(full);
                    }
                }

                return (combinedShape, flatSlices, flatArrays);
            }
            else
            {
                // We've 'bottomed out' - arrays is a leaf NDArray. Return the
                // slice and the array inside a list to be consistent with the
                // recursive case.
                var arr = AtLeastNdView((NDArray)arrays, resultNdim);
                return ((long[])arr.Shape.dimensions.Clone(),
                        new List<(long, long)[]> { System.Array.Empty<(long, long)>() },
                        new List<NDArray> { arr });
            }
        }

        /// <summary>NumPy's <c>_block_slicing</c>: the single-allocation route - the result allocated once in the leaves'
        ///     result type, then every leaf copied into its window.</summary>
        /// <param name="normalized">The normalized tree.</param>
        /// <param name="listNdim">The list depth.</param>
        /// <param name="resultNdim">The result's rank.</param>
        /// <returns>A new array: F-ordered only when every leaf is F-contiguous and not all are C-contiguous, else C.</returns>
        /// <exception cref="ValueError">Children whose extents off a join axis differ.</exception>
        /// <exception cref="DTypePromotionError">Leaves whose dtypes have no common type.</exception>
        private static NDArray _block_slicing(object normalized, int listNdim, int resultNdim)
        {
            var (shape, slices, leaves) =
                _block_info_recursion(normalized, listNdim, resultNdim, 0);

            // np.result_type(*leaves)'s body for two or more arrays: the promotion by descriptor (one leaf keeps its own
            // dtype, as np.result_type's single-value path would).
            DType dtype = leaves.Count == 1
                ? leaves[0].dtype
                : DTypePromotion.ResultType(leaves.ConvertAll(static leaf => (object)leaf).ToArray());

            // Prefer F only in the case that all input arrays are F (and not
            // all C, i.e. genuinely Fortran-laid-out) — NumPy's _block_slicing.
            bool fOrder = true, cOrder = true;
            for (int i = 0; i < leaves.Count; i++)
            {
                var sh = leaves[i].Shape;
                if (!sh.IsFContiguous) fOrder = false;
                if (!sh.IsContiguous) cOrder = false;
                if (!fOrder && !cOrder) break;
            }

            var retShape = fOrder && !cOrder ? Shape.Adopt(shape, 'F') : Shape.Adopt(shape);
            // fillZeros: false — the slice windows tile the output exactly
            // (validated level-by-level in _concatenate_shapes), so every byte
            // is written below (NumPy uses np.empty for the same reason).
            var result = new NDArray(dtype, retShape, fillZeros: false);

            for (int i = 0; i < leaves.Count; i++)
            {
                var leaf = leaves[i];
                if (leaf.size == 0)
                    continue; // nothing to assign

                var window = slices[i];
                var accessor = new Slice[resultNdim];
                int lead = resultNdim - window.Length; // (Ellipsis,) prefix
                for (int j = 0; j < lead; j++)
                    accessor[j] = Slice.All;
                for (int j = 0; j < window.Length; j++)
                    accessor[lead + j] = new Slice(window[j].start, window[j].stop);

                // dstView is an owning intermediate wrapper over shared storage;
                // release it eagerly like concatenate's general path does.
                using var dstView = result[accessor];
                NDIter.Copy(dstView, leaf);
            }

            return result;
        }
    }
}
