using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Assemble an nd-array from nested lists of blocks.
        ///     Blocks in the innermost lists are concatenated (see
        ///     <see cref="concatenate(NDArray[], int?, NDArray, NPTypeCode?, string)"/>)
        ///     along the last dimension (-1), then these are concatenated along
        ///     the second-last dimension (-2), and so on until the outermost
        ///     list is reached.
        ///     Blocks can be of any dimension, but will not be broadcasted using
        ///     the normal rules. Instead, leading axes of size 1 are inserted,
        ///     to make <c>block.ndim</c> the same for all blocks.
        /// </summary>
        /// <param name="arrays">
        ///     Nested "lists" of blocks. The C# mapping of NumPy's nested Python
        ///     lists:
        ///     <list type="bullet">
        ///     <item><see cref="NDArray"/> — a leaf block.</item>
        ///     <item>numeric scalar (int, double, bool, …) — a 0-d leaf.</item>
        ///     <item>any 1-D <see cref="Array"/> (<c>object[]</c>, <c>NDArray[]</c>,
        ///     <c>int[]</c>, jagged <c>int[][]</c>, …) or non-generic
        ///     <see cref="IList"/> (<c>List&lt;T&gt;</c>) — a nested LIST whose
        ///     elements are classified recursively (mirrors a Python list).</item>
        ///     <item>rank ≥ 2 rectangular arrays (<c>int[,]</c>, …) — a leaf,
        ///     converted via <see cref="asanyarray(in object, Type)"/>.</item>
        ///     <item>tuples (<see cref="ITuple"/>) — rejected with
        ///     <see cref="TypeError"/>, matching NumPy ("np.block does not allow
        ///     implicit conversion from tuple to ndarray").</item>
        ///     </list>
        ///     If passed a single NDArray or scalar (a nested list of depth 0),
        ///     a copy is returned (matching NumPy 2.x behavior).
        /// </param>
        /// <returns>
        ///     The array assembled from the given blocks. The dimensionality of
        ///     the output is equal to the greatest of the dimensionality of all
        ///     the inputs and the depth to which the input list is nested.
        /// </returns>
        /// <exception cref="ValueError">
        ///     If list depths are mismatched — for instance <c>[[a, b], c]</c> is
        ///     illegal and should be spelt <c>[[a, b], [c]]</c> — or if lists are
        ///     empty — for instance <c>[[a, b], []]</c>.
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.block.html</remarks>
        public static NDArray block(object arrays)
        {
            if (arrays is null)
                throw new ArgumentNullException(nameof(arrays));

            var (normalized, listNdim, resultNdim, finalSize) = _block_setup(arrays);

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

        // ------------------------------------------------------------------
        //  Port of numpy/_core/shape_base.py — same helper names, same flow.
        // ------------------------------------------------------------------

        /// <summary>Sentinel for Python's <c>None</c> marking an empty list at the bottom of the nesting.</summary>
        private const int _blockEmptyListMarker = int.MinValue;

        /// <summary>Convert an index chain <c>[0, 1]</c> into <c>"arrays[0][1]"</c> (empty-list markers skipped).</summary>
        private static string _block_format_index(int[] index)
        {
            var sb = new System.Text.StringBuilder("arrays");
            foreach (var i in index)
                if (i != _blockEmptyListMarker)
                    sb.Append('[').Append(i).Append(']');
            return sb.ToString();
        }

        private static int[] _block_append_index(int[] index, int i)
        {
            var next = new int[index.Length + 1];
            Array.Copy(index, next, index.Length);
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
            if (arrays is Array clrArray)
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

            // We've 'bottomed out' - arrays is either a scalar or an array
            var nd = arrays as NDArray ?? asanyarray(arrays);
            return (parentIndex, nd.ndim, nd.size, nd);
        }

        /// <summary>
        ///     Ensures <paramref name="a"/> has at least <paramref name="ndim"/>
        ///     dimensions by prepending ones to its shape as necessary — always
        ///     as a VIEW (NumPy's <c>array(a, ndmin=ndim, copy=None)</c>).
        /// </summary>
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
                    dims[pad + i] = a.shape[i];
                return a.reshape(dims);
            }

            while (a.ndim < ndim)
                a = np.expand_dims(a, 0);
            return a;
        }

        /// <summary>
        ///     Returns <c>(normalized, list_ndim, result_ndim, final_size)</c> —
        ///     NumPy's <c>_block_setup</c>.
        /// </summary>
        private static (object normalized, int listNdim, int resultNdim, long finalSize)
            _block_setup(object arrays)
        {
            var (bottomIndex, arrNdim, finalSize, normalized) =
                _block_check_depths_match(arrays, Array.Empty<int>());
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
        private static NDArray _block(object arrays, int maxDepth, int resultNdim, int depth)
        {
            if (depth < maxDepth)
            {
                var list = (object[])arrays;
                var arrs = new NDArray[list.Length];
                for (int i = 0; i < list.Length; i++)
                    arrs[i] = _block(list[i], maxDepth, resultNdim, depth + 1);
                var joined = np.concatenate(arrs, -(maxDepth - depth));

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

            // We've 'bottomed out' - arrays is a leaf NDArray
            return AtLeastNdView((NDArray)arrays, resultNdim);
        }

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
            Array.Copy(firstShape, shape, ndim);
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
                        Array.Copy(innerSlice, 0, full, 1, innerSlice.Length);
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
                return ((long[])arr.shape.Clone(),
                        new List<(long, long)[]> { Array.Empty<(long, long)>() },
                        new List<NDArray> { arr });
            }
        }

        private static NDArray _block_slicing(object normalized, int listNdim, int resultNdim)
        {
            var (shape, slices, leaves) =
                _block_info_recursion(normalized, listNdim, resultNdim, 0);

            var dtype = leaves.Count == 1
                ? leaves[0].GetTypeCode
                : np.result_type(leaves.ToArray());

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

            var retShape = fOrder && !cOrder ? new Shape(shape, 'F') : new Shape(shape);
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
