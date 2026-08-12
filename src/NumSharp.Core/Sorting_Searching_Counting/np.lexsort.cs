using System;
using NumSharp.Backends.Sorting;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Perform an indirect stable sort using a sequence of keys — the LAST key is the
        ///     PRIMARY sort key, the second-to-last breaks its ties, and so on (NumPy
        ///     <c>np.lexsort</c>). Returns int64 indices that sort every key line lexicographically.
        /// </summary>
        /// <param name="keys">The k sort keys, all the same shape. Keys are only read.</param>
        /// <param name="axis">Axis to sort along (default -1, the last axis).</param>
        /// <returns>int64 index array of the keys' shape.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html<br></br>
        ///     Port of <c>PyArray_LexSort</c> (item_selection.c): each key is run through a STABLE
        ///     argsort from FIRST to LAST, every pass re-sorting the running permutation — NumSharp's
        ///     radix argsort is stable, so the composition
        ///     <c>perm = take_along_axis(perm, argsort(take_along_axis(key, perm)))</c> reproduces
        ///     NumPy's mechanism pass for pass. Validation order is NumPy's: non-empty keys
        ///     (TypeError "need sequence of keys with len &gt; 0 in lexsort") → same shape
        ///     (ValueError "all keys need to be the same shape") → axis bounds (0-d keys let axis
        ///     0/-1 slip through, NumPy's backwards-compat quirk) → the size ≤ 1 early return
        ///     (a 0-filled int64 array of the keys' shape). NaN keys sort last (stable argsort policy);
        ///     ties across ALL keys keep ascending index order.
        /// </remarks>
        public static NDArray lexsort(NDArray[] keys, int axis = -1)
        {
            if (keys is null || keys.Length == 0)
                throw new TypeError("need sequence of keys with len > 0 in lexsort");
            for (int i = 0; i < keys.Length; i++)
                if (keys[i] is null)
                    throw new ArgumentNullException(nameof(keys), $"lexsort key {i} is null");

            var first = keys[0];
            for (int i = 1; i < keys.Length; i++)
            {
                if (keys[i].ndim != first.ndim)
                    throw new ValueError("all keys need to be the same shape");
                for (int d = 0; d < first.ndim; d++)
                    if (keys[i].shape[d] != first.shape[d])
                        throw new ValueError("all keys need to be the same shape");
            }

            int nd = first.ndim;
            int ax = axis;
            if (nd == 0 && (axis == 0 || axis == -1))
            {
                // NumPy lets axis={-1,0} slip through for 0-d keys (backwards compatibility).
            }
            else
            {
                ax = AxisSort.NormalizeAxis(axis, nd);
            }

            if (nd == 0)
                return NDArray.Scalar(0L);
            if (first.size <= 1)
                return np.zeros(first.Shape.dimensions, NPTypeCode.Int64); // [0] for one element, [] for none

            // First pass over an identity permutation IS the first key's stable argsort;
            // every later key re-sorts the permutation it receives (stably), so the LAST
            // key decides first and earlier keys only break its ties — NumPy's ordering.
            var perm = np.argsort(keys[0], ax);
            for (int j = 1; j < keys.Length; j++)
            {
                var g = np.argsort(np.take_along_axis(keys[j], perm, ax), ax);
                perm = np.take_along_axis(perm, g, ax);
            }
            return perm;
        }

        /// <summary>
        ///     <c>np.lexsort</c> with the keys packed in ONE array, exactly as NumPy reads it as a
        ///     sequence: a (k, …) array contributes its k SUB-ARRAYS as the keys (last row = primary).
        ///     A 1-D input therefore degenerates into N scalar (0-d) keys and returns the 0-d
        ///     <c>0</c> — NumPy's probed quirk, not an error.
        /// </summary>
        /// <param name="keys">Array whose first-axis sub-arrays are the sort keys.</param>
        /// <param name="axis">Axis to sort along (default -1).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html</remarks>
        public static NDArray lexsort(NDArray keys, int axis = -1)
        {
            if (keys is null)
                throw new ArgumentNullException(nameof(keys));
            // a 0-d array is not a sequence; an empty first axis is a zero-length key sequence
            if (keys.ndim == 0 || keys.shape[0] == 0)
                throw new TypeError("need sequence of keys with len > 0 in lexsort");

            var list = new NDArray[keys.shape[0]];
            for (int i = 0; i < list.Length; i++)
                list[i] = keys[i];
            return lexsort(list, axis);
        }
    }
}
