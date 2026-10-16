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
        /// <returns>int64 index array of the keys' shape.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
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
        /// <exception cref="TypeError">No key (<c>need sequence of keys with len &gt; 0 in lexsort</c>).</exception>
        /// <exception cref="ValueError">Keys of different shapes (<c>all keys need to be the same shape</c>).</exception>
        /// <exception cref="AxisError">The axis is out of bounds for the keys' rank.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> key line of two or more elements holding an NA whose NA object
        ///     is neither NaN-like nor a string (<c>Cannot compare null that is not a nan-like value</c>). Text keys sort by
        ///     their text: fixed-width ones (<c>S</c> / <c>U</c>) with the stable string argsort, <c>StringDType</c> ones with
        ///     NumPy's timsort (two adjacent NaN-like NAs of a line come out reversed, as in NumPy).</exception>
        [NDScoped]
        public static NDArray lexsort(NDArray[] keys, int axis = -1)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(keys))
                return NDMatrixBoundary.Plain((keys, axis), static s => lexsort(s.keys, s.axis));
            // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(keys))
                return NDCharArrayBoundary.Plain((keys, axis), static s => lexsort(s.keys, s.axis));
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
                    if (keys[i].Shape.dimensions[d] != first.Shape.dimensions[d])
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
                return np.zeros(Shape.Adopt(first.Shape.dimensions), NPTypeCode.Int64); // [0] for one element, [] for none

            // First pass over an identity permutation IS the first key's stable argsort;
            // every later key re-sorts the permutation it receives (stably), so the LAST
            // key decides first and earlier keys only break its ties — NumPy's ordering.
            // Every pass asks for the STABLE kind: NumPy's PyArray_LexSort runs each key's NPY_STABLESORT argsort, and a
            // text key's default kind would be NumPy's introsort, whose order of equal strings is not the stable one (a
            // numeric key's argsort is one stable order for every kind).
            // A StringDType key has no legacy stable argsort: NumPy runs npy_atimsort with the dtype's compare over the
            // running index array, whose run detection reverses adjacent NaN-like NAs — VStringSort.LexPass re-sorts the
            // permutation in place exactly so (the first key's pass starts from the identity, as NumPy's index array does).
            NDArray perm;
            if (Backends.Strings.NDStringUfuncs.IsVString(keys[0]))
            {
                perm = IdentityPermutation(first, ax);
                VStringSort.LexPass(keys[0], perm, ax);
            }
            else
                perm = np.argsort(keys[0], ax, kind: "stable");
            for (int j = 1; j < keys.Length; j++)
            {
                if (Backends.Strings.NDStringUfuncs.IsVString(keys[j]))
                {
                    VStringSort.LexPass(keys[j], perm, ax);
                    continue;
                }
                var g = np.argsort(np.take_along_axis(keys[j], perm, ax), ax, kind: "stable");
                perm = np.take_along_axis(perm, g, ax);
            }
            return perm;
        }

        /// <summary>
        ///     NumPy's initial lexsort index array (<c>iptr[i] = i</c> along the axis): a fresh, writeable, C-contiguous int64
        ///     array of <paramref name="like"/>'s dimensions holding each element's index along <paramref name="axis"/>.
        /// </summary>
        /// <param name="like">The keys' shape donor (at least one dimension).</param>
        /// <param name="axis">The normalised axis.</param>
        /// <returns>The identity permutation of every line.</returns>
        private static NDArray IdentityPermutation(NDArray like, int axis)
        {
            var dims = like.Shape.dimensions;
            var lineShape = new long[dims.Length];
            for (int d = 0; d < dims.Length; d++)
                lineShape[d] = d == axis ? dims[d] : 1;
            // arange along the axis, broadcast over the other axes and materialized (the pass rewrites it in place).
            using var line = np.arange(dims[axis]).reshape(Shape.Adopt(lineShape));
            using var broadcast = np.broadcast_to(line, Shape.Adopt((long[])dims.Clone()));
            return broadcast.copy();
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
        /// <exception cref="TypeError">A 0-d array or an empty first axis (<c>need sequence of keys with len &gt; 0 in
        ///     lexsort</c>).</exception>
        /// <exception cref="AxisError">The axis is out of bounds for a key's rank.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> key line holding a None-like NA (see
        ///     <see cref="lexsort(NDArray[], int)"/>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.
        /// </remarks>
        public static NDArray lexsort(NDArray keys, int axis = -1)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(keys))
                return NDMatrixBoundary.Plain((keys, axis), static s => lexsort(s.keys, s.axis));
            // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(keys))
                return NDCharArrayBoundary.Plain((keys, axis), static s => lexsort(s.keys, s.axis));
            // A string array is a sequence of text keys, each sorted by its text (the key list below).
            if (keys is null)
                throw new ArgumentNullException(nameof(keys));
            // a 0-d array is not a sequence; an empty first axis is a zero-length key sequence
            if (keys.ndim == 0 || keys.Shape.dimensions[0] == 0)
                throw new TypeError("need sequence of keys with len > 0 in lexsort");

            var list = new NDArray[keys.Shape.dimensions[0]];
            for (int i = 0; i < list.Length; i++)
                list[i] = keys[i];
            return lexsort(list, axis);
        }
    }
}
