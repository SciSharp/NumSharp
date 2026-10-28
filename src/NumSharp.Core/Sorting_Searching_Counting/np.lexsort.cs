using System;

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
        ///     Port of <c>PyArray_LexSort</c> (item_selection.c), the engine's
        ///     <see cref="TensorEngine.LexSort(NDArray[], int)"/> (the np-layer dtype plan, phase PE, E5): each key is run
        ///     through a STABLE argsort from FIRST to LAST, every pass re-sorting the running permutation — NumSharp's
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
        /// <exception cref="ArgumentNullException">A key is null (<c>lexsort key 1 is null</c>).</exception>
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
            // NumPy's lexsort is PyArray_LexSort, the engine's: TensorEngine.LexSort (the first key's engine, or the default
            // one when there is none, so a null or empty list still meets the body's checks).
            return EngineOf(keys is { Length: > 0 } ? keys[0] : null).LexSort(keys, axis);
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
