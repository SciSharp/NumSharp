using System;
using System.Collections;

namespace NumSharp
{
    /// <summary>
    /// Container protocol implementation for NDArray.
    /// Provides Python-compatible container protocol methods:
    /// __contains__, __hash__, __len__, __iter__, __getitem__, __setitem__
    /// </summary>
    public partial class NDArray
    {
        #region Contains (__contains__)

        /// <summary>
        /// Returns true if value is found in the array (linear search).
        /// Equivalent to NumPy's <c>value in arr</c>.
        /// </summary>
        /// <param name="value">Value to search for.</param>
        /// <returns>True if value exists in the array.</returns>
        /// <remarks>
        /// This is a linear O(n) search. For sorted arrays, consider using np.searchsorted.
        /// NaN handling: NaN == NaN is false in IEEE 754, so Contains(float.NaN) returns false
        /// for arrays containing NaN. Use np.any(np.isnan(arr)) to check for NaN.
        /// </remarks>
        /// <example>
        /// <code>
        /// var arr = np.array(new[] { 1, 2, 3, 4, 5 });
        /// arr.Contains(3);  // true
        /// arr.Contains(10); // false
        /// </code>
        /// </example>
        [NDScoped] // bool boundary: reclaims asanyarray's fresh conversion while an NDArray-typed value passes through untracked
        public bool Contains(object value)
        {
            if (value is null)
                return false;

            // NumPy: (self == el).any()
            // NumPy compares the raw object directly, not an array-converted version.
            // When comparing incompatible types (e.g., int array with string),
            // NumPy returns False for each element.

            // A string searched in a numeric array: NumPy's == has no loop for text against numbers and answers False
            // everywhere (its failed-comparison workaround), so the answer is False without building anything. (A fixed-
            // width string array compares text: np.asanyarray reads the string as NumPy's 0-d str array and the == below
            // runs the string comparison loop — a bytes array searched for a str is False everywhere, as in NumPy.)
            if (value is string && typecode != NPTypeCode.Char && typecode != NPTypeCode.String)
                return false;

            // Convert value to NDArray for comparison
            // NumPy: (self == el).any()
            // If shapes are incompatible for broadcasting, let the exception propagate
            // (matches NumPy behavior: ValueError for shape mismatch)
            // `scalar` is NOT `using`-bound — np.asanyarray may return `value` itself
            // when it was already an NDArray (would dispose caller's array, Rule 1).
            // NumSharp's Char arrays are character vectors: a string is searched as its characters (the legacy semantics
            // these NumSharp-only arrays keep), not as NumPy's 0-d text array (np.asanyarray("abc") is <U3), which a char
            // array has no comparison with.
            var scalar = value is string text && typecode == NPTypeCode.Char ? NDArray.FromString(text) : np.asanyarray(value);

            // `comparison` IS using-bound: operator== always allocates a fresh bool
            // NDArray sized broadcast(this, scalar); after the engine's Any reads it, the buffer
            // is dead. This is the hot path for `value in arr` so atomic release
            // matters in tight loops.
            using var comparison = this == scalar;
            return comparison.TensorEngine.Any(comparison);
        }

        /// <summary>
        /// Python-compatible __contains__ method.
        /// Equivalent to <see cref="Contains(object)"/>.
        /// </summary>
        /// <param name="value">Value to search for.</param>
        /// <returns>True if value exists in the array.</returns>
        /// <remarks>
        /// This method exists for Python interoperability and naming consistency.
        /// In Python: <c>value in arr</c> calls <c>arr.__contains__(value)</c>
        /// </remarks>
        public bool __contains__(object value) => Contains(value);

        #endregion

        #region Hash (__hash__)

        /// <summary>
        /// NDArray is unhashable because it is mutable.
        /// </summary>
        /// <returns>Never returns - always throws.</returns>
        /// <exception cref="NotSupportedException">Always thrown.</exception>
        /// <remarks>
        /// <para>
        /// NumPy arrays are unhashable because they are mutable. If an array were used
        /// as a dictionary key and then modified, the hash would change, breaking the
        /// dictionary's invariants.
        /// </para>
        /// <para>
        /// This matches NumPy behavior:
        /// <code>
        /// >>> hash(np.array([1, 2, 3]))
        /// TypeError: unhashable type: 'numpy.ndarray'
        /// </code>
        /// </para>
        /// <para>
        /// Workarounds:
        /// <list type="bullet">
        /// <item>Use <c>arr.tobytes()</c> as a hashable key (immutable snapshot)</item>
        /// <item>Use <c>ReferenceEqualityComparer.Instance</c> for identity-based dictionaries</item>
        /// <item>Convert to tuple: <c>tuple(arr.ToArray())</c></item>
        /// </list>
        /// </para>
        /// </remarks>
        public override int GetHashCode()
        {
            throw new NotSupportedException(
                "NDArray is unhashable because it is mutable. " +
                "Workarounds: use arr.tobytes() as key, use ReferenceEqualityComparer.Instance, " +
                "or convert to an immutable type.");
        }

        /// <summary>
        /// Python-compatible __hash__ method.
        /// NDArray is unhashable because it is mutable.
        /// </summary>
        /// <returns>Never returns - always throws.</returns>
        /// <exception cref="NotSupportedException">Always thrown.</exception>
        /// <remarks>
        /// This method exists for Python interoperability and naming consistency.
        /// In Python: <c>hash(arr)</c> calls <c>arr.__hash__()</c>
        ///
        /// NumPy behavior:
        /// <code>
        /// >>> arr = np.array([1, 2, 3])
        /// >>> hash(arr)
        /// TypeError: unhashable type: 'numpy.ndarray'
        /// </code>
        /// </remarks>
        public int __hash__()
        {
            throw new NotSupportedException(
                "NDArray is unhashable because it is mutable. " +
                "Workarounds: use arr.tobytes() as key, use ReferenceEqualityComparer.Instance, " +
                "or convert to an immutable type.");
        }

        #endregion

        #region Length (__len__)

        /// <summary>
        /// Python-compatible __len__ method.
        /// Returns the length of the first dimension (like Python's len()).
        /// </summary>
        /// <returns>Length of the first dimension, or 1 for scalars.</returns>
        /// <remarks>
        /// This matches NumPy behavior:
        /// <code>
        /// >>> len(np.array([1, 2, 3]))
        /// 3
        /// >>> len(np.array([[1, 2], [3, 4]]))
        /// 2  # First dimension
        /// >>> len(np.array(5))
        /// TypeError: len() of unsized object
        /// </code>
        ///
        /// Note: For scalars (0-d arrays), NumPy raises <c>TypeError: len() of unsized object</c>
        /// and NumSharp raises the same, message-verbatim (oracle-gated by the instance tier).
        /// Use <see cref="size"/> for total element count.
        /// </remarks>
        public long __len__()
        {
            if (ndim == 0)
                throw new TypeError("len() of unsized object");
            return this.Shape.dimensions[0];
        }

        #endregion

        #region Iteration (__iter__)

        /// <summary>
        /// Python-compatible __iter__ method.
        /// Returns an enumerator over the first axis.
        /// </summary>
        /// <returns>Enumerator yielding NDArray slices along the first axis.</returns>
        /// <remarks>
        /// This matches NumPy behavior:
        /// <code>
        /// >>> for row in np.array([[1, 2], [3, 4]]):
        /// ...     print(row)
        /// [1 2]
        /// [3 4]
        /// </code>
        ///
        /// For 1-D arrays, iterates over scalar elements.
        /// For N-D arrays, iterates over (N-1)-D slices.
        /// </remarks>
        public IEnumerator __iter__() => GetEnumerator();

        #endregion

        #region Indexing (__getitem__, __setitem__)

        /// <summary>
        ///     NumPy's <c>ndarray.__getitem__(key, /)</c>: the same as <c>a[key]</c> (<see cref="this[object[]]"/>).
        /// </summary>
        /// <param name="key">The index, as Python passes it to <c>__getitem__</c>: one entry (an integer, a slice-notation
        ///     string such as <c>"1:3"</c> or <c>"..., 0"</c>, a <see cref="Slice"/>, an index array, a mask, a C# bool) or
        ///     the whole index tuple — an <c>object[]</c> / <see cref="Slice"/>[] (each element an entry) or a C# tuple
        ///     (<c>a.__getitem__((1, "0:2"))</c>). A typed C# array other than <see cref="Slice"/>[] (<c>int[]</c>,
        ///     <c>bool[]</c>) is ONE entry, NumPy's list: an index array or a mask.</param>
        /// <returns>The selection: a view for an index of integers and slices (0-d for one integer per axis), a new array
        ///     for an advanced index.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <c>null</c> (NumPy's <c>a[None]</c> is
        ///     <see cref="Slice.NewAxis"/>).</exception>
        /// <exception cref="IndexError">The index is invalid for this array (NumPy's <c>prepare_index</c> texts).</exception>
        /// <remarks>One member, NumPy's signature. An integer array used to bind a <c>params int[]</c> form that read it
        ///     as COORDINATES (<c>a.__getitem__(1, 2)</c>); NumPy reads a list key as an index array, and so does this
        ///     member — pass a tuple for coordinates (<c>a.__getitem__((1, 2))</c>).</remarks>
        public NDArray __getitem__(object key) => this[ItemKey(key)];

        /// <summary>
        ///     NumPy's <c>ndarray.__setitem__(key, value, /)</c>: stores exactly what <c>a[key] = value</c> stores, for any
        ///     value kind (the indexer's setter takes an <see cref="NDArray"/>).
        /// </summary>
        /// <param name="key">The index, as for <see cref="__getitem__(object)"/>.</param>
        /// <param name="value">Value to set — an NDArray, a C# scalar (NumPy's Python scalar), a C# array or list, or null
        ///     (Python's <c>None</c>); see <see cref="SetItemValue"/>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">The index is invalid for this array (NumPy's texts).</exception>
        /// <exception cref="TypeError"><paramref name="value"/> is null (Python's <c>None</c>) and this is an integer (or
        ///     <c>decimal</c>) array: <c>int(None)</c>'s text, after the index's bounds check. A float array stores NaN,
        ///     complex <c>nan+nanj</c>, bool <c>False</c>, text <c>'None'</c> or a <c>StringDType</c> NA object.</exception>
        /// <exception cref="OverflowException">A C# integer outside this array's integer dtype (NumPy's OverflowError).</exception>
        /// <exception cref="ValueError">The array is read-only, the value does not broadcast to the selection, or does not
        ///     parse into its dtype.</exception>
        public void __setitem__(object key, object value)
        {
            var arr = SetItemValue(value);
            try
            {
                this[ItemKey(key)] = arr;
            }
            finally
            {
                // The 0-d array made for a C# scalar is this call's own temporary (the store copies from it).
                if (!ReferenceEquals(arr, value))
                    arr?.Dispose();
            }
        }

        /// <summary>
        ///     The indexer's tuple for a <c>__getitem__</c> / <c>__setitem__</c> key: an <c>object[]</c> (a
        ///     <see cref="Slice"/>[] and an <see cref="NDArray"/>[] by array covariance) is the tuple itself, any other key is
        ///     one entry (a C# tuple entry is spread by the indexer, as Python reads <c>a[(1, 2)]</c> as <c>a[1, 2]</c>).
        /// </summary>
        /// <param name="key">The key Python would pass.</param>
        /// <returns>The index tuple.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <c>null</c>.</exception>
        internal static object[] ItemKey(object key) => key switch
        {
            null => throw new ArgumentNullException(nameof(key)),
            object[] tuple => tuple,
            _ => new[] { key },
        };

        /// <summary>
        ///     The value of a <c>__setitem__</c> call as NumPy's <c>ndarray.__setitem__</c> receives it, so that
        ///     <c>a.__setitem__(i, v)</c> stores exactly what <c>a[i] = v</c> stores: an <see cref="NDArray"/> as it is; a C#
        ///     number, bool or string as the Python scalar it stands for (<see cref="FromPythonOperand"/>: NEP 50 weak, stored
        ///     by setitem's range-checked pack — <c>__setitem__(0, 300)</c> into an int8 array raises NumPy's OverflowError
        ///     where a plain 0-d int32 array would wrap to 44); a C# <see cref="Half"/> or <see cref="char"/> as the NumPy
        ///     scalar it stands for; any other object (a C# array, a list) through <c>np.asanyarray</c>.
        /// </summary>
        /// <param name="value">The value, or null for Python's <c>None</c> (passed on as null: every store converts it into
        ///     the array's dtype — NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>, a <c>StringDType</c> NA object — or
        ///     raises <c>int(None)</c>'s TypeError for an integer array).</param>
        /// <returns>The value as an array, or null.</returns>
        private static NDArray SetItemValue(object value)
            => value is null ? null : FromPythonOperand(value);

        #endregion
    }
}
