using System;
using System.Collections;
using System.Numerics;

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
            // None, a Python int past every C# integer and a Python list have no literal array of their own here (NumPy's
            // coercion would discover an object array for them): NumPy converts them with THIS array's descriptor, after
            // the index, which SetItemConverted reproduces.
            if (value is null || value is BigInteger || IsPythonListValue(value))
            {
                SetItemConverted(ItemKey(key), value);
                return;
            }
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

        /// <summary>
        ///     Whether a <c>__setitem__</c> value is a Python list or tuple: an <c>object[]</c>, a C# tuple, a <c>List</c> or
        ///     any other enumerable that is not a string, bytes, an <see cref="NDArray"/> or a C# array of a NumSharp dtype
        ///     (an <c>int[]</c> or <c>double[,]</c> stands for an ndarray of that dtype, NumSharp's house map, and keeps the
        ///     array route).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>True for a Python sequence.</returns>
        /// <remarks>The difference is NumPy's: a list stored into one element is the dtype's setitem's to answer (an integer
        ///     dtype raises <c>int()</c>'s TypeError, StringDType stores <c>str(list)</c>), an ndarray the array's
        ///     (<c>setting an array element with a sequence.</c>, <c>str(ndarray)</c>).</remarks>
        private static bool IsPythonListValue(object value)
            => value is System.Runtime.CompilerServices.ITuple
               || (value is IEnumerable && value is not string && value is not byte[] && value is not NDArray
                   && (value is not System.Array array || array.GetType().GetElementType().GetTypeCode() == NPTypeCode.Empty));

        /// <summary>
        ///     <c>__setitem__</c> of a value NumPy converts with this array's descriptor (None, a Python int past every C#
        ///     integer, a Python list): NumPy's <c>array_assign_subscript</c> order — the writeability, the index
        ///     (<c>prepare_index</c>: its entries, its rank, a boolean index's shape, an integer's bounds), the value's
        ///     conversion, then the store, which checks an index array's values.
        /// </summary>
        /// <param name="key">The index tuple (<see cref="ItemKey"/>).</param>
        /// <param name="value">null (None), a <see cref="BigInteger"/>, or a Python sequence (<see cref="IsPythonListValue"/>).</param>
        /// <exception cref="ValueError">The array is read-only (first), or the value does not fit the selection.</exception>
        /// <exception cref="IndexError">The index is invalid (before the value is converted), or an index array's value is
        ///     out of bounds (after it).</exception>
        /// <exception cref="ArgumentNullException">An entry of the key is null.</exception>
        /// <exception cref="TypeError">The value does not convert: <c>int()</c>'s text for None or a list into an integer
        ///     dtype, <c>must be real number, not list</c> for complex.</exception>
        /// <exception cref="OverflowException">A Python int an integer dtype cannot hold (NumPy's OverflowError, LP64
        ///     texts).</exception>
        /// <remarks>
        ///     <para>
        ///     <b>The conversion.</b> A full integer index (NumPy's <c>index_type == HAS_INTEGER</c>: one integer or 0-d integer
        ///     array per axis) packs the value into its one element with the dtype's <see cref="DTypeMeta.setitem"/>
        ///     (<c>array_assign_item</c> → <c>PyArray_Pack</c>), so a list is the setitem's to answer. None and a Python int
        ///     are packed the same way for any selection: <c>PyArray_FromAny(value, descr)</c> of a scalar is that pack into a
        ///     0-d array, which the store broadcasts — before it looks at the selection, so an empty selection still raises
        ///     (<c>int32[False] = None</c> is the TypeError).
        ///     </para>
        ///     <para>
        ///     <b>A list into a selection of several elements</b> is converted with this array's descriptor too — every item
        ///     by its setitem (a Python int range-checked, a str parsed, None NaN or <c>int()</c>'s TypeError) — by the route
        ///     NumPy's assignment takes for the index: a view assignment's <c>PyArray_CopyObject</c>, whose discovery stops at
        ///     the view's rank (<see cref="Backends.UnmanagedStorage.ThrowIfSequenceExceedsDims"/>: <c>a[1:] = [[1, 2, 3]]</c> raises);
        ///     for a <c>StringDType</c> array's fancy index the same CopyObject into MapIter's temporary of the result shape
        ///     (<c>PyDataType_REFCHK</c>: its broadcast text, "could not broadcast input array from shape (2,) into shape
        ///     (2,4)"); for any other fancy index and a mask of this array's shape <c>PyArray_FromAny(value, descr)</c>,
        ///     unlimited. Until 2026-10-29 the list became the array NumSharp's coercion discovers, cast by the store:
        ///     <c>uint8[0:2] = [1, -1]</c> stored 255 where NumPy raises the OverflowError, a list nested deeper than a view
        ///     was broadcast, and None in a list was refused.
        ///     </para>
        ///     <para>
        ///     <b>The order.</b> The index is validated by <see cref="PrepareIndex"/> without its index-array value scan:
        ///     NumPy bounds-checks those values in its MapIter, after the conversion (<c>int32[[0, 99]] = 2**70</c> is the
        ///     OverflowError, <c>float64[[0, 99]] = 2**70</c> the IndexError).
        ///     </para>
        ///     <para>Until 2026-10-29 a Python int past every C# integer raised <c>NotSupportedException</c> ("Unable to resolve
        ///     asanyarray for type BigInteger"), a list into one element of an integer array raised the sequence ValueError and
        ///     into a StringDType element stored <c>'[1 2]'</c>, and None was converted before the index was checked (or not at
        ///     all for an empty selection).</para>
        /// </remarks>
        [NDScoped]
        private void SetItemConverted(object[] key, object value)
        {
            ThrowIfNotWriteable();
            // A null entry is the setter's own C# refusal (ArgumentNullException; NumPy's None is Slice.NewAxis), raised there
            // before the value is looked at.
            if (System.Array.IndexOf(key, null) >= 0)
            {
                this[key] = null;
                return;
            }
            var prepared = PrepareIndex(Shape, StoreIndexEntries(key), scanFancyBounds: false);
            if (prepared.Flags == IndexType.Integer || value is null || value is BigInteger)
            {
                // PyArray_Pack into a 0-d array of this array's own descriptor (the scope reclaims it after the store).
                this[key] = PackForStore(value);
                return;
            }

            // A Python list into a selection of several elements (the remarks): converted with this array's descriptor, by
            // the route NumPy's assignment takes for this index.
            bool fancy = (prepared.Flags & IndexType.Fancy) != 0;
            if (!fancy && prepared.Flags != IndexType.Bool)
            {
                // A view assignment: PyArray_CopyObject discovers the list with at most the view's rank.
                Backends.UnmanagedStorage.ThrowIfSequenceExceedsDims(value, prepared.NewNdim);
            }
            else if (fancy && Storage.IsVString)
            {
                // PyDataType_REFCHK: MapIter allocates the result-shaped temporary and PyArray_CopyObject fills it — before
                // MapIter checks the index values, so the result's shape comes from the prepared index, not from a gather —
                // and the list is bounded by the result's rank and must broadcast to its shape with AssignArray's text.
                long[] resultDims = MapIterResultDims(Shape, prepared);
                Backends.UnmanagedStorage.ThrowIfSequenceExceedsDims(value, resultDims.Length);
                var filled = AnyArrayFromObject(value, Storage.Descr, 'K');
                ThrowIfCannotAssignInto(filled, resultDims);
                this[key] = filled;
                return;
            }
            this[key] = AnyArrayFromObject(value, Storage.Descr, 'K');
        }

        /// <summary>
        ///     NumPy's <c>PyArray_AssignArray</c> check of a converted value against a destination it is copied into: the
        ///     value's leading unit dimensions are dropped while it has more than the destination, then each remaining axis
        ///     must be 1 or the destination's length (right-aligned).
        /// </summary>
        /// <param name="value">The converted value.</param>
        /// <param name="destination">The destination's dimensions.</param>
        /// <exception cref="ValueError"><c>could not broadcast input array from shape (3,) into shape (2,)</c>, NumPy's
        ///     <c>broadcast_strides</c> text (the value's shape after the strip).</exception>
        private static void ThrowIfCannotAssignInto(NDArray value, long[] destination)
        {
            long[] dims = value.Shape.dimensions;
            int strip = 0;
            while (dims.Length - strip > destination.Length && dims[strip] == 1)
                strip++;
            var kept = new long[dims.Length - strip];
            System.Array.Copy(dims, strip, kept, 0, kept.Length);
            bool fits = kept.Length <= destination.Length;
            for (int i = kept.Length - 1, j = destination.Length - 1; fits && i >= 0; i--, j--)
                fits = kept[i] == 1 || kept[i] == destination[j];
            if (!fits)
                throw new ValueError($"could not broadcast input array from shape {PyShapeTuple(kept)} " +
                                     $"into shape {PyShapeTuple(destination)}");
        }

        /// <summary>
        ///     The index tuple <see cref="PrepareIndex"/> validates for a store: the entries the setter reads, in NumPy's terms.
        ///     A key the basic indexer takes (<see cref="TryBasicIndex"/>: a lone string of slice notation holds the WHOLE
        ///     index, <c>"..., 0"</c>) is its parsed slices, an integer slice (<see cref="Slice.Index(long)"/>) given back as
        ///     the integer it is, so that a full integer index is classified as one; any other key is
        ///     <see cref="NormalizeIndexInputs"/>' form of a copy (the setter re-normalizes the caller's own key; the copy
        ///     dates from when that method rewrote its argument in place, until 2026-10-29).
        /// </summary>
        /// <param name="key">The caller's index tuple (not modified).</param>
        /// <returns>A new tuple for <see cref="PrepareIndex"/>.</returns>
        /// <exception cref="ArgumentException">A lone string is not slice notation (the basic indexer's own refusal).</exception>
        private static object[] StoreIndexEntries(object[] key)
        {
            if (!TryBasicIndex(key, out Slice[] slices))
                return NormalizeIndexInputs((object[])key.Clone());
            var entries = new object[slices.Length];
            for (int i = 0; i < slices.Length; i++)
                // (object) on the integer branch: Slice converts implicitly from an integer, so without it the conditional's
                // natural type is Slice and the integer would be turned back into a slice.
                entries[i] = slices[i] is { IsIndex: true, Start: long index } ? (object)index : slices[i];
            return entries;
        }

        /// <summary>
        ///     A new 0-d array of this array's descriptor holding <paramref name="value"/> packed by the dtype's
        ///     <see cref="DTypeMeta.setitem"/> (NumPy's <c>PyArray_Pack</c>).
        /// </summary>
        /// <param name="value">The Python value (None, an int, a list, ...).</param>
        /// <returns>The 0-d array; the caller owns it.</returns>
        /// <exception cref="TypeError">See <see cref="SetItemConverted"/>.</exception>
        /// <exception cref="OverflowException">See <see cref="SetItemConverted"/>.</exception>
        /// <exception cref="ValueError">The dtype's setitem refuses the value (a sequence into a fixed-width text or float
        ///     dtype).</exception>
        private unsafe NDArray PackForStore(object value)
        {
            var descr = Storage.Descr;
            var packed = new NDArray(descr, Shape.Scalar);
            try
            {
                // The element is the 0-d array's only one; a StringDType element allocates through the array's own buffer.
                descr.Meta.setitem(descr, value, (byte*)packed.Storage.Address, packed.Storage.InternalArray);
                return packed;
            }
            catch
            {
                // The pack refused the value: the temporary never reaches the caller.
                packed.Dispose();
                throw;
            }
        }

        #endregion
    }
}
