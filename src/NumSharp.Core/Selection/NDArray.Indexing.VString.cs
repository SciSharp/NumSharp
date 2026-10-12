using System;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    public partial class NDArray
    {
        // =============================================================================================================
        // Advanced (fancy / boolean) indexing of NEP 55 StringDType arrays.
        //
        // A StringDType element is a 16-byte cell tied to its buffer's allocator, so neither the fixed-width unit route
        // nor any raw element kernel may move it. A GET runs the same index on the array's CELL VIEW (the cells typed as
        // complex128, this array's exact shape — so every index error names this array's shapes) and duplicates the picked
        // cells into a fresh buffer (NDVStringMove.Gather). A SET runs the index on MARKS (an int64 array of this shape) with
        // the value's element numbers as the value, so the numeric route applies NumPy's broadcast and value rules, then
        // NDVStringMove.ScatterMarks frees the old strings and duplicates the assigned ones.
        // =============================================================================================================

        /// <summary>
        ///     <c>this[indices]</c> GET for a <c>StringDType</c> array whose (normalized) index has an advanced component.
        /// </summary>
        /// <param name="normalized">The normalized index tuple.</param>
        /// <returns>The selected array (a fresh buffer with its own allocator; a view when the selection is one).</returns>
        /// <exception cref="Exception">The numeric route's indexing errors for this array's shape (IndexError, ValueError, …).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's string <c>MemoryError</c> texts.</exception>
        private NDArray VStringFetchIndices(object[] normalized)
            => NDVStringMove.Gather(this, cells => cells.FetchIndices((object[])normalized.Clone()));

        /// <summary>
        ///     <c>this[indices] = value</c> for a <c>StringDType</c> array whose (normalized) index has an advanced component.
        /// </summary>
        /// <param name="normalized">The normalized index tuple.</param>
        /// <param name="value">The value (text, numbers or another <c>StringDType</c> array; converted to this descriptor).</param>
        /// <exception cref="ValueError">This array is read-only, or the value does not broadcast to the selection.</exception>
        /// <exception cref="Exception">The numeric route's indexing errors, and the value conversion's errors.</exception>
        private void VStringSetIndices(object[] normalized, NDArray value)
            => VStringAdvancedSet(value, (marks, index) => marks.SetIndices((object[])normalized.Clone(), index));

        /// <summary>
        ///     An advanced SET on this <c>StringDType</c> array. The value is converted to THIS array's descriptor the way NumPy's
        ///     setitem converts it (<c>U</c> / <c>S</c> text, numbers as their <c>str()</c>, another <c>StringDType</c> with its
        ///     NA rules); <paramref name="onMarks"/> runs the same selection on MARKS (an <c>int64</c> array of this shape, -1 =
        ///     not written) with the value's flat element numbers as the value, and the marked elements receive the strings.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="onMarks">The assignment, applied to (marks, value element numbers of the value's shape).</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="ValueError">This array is read-only, or the value does not broadcast to the selection.</exception>
        /// <exception cref="Exception">The selection's indexing errors, and the value conversion's errors.</exception>
        /// <remarks>
        ///     <b>When the value is converted</b> (NumPy's <c>array_assign_subscript</c>, as the fixed-width route): a SCALAR
        ///     text value is converted before the index is resolved, so its conversion error wins; any other value is converted
        ///     after the index and broadcast checks, and not at all when the selection is empty. A value that already has this
        ///     array's descriptor is used as it is (copied first only when it may share memory with this array).
        /// </remarks>
        private void VStringAdvancedSet(NDArray value, Action<NDArray, NDArray> onMarks)
        {
            ThrowIfNotWriteable();
            if (value is null)
                throw new ArgumentNullException(nameof(value));

            // A C# literal is NumPy's Python scalar: setitem stores it (the NA object, or its str(); refused under
            // coerce=False) BEFORE the index is resolved, as NumPy packs a scalar first.
            NDArray converted = NDArray.PackForVStringStore(value, dtype);
            if (ReferenceEquals(converted, value))
                converted = null;
            try
            {
                var current = converted ?? value;
                bool same = current.Storage.IsVString && current.dtype.Equals(dtype);
                if (!same && current.typecode == NPTypeCode.String && current.ndim == 0)
                {
                    // A 0-d text array converts before the index is resolved, as the fixed-width route does.
                    var text = ConvertForVStringStore(current);
                    converted?.Dispose();
                    converted = text;
                    current = text;
                    same = true;
                }
                var source = same ? current : null;

                using var marks = Backends.Strings.NDVStringMove.NewMarks(this);
                using (var index = Backends.Strings.NDVStringMove.ValueIndex(source ?? current))
                    onMarks(marks, index); // NumPy's index, broadcast and value rules, on the marks

                if (source is null)
                {
                    // Converted while the selection is written: only when the selection holds an element.
                    if (marks.size == 0)
                        return;
                    using (var top = np.amax(marks))
                        if (Convert.ToInt64(top.GetAtIndex(0)) < 0)
                            return;
                    var cast = ConvertForVStringStore(current);
                    converted?.Dispose();
                    converted = cast;
                    source = cast;
                }
                Backends.Strings.NDVStringMove.ScatterMarks(this, marks, source);
            }
            finally
            {
                converted?.Dispose();
            }
        }

        /// <summary>
        ///     A value converted to this <c>StringDType</c> array's descriptor for a store (NumPy's setitem conversion: an
        ///     unsafe cast, so a number becomes its <c>str()</c> and an NA value of another descriptor follows the NA rules).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>A fresh array of this descriptor (the caller disposes it).</returns>
        /// <exception cref="Exception">The cast's errors (a non-UTF-8 <c>S</c> value, a non-text value under
        ///     <c>coerce=False</c>).</exception>
        private NDArray ConvertForVStringStore(NDArray value)
            => value.astype(dtype, true, 'K', "unsafe");
    }
}
