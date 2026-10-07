using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
    public partial class NDArray
    {
        /// <summary>
        ///     Throws if this NDArray is not writeable (e.g., broadcast arrays).
        ///     NumPy raises: ValueError: assignment destination is read-only
        /// </summary>
        [MethodImpl(Inline)]
        private void ThrowIfNotWriteable()
        {
            NumSharpException.ThrowIfNotWriteable(Shape);
        }

        /// <summary>
        ///     Used to perform selection based on given indices.
        /// </summary>
        /// <param name="dims">The pointer to the dimensions</param>
        /// <param name="ndims">The count of longs in <paramref name="dims"/></param>
        public unsafe NDArray this[long* dims, int ndims]
        {
            get => new NDArray(Storage.GetData(dims, ndims));
            set
            {
                ThrowIfNotWriteable();
                AssignThroughView(Storage.GetData(dims, ndims), value);
            }
        }

        /// <summary>
        ///     Used to perform selection based on a selection indices.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        public NDArray this[params NDArray<int>[] selection]
        {
            get => typecode == NPTypeCode.String
                // A string element has no element kernel: the gather runs on the unit view (NDArray.Indexing.Strings.cs).
                ? StringFetchIndices(selection.Select(array => (NDArray)array).ToArray())
                : FetchIndices(this, selection.Select(array => (NDArray)array).ToArray(), null, true);
            set
            {
                ThrowIfNotWriteable();
                if (typecode == NPTypeCode.String)
                {
                    StringSetIndices(selection.Select(array => (NDArray)array).ToArray(), value);
                    return;
                }
                SetIndices(this, selection, value);
            }
        }

        /// <summary>
        ///     Slice the array with Python slice notation like this: ":, 2:7:1, ..., np.newaxis"
        /// </summary>
        /// <param name="slice">A string containing slice notations for every dimension, delimited by comma</param>
        /// <returns>A sliced view</returns>
        public NDArray this[string slice]
        {
            get
            {
                return new NDArray(Storage.GetView(Slice.ParseSlices(slice)));
            }
            set
            {
                ThrowIfNotWriteable();
                AssignThroughView(Storage.GetView(Slice.ParseSlices(slice)), value);
            }
        }


        /// <summary>
        ///     Slice the array with Python slice notation like this: ":, 2:7:1, ..., np.newaxis"
        /// </summary>
        /// <param name="slice">A string containing slice notations for every dimension, delimited by comma</param>
        /// <returns>A sliced view</returns>
        public NDArray this[params Slice[] slice]
        {
            get
            {
                return new NDArray(Storage.GetView(slice));
            }
            set
            {
                ThrowIfNotWriteable();
                AssignThroughView(Storage.GetView(slice), value);
            }
        }

        /// <summary>
        ///     Assigns <paramref name="value"/> to every element of <paramref name="view"/> (a basic-index view of this
        ///     array) and releases the view's own reference afterwards.
        /// </summary>
        /// <param name="view">The view storage the index selected: it shares this array's buffer, except an EMPTY
        ///     selection (<c>a["4:"]</c> on 4 elements), which <see cref="UnmanagedStorage.GetView(Slice[])"/> backs with a
        ///     fresh zero-length pooled buffer.</param>
        /// <param name="value">The value (a C# literal is packed by NumPy's rules inside <c>SetData</c>, an empty
        ///     selection included).</param>
        /// <exception cref="ValueError">The value does not broadcast to the selection, or a sequence into one element.</exception>
        /// <exception cref="OverflowException">A C# literal out of this dtype's range.</exception>
        /// <exception cref="TypeError">A complex literal into a real dtype.</exception>
        /// <remarks>
        ///     The view is wrapped in an <see cref="NDArray"/> for the assignment and disposed after it, exceptions included:
        ///     the wrapper takes one counted reference on the view's slice and gives it back. A bare view storage was never
        ///     released, which stranded the empty selection's fresh buffer (one pooled buffer per <c>a[5:] = v</c>, found by
        ///     the zero-leak gate's store cases); a non-empty view shares the parent's buffer and is unaffected.
        /// </remarks>
        private static void AssignThroughView(UnmanagedStorage view, NDArray value)
        {
            using var target = new NDArray(view);
            target.Storage.SetData(value, new int[0]);
        }

        ///// <summary>
        /////     todo: doc
        ///// </summary>
        ///// <param name="slice">A string containing slice notations for every dimension, delimited by comma</param>
        ///// <returns>A sliced view</returns>
        //public NDArray this[params IIndex[] slice] //TODO IIndex is NDArray and 
        //{
        //    get => new NDArray(Storage.GetView(slice)); 
        //    set => Storage.GetView(slice).SetData(value);
        //}


        /// <summary>
        /// Perform slicing, index extraction, masking and indexing all at the same time with mixed index objects
        /// </summary>
        /// <param name="indicesObjects"></param>
        /// <returns></returns>
        public NDArray this[params object[] indicesObjects]
        {
            get
            {
                // A string array's ADVANCED index (arrays, masks, bools) runs on its unit view; basic indices (slices,
                // integers) are views and take the ordinary path.
                if (typecode == NPTypeCode.String && TryStringAdvancedIndex(ref indicesObjects, out var normalized))
                {
                    if (normalized is null)
                        return this[indicesObjects];
                    // NumPy's prepare_index checks the index against THIS array's rank before the unit route (whose extra
                    // character axis would otherwise accept one index too many).
                    ThrowIfTooManyIndices(normalized);
                    return StringFetchIndices(normalized);
                }
                return this.FetchIndices(indicesObjects);
            }
            set
            {
                ThrowIfNotWriteable();
                if (typecode == NPTypeCode.String && TryStringAdvancedIndex(ref indicesObjects, out var normalized))
                {
                    // NumPy validates the index (prepare_index) before it converts the value.
                    if (normalized != null)
                        ThrowIfTooManyIndices(normalized);
                    if (normalized is null)
                        this[indicesObjects] = value;
                    else if (normalized.Length == 1 && normalized[0] is NDArray m && m.typecode == NPTypeCode.Boolean)
                        // A lone boolean mask takes the mask setter: NumPy's single-mask assignment rules and texts.
                        this[m.MakeGeneric<bool>()] = value;
                    else
                        StringSetIndices(normalized, value);
                    return;
                }
                SetIndices(indicesObjects, value);
            }
        }

        /// <summary>
        ///     Classifies a string array's index tuple for the unit route. A tuple that IS one nested tuple
        ///     (<c>NDArray[]</c> / <c>object[]</c>, NumPy's index-as-tuple) is unwrapped into <paramref name="indicesObjects"/>
        ///     with <paramref name="normalized"/> null (the caller re-indexes with it); a tuple with an advanced component
        ///     yields its normalized form; a basic-only tuple returns false (the view path serves it).
        /// </summary>
        /// <param name="indicesObjects">The caller's index tuple; replaced by the inner tuple when it was a nested one.</param>
        /// <param name="normalized">The normalized tuple for the unit route, or null for "re-index with the unwrapped tuple".</param>
        /// <returns>True when the caller must take the string route (or re-index), false for the ordinary path.</returns>
        private bool TryStringAdvancedIndex(ref object[] indicesObjects, out object[] normalized)
        {
            normalized = null;
            if (indicesObjects.Length == 1)
            {
                switch (indicesObjects[0])
                {
                    case NDArray[] nds:
                        indicesObjects = nds;
                        return true;
                    case object[] objs:
                        indicesObjects = objs;
                        return true;
                }
            }
            var candidate = NormalizeIndexInputs((object[])indicesObjects.Clone());
            if (!HasAdvancedIndex(candidate))
                return false;
            normalized = candidate;
            return true;
        }
    }
}
