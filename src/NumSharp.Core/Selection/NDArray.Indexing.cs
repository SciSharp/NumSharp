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
        ///     NumPy's <c>ndarray.__getitem__(key)</c> and <c>ndarray.__setitem__(key, value)</c>: the one indexer of an
        ///     array. The C# argument list is NumPy's index tuple (<c>a[1, 2]</c> is NumPy's <c>a[(1, 2)]</c>), and an entry
        ///     is any index NumPy takes: an integer, a <see cref="Slice"/>, slice notation (<c>a[":, 2:7:1, ..., np.newaxis"]</c>),
        ///     <see cref="Slice.Ellipsis"/> and <see cref="Slice.NewAxis"/>, an integer index array (an <see cref="NDArray"/>,
        ///     <c>int[]</c>, <c>long[]</c> or any integer sequence), a boolean mask (an <see cref="NDArray"/> of bool,
        ///     <c>bool[]</c>, <c>bool[,]</c> or any bool sequence) and a C# <see cref="bool"/> (NumPy's 0-d bool).
        /// </summary>
        /// <param name="key">The index tuple. One entry per C# argument; a <see cref="Slice"/>[] or <c>object[]</c> passed
        ///     alone is the tuple itself, and a single C# tuple entry (<c>a[(1, "0:2")]</c>) is spread into its entries, as
        ///     Python reads <c>a[(1, 2)]</c> as <c>a[1, 2]</c>. A lone string holds the whole index in slice notation
        ///     (comma-separated); a string among other entries is one slice.</param>
        /// <value>
        ///     GET: a VIEW for an index of integers, slices, the ellipsis and new axes (one integer per axis gives a 0-d view,
        ///     NumSharp's scalar); a new array for an index array, a mask or a C# bool (NumPy's advanced indexing).
        ///     SET: stores the value into the selection by NumPy's rules: a value broadcasts to the selection, a C# number is
        ///     NumPy's Python scalar (range-checked, never wrapped), one integer per axis stores one element (a
        ///     <c>StringDType</c> element takes a Python object by setitem).
        /// </value>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <c>null</c>, or an entry is <c>null</c>
        ///     (NumPy's <c>None</c> is <see cref="Slice.NewAxis"/>).</exception>
        /// <exception cref="IndexError">An integer is out of bounds, there are more indices than axes, a mask does not match
        ///     the axes it indexes, index arrays do not broadcast together, or an entry is not a valid index (NumPy's
        ///     <c>prepare_index</c> texts).</exception>
        /// <exception cref="ValueError">SET: the array is read-only (checked first, as NumPy's setitem does), the value does
        ///     not broadcast to the selection, or a sequence is stored into one element.</exception>
        /// <exception cref="OverflowException">SET: a C# integer outside this array's integer dtype (NumPy's OverflowError).</exception>
        /// <exception cref="TypeError">SET: a complex number into a real array, or a value of two or more dimensions through a
        ///     mask of this array's own shape.</exception>
        /// <remarks>
        ///     <para>
        ///     <b>Dispatch.</b> The key is sorted before any work, so each kind runs the code of the typed indexer that served
        ///     it until this one replaced them (<c>this[string]</c>, <c>this[params Slice[]]</c>,
        ///     <c>this[NDArray&lt;bool&gt;]</c>, <c>this[params NDArray&lt;int&gt;[]]</c>, <c>this[long*, int]</c>), and a
        ///     call keeps its result and its error: a basic index (see <see cref="TryBasicIndex"/>) is one view of the
        ///     storage; a lone <see cref="NDArray{T}"/> of bool is the mask path; every other key is the general advanced
        ///     path. The pre-dispatch matters beyond speed: the general path has no case for a lone <see cref="long"/> or
        ///     <see cref="short"/>, which C# used to bind to the <see cref="Slice"/> indexer at compile time.
        ///     </para>
        ///     <para>
        ///     <b>Virtual.</b> <see cref="NDArray{T}"/> overrides the GETTER to type its result (<c>typed["1:3"]</c> is an
        ///     <see cref="NDArray{T}"/>) and inherits this setter. <see cref="NDCharArray"/> and <see cref="NDMatrix"/> declare
        ///     their own indexer instead of overriding: their rules (a stripped element, a matrix row or column) apply through
        ///     their static type, and through an <see cref="NDArray"/> reference they are indexed as plain arrays, which the
        ///     library's own code relies on (it receives such objects as <see cref="NDArray"/>, see
        ///     <see cref="NDMatrixBoundary"/>).
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/user/basics.indexing.html</para>
        /// </remarks>
        public virtual NDArray this[params object[] key]
        {
            get
            {
                if (key is null)
                    throw new ArgumentNullException(nameof(key));
                // The old this[string] / this[params Slice[]] body: one view of the storage, no index normalization.
                if (TryBasicIndex(key, out Slice[] slices))
                    return new NDArray(Storage.GetView(slices));
                // The old this[NDArray<bool>] body: a mask from a comparison (statically NDArray<bool>) is the common
                // advanced index, and the general path would only route it here after its normalization pass.
                if (key.Length == 1 && key[0] is NDArray<bool> mask)
                    return GetMasked(mask);
                return GetAdvanced(key);
            }
            set
            {
                // NumPy's setitem tests writeability before it looks at the index; every path below did so first too.
                ThrowIfNotWriteable();
                if (key is null)
                    throw new ArgumentNullException(nameof(key));
                if (TryBasicIndex(key, out Slice[] slices))
                {
                    AssignThroughView(Storage.GetView(slices), value, IsItemIndex(slices));
                    return;
                }
                if (key.Length == 1 && key[0] is NDArray<bool> mask)
                {
                    SetMasked(mask, value);
                    return;
                }
                SetAdvanced(key, value);
            }
        }

        /// <summary>
        ///     Whether an index tuple is a BASIC index that C# used to bind to the typed indexers <c>this[string]</c> and
        ///     <c>this[params Slice[]]</c>, and that index as slices: a <see cref="Slice"/>[] passed as the whole tuple; ONE
        ///     string, read as slice notation (<see cref="Slice.ParseSlices"/>: comma-separated entries); or entries that are
        ///     each a <see cref="Slice"/>, a string (ONE slice, the implicit <c>string</c> → <see cref="Slice"/> conversion)
        ///     or an integer of a type C# converts implicitly to <see cref="int"/> or <see cref="long"/> (an integer
        ///     <see cref="Slice.Index(long)"/>).
        /// </summary>
        /// <param name="key">The index tuple (not <c>null</c>).</param>
        /// <param name="slices">The index as slices when the result is true: the caller's own <see cref="Slice"/>[] when
        ///     it passed one (as the typed indexer received it), else a new array.</param>
        /// <returns>True for a basic index; false when an entry is anything else (an index array, a mask, a C# bool, a
        ///     C# tuple, a <see cref="ulong"/>, a floating-point number, <c>null</c>), which the general path serves.</returns>
        /// <exception cref="ArgumentException">A lone string is not slice notation.</exception>
        /// <remarks>
        ///     Shared by every indexer that keeps the basic-index meaning of the old typed indexers (<see cref="NDArray{T}"/>'s
        ///     override, <see cref="NDListArray"/>'s ids). <see cref="ulong"/> and <see cref="nuint"/> are left out because no
        ///     implicit conversion takes them to <see cref="long"/>: C# sent them to the general indexer, whose
        ///     <see cref="ulong"/> case stays theirs. A <see cref="char"/> is an integer here, as C#'s <c>char</c> →
        ///     <c>int</c> conversion made it.
        /// </remarks>
        internal static bool TryBasicIndex(object[] key, out Slice[] slices)
        {
            if (key is Slice[] given)
            {
                slices = given;
                return true;
            }

            // One entry, the common case (a[i], a["1:3"], a[slice]): sorted without the validation pass below.
            if (key.Length == 1)
            {
                switch (key[0])
                {
                    case int i32:
                        slices = new[] { Slice.Index(i32) };
                        return true;
                    case long i64:
                        slices = new[] { Slice.Index(i64) };
                        return true;
                    case string notation:
                        slices = Slice.ParseSlices(notation);
                        return true;
                    case Slice slice:
                        slices = new[] { slice };
                        return true;
                }
            }

            // Validate every entry before converting any: a key with one advanced entry is not basic at all, and the
            // general path must see the caller's entries unchanged.
            foreach (object entry in key)
            {
                if (entry is not (Slice or string or int or long or short or ushort or sbyte or byte or uint or char or nint))
                {
                    slices = null;
                    return false;
                }
            }

            slices = new Slice[key.Length];
            for (int i = 0; i < key.Length; i++)
            {
                slices[i] = key[i] switch
                {
                    Slice slice => slice,
                    string one => new Slice(one),
                    int i32 => Slice.Index(i32),
                    long i64 => Slice.Index(i64),
                    char ch => Slice.Index(ch),
                    nint native => Slice.Index((long)native),
                    // sbyte, byte, short, ushort, uint (validated above): each widens to long without loss.
                    _ => Slice.Index(Convert.ToInt64(key[i], CultureInfo.InvariantCulture)),
                };
            }
            return true;
        }

        /// <summary>
        ///     Whether a basic index names ONE element by integers alone (one integer per axis, no slice, ellipsis or new
        ///     axis) — NumPy's "full integer index", which its assignment routes to the dtype's setitem
        ///     (<c>array_assign_item</c>) instead of the copy of a view.
        /// </summary>
        /// <param name="slices">The parsed index.</param>
        /// <returns>True for a full integer index of this array.</returns>
        private bool IsItemIndex(Slice[] slices)
        {
            if (slices.Length != ndim)
                return false;
            foreach (var s in slices)
                if (s is null || !s.IsIndex || s.IsEllipsis || s.IsNewAxis)
                    return false;
            return true;
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
        /// <param name="itemIndex">True when the index named one element by integers alone (<see cref="IsItemIndex"/>): a
        ///     <c>StringDType</c> element then takes a Python object by setitem, as NumPy's <c>array_assign_item</c> does.</param>
        /// <exception cref="ValueError">The value does not broadcast to the selection, or a sequence into one element.</exception>
        /// <exception cref="OverflowException">A C# literal out of this dtype's range.</exception>
        /// <exception cref="TypeError">A complex literal into a real dtype.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> store's <c>MemoryError</c> texts.</exception>
        /// <remarks>
        ///     The view is wrapped in an <see cref="NDArray"/> for the assignment and disposed after it, exceptions included:
        ///     the wrapper takes one counted reference on the view's slice and gives it back. A bare view storage was never
        ///     released, which stranded the empty selection's fresh buffer (one pooled buffer per <c>a[5:] = v</c>, found by
        ///     the zero-leak gate's store cases); a non-empty view shares the parent's buffer and is unaffected.
        /// </remarks>
        private static void AssignThroughView(UnmanagedStorage view, NDArray value, bool itemIndex = false)
        {
            using var target = new NDArray(view);
            // NumPy's array_assign_item: one StringDType element given a Python object is packed by setitem into the element
            // itself (setitem's error texts, one allocation); a view of a 0-d array selected by an ellipsis or a slice keeps
            // the view assignment below (the copy loop), as in NumPy.
            if (itemIndex && view.IsVString && UnmanagedStorage.IsVStringItemValue(value))
            {
                view.SetVStringItem(value, view.Shape.offset);
                return;
            }
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
        ///     The general (advanced) GET of the indexer: slicing, index arrays, masks and bools in any mix — the body of the
        ///     old <c>this[params object[]]</c> indexer, reached for every key that is neither basic nor a lone typed mask.
        /// </summary>
        /// <param name="indicesObjects">The index tuple (not <c>null</c>).</param>
        /// <returns>The selection: a new array for an advanced index, a view for the basic remainder the general path
        ///     still meets (a C# tuple entry, an <see cref="ulong"/>).</returns>
        /// <exception cref="ArgumentNullException">An entry is <c>null</c>.</exception>
        /// <exception cref="IndexError">An index is out of bounds, too many indices, a mask that does not match, index
        ///     arrays that do not broadcast together, or an entry that is not a valid index (NumPy's texts).</exception>
        /// <remarks>A fixed-width string array's ADVANCED index runs on its unit view (NumPy's <c>prepare_index</c> first,
        ///     against this array's rank); every other array takes <see cref="FetchIndices(object[])"/>, whose result keeps
        ///     this array's lane-sharing C-type class.</remarks>
        private protected NDArray GetAdvanced(object[] indicesObjects)
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
            return KeepLaneSharingClass(this.FetchIndices(indicesObjects));
        }

        /// <summary>
        ///     The general (advanced) SET of the indexer — the body of the old <c>this[params object[]]</c> setter, reached
        ///     for every key that is neither basic nor a lone typed mask.
        /// </summary>
        /// <param name="indicesObjects">The index tuple (not <c>null</c>).</param>
        /// <param name="value">The value (NumPy's assignment rules: broadcast to the selection, a C# literal packed as
        ///     NumPy's Python scalar).</param>
        /// <exception cref="ValueError">The array is read-only, or the value does not fit the selection (NumPy's
        ///     <c>shape mismatch: value array of shape ... could not be broadcast to indexing result of shape ...</c>).</exception>
        /// <exception cref="IndexError">The index is invalid (NumPy's <c>prepare_index</c> texts).</exception>
        /// <exception cref="OverflowException">A C# integer outside this array's integer dtype.</exception>
        private void SetAdvanced(object[] indicesObjects, NDArray value)
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
