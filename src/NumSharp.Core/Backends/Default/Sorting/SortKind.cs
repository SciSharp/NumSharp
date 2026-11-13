using NumSharp.Backends.Printing;

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     NumPy's <c>NPY_SORTKIND</c> (<c>ndarraytypes.h</c>): the algorithm family a sort or argsort is asked for, as
    ///     <c>PyArray_SortkindConverter</c> parses the <c>kind=</c> string and <c>array_sort</c> / <c>array_argsort</c> fold
    ///     <c>stable=</c> and <c>descending=</c> into it — a set of flags since NumPy 2.5 (<see cref="NPY_SORT_STABLE"/>
    ///     and <see cref="NPY_SORT_DESCENDING"/> combine).
    /// </summary>
    /// <remarks>
    ///     What each kind changes is the dtype's business: <c>PyArray_Sort</c> / <c>PyArray_ArgSort</c> clear the heapsort bit
    ///     first, so the heapsort kind IS the default kind for every dtype (NumPy 2.4's <c>StringDType</c> loop refused it);
    ///     over <c>S</c> / <c>U</c> text the default kind is NumPy's introsort and the stable kind its timsort;
    ///     <c>StringDType</c> and object arrays refuse the descending flag; NumSharp's numeric ascending sorts give one answer
    ///     for every kind but keep each float NaN's bits, as NumPy's do (the default kind in the order NumPy's x86-simd-sort
    ///     moves the NaNs to the end), and the descending sorts give NumPy's own introsort order for the default kind.
    ///     <para>
    ///     The members are NumPy 2.5.4's header enum, in its order and with its values, aliases and NumPy's new-style names
    ///     included (the dtype oracle's C-enum gate holds the mirror to the header). Compare values, never names: .NET's
    ///     <c>ToString()</c> of an aliased value may print either name.
    ///     </para>
    /// </remarks>
    internal enum NPY_SORTKIND
    {
        /// <summary>NumPy's private "no kind given" sentinel; no converter returns it.</summary>
        _NPY_SORT_UNDEFINED = -1,

        /// <summary>The default kind: <c>kind=None</c> / <c>'quicksort'</c> (NumPy's introsort, not stable).</summary>
        NPY_QUICKSORT = 0,

        /// <summary><c>kind='heapsort'</c>: the default kind for every dtype (<c>PyArray_Sort</c> clears this bit before it
        ///     resolves the loop; NumPy 2.4's <c>StringDType</c> loop refused it).</summary>
        NPY_HEAPSORT = 1,

        /// <summary>The legacy name of the stable kind (<c>kind='mergesort'</c>): <see cref="NPY_STABLESORT"/>'s value.</summary>
        NPY_MERGESORT = 2,

        /// <summary><c>kind='stable'</c> / <c>'mergesort'</c> / <c>stable=True</c>: the stable sort.</summary>
        NPY_STABLESORT = 2,

        /// <summary>NumPy's private new-style name of the heapsort kind (<see cref="NPY_HEAPSORT"/>'s value).</summary>
        _NPY_SORT_HEAPSORT = 1,

        /// <summary>NumPy's new-style name of the default kind (<see cref="NPY_QUICKSORT"/>'s value).</summary>
        NPY_SORT_DEFAULT = 0,

        /// <summary>NumPy's new-style name of the stable kind (<see cref="NPY_STABLESORT"/>'s value).</summary>
        NPY_SORT_STABLE = 2,

        /// <summary>
        ///     NumPy's new-style descending flag, set by <c>descending=True</c> (NumPy 2.5) on top of
        ///     <see cref="NPY_SORT_DEFAULT"/> or <see cref="NPY_SORT_STABLE"/>: the largest value first, NaN still last.
        ///     <c>StringDType</c> loops refuse it (<c>RuntimeError: unsupported sort kind</c>), and so do object arrays
        ///     (<c>TypeError: no current sort function meets the requirements</c>).
        /// </summary>
        NPY_SORT_DESCENDING = 4,
    }

    /// <summary>
    ///     The argument rules NumPy's <c>ndarray.sort</c> / <c>ndarray.argsort</c> (<c>array_sort</c> / <c>array_argsort</c>,
    ///     <c>methods.c</c>) apply before they touch the array, shared by <c>np.sort</c>, <c>np.argsort</c> and the two methods.
    /// </summary>
    internal static class SortKindArguments
    {
        /// <summary>The text of NumPy's refusal of <c>kind</c> together with <c>stable</c>.</summary>
        internal const string KindAndKeywordsMessage =
            "`kind` and keyword parameters can't be provided at the same time. Use only one of them.";

        /// <summary>The text of NumPy's refusal of <c>order</c> on an array without fields (every NumSharp dtype).</summary>
        internal const string OrderWithoutFieldsMessage = "Cannot specify order when the array has no fields.";

        /// <summary>
        ///     NumPy's argument checks in their order: <c>kind</c> parsed while the arguments are read, then <c>kind</c> with
        ///     the keywords (<c>stable</c>, <c>descending</c>), then <c>order</c>.
        /// </summary>
        /// <param name="kind">The <c>kind=</c> argument, or null (NumPy's <c>None</c>).</param>
        /// <param name="order">The <c>order=</c> argument, or null; any other value is refused (no NumSharp dtype has fields).</param>
        /// <param name="stable">The keyword-only <c>stable=</c> argument, or null (NumPy's <c>None</c>).</param>
        /// <param name="descending">The keyword-only <c>descending=</c> argument (NumPy 2.5), or null (NumPy's <c>None</c>,
        ///     ascending).</param>
        /// <returns>The kind the call sorts with: the parsed <paramref name="kind"/>; else the stable kind when
        ///     <paramref name="stable"/> is true, the default otherwise, with <see cref="NPY_SORTKIND.NPY_SORT_DESCENDING"/>
        ///     added when <paramref name="descending"/> is true.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/> (<c>sort kind must be one of 'quick', 'heap', or
        ///     'stable' (got 'x')</c>), <paramref name="kind"/> given with a non-null <paramref name="stable"/> or
        ///     <paramref name="descending"/> (false included, as NumPy's <c>PyArray_OptionalBoolConverter</c> reads any
        ///     non-None value as given: <c>`kind` and keyword parameters can't be provided at the same time. Use only one of
        ///     them.</c>), or an <paramref name="order"/>.</exception>
        internal static NPY_SORTKIND Resolve(string kind, object order, bool? stable, bool? descending)
        {
            NPY_SORTKIND resolved;
            if (kind is not null)
            {
                resolved = Parse(kind);
                // array_sort / array_argsort: a kind leaves no room for the keywords — either keyword given (False
                // included) is refused, before the order is looked at.
                if (stable is not null || descending is not null)
                    throw new ValueError(KindAndKeywordsMessage);
            }
            else
            {
                // The keywords fold into NumPy 2.5's flags: NPY_SORT_STABLE | NPY_SORT_DESCENDING.
                resolved = stable == true ? NPY_SORTKIND.NPY_SORT_STABLE : NPY_SORTKIND.NPY_SORT_DEFAULT;
                if (descending == true)
                    resolved |= NPY_SORTKIND.NPY_SORT_DESCENDING;
            }
            if (order is not null)
                throw new ValueError(OrderWithoutFieldsMessage);
            return resolved;
        }

        /// <summary>
        ///     <c>PyArray_Sort</c> / <c>PyArray_ArgSort</c>'s first statement, <c>flags &amp;= ~_NPY_SORT_HEAPSORT</c>: the
        ///     heapsort kind becomes the default kind (its bit cleared, the stable and descending bits kept).
        /// </summary>
        /// <param name="kind">The resolved kind.</param>
        /// <returns>The kind the sort loop sees.</returns>
        internal static NPY_SORTKIND ClearHeapsort(NPY_SORTKIND kind) => kind & ~NPY_SORTKIND._NPY_SORT_HEAPSORT;

        /// <summary>Whether a resolved kind asks for the stable sort (its stable bit, whatever the descending bit).</summary>
        /// <param name="kind">The resolved kind.</param>
        /// <returns>True for <c>kind='stable'</c> / <c>'mergesort'</c> and <c>stable=True</c>.</returns>
        internal static bool IsStable(NPY_SORTKIND kind) => (kind & NPY_SORTKIND.NPY_SORT_STABLE) != 0;

        /// <summary>Whether a resolved kind asks for a descending sort (NumPy 2.5's <c>descending=True</c>).</summary>
        /// <param name="kind">The resolved kind.</param>
        /// <returns>True when the descending bit is set.</returns>
        internal static bool IsDescending(NPY_SORTKIND kind) => (kind & NPY_SORTKIND.NPY_SORT_DESCENDING) != 0;

        /// <summary>
        ///     NumPy's <c>sortkind_parser</c>: the FIRST character decides, case-insensitively — <c>q</c> quicksort,
        ///     <c>h</c> heapsort, <c>m</c> mergesort (the stable kind), <c>s</c> stable; anything else (an empty string
        ///     included, and <c>'timsort'</c> / <c>'radixsort'</c>) is refused.
        /// </summary>
        /// <param name="kind">The <c>kind=</c> string.</param>
        /// <returns>The kind.</returns>
        /// <exception cref="ValueError">The first character names no kind; the text quotes the argument as Python's
        ///     <c>repr</c> does.</exception>
        internal static NPY_SORTKIND Parse(string kind)
        {
            if (kind.Length >= 1)
            {
                switch (kind[0])
                {
                    case 'q':
                    case 'Q':
                        return NPY_SORTKIND.NPY_QUICKSORT;
                    case 'h':
                    case 'H':
                        return NPY_SORTKIND.NPY_HEAPSORT;
                    case 'm':
                    case 'M':
                    case 's':
                    case 'S':
                        return NPY_SORTKIND.NPY_STABLESORT;
                }
            }
            throw new ValueError($"sort kind must be one of 'quick', 'heap', or 'stable' (got {PyRepr.Str(kind)})");
        }
    }
}
