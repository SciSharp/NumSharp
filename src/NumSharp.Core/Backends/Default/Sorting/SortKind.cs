using NumSharp.Backends.Printing;

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    ///     NumPy's <c>NPY_SORTKIND</c> (<c>ndarraytypes.h</c>): the algorithm family a sort or argsort is asked for, as
    ///     <c>PyArray_SortkindConverter</c> parses the <c>kind=</c> string and <c>array_sort</c> / <c>array_argsort</c> fold
    ///     <c>stable=</c> into it.
    /// </summary>
    /// <remarks>
    ///     What each kind changes is the dtype's business: over <c>S</c> / <c>U</c> text the default and the heapsort kinds are
    ///     one algorithm (NumPy clears the heapsort bit before it looks a legacy sort function up) and the stable kind another;
    ///     <c>StringDType</c> refuses the heapsort kind; NumSharp's numeric sorts give one answer for every kind but keep the
    ///     input NaN bits in the stable kind's float value sort, as NumPy's do.
    ///     <para>
    ///     The members are NumPy 2.4.2's header enum, in its order and with its values, aliases and NumPy's new-style names
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

        /// <summary><c>kind='heapsort'</c>: the default kind for every legacy dtype, refused by <c>StringDType</c>.</summary>
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

        /// <summary>NumPy's new-style descending flag; no NumPy 2.4.2 Python call sets it, and no NumSharp sort reads it.</summary>
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
        ///     <c>stable</c>, then <c>order</c>.
        /// </summary>
        /// <param name="kind">The <c>kind=</c> argument, or null (NumPy's <c>None</c>).</param>
        /// <param name="order">The <c>order=</c> argument, or null; any other value is refused (no NumSharp dtype has fields).</param>
        /// <param name="stable">The keyword-only <c>stable=</c> argument, or null (NumPy's <c>None</c>).</param>
        /// <returns>The kind the call sorts with: the parsed <paramref name="kind"/>, else the stable kind when
        ///     <paramref name="stable"/> is true, else the default.</returns>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/> (<c>sort kind must be one of 'quick', 'heap', or
        ///     'stable' (got 'x')</c>), <paramref name="kind"/> given with a non-null <paramref name="stable"/> (false
        ///     included, as NumPy reads any non-None value as given), or an <paramref name="order"/>.</exception>
        internal static NPY_SORTKIND Resolve(string kind, object order, bool? stable)
        {
            NPY_SORTKIND resolved;
            if (kind is not null)
            {
                resolved = Parse(kind);
                if (stable is not null)
                    throw new ValueError(KindAndKeywordsMessage);
            }
            else
                resolved = stable == true ? NPY_SORTKIND.NPY_STABLESORT : NPY_SORTKIND.NPY_QUICKSORT;
            if (order is not null)
                throw new ValueError(OrderWithoutFieldsMessage);
            return resolved;
        }

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
