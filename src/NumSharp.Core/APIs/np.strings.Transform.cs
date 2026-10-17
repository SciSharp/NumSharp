using System;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class strings
        {
            // =========================================================================================================
            // strip / lstrip / rstrip
            // =========================================================================================================

            /// <summary>
            ///     NumPy's <c>np.strings.strip(a, chars=None)</c>: each element without its leading and trailing characters
            ///     found in <paramref name="chars"/> (Python's whitespace for null), keeping <paramref name="a"/>'s dtype.
            /// </summary>
            /// <param name="a">The strings (<c>S</c>, <c>U</c> or <c>StringDType</c>).</param>
            /// <param name="chars">The characters to remove, as a set (the same class as <paramref name="a"/>, or <c>U</c> with
            ///     a <c>StringDType</c> <paramref name="a"/>); null strips whitespace (NumPy's <c>None</c>).</param>
            /// <returns>A fresh array of <paramref name="a"/>'s dtype and the broadcast shape (a <c>StringDType</c> result
            ///     takes the compatible NA object).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">No loop for the operand classes (NumPy's no-loop text); two different
            ///     <c>StringDType</c> NA objects.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">A <c>StringDType</c> NA its rule refuses.</exception>
            /// <remarks>
            ///     Whitespace is the class's: ASCII whitespace for <c>S</c>, Python's <c>str.isspace</c> for <c>U</c> /
            ///     <c>StringDType</c>. NumPy's trailing scan also strips NUL characters, its leading scan does not
            ///     (<c>strip(' \0ab')</c> is <c>'\0ab'</c>). A <c>numpy.char.chararray</c> operand gives a chararray view of
            ///     the result.
            /// </remarks>
            public static NDArray strip(NDArray a, NDArray chars = null)
                => NDCharArrayBoundary.Sees(a, chars)
                    ? NDCharArrayBoundary.Wrapped((a, chars), static s => strip(s.a, s.chars))
                    : Strip("_strip", NDTextElements.BothStrip, a, chars);

            /// <summary>
            ///     NumPy's <c>np.strings.lstrip(a, chars=None)</c>: each element without its LEADING characters found in
            ///     <paramref name="chars"/> (whitespace for null).
            /// </summary>
            /// <param name="a">The strings.</param>
            /// <param name="chars">The characters to remove (see <see cref="strip"/>), or null for whitespace.</param>
            /// <returns>A fresh array of <paramref name="a"/>'s dtype.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">A <c>StringDType</c> NA its rule refuses.</exception>
            public static NDArray lstrip(NDArray a, NDArray chars = null)
                => NDCharArrayBoundary.Sees(a, chars)
                    ? NDCharArrayBoundary.Wrapped((a, chars), static s => lstrip(s.a, s.chars))
                    : Strip("_lstrip", NDTextElements.LeftStrip, a, chars);

            /// <summary>
            ///     NumPy's <c>np.strings.rstrip(a, chars=None)</c>: each element without its TRAILING characters found in
            ///     <paramref name="chars"/> (whitespace for null).
            /// </summary>
            /// <param name="a">The strings.</param>
            /// <param name="chars">The characters to remove (see <see cref="strip"/>), or null for whitespace.</param>
            /// <returns>A fresh array of <paramref name="a"/>'s dtype.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">A <c>StringDType</c> NA its rule refuses.</exception>
            public static NDArray rstrip(NDArray a, NDArray chars = null)
                => NDCharArrayBoundary.Sees(a, chars)
                    ? NDCharArrayBoundary.Wrapped((a, chars), static s => rstrip(s.a, s.chars))
                    : Strip("_rstrip", NDTextElements.RightStrip, a, chars);

            /// <summary>
            ///     The three strip wrappers' shared body: <c>_{name}_whitespace(a)</c> when <paramref name="chars"/> is null,
            ///     <c>_{name}_chars(a, chars)</c> otherwise. The public wrappers apply the chararray rule themselves: the rule
            ///     belongs to the outermost public call, which the boundary audit identifies by its name.
            /// </summary>
            /// <param name="stem">The ufunc name stem (<c>_strip</c>, <c>_lstrip</c>, <c>_rstrip</c>).</param>
            /// <param name="type">The strip type.</param>
            /// <param name="a">The strings.</param>
            /// <param name="chars">The set, or null.</param>
            /// <returns>The stripped strings.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">No loop; two different NA objects.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">A <c>StringDType</c> NA its rule refuses.</exception>
            private static NDArray Strip(string stem, int type, NDArray a, NDArray chars)
            {
                RequireOperand(a, nameof(a));
                // The chararray rule belongs to the public call above: the public functions this body calls compute plain
                // (a chararray in another position must not reach their rules from inside the library).
                using var plain = NDCharArrayBoundary.Inside();
                return chars is null
                    ? NDStringMethods.StripWhitespace(stem + "_whitespace", type, a)
                    : NDStringMethods.StripChars(stem + "_chars", type, a, chars);
            }

            // =========================================================================================================
            // center / ljust / rjust / zfill / expandtabs
            // =========================================================================================================

            /// <summary>
            ///     NumPy's <c>np.strings.center(a, width, fillchar=' ')</c>: each element centered in a string of
            ///     <paramref name="width"/> characters, padded with <paramref name="fillchar"/> (an odd padding puts the extra
            ///     character on the left when the width is odd, on the right otherwise — CPython's rule). An element longer
            ///     than the width is unchanged.
            /// </summary>
            /// <param name="a">The strings (<c>S</c>, <c>U</c> or <c>StringDType</c>).</param>
            /// <param name="width">The widths: an integer array (any width, not bool) or literal, broadcast.</param>
            /// <param name="fillchar">The fill characters, each exactly one character long; null is NumPy's default
            ///     <c>' '</c>. A <c>U</c> fill beside an <c>S</c> operand is cast to <c>S</c> (ASCII only).</param>
            /// <returns>A fresh array: for <c>S</c> / <c>U</c> the class of <paramref name="a"/> at the widest result width,
            ///     laid out like <paramref name="a"/>; for <c>StringDType</c> a <c>StringDType</c> array.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="width"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="width"/> is not an integer array (<c>unsupported type float64 for
            ///     operand 'width'</c>); a fill that is not one character (<c>The fill character must be exactly one
            ///     character long</c>); a non-text operand; no common dtype or no loop for the classes; a uint64 width (NumPy
            ///     builds a <c>'U6.0'</c> descriptor: <c>data type 'U6.0' not understood</c>).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">An empty broadcast (<c>zero-size array to reduction operation maximum which has
            ///     no identity</c>); a <c>StringDType</c> NA its rule refuses.</exception>
            /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> operand.</exception>
            /// <exception cref="OverflowException">A <c>StringDType</c> result past <c>sys.maxsize</c> bytes.</exception>
            public static NDArray center(NDArray a, NDArray width, NDArray fillchar = null)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Wrapped((a, width, fillchar), static s => center(s.a, s.width, s.fillchar));
                return Justify("_center", NDTextElements.Center, a, width, fillchar);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.ljust(a, width, fillchar=' ')</c>: each element left-justified in a string of
            ///     <paramref name="width"/> characters, padded on the right with <paramref name="fillchar"/>.
            /// </summary>
            /// <param name="a">The strings.</param>
            /// <param name="width">The integer widths, broadcast.</param>
            /// <param name="fillchar">The one-character fills (null = <c>' '</c>).</param>
            /// <returns>A fresh array (see <see cref="center"/>).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="width"/> is null.</exception>
            /// <exception cref="TypeError">See <see cref="center"/>.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">An empty broadcast; a <c>StringDType</c> NA its rule refuses.</exception>
            /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> operand.</exception>
            /// <exception cref="OverflowException">A <c>StringDType</c> result past <c>sys.maxsize</c> bytes.</exception>
            public static NDArray ljust(NDArray a, NDArray width, NDArray fillchar = null)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Wrapped((a, width, fillchar), static s => ljust(s.a, s.width, s.fillchar));
                return Justify("_ljust", NDTextElements.Left, a, width, fillchar);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.rjust(a, width, fillchar=' ')</c>: each element right-justified in a string of
            ///     <paramref name="width"/> characters, padded on the left with <paramref name="fillchar"/>.
            /// </summary>
            /// <param name="a">The strings.</param>
            /// <param name="width">The integer widths, broadcast.</param>
            /// <param name="fillchar">The one-character fills (null = <c>' '</c>).</param>
            /// <returns>A fresh array (see <see cref="center"/>).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="width"/> is null.</exception>
            /// <exception cref="TypeError">See <see cref="center"/>.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">An empty broadcast; a <c>StringDType</c> NA its rule refuses.</exception>
            /// <exception cref="UnicodeEncodeError">A non-ASCII fill for an <c>S</c> operand.</exception>
            /// <exception cref="OverflowException">A <c>StringDType</c> result past <c>sys.maxsize</c> bytes.</exception>
            public static NDArray rjust(NDArray a, NDArray width, NDArray fillchar = null)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Wrapped((a, width, fillchar), static s => rjust(s.a, s.width, s.fillchar));
                return Justify("_rjust", NDTextElements.Right, a, width, fillchar);
            }

            /// <summary>
            ///     The three justify wrappers, statement for statement: <c>width = np.asanyarray(width)</c> and its integer
            ///     check, <c>a = np.asanyarray(a)</c>, <c>fillchar = np.asanyarray(fillchar)</c>, the one-character check
            ///     (<c>np.any(str_len(fillchar) != 1)</c>), then the <c>StringDType</c> ufunc when
            ///     <c>np.result_type(a, fillchar)</c> is <c>StringDType</c>, else the fixed-width sizing and loop.
            /// </summary>
            /// <param name="ufunc">The ufunc name (<c>_center</c>, <c>_ljust</c>, <c>_rjust</c>).</param>
            /// <param name="pos">The justification.</param>
            /// <param name="a">The strings.</param>
            /// <param name="width">The widths.</param>
            /// <param name="fillchar">The fills, or null for <c>' '</c>.</param>
            /// <returns>The padded strings.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="width"/> is null.</exception>
            /// <exception cref="TypeError">See <see cref="center"/>.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">An empty broadcast; a <c>StringDType</c> NA its rule refuses.</exception>
            private static NDArray Justify(string ufunc, int pos, NDArray a, NDArray width, NDArray fillchar)
            {
                RequireOperand(a, nameof(a));
                // The chararray rule belongs to the public call above: the public functions this body calls compute plain
                // (a chararray in another position must not reach their rules from inside the library).
                using var plain = NDCharArrayBoundary.Inside();
                RequireOperand(width, nameof(width));
                NDArray w = null, arr = null, fill = null;
                try
                {
                    // width = np.asanyarray(width); the integer check (bool is not a NumPy integer)
                    w = NDStringMethods.AsAnyArray(width);
                    if (!w.typecode.IsInteger())
                        throw new TypeError($"unsupported type {w.dtype} for operand 'width'");
                    arr = NDStringMethods.AsAnyArray(a);
                    fill = fillchar is null ? NDArray.Scalar(" ") : NDStringMethods.AsAnyArray(fillchar);
                    // if np.any(str_len(fillchar) != 1): raise TypeError(...)
                    using (var lengths = str_len(fill))
                    using (var wrong = np.not_equal(lengths, 1))
                    {
                        if (np.any(wrong))
                            throw new TypeError("The fill character must be exactly one character long");
                    }
                    if (np.result_type(arr, fill).@char == 'T')
                        return NDVStringMethods.Pad(ufunc, pos, arr, w, fill);
                    return NDStringMethods.Pad(ufunc, pos, arr, w, fill);
                }
                finally
                {
                    NDStringMethods.DisposeIfTemp(w, width);
                    NDStringMethods.DisposeIfTemp(arr, a);
                    NDStringMethods.DisposeIfTemp(fill, fillchar);
                }
            }

            /// <summary>
            ///     NumPy's <c>np.strings.zfill(a, width)</c>: each element left-filled with <c>'0'</c> to
            ///     <paramref name="width"/> characters, a leading <c>'+'</c> / <c>'-'</c> moved in front of the zeros
            ///     (<c>zfill('-42', 5)</c> is <c>'-0042'</c>).
            /// </summary>
            /// <param name="a">The strings (<c>S</c>, <c>U</c> or <c>StringDType</c>).</param>
            /// <param name="width">The widths: an integer array or literal, broadcast.</param>
            /// <returns>A fresh array (for <c>S</c> / <c>U</c> at the widest result width, laid out like
            ///     <paramref name="a"/>).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="width"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="width"/> is not an integer array; a non-text operand; a uint64
            ///     width (the float descriptor); a <c>StringDType</c> NA that is neither a string nor NaN-like.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">An empty broadcast.</exception>
            public static NDArray zfill(NDArray a, NDArray width)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Wrapped((a, width), static s => zfill(s.a, s.width));
                RequireOperand(a, nameof(a));
                // The chararray rule belongs to the public call above: the public functions this body calls compute plain
                // (a chararray in another position must not reach their rules from inside the library).
                using var plain = NDCharArrayBoundary.Inside();
                RequireOperand(width, nameof(width));
                NDArray w = null, arr = null;
                try
                {
                    w = NDStringMethods.AsAnyArray(width);
                    if (!w.typecode.IsInteger())
                        throw new TypeError($"unsupported type {w.dtype} for operand 'width'");
                    arr = NDStringMethods.AsAnyArray(a);
                    // if a.dtype.char == "T": return _zfill(a, width)
                    return NDStringUfuncs.IsVString(arr) ? NDVStringMethods.ZFill(arr, w) : NDStringMethods.ZFill(arr, w);
                }
                finally
                {
                    NDStringMethods.DisposeIfTemp(w, width);
                    NDStringMethods.DisposeIfTemp(arr, a);
                }
            }

            /// <summary>
            ///     NumPy's <c>np.strings.expandtabs(a, tabsize=8)</c>: each tab replaced by spaces up to the next multiple of
            ///     <paramref name="tabsize"/> columns (the column restarts after <c>'\n'</c> and <c>'\r'</c>; a tab size of 0 or
            ///     less removes the tabs).
            /// </summary>
            /// <param name="a">The strings (<c>S</c>, <c>U</c> or <c>StringDType</c>).</param>
            /// <param name="tabsize">The tab sizes (any integer dtype), broadcast; null is NumPy's default 8.</param>
            /// <returns>A fresh array (for <c>S</c> / <c>U</c> at the widest result width, laid out like
            ///     <paramref name="a"/>).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">A non-text operand (NumPy's no-loop text).</exception>
            /// <exception cref="ArgumentException">A tab size that cannot reach int64 by a same_kind cast (a float).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="OverflowException">An expanded string longer than 2^31 - 1 (<c>new string is too
            ///     long</c>).</exception>
            /// <exception cref="ValueError">An empty broadcast; a <c>StringDType</c> NA whose NA object is not a string.</exception>
            public static NDArray expandtabs(NDArray a, NDArray tabsize = null)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Wrapped((a, tabsize), static s => expandtabs(s.a, s.tabsize));
                RequireOperand(a, nameof(a));
                // The chararray rule belongs to the public call above: the public functions this body calls compute plain
                // (a chararray in another position must not reach their rules from inside the library).
                using var plain = NDCharArrayBoundary.Inside();
                NDArray arr = null, t = null;
                try
                {
                    arr = NDStringMethods.AsAnyArray(a);
                    t = tabsize is null ? NDArray.Scalar(8L) : NDStringMethods.AsAnyArray(tabsize);
                    return NDStringUfuncs.IsVString(arr) ? NDVStringMethods.ExpandTabs(arr, t) : NDStringMethods.ExpandTabs(arr, t);
                }
                finally
                {
                    NDStringMethods.DisposeIfTemp(arr, a);
                    NDStringMethods.DisposeIfTemp(t, tabsize);
                }
            }

            // =========================================================================================================
            // replace / partition / rpartition / slice
            // =========================================================================================================

            /// <summary>
            ///     NumPy's <c>np.strings.replace(a, old, new, count=-1)</c>: each element with its occurrences of
            ///     <paramref name="old"/> replaced by <paramref name="new"/>, left to right, at most <paramref name="count"/>
            ///     of them (a negative count replaces all). An empty <paramref name="old"/> inserts <paramref name="new"/>
            ///     between every character and at both ends.
            /// </summary>
            /// <param name="a">The strings (<c>S</c>, <c>U</c> or <c>StringDType</c>).</param>
            /// <param name="old">The searched strings. A C# string (NumPy's Python <c>str</c>, which has no dtype) is cast to
            ///     <paramref name="a"/>'s dtype first — truncated to its width; an array keeps its own.</param>
            /// <param name="new">The replacements (the same rule as <paramref name="old"/>).</param>
            /// <param name="count">The maximum replacements: an integer array or literal, broadcast; null is NumPy's -1.</param>
            /// <returns>A fresh array (for <c>S</c> / <c>U</c> of <paramref name="a"/>'s class at the widest result, laid
            ///     out like <paramref name="a"/>).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/>, <paramref name="old"/> or <paramref name="new"/>
            ///     is null.</exception>
            /// <exception cref="TypeError"><paramref name="count"/> is not an integer array (<c>unsupported type float64 for
            ///     operand 'count'</c>); no common dtype or no loop for the classes; a uint64 count (the float descriptor);
            ///     two different <c>StringDType</c> NA objects.</exception>
            /// <exception cref="IncorrectShapeException">A statement's operands do not broadcast.</exception>
            /// <exception cref="ValueError">An empty broadcast; a numeric operand cast to text; a <c>StringDType</c> NA its
            ///     rule refuses.</exception>
            /// <exception cref="UnicodeEncodeError">A non-ASCII <paramref name="old"/> / <paramref name="new"/> cast to <c>S</c>.</exception>
            public static NDArray replace(NDArray a, NDArray old, NDArray @new, NDArray count = null)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Wrapped((a, old, @new, count), static s => replace(s.a, s.old, s.@new, s.count));
                RequireOperand(a, nameof(a));
                // The chararray rule belongs to the public call above: the public functions this body calls compute plain
                // (a chararray in another position must not reach their rules from inside the library).
                using var plain = NDCharArrayBoundary.Inside();
                RequireOperand(old, nameof(old));
                RequireOperand(@new, nameof(@new));
                NDArray c = null, arr = null, o = null, n = null;
                try
                {
                    // count = np.asanyarray(count); the integer check
                    c = count is null ? NDArray.Scalar(-1L) : NDStringMethods.AsAnyArray(count);
                    if (!c.typecode.IsInteger())
                        throw new TypeError($"unsupported type {c.dtype} for operand 'count'");
                    arr = NDStringMethods.AsAnyArray(a);
                    // old_dtype = getattr(old, 'dtype', None): a Python str / int has none
                    bool oldHasDtype = !(old.WasPythonStr || old.WasPythonLiteral);
                    bool newHasDtype = !(@new.WasPythonStr || @new.WasPythonLiteral);
                    o = NDStringMethods.AsAnyArray(old);
                    n = NDStringMethods.AsAnyArray(@new);
                    if (np.result_type(arr, o, n).@char == 'T')
                        return NDVStringMethods.Replace(arr, o, n, c);
                    return NDStringMethods.Replace(arr, o, n, oldHasDtype, newHasDtype, c);
                }
                finally
                {
                    NDStringMethods.DisposeIfTemp(c, count);
                    NDStringMethods.DisposeIfTemp(arr, a);
                    NDStringMethods.DisposeIfTemp(o, old);
                    NDStringMethods.DisposeIfTemp(n, @new);
                }
            }

            /// <summary>
            ///     NumPy's <c>np.strings.partition(a, sep)</c>: each element split around the FIRST occurrence of
            ///     <paramref name="sep"/> into (before, separator, after); without a match, (the element, empty, empty).
            /// </summary>
            /// <param name="a">The strings (<c>S</c>, <c>U</c> or <c>StringDType</c>).</param>
            /// <param name="sep">The separators: cast to <paramref name="a"/>'s dtype for <c>S</c> / <c>U</c>; for
            ///     <c>StringDType</c> it must be <c>StringDType</c> too.</param>
            /// <returns>The three parts of the broadcast shape. For <c>S</c> / <c>U</c> each part has its own width (the widest
            ///     of its values, 1 at least) and is laid out like <paramref name="a"/>.</returns>
            /// <exception cref="ArgumentNullException">An operand is null.</exception>
            /// <exception cref="TypeError">No common dtype or no loop for the classes; two different <c>StringDType</c> NA
            ///     objects.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">An empty separator (<c>empty separator</c>); an empty broadcast; a numeric operand
            ///     cast to text; a <c>StringDType</c> NA whose NA object is not a string.</exception>
            /// <exception cref="UnicodeEncodeError">A non-ASCII separator cast to <c>S</c>.</exception>
            /// <remarks>NumPy returns three field views of one structured array for <c>S</c> / <c>U</c>; NumSharp has no
            ///     structured dtype, so the parts are three arrays: NumPy's values and descriptors, not its strides or
            ///     OWNDATA, and a part NumPy sizes at width 0 has width 1.</remarks>
            public static (NDArray Before, NDArray Separator, NDArray After) partition(NDArray a, NDArray sep)
            {
                // The parts' class follows a alone: NumPy's out= is np.empty_like(a) (a separator's class never reaches it),
                // and a StringDType result is never a chararray.
                if (NDCharArrayBoundary.Sees(a))
                {
                    var parts = NDCharArrayBoundary.WrappedAll((a, sep), static s =>
                    {
                        var (b, m, e) = partition(s.a, s.sep);
                        return new[] { b, m, e };
                    });
                    return (parts[0], parts[1], parts[2]);
                }
                return Partition(a, sep, back: false);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.rpartition(a, sep)</c>: each element split around the LAST occurrence of
            ///     <paramref name="sep"/> into (before, separator, after); without a match, (empty, empty, the element).
            /// </summary>
            /// <param name="a">The strings.</param>
            /// <param name="sep">The separators (see <see cref="partition"/>).</param>
            /// <returns>The three parts of the broadcast shape.</returns>
            /// <exception cref="ArgumentNullException">An operand is null.</exception>
            /// <exception cref="TypeError">No common dtype or no loop for the classes; two different <c>StringDType</c> NA
            ///     objects.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">An empty separator; an empty broadcast; a numeric operand cast to text; a
            ///     <c>StringDType</c> NA whose NA object is not a string.</exception>
            /// <exception cref="UnicodeEncodeError">A non-ASCII separator cast to <c>S</c>.</exception>
            public static (NDArray Before, NDArray Separator, NDArray After) rpartition(NDArray a, NDArray sep)
            {
                if (NDCharArrayBoundary.Sees(a))
                {
                    var parts = NDCharArrayBoundary.WrappedAll((a, sep), static s =>
                    {
                        var (b, m, e) = rpartition(s.a, s.sep);
                        return new[] { b, m, e };
                    });
                    return (parts[0], parts[1], parts[2]);
                }
                return Partition(a, sep, back: true);
            }

            /// <summary>
            ///     The partition wrappers: <c>np.asanyarray</c> of both operands, the <c>StringDType</c> ufunc when
            ///     <c>np.result_type(a, sep)</c> is <c>StringDType</c>, else the fixed-width sizing and loop. The public
            ///     wrappers apply the chararray rule (three chararray views).
            /// </summary>
            /// <param name="a">The strings.</param>
            /// <param name="sep">The separators.</param>
            /// <param name="back">True for <c>rpartition</c>.</param>
            /// <returns>The three parts.</returns>
            /// <exception cref="ArgumentNullException">An operand is null.</exception>
            /// <exception cref="TypeError">See <see cref="partition"/>.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">See <see cref="partition"/>.</exception>
            private static (NDArray Before, NDArray Separator, NDArray After) Partition(NDArray a, NDArray sep, bool back)
            {
                RequireOperand(a, nameof(a));
                // The chararray rule belongs to the public call above: the public functions this body calls compute plain
                // (a chararray in another position must not reach their rules from inside the library).
                using var plain = NDCharArrayBoundary.Inside();
                RequireOperand(sep, nameof(sep));
                NDArray arr = null, s = null;
                try
                {
                    arr = NDStringMethods.AsAnyArray(a);
                    s = NDStringMethods.AsAnyArray(sep);
                    if (np.result_type(arr, s).@char == 'T')
                        return NDVStringMethods.Partition(arr, s, back);
                    return NDStringMethods.Partition(arr, s, back);
                }
                finally
                {
                    NDStringMethods.DisposeIfTemp(arr, a);
                    NDStringMethods.DisposeIfTemp(s, sep);
                }
            }

            /// <summary>NumPy's <c>np.strings.slice(a)</c>: every element whole (<c>a[:]</c> of each string).</summary>
            /// <param name="a">The strings (<c>S</c>, <c>U</c> or <c>StringDType</c>).</param>
            /// <returns>A fresh array of <paramref name="a"/>'s dtype.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">A non-text operand; a <c>StringDType</c> NA (<c>Cannot slice null
            ///     string</c>).</exception>
            public static NDArray slice(NDArray a)
                => NDCharArrayBoundary.Sees(a)
                    ? NDCharArrayBoundary.Wrapped(a, static s => slice(s))
                    : SliceCore(a, null, null, null);

            /// <summary>
            ///     NumPy's <c>np.strings.slice(a, stop)</c>: one argument after <paramref name="a"/> is the STOP, as in
            ///     Python's <c>slice(stop)</c> — each element's <c>s[:stop]</c>; null is NumPy's <c>None</c> (the whole
            ///     string).
            /// </summary>
            /// <param name="a">The strings.</param>
            /// <param name="stop">The stops (characters; negative counts from the end), any integer dtype, or null.</param>
            /// <returns>A fresh array of <paramref name="a"/>'s dtype.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">A non-text operand; a <c>StringDType</c> NA.</exception>
            /// <exception cref="ArgumentException">A stop that cannot reach int64 by a same_kind cast.</exception>
            /// <exception cref="OverflowException">A stop literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            public static NDArray slice(NDArray a, NDArray stop)
                => NDCharArrayBoundary.Sees(a)
                    ? NDCharArrayBoundary.Wrapped((a, stop), static s => slice(s.a, s.stop))
                    : SliceCore(a, null, stop, null);

            /// <summary>NumPy's <c>np.strings.slice(a, start, stop)</c>: each element's <c>s[start:stop]</c> (null bounds are
            ///     NumPy's <c>None</c>).</summary>
            /// <param name="a">The strings.</param>
            /// <param name="start">The starts, or null.</param>
            /// <param name="stop">The stops, or null.</param>
            /// <returns>A fresh array of <paramref name="a"/>'s dtype.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">A non-text operand; a <c>StringDType</c> NA.</exception>
            /// <exception cref="ArgumentException">A bound that cannot reach int64 by a same_kind cast.</exception>
            /// <exception cref="OverflowException">A bound literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            public static NDArray slice(NDArray a, NDArray start, NDArray stop)
                => NDCharArrayBoundary.Sees(a)
                    ? NDCharArrayBoundary.Wrapped((a, start, stop), static s => slice(s.a, s.start, s.stop))
                    : SliceCore(a, start, stop, null);

            /// <summary>
            ///     NumPy's <c>np.strings.slice(a, start, stop, step)</c>: each element's <c>s[start:stop:step]</c>, characters
            ///     (code points for <c>StringDType</c>), every argument broadcast.
            /// </summary>
            /// <param name="a">The strings.</param>
            /// <param name="start">The starts, or null (NumPy's <c>None</c>: the end for a negative step, 0 otherwise).</param>
            /// <param name="stop">The stops, or null (past the beginning for a negative step, past the end otherwise).</param>
            /// <param name="step">The steps (an integer array or literal, never 0), or null for 1.</param>
            /// <returns>A fresh array of <paramref name="a"/>'s dtype.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="step"/> is not an integer array (<c>unsupported type float64 for
            ///     operand 'step'</c>); a non-text operand; a <c>StringDType</c> NA (<c>Cannot slice null string</c>).</exception>
            /// <exception cref="ValueError">A zero step (<c>slice step cannot be zero</c>).</exception>
            /// <exception cref="ArgumentException">A bound that cannot reach int64 by a same_kind cast.</exception>
            /// <exception cref="OverflowException">A bound literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            public static NDArray slice(NDArray a, NDArray start, NDArray stop, NDArray step)
                => NDCharArrayBoundary.Sees(a)
                    ? NDCharArrayBoundary.Wrapped((a, start, stop, step), static s => slice(s.a, s.start, s.stop, s.step))
                    : SliceCore(a, start, stop, step);

            /// <summary>
            ///     The slice wrapper after its arity rule (one argument is the stop), statement for statement: the step's
            ///     <c>np.asanyarray</c>, integer and zero checks, the <c>np.where</c> defaults of a missing start / stop, then
            ///     <c>_slice</c>. The public overloads apply the chararray rule.
            /// </summary>
            /// <param name="a">The strings.</param>
            /// <param name="start">The starts, or null.</param>
            /// <param name="stop">The stops, or null.</param>
            /// <param name="step">The steps, or null.</param>
            /// <returns>The sliced strings.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError">See <see cref="slice(NDArray, NDArray, NDArray, NDArray)"/>.</exception>
            /// <exception cref="ValueError">A zero step.</exception>
            /// <exception cref="ArgumentException">A bound that cannot reach int64.</exception>
            /// <exception cref="OverflowException">A bound literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            private static NDArray SliceCore(NDArray a, NDArray start, NDArray stop, NDArray step)
            {
                RequireOperand(a, nameof(a));
                // The chararray rule belongs to the public call above: the public functions this body calls compute plain
                // (a chararray in another position must not reach their rules from inside the library).
                using var plain = NDCharArrayBoundary.Inside();
                NDArray st = null, s0 = null, s1 = null;
                try
                {
                    // if step is None: step = 1; step = np.asanyarray(step); the integer and zero checks
                    st = step is null ? NDArray.Scalar(1L) : NDStringMethods.AsAnyArray(step);
                    if (!st.typecode.IsInteger())
                        throw new TypeError($"unsupported type {st.dtype} for operand 'step'");
                    using (var zero = np.equal(st, 0))
                    {
                        if (np.any(zero))
                            throw new ValueError("slice step cannot be zero");
                    }
                    s0 = start;
                    s1 = stop;
                    if (start is null || stop is null)
                    {
                        // start = np.where(step < 0, np.iinfo(np.intp).max, 0); stop = np.where(step < 0, intp.min, intp.max)
                        using var negative = np.less(st, 0);
                        if (start is null)
                        {
                            using var largest = NDArray.Scalar(long.MaxValue);
                            using var zero = NDArray.Scalar(0L);
                            s0 = np.where(negative, largest, zero);
                        }
                        if (stop is null)
                        {
                            using var smallest = NDArray.Scalar(long.MinValue);
                            using var largest = NDArray.Scalar(long.MaxValue);
                            s1 = np.where(negative, smallest, largest);
                        }
                    }
                    return NDStringMethods.Slice(a, s0, s1, st);
                }
                finally
                {
                    NDStringMethods.DisposeIfTemp(st, step);
                    NDStringMethods.DisposeIfTemp(s0, start);
                    NDStringMethods.DisposeIfTemp(s1, stop);
                }
            }
        }
    }
}
