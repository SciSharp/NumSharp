using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text;

namespace NumSharp.Backends.Printing
{
    /// <summary>
    ///     C# port of NumPy's array-to-string machinery in <c>numpy/_core/arrayprint.py</c>:
    ///     the recursive layout engine (<c>_formatArray</c>), the per-dtype formatter selection
    ///     (<c>_get_format_function</c>), and the <c>array2string</c> / <c>array_str</c> /
    ///     <c>array_repr</c> entry points.
    /// </summary>
    internal static class ArrayFormatter
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        /// <summary>
        ///     NumPy's <c>dtype_is_implied</c>: whether an array's repr can leave out <c>dtype=</c> — true when the dtype's
        ///     CLASS is in NumPy's <c>_typelessdata</c> (<c>int_</c>, the pointer-sized integer, <c>float64</c>,
        ///     <c>complex128</c> and <c>bool</c>; NumPy tests <c>dtype.type in _typelessdata</c>). It is the class, not the
        ///     storage lane, that decides: a <c>longdouble</c> array is float64 storage and prints <c>dtype=float64</c>, as in
        ///     NumPy (<c>array([0., 1.], dtype=float64)</c>), and so does a Linux <c>longlong</c> array (<c>dtype=int64</c>).
        /// </summary>
        /// <param name="dtype">The array's descriptor (NumSharp's arrays are always native-order, which NumPy also requires).</param>
        /// <returns>True when the repr shows no <c>dtype=</c>.</returns>
        internal static bool DtypeIsImplied(DType dtype)
        {
            var meta = dtype?.Meta;
            return meta != null
                   && (ReferenceEquals(meta, DTypeRegistry.Bool) || ReferenceEquals(meta, DTypeRegistry.Double)
                       || ReferenceEquals(meta, DTypeRegistry.Complex128)
                       || ReferenceEquals(meta, DTypeRegistry.FromTypeCode(IntPtr.Size == 8 ? NPTypeCode.Int64 : NPTypeCode.Int32)));
        }

        #region public entry points

        /// <summary>Port of <c>array_str</c> — the data of an array as a single string (NumPy <c>str</c>).</summary>
        public static string ArrayStr(NDArray a, PrintOptions opts)
        {
            // 0d arrays print like a scalar: floats are not truncated by precision and strings are
            // not quoted, so we return the scalar's str.
            if (a.ndim == 0)
                // A StringDType element is a Python str (or the descriptor's NA object): its str() is the text itself,
                // 'None', 'nan', ...
                return a.Storage.IsVString
                    ? Strings.NDPyObject.Str(a.GetAtIndex(0))
                    : ScalarStr(a.GetAtIndex(0), a.typecode);

            return Array2String(a, opts, " ", "", "");
        }

        /// <summary>Port of <c>array_repr</c> — the array with class/dtype/shape info (NumPy <c>repr</c>).</summary>
        public static string ArrayRepr(NDArray a, PrintOptions opts)
        {
            // NumPy: class_name = type(arr).__name__ for a subclass ("matrix"), "array" for an ndarray. The prefix
            // length indents every continuation line, so a subclass name moves the whole layout.
            string prefix = a.ReprClassName + "(";
            string lst = Array2String(a, opts, ", ", prefix, ")");

            var extras = new List<string>();
            bool shapeShown = (a.size == 0 && !(a.ndim == 1 && a.Shape.dimensions[0] == 0))
                              || (opts.legacy > 210 && a.size > opts.threshold);
            if (shapeShown)
                extras.Add("shape=" + ShapeTuple(a.Shape.dimensions));
            if (!DtypeIsImplied(a.dtype) || a.size == 0)
                extras.Add("dtype=" + DtypeShortRepr(a));

            if (extras.Count == 0)
                return prefix + lst + ")";

            string arrStr = prefix + lst + ",";
            string extraStr = string.Join(", ", extras) + ")";

            int lastNewline = arrStr.LastIndexOf('\n');
            // Python's len() of the last line: code points, so an astral character in a printed string counts once.
            int lastLineLen = PyRepr.CodePointLength(arrStr.Substring(lastNewline + 1));
            string spacer = " ";
            if (lastLineLen + extraStr.Length + 1 > opts.linewidth)
                spacer = "\n" + new string(' ', prefix.Length);

            return arrStr + spacer + extraStr;
        }

        /// <summary>Port of <c>array2string</c>.</summary>
        public static string Array2String(NDArray a, PrintOptions opts, string separator, string prefix, string suffix)
        {
            int linewidth = opts.linewidth;
            if (opts.legacy > 113)
                linewidth -= suffix.Length;

            // treat as a null array if any of the shape elements == 0
            if (a.size == 0)
                return "[]";

            bool summarize = a.size > opts.threshold;

            var reduced = CollectReducedValues(a, opts.edgeitems, summarize);
            // StringDType's scalar type is Python's str, not np.str_: NumPy's _get_format_function falls through to
            // 'numpystr', which prints repr() of each element object - the quoted text, or the NA object's repr (None,
            // nan, 'NA').
            IElementFormatter formatFn = a.Storage.IsVString
                ? new VStringReprFormat()
                : BuildFormatFunction(reduced, a.typecode, a.ndim, opts);
            // numpy.char.chararray: NumPy's _formatArray reads every element through the array's own __getitem__, which
            // strips a chararray's text (the format function itself is built over np.asarray's unstripped data, and a 0-d
            // array is formatted from np.asarray: unstripped). The run-time class decides, as type(arr) does in NumPy.
            if (a is NDCharArray && a.ndim > 0 && a.Storage.IsFixedString)
                formatFn = new StrippedTextReprFormat(formatFn);

            string nextLinePrefix = " " + new string(' ', prefix.Length);
            string summaryInsert = summarize ? "..." : "";

            return FormatArray(a, formatFn, linewidth, nextLinePrefix, separator,
                opts.edgeitems, summaryInsert, opts.legacy);
        }

        /// <summary>
        ///     Formats a masked array's DATA the way NumPy's <c>MaskedArray._insert_masked_print</c> does when a
        ///     mask is present and printing is enabled: the data is treated as an OBJECT array (each element its
        ///     Python-scalar repr, with NO numeric column alignment) and every masked slot is replaced by
        ///     <paramref name="display"/> (the <c>masked_print_option</c> token, e.g. <c>--</c>). This is why a
        ///     masked print looks unaligned (<c>[1 -- 3 --]</c>) versus the aligned numeric print of the same data.
        /// </summary>
        /// <param name="data">The data array (any layout); read positionally in lockstep with <paramref name="mask"/>.</param>
        /// <param name="mask">The boolean mask, same shape as <paramref name="data"/>. Must be non-null — the
        /// caller uses <see cref="Array2String"/> on the raw data (nomask) or the filled data (disabled) instead.</param>
        /// <param name="display">The token shown at masked positions (NumPy's <c>masked_print_option.display()</c>).</param>
        /// <param name="opts">Print options (threshold/edgeitems/linewidth drive summarization and wrapping).</param>
        /// <param name="separator">Element separator — <c>" "</c> for the str form, <c>", "</c> for the repr form.</param>
        /// <param name="prefix">Alignment prefix (its length only) for line wrapping, exactly as <see cref="Array2String"/>.</param>
        /// <param name="suffix">Trailing text (its length only) reserved on the last line.</param>
        /// <returns>The bracketed, wrapped, summarized object-style rendering with <c>--</c> at masked slots.</returns>
        public static string MaskedObjectArray2String(NDArray data, NDArray mask, string display,
            PrintOptions opts, string separator, string prefix, string suffix)
        {
            int linewidth = opts.linewidth;
            if (opts.legacy > 113)
                linewidth -= suffix.Length;

            if (data.size == 0)
                return "[]";

            bool summarize = data.size > opts.threshold;
            // Object-array element formatter: Python-scalar repr per element (no shared width), matching what
            // NumPy prints once the data is cast to dtype=object. No CollectReducedValues pass is needed — unlike
            // the numeric formatters, ScalarStr computes each element independently of the others.
            IElementFormatter formatFn = new ObjectReprFormat(data.typecode);

            string nextLinePrefix = " " + new string(' ', prefix.Length);
            string summaryInsert = summarize ? "..." : "";

            return FormatArray(data, formatFn, linewidth, nextLinePrefix, separator,
                opts.edgeitems, summaryInsert, opts.legacy, mask, display);
        }

        /// <summary>
        ///     <c>array_str</c> of an OBJECT array whose elements NumSharp keeps outside an <see cref="NDArray"/> (the lists of
        ///     <see cref="NDListArray"/>): NumPy's <c>_formatArray</c> layout over <paramref name="positions"/> — an int64 array of
        ///     the object array's shape holding each element's C-order position — with every element printed by
        ///     <paramref name="formatter"/> (NumPy's <c>_object_format</c>: no shared width, no alignment).
        /// </summary>
        /// <param name="positions">The positions (any shape of at least one axis; a 0-d object array prints its object's str()
        ///     itself, which the caller produces).</param>
        /// <param name="formatter">Prints the element at a position (the boxed int64 the layout engine reads).</param>
        /// <param name="opts">Print options (threshold / edgeitems summarize, linewidth wraps).</param>
        /// <returns>The bracketed text, <c>[]</c> for an empty array.</returns>
        internal static string ObjectArrayStr(NDArray positions, IElementFormatter formatter, PrintOptions opts)
            => ObjectArray2String(positions, formatter, opts, " ", "", "");

        /// <summary>
        ///     <c>array_repr</c> of such an object array: <c>array([...], dtype=object)</c> — the dtype is never implied, a
        ///     <c>shape=</c> extra appears for an empty array of another shape than <c>(0,)</c> and for a summarized one, and
        ///     the extras move to their own line when the last line would pass the line width (NumPy's spacer rule).
        /// </summary>
        /// <param name="positions">The positions (see <see cref="ObjectArrayStr"/>; a 0-d array is formatted too).</param>
        /// <param name="formatter">Prints the element at a position.</param>
        /// <param name="opts">Print options.</param>
        /// <returns>The repr.</returns>
        internal static string ObjectArrayRepr(NDArray positions, IElementFormatter formatter, PrintOptions opts)
        {
            const string prefix = "array(";
            string lst = ObjectArray2String(positions, formatter, opts, ", ", prefix, ")");
            var extras = new List<string>();
            bool shapeShown = (positions.size == 0 && !(positions.ndim == 1 && positions.Shape.dimensions[0] == 0))
                              || (opts.legacy > 210 && positions.size > opts.threshold);
            if (shapeShown)
                extras.Add("shape=" + ShapeTuple(positions.Shape.dimensions));
            // An object dtype is never one of NumPy's _typelessdata.
            extras.Add("dtype=object");

            string arrStr = prefix + lst + ",";
            string extraStr = string.Join(", ", extras) + ")";
            int lastNewline = arrStr.LastIndexOf('\n');
            int lastLineLen = PyRepr.CodePointLength(arrStr.Substring(lastNewline + 1));
            string spacer = lastLineLen + extraStr.Length + 1 > opts.linewidth ? "\n" + new string(' ', prefix.Length) : " ";
            return arrStr + spacer + extraStr;
        }

        /// <summary><c>array2string</c> of an object array kept outside an <see cref="NDArray"/> (see <see cref="ObjectArrayStr"/>).</summary>
        /// <param name="positions">The positions.</param>
        /// <param name="formatter">Prints the element at a position.</param>
        /// <param name="opts">Print options.</param>
        /// <param name="separator"><c>" "</c> for str, <c>", "</c> for repr.</param>
        /// <param name="prefix">The repr prefix (its length indents continuation lines).</param>
        /// <param name="suffix">The repr suffix (its length is reserved on the last line).</param>
        /// <returns>The bracketed text.</returns>
        private static string ObjectArray2String(NDArray positions, IElementFormatter formatter, PrintOptions opts,
                                                 string separator, string prefix, string suffix)
        {
            int linewidth = opts.linewidth;
            if (opts.legacy > 113)
                linewidth -= suffix.Length;
            if (positions.size == 0)
                return "[]";
            bool summarize = positions.size > opts.threshold;
            string nextLinePrefix = " " + new string(' ', prefix.Length);
            string summaryInsert = summarize ? "..." : "";
            return FormatArray(positions, formatter, linewidth, nextLinePrefix, separator, opts.edgeitems, summaryInsert, opts.legacy);
        }

        #endregion

        #region layout engine (_formatArray, as an explicit-stack walk)

        /// <summary>
        ///     NumPy's <c>_formatArray</c>: the bracketed, wrapped and summarized text of <paramref name="a"/>, the same
        ///     characters NumPy's recursion builds, produced by ONE explicit-stack walk that writes them in order into one
        ///     buffer. Its cost is linear in the text, whatever the rank.
        /// </summary>
        /// <param name="a">The array (any layout, any rank; a 0-d array prints its only element).</param>
        /// <param name="formatFn">Formats one element (the boxed value read at its storage offset).</param>
        /// <param name="lineWidth">The line width for the outermost level (each deeper level gets one less, NumPy's
        ///     <c>len(']')</c>, unless <paramref name="legacy"/> is 1.13).</param>
        /// <param name="nextLinePrefix">The outermost hanging indent; each deeper level indents one more space.</param>
        /// <param name="separator">The element separator (<c>" "</c> for str, <c>", "</c> for repr).</param>
        /// <param name="edgeItems">How many leading / trailing entries a summarized axis keeps.</param>
        /// <param name="summaryInsert"><c>"..."</c> when the array is summarized, else empty (no axis summarizes).</param>
        /// <param name="legacy">NumPy's legacy print mode as a number (<c>long.MaxValue</c> for none).</param>
        /// <param name="mask">The masked-array hook: a boolean array of <paramref name="a"/>'s shape (its own layout) walked
        ///     in lockstep; a true element prints <paramref name="display"/> instead of the value. Null on the plain path.</param>
        /// <param name="display">The masked slot's token (NumPy's <c>masked_print_option</c>, <c>--</c>).</param>
        /// <returns>The text.</returns>
        /// <exception cref="NotSupportedException">The storage has no element type a formatter can read.</exception>
        /// <remarks>
        ///     <para>
        ///         NumPy recurses once per axis, building every level's text as a string: <c>s = '[' +
        ///         s[len(hanging_indent):] + ']'</c> over children that each carry the level's hanging indent. Its 64-dimension
        ///         cap bounds that; NumSharp's uncapped rank did not, so the former port of the recursion (a frame per axis in
        ///         the value collection and the layout engine, a view per level, a new indent string per level, each level's
        ///         text copied into its parent's) overflowed the stack and ended the process from about 2,000 dimensions and
        ///         was quadratic in the rank below that (parity QA-50).
        ///     </para>
        ///     <para>
        ///         The walk keeps NumPy's text exactly: a level writes <c>[</c>, then its pieces (children, and <c>...</c> in a
        ///         summarized axis) separated by <c>separator.rstrip() + '\n' * (axes_left - 1)</c> plus the level's hanging
        ///         indent — the first piece's indent is the one NumPy strips — then <c>]</c>. A row (the last axis) is NumPy's
        ///         <c>_extendLine</c> loop, written by <see cref="FormatRow"/>. Elements are read at
        ///         <c>Shape.offset + Σ index·stride</c>, the offset the former <c>cur[i]</c> views resolved.
        ///     </para>
        /// </remarks>
        private static string FormatArray(NDArray a, IElementFormatter formatFn, int lineWidth,
            string nextLinePrefix, string separator, int edgeItems, string summaryInsert, long legacy,
            NDArray mask = null, string display = null)
        {
            UnmanagedStorage data = a.Storage;
            UnmanagedStorage maskData = mask?.Storage;
            long maskOrigin = mask is null ? 0 : mask.Shape.offset;
            int nd = a.ndim;

            // axes_left == 0 at the top: a 0-d array is its only element.
            if (nd == 0)
                return FormatLeafAt(formatFn, data, a.Shape.offset, maskData, maskOrigin, display);

            long[] dims = a.Shape.dimensions;
            long[] strides = a.Shape.strides;
            long[] maskStrides = mask?.Shape.strides;
            var layout = new LayoutText(nextLinePrefix, separator, summaryInsert, edgeItems, legacy, formatFn, display);
            var sb = new StringBuilder();

            // A 1-D array is one row at depth 0.
            if (nd == 1)
            {
                FormatRow(sb, layout, data, a.Shape.offset, strides[0], maskData, maskOrigin, maskStrides?[0] ?? 0, dims[0], 0,
                    lineWidth);
                return sb.ToString();
            }

            // Per depth: the next piece of the level and the data / mask offsets of the level's first element. One slot
            // per axis on the heap: the walk's state is O(ndim) memory, never stack.
            var piece = new long[nd];
            var origin = new long[nd];
            var maskAt = new long[nd];
            origin[0] = a.Shape.offset;
            maskAt[0] = maskOrigin;
            int depth = 0;
            sb.Append('[');

            while (depth >= 0)
            {
                int d = depth;
                Pieces(dims[d], edgeItems, summaryInsert, out long leading, out bool showSummary, out long trailing);
                long count = leading + (showSummary ? 1 : 0) + trailing;
                long k = piece[d];

                // The level is done: close it and move the parent on to its next piece.
                if (k == count)
                {
                    sb.Append(']');
                    depth--;
                    if (depth >= 0)
                        piece[depth]++;
                    continue;
                }

                // Between pieces: line_sep, then this level's hanging indent (nextLinePrefix + d spaces). NumPy prefixes
                // EVERY piece with the indent and strips the first one, which is the same as writing it from the second on.
                if (k > 0)
                {
                    sb.Append(layout.SeparatorRStrip).Append('\n', nd - d - 1);
                    sb.Append(nextLinePrefix).Append(' ', d);
                }

                long child = ChildOf(k, dims[d], leading, showSummary, trailing);
                if (child < 0)
                {
                    sb.Append(summaryInsert);
                    piece[d]++;
                    continue;
                }

                long childOrigin = origin[d] + child * strides[d];
                long childMask = maskData is null ? 0 : maskAt[d] + child * maskStrides[d];
                if (d + 1 == nd - 1)
                {
                    // The child is a row: written whole, then this level moves on. The row's width shrinks by one per level
                    // above it (NumPy's next_width), except in the 1.13 legacy mode.
                    int rowWidth = legacy <= 113 ? lineWidth : lineWidth - (d + 1);
                    FormatRow(sb, layout, data, childOrigin, strides[nd - 1], maskData, childMask,
                        maskStrides?[nd - 1] ?? 0, dims[nd - 1], d + 1, rowWidth);
                    piece[d]++;
                }
                else
                {
                    // Descend into a level of two or more axes.
                    depth = d + 1;
                    origin[depth] = childOrigin;
                    maskAt[depth] = childMask;
                    piece[depth] = 0;
                    sb.Append('[');
                }
            }

            return sb.ToString();
        }

        /// <summary>
        ///     The constant inputs of one <see cref="FormatArray"/> call, gathered so the row writer does not take a dozen
        ///     parameters; built once per call.
        /// </summary>
        private sealed class LayoutText
        {
            /// <summary>The outermost hanging indent; a level at depth d indents with it plus d spaces.</summary>
            internal readonly string NextLinePrefix;

            /// <summary>Python's <c>len()</c> of <see cref="NextLinePrefix"/> (code points, as NumPy measures a line).</summary>
            internal readonly long PrefixCodePoints;

            /// <summary>The element separator.</summary>
            internal readonly string Separator;

            /// <summary>Python's <c>len()</c> of <see cref="Separator"/>.</summary>
            internal readonly long SeparatorCodePoints;

            /// <summary><c>separator.rstrip()</c>: the end of a line before a row break, and of <c>line_sep</c>.</summary>
            internal readonly string SeparatorRStrip;

            /// <summary><c>"..."</c> for a summarized array, else empty.</summary>
            internal readonly string SummaryInsert;

            /// <summary>How many leading / trailing entries a summarized axis keeps.</summary>
            internal readonly int EdgeItems;

            /// <summary>NumPy's legacy mode as a number.</summary>
            internal readonly long Legacy;

            /// <summary>The element formatter.</summary>
            internal readonly IElementFormatter Format;

            /// <summary>The masked slot's token (unused on the plain path).</summary>
            internal readonly string Display;

            /// <summary>Gathers the call's constants and measures the strings the row writer measures per line.</summary>
            /// <param name="nextLinePrefix">The outermost hanging indent.</param>
            /// <param name="separator">The element separator.</param>
            /// <param name="summaryInsert">The summary marker or empty.</param>
            /// <param name="edgeItems">The edge item count.</param>
            /// <param name="legacy">The legacy mode.</param>
            /// <param name="format">The element formatter.</param>
            /// <param name="display">The masked slot's token.</param>
            internal LayoutText(string nextLinePrefix, string separator, string summaryInsert, int edgeItems, long legacy,
                IElementFormatter format, string display)
            {
                NextLinePrefix = nextLinePrefix;
                PrefixCodePoints = PyRepr.CodePointLength(nextLinePrefix);
                Separator = separator;
                SeparatorCodePoints = PyRepr.CodePointLength(separator);
                SeparatorRStrip = RStrip(separator);
                SummaryInsert = summaryInsert;
                EdgeItems = edgeItems;
                Legacy = legacy;
                Format = format;
                Display = display;
            }
        }

        /// <summary>
        ///     NumPy's piece plan for one axis of <c>_formatArray</c>: <c>leading_items</c> children from the front, the
        ///     summary marker when <c>summary_insert and 2*edge_items &lt; a_len</c>, then the trailing children
        ///     (<c>range(trailing_items, 1, -1)</c> plus the always-printed last one).
        /// </summary>
        /// <param name="aLen">The axis length (at least 1: an empty array never reaches the layout engine).</param>
        /// <param name="edgeItems">The edge item count.</param>
        /// <param name="summaryInsert">The summary marker or empty.</param>
        /// <param name="leading">The children printed before the marker.</param>
        /// <param name="showSummary">Whether the marker is printed.</param>
        /// <param name="trailing">The children printed after the marker, the last one included: <c>edge_items</c> (one when
        ///     <c>edge_items</c> is 0, because NumPy always prints the last child) for a summarized axis, else all of
        ///     them.</param>
        private static void Pieces(long aLen, int edgeItems, string summaryInsert, out long leading, out bool showSummary,
            out long trailing)
        {
            showSummary = summaryInsert.Length != 0 && 2L * edgeItems < aLen;
            leading = showSummary ? edgeItems : 0L;
            trailing = showSummary ? Math.Max(edgeItems, 1L) : aLen;
        }

        /// <summary>The axis index of piece <paramref name="k"/> of a level, or -1 for the summary marker.</summary>
        /// <param name="k">The piece's position in the level (0-based).</param>
        /// <param name="aLen">The axis length.</param>
        /// <param name="leading">The children before the marker.</param>
        /// <param name="showSummary">Whether the level prints the marker.</param>
        /// <param name="trailing">The children after it (see <see cref="Pieces"/>).</param>
        /// <returns>The child's index along the axis, or -1 when the piece is the marker.</returns>
        private static long ChildOf(long k, long aLen, long leading, bool showSummary, long trailing)
        {
            if (k < leading)
                return k;
            if (showSummary && k == leading)
                return -1;
            return aLen - trailing + (k - leading - (showSummary ? 1 : 0));
        }

        /// <summary>
        ///     One row of <c>_formatArray</c> (<c>axes_left == 1</c>), appended to <paramref name="sb"/> as
        ///     <c>'[' + s[len(hanging_indent):] + ']'</c>: the elements (and the summary marker) through NumPy's
        ///     <c>_extendLine</c> loop, which breaks the line before a word that would pass the row's element width and starts
        ///     the next line with the row's hanging indent.
        /// </summary>
        /// <param name="sb">The output buffer.</param>
        /// <param name="layout">The call's constants.</param>
        /// <param name="data">The array's storage.</param>
        /// <param name="first">The storage offset of the row's element 0.</param>
        /// <param name="stride">The row's element stride (elements; 0 for a broadcast row).</param>
        /// <param name="mask">The mask's storage, or null.</param>
        /// <param name="maskFirst">The mask's offset of the row's element 0.</param>
        /// <param name="maskStride">The mask row's element stride.</param>
        /// <param name="aLen">The row length.</param>
        /// <param name="depth">The row's depth (its hanging indent is <c>nextLinePrefix</c> plus this many spaces).</param>
        /// <param name="currWidth">The row's width (NumPy's <c>curr_width</c> at this depth).</param>
        /// <exception cref="NotSupportedException">The storage has no element type a formatter can read.</exception>
        /// <remarks>
        ///     The hanging indent is never built as a string: its length is counted and its characters written as a space run
        ///     where they appear. NumPy strips the row text's first <c>len(hanging_indent)</c> characters; the writer drops
        ///     them as it goes (<c>toStrip</c>), which is the first line's indent in every real case and stays exact in the
        ///     pathological one where the first line strips to nothing. Line lengths are counted additively in code points
        ///     (NumPy's <c>len(line)</c>): the words are whole element reprs and ASCII separators, so no surrogate pair
        ///     straddles two of them.
        /// </remarks>
        private static void FormatRow(StringBuilder sb, LayoutText layout, UnmanagedStorage data, long first, long stride,
            UnmanagedStorage mask, long maskFirst, long maskStride, long aLen, int depth, int currWidth)
        {
            string prefix = layout.NextLinePrefix;
            long indentLength = prefix.Length + (long)depth;            // len(hanging_indent), UTF-16 units
            long indentCodePoints = layout.PrefixCodePoints + depth;   // Python's len(hanging_indent)
            // elem_width = curr_width - max(len(separator.rstrip()), len(']'))
            long elemWidth = currWidth - Math.Max(layout.SeparatorRStrip.Length, 1);

            Pieces(aLen, layout.EdgeItems, layout.SummaryInsert, out long leading, out bool showSummary, out long trailing);
            long count = leading + (showSummary ? 1 : 0) + trailing;

            sb.Append('[');
            long toStrip = indentLength;   // the characters of the row text NumPy's s[len(hanging_indent):] drops
            var line = new StringBuilder();  // the current line AFTER its hanging indent
            long lineCodePoints = indentCodePoints;

            for (long k = 0; k < count; k++)
            {
                long child = ChildOf(k, aLen, leading, showSummary, trailing);
                string word = child < 0
                    ? layout.SummaryInsert
                    : FormatLeafAt(layout.Format, data, first + child * stride, mask, maskFirst + child * maskStride,
                        layout.Display);

                // _extendLine: wrap before a word that passes the width, unless the line holds only its indent (a wrap
                // there would not help; not in the 1.13 legacy mode).
                long wordCodePoints = PyRepr.CodePointLength(word);
                bool needsWrap = lineCodePoints + wordCodePoints > elemWidth;
                if (layout.Legacy > 113 && lineCodePoints <= indentLength)
                    needsWrap = false;
                if (needsWrap)
                {
                    // s += line.rstrip() + "\n", where line = hanging indent + the words so far.
                    string content = RStrip(line.ToString());
                    if (content.Length > 0)
                    {
                        AppendIndentDropping(sb, prefix, depth, ref toStrip);
                        AppendDropping(sb, content, ref toStrip);
                    }
                    else
                    {
                        // The whole line is spaces: rstrip keeps only the part of the indent before its trailing spaces.
                        AppendDropping(sb, RStrip(prefix), ref toStrip);
                    }
                    AppendDropping(sb, "\n", ref toStrip);
                    line.Clear();
                    lineCodePoints = indentCodePoints;
                }

                line.Append(word);
                lineCodePoints += wordCodePoints;
                // Every piece but the last element is followed by the separator (the marker included).
                if (k + 1 < count)
                {
                    line.Append(layout.Separator);
                    lineCodePoints += layout.SeparatorCodePoints;
                }
            }

            // s += line: the last line keeps its indent and is not stripped.
            AppendIndentDropping(sb, prefix, depth, ref toStrip);
            AppendDropping(sb, line.ToString(), ref toStrip);
            sb.Append(']');
        }

        /// <summary>
        ///     Appends a hanging indent (<paramref name="prefix"/> followed by <paramref name="depth"/> spaces) to the row text,
        ///     dropping as many of its leading characters as <paramref name="toStrip"/> still owes.
        /// </summary>
        /// <param name="sb">The output buffer.</param>
        /// <param name="prefix">The outermost hanging indent.</param>
        /// <param name="depth">The spaces added after it.</param>
        /// <param name="toStrip">The characters still to drop; decreased by what this call drops.</param>
        private static void AppendIndentDropping(StringBuilder sb, string prefix, int depth, ref long toStrip)
        {
            AppendDropping(sb, prefix, ref toStrip);
            if (toStrip >= depth)
            {
                toStrip -= depth;
                return;
            }
            sb.Append(' ', depth - (int)toStrip);
            toStrip = 0;
        }

        /// <summary>
        ///     Appends <paramref name="text"/> to the row text, dropping as many of its leading characters as
        ///     <paramref name="toStrip"/> still owes (NumPy's <c>s[len(hanging_indent):]</c>, applied while writing).
        /// </summary>
        /// <param name="sb">The output buffer.</param>
        /// <param name="text">The text.</param>
        /// <param name="toStrip">The characters still to drop; decreased by what this call drops.</param>
        private static void AppendDropping(StringBuilder sb, string text, ref long toStrip)
        {
            if (toStrip >= text.Length)
            {
                toStrip -= text.Length;
                return;
            }
            sb.Append(text, (int)toStrip, text.Length - (int)toStrip);
            toStrip = 0;
        }

        /// <summary>Python's <c>str.rstrip()</c> as NumPy's layout uses it here: trailing SPACES only (a separator's, a
        ///     line's before a break).</summary>
        /// <param name="s">The text.</param>
        /// <returns><paramref name="s"/> without its trailing spaces.</returns>
        private static string RStrip(string s) => s.TrimEnd(' ');

        /// <summary>
        ///     Formats the element at a storage offset. On the masked path (<paramref name="mask"/> not null) a true mask
        ///     element renders as <paramref name="display"/> (NumPy's <c>masked_print_option</c>, e.g. <c>--</c>) instead of
        ///     the value — the substitution <c>_insert_masked_print</c> performs; otherwise it is the formatter's own output.
        /// </summary>
        /// <param name="fn">The element formatter.</param>
        /// <param name="data">The array's storage.</param>
        /// <param name="offset">The element's storage offset (<c>Shape.offset + Σ index·stride</c>).</param>
        /// <param name="mask">The mask's storage, or null.</param>
        /// <param name="maskOffset">The mask element's storage offset (ignored without a mask).</param>
        /// <param name="display">The masked slot's token.</param>
        /// <returns>The element's text.</returns>
        /// <exception cref="NotSupportedException">The storage has no element type a formatter can read.</exception>
        /// <remarks><see cref="Convert.ToBoolean(object)"/> reads the boolean mask element whatever its storage layout, as
        ///     the former view-based read did.</remarks>
        private static string FormatLeafAt(IElementFormatter fn, UnmanagedStorage data, long offset, UnmanagedStorage mask,
            long maskOffset, string display)
            => mask is not null && Convert.ToBoolean(mask.ElementAtResolvedOffset(maskOffset))
                ? display
                : fn.Format(data.ElementAtResolvedOffset(offset));

        #endregion

        #region formatter selection (_get_format_function)

        private static IElementFormatter BuildFormatFunction(IReadOnlyList<object> reduced,
            NPTypeCode tc, int rank, PrintOptions opts)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean:
                    return new BoolFormat(rank);
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return new IntegerFormat(reduced, opts.sign);
                case NPTypeCode.Half:
                    return new FloatingFormat(reduced, FloatKind.Half, 3, opts, opts.sign);
                case NPTypeCode.Single:
                    return new FloatingFormat(reduced, FloatKind.Single, 6, opts, opts.sign);
                case NPTypeCode.Double:
                    return new FloatingFormat(reduced, FloatKind.Double, 15, opts, opts.sign);
                case NPTypeCode.Decimal:
                    return new FloatingFormat(reduced, FloatKind.Double, 15, opts, opts.sign);
                case NPTypeCode.Complex:
                    return new ComplexFloatingFormat(reduced, opts, opts.sign);
                // NumPy's 'numpystr' formatter: repr() of each str / bytes scalar, never padded.
                case NPTypeCode.String:
                    return new StringReprFormat();
                default:
                    throw new NotSupportedException($"Cannot format dtype {tc}.");
            }
        }

        /// <summary>
        ///     The leading / trailing "corner" values NumPy's <c>_leading_trailing</c> feeds to <c>_get_format_function</c>
        ///     (only these take part in the column-width and exponent-format decisions), in NumPy's order: an odometer over
        ///     the visited indices of every axis, the last axis fastest.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="edgeitems">How many leading / trailing indices a summarized axis keeps.</param>
        /// <param name="summarize">Whether the array is summarized (an axis longer than <c>2·edgeitems</c> then visits only
        ///     its first and last <paramref name="edgeitems"/> indices; a summarized array with <c>edgeitems == 0</c>
        ///     collects nothing).</param>
        /// <returns>The values, boxed as the storage boxes them.</returns>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        /// <remarks>
        ///     One loop with O(ndim) state, reading each value at <c>Shape.offset + Σ index·stride</c>: the former recursion
        ///     (a frame and a view per axis) overflowed the stack from about 2,000 dimensions and was quadratic in the rank
        ///     (parity QA-50). Axes that visit a single index never move, so the odometer carries only through the others.
        /// </remarks>
        private static List<object> CollectReducedValues(NDArray a, int edgeitems, bool summarize)
        {
            var result = new List<object>();
            if (a.ndim == 0)
            {
                result.Add(a.GetAtIndex(0));
                return result;
            }

            int nd = a.ndim;
            long[] dims = a.Shape.dimensions;
            long[] strides = a.Shape.strides;
            UnmanagedStorage data = a.Storage;

            // The axes that visit more than one index, outermost first; an axis that visits none empties the result
            // (the recursion visited no element under it).
            var moving = new List<int>();
            var visits = new long[nd];
            var cut = new bool[nd];
            for (int d = 0; d < nd; d++)
            {
                cut[d] = summarize && dims[d] > 2L * edgeitems;
                visits[d] = cut[d] ? 2L * edgeitems : dims[d];
                if (visits[d] == 0)
                    return result;
                if (visits[d] > 1)
                    moving.Add(d);
            }

            // Every axis starts at its first visited index, 0, so the walk starts at the array's own offset.
            var position = new long[nd];
            long offset = a.Shape.offset;
            while (true)
            {
                result.Add(data.ElementAtResolvedOffset(offset));

                // Advance the odometer over the moving axes, last fastest, keeping the offset in step.
                int m = moving.Count - 1;
                for (; m >= 0; m--)
                {
                    int d = moving[m];
                    long from = VisitedIndex(position[d], dims[d], edgeitems, cut[d]);
                    if (++position[d] < visits[d])
                    {
                        offset += (VisitedIndex(position[d], dims[d], edgeitems, cut[d]) - from) * strides[d];
                        break;
                    }
                    // Wrap back to index 0 and carry into the next axis out.
                    offset -= from * strides[d];
                    position[d] = 0;
                }
                if (m < 0)
                    return result;
            }
        }

        /// <summary>The axis index of an axis's <paramref name="position"/>-th visited index.</summary>
        /// <param name="position">The position in the axis's visit list.</param>
        /// <param name="dim">The axis length.</param>
        /// <param name="edgeitems">The edge item count.</param>
        /// <param name="cut">Whether the axis is summarized (it visits <c>0..edgeitems-1</c>, then
        ///     <c>dim-edgeitems..dim-1</c>).</param>
        /// <returns>The index along the axis.</returns>
        private static long VisitedIndex(long position, long dim, int edgeitems, bool cut)
            => cut && position >= edgeitems ? dim - 2L * edgeitems + position : position;

        #endregion

        #region scalar str / dtype-shape helpers

        /// <summary>Element formatter used for the masked (object-array) print path: renders each element with its
        /// Python-scalar repr via <see cref="ScalarStr"/> — no shared/padded width, since object arrays are not
        /// column-aligned. This is what makes a masked print unaligned versus the numeric formatters' padded output.</summary>
        private sealed class ObjectReprFormat : IElementFormatter
        {
            private readonly NPTypeCode _tc;
            /// <summary>Captures the data typecode so each element is stringified with the right scalar rules.</summary>
            /// <param name="tc">The data array's element typecode.</param>
            public ObjectReprFormat(NPTypeCode tc) => _tc = tc;
            /// <summary>The Python-scalar repr of one element (int decimal, shortest-round-trip float, complex "(a+bj)", etc.).
            /// A string element of an object array prints as Python's repr of the str / bytes (quoted), which differs from
            /// the unquoted str() a 0-d string array prints.</summary>
            /// <param name="value">The boxed element value.</param>
            /// <returns>The element's object-array string form.</returns>
            /// <remarks>A float16 or float32 element prints as the Python FLOAT it becomes: NumPy's <c>astype(object)</c>
            /// converts it through <c>PyFloat_FromDouble</c>, so float16 1.732 prints <c>1.732421875</c> and float32 0.1
            /// <c>0.10000000149011612</c> (the double's shortest repr), not the narrow type's shortest digits (they did until
            /// 2026-10-08; found by the ma_fill tier's repr cases).</remarks>
            public string Format(object value) => _tc switch
            {
                // A str / bytes element by its repr; a StringDType NA element is its NA object (an object array holds the
                // na_object itself: None, nan, or a sentinel's repr).
                NPTypeCode.String => Strings.NDPyObject.Repr(value),
                NPTypeCode.Half => PythonFloatRepr((double)(Half)value, FloatKind.Double),
                NPTypeCode.Single => PythonFloatRepr((double)(float)value, FloatKind.Double),
                _ => ScalarStr(value, _tc),
            };
        }

        // str() of a 0d array element (Python scalar str semantics).
        internal static string ScalarStr(object value, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean:
                    return Convert.ToBoolean(value) ? "True" : "False";
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return IntegerFormat.DecimalString(value);
                case NPTypeCode.Half:
                    return PythonFloatRepr(value, FloatKind.Half);
                case NPTypeCode.Single:
                    return PythonFloatRepr(value, FloatKind.Single);
                case NPTypeCode.Double:
                    return PythonFloatRepr(value, FloatKind.Double);
                case NPTypeCode.Decimal:
                    return PythonFloatRepr((double)(decimal)value, FloatKind.Double);
                case NPTypeCode.Complex:
                    return PythonComplexRepr((Complex)value);
                // str() of NumPy's string scalars: np.str_ prints its text raw (no quotes, no escapes); np.bytes_ has no
                // str of its own, so NumPy's _guarded_repr_or_str prints its repr, np.bytes_(b'...').
                case NPTypeCode.String:
                    return value is byte[] bytes ? "np.bytes_(" + PyRepr.Bytes(bytes) + ")" : (string)value;
                default:
                    return value?.ToString() ?? "";
            }
        }

        // NumPy scalar float repr (scalartypes.c.src @name@type_@kind@_either): shortest digits, and
        // positional iff 1e-4 <= |value| < max_positional, else scientific. max_positional is per dtype —
        // float16 -> 1e3, float32 -> 1e6, float64 -> 1e16 — and the cutoff is tested on the VALUE, not on
        // the shortest-repr exponent: e.g. float32(1e-4) rounds to 9.999999e-5 < 1e-4 so it prints "1e-04".
        // For float64 this is exactly equivalent to the old decExp in [-4, 16) test (no behavior change);
        // it only tightens float32/float16 to their true (narrower) positional windows.
        internal static string PythonFloatRepr(object v, FloatKind kind)
        {
            double d = kind switch
            {
                FloatKind.Half => (double)(Half)v,
                FloatKind.Single => (float)v,
                _ => Convert.ToDouble(v, CI)
            };

            if (double.IsNaN(d)) return "nan";
            if (double.IsPositiveInfinity(d)) return "inf";
            if (double.IsNegativeInfinity(d)) return "-inf";

            var (digits, decExp, neg) = Dragon4.ExtractShortest(v, kind);
            string sign = neg ? "-" : "";
            if (digits == "0")
                return sign + "0.0";

            double maxPositional = kind switch
            {
                FloatKind.Half => 1e3,
                FloatKind.Single => 1e6,
                _ => 1e16
            };
            double absval = Math.Abs(d);

            if (absval >= 1e-4 && absval < maxPositional)
            {
                var (intStr, fracStr) = Dragon4.SplitPositional(digits, decExp);
                return fracStr.Length == 0 ? sign + intStr + ".0" : sign + intStr + "." + fracStr;
            }

            string lead = digits.Substring(0, 1);
            string rest = digits.Length > 1 ? "." + digits.Substring(1) : "";
            string es = (decExp >= 0 ? "+" : "-") + Math.Abs(decExp).ToString(CI).PadLeft(2, '0');
            return sign + lead + rest + "e" + es;
        }

        // str() of a 0d complex array, matching Python's complex repr "(a+bj)". Internal: numpy.polynomial's error texts
        // (PolyIndexArgument.Format) interpolate a Python complex with the same repr.
        internal static string PythonComplexRepr(Complex c)
        {
            // Pure-imaginary form when the real part is +0.0: just "<imag>j" carrying the imaginary's
            // OWN sign (no forced '+'). Python: str(0j)='0j', str(1j)='1j', str(-1j)='-1j', str(0-0j)='-0j'.
            // A negative-zero real (-0.0) is NOT this case — it falls through to the parenthesized form
            // ("(-0+0j)"), matching CPython's complex_repr.
            if (c.Real == 0.0 && !double.IsNegative(c.Real))
                return PythonFloatPart(c.Imaginary) + "j";

            // Parenthesized "(a±bj)": the imaginary part always carries an explicit sign.
            string im = PythonFloatPart(c.Imaginary);
            if (!(im.StartsWith("-") || im.StartsWith("+")))
                im = "+" + im;

            return "(" + PythonFloatPart(c.Real) + im + "j)";
        }

        // float repr used inside a complex: like Python's, integers drop the ".0".
        private static string PythonFloatPart(double d)
        {
            if (double.IsNaN(d)) return "nan";
            if (double.IsPositiveInfinity(d)) return "inf";
            if (double.IsNegativeInfinity(d)) return "-inf";
            string s = PythonFloatRepr(d, FloatKind.Double);
            if (s.EndsWith(".0"))
                s = s.Substring(0, s.Length - 2);
            return s;
        }

        /// <summary>
        ///     NumPy's <c>dtype_short_repr</c>: a builtin numeric dtype prints its name (<c>dtype=int32</c>); a flexible
        ///     dtype prints its quoted descriptor string (<c>dtype='&lt;U2'</c>, <c>dtype='|S3'</c>), because its name
        ///     alone ("str96") would not name the width.
        /// </summary>
        /// <param name="a">The printed array.</param>
        /// <returns>The text after <c>dtype=</c>.</returns>
        /// <remarks>Internal so <see cref="NDMaskedArray"/>'s repr prints the <c>dtype=</c> line exactly as the plain array
        ///     repr does.</remarks>
        internal static string DtypeShortRepr(NDArray a)
            => a.Storage.IsVString
                // A DType with its own __repr__ (StringDType) prints that repr, unquoted: dtype=StringDType(na_object=None).
                ? a.dtype.ToString(true)
                : a.typecode == NPTypeCode.String ? "'" + a.dtype.str + "'" : a.typecode.AsNumpyDtypeName();

        /// <summary>
        ///     A shape as Python's <c>repr</c> of a tuple prints it: <c>()</c>, <c>(3,)</c>, <c>(2, 3)</c> (a space after each
        ///     comma, a trailing comma for one element). NumPy's Python-level messages format shapes this way
        ///     (<c>"... (got %s and %s)" % (cshape, ashape)</c>); the C-level ones use <see cref="Shape.ToPythonTuple"/>'s
        ///     spaceless form.
        /// </summary>
        /// <param name="shape">The dimensions.</param>
        /// <returns>The tuple text.</returns>
        internal static string ShapeTuple(long[] shape)
        {
            if (shape.Length == 1)
                return "(" + shape[0].ToString(CI) + ",)";
            var sb = new StringBuilder("(");
            for (int i = 0; i < shape.Length; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(shape[i].ToString(CI));
            }
            sb.Append(')');
            return sb.ToString();
        }

        #endregion
    }
}
