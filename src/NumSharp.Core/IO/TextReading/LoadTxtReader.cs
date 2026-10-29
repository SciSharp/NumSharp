using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp.Backends.Strings;

namespace NumSharp.IO
{
    /// <summary>
    ///     The body of <see cref="np.loadtxt(string,DType,object,string,object,int,int[],bool,int,string,int?,string)"/>: NumPy's
    ///     <c>_read</c> (<c>numpy/lib/_npyio_impl.py</c>: the argument checks, in NumPy's order, before the input is opened)
    ///     and the C reader it calls - <c>read_rows</c> (<c>textreading/rows.c</c>), the field converters of
    ///     <c>conversions.c</c> / <c>str_to_int.c</c> and the per-dtype destinations of the converted fields.
    /// </summary>
    /// <remarks>
    ///     The <c>np.loadtxt</c> overloads convert their C# arguments, open the input (a file name through NumPy's
    ///     <c>DataSource</c>, a byte stream, a text reader, a sequence of lines) and call <see cref="BuildLoadTxtConfig"/>,
    ///     then <see cref="LoadTxtCore"/>. The partial files: this one (the configuration and <c>read_rows</c>),
    ///     <c>LoadTxtReader.Conversions.cs</c> (the numeric field parsers) and <c>LoadTxtReader.Sinks.cs</c> (the sinks).
    /// </remarks>
    internal static partial class LoadTxtReader
    {
        #region config (the argument checks of _read and _load_from_filelike, in NumPy's order)

        /// <summary>The resolved arguments of one <c>np.loadtxt</c> call.</summary>
        internal sealed class LoadTxtConfig
        {
            /// <summary>The result's descriptor.</summary>
            public DType Descr;
            /// <summary>The descriptor's storage lane.</summary>
            public NPTypeCode Tc;
            /// <summary>The delimiter character (<see cref="LoadTxtTokenizer.None"/>: whitespace).</summary>
            public int Delimiter = LoadTxtTokenizer.None;
            /// <summary>The quote character (<see cref="LoadTxtTokenizer.None"/>: no quoting).</summary>
            public int Quote = LoadTxtTokenizer.None;
            /// <summary>The single-character comment (<see cref="LoadTxtTokenizer.None"/>: none, or stripped per line).</summary>
            public int Comment = LoadTxtTokenizer.None;
            /// <summary>The comments stripped from each line before tokenizing (NumPy's <c>_preprocess_comments</c>), or null:
            ///     Python objects in order — a string cuts the line at its first occurrence, null (<c>None</c>) is
            ///     <c>str.split(None, 1)[0]</c>, any other object raises <c>str.split</c>'s TypeError at the first line.</summary>
            public object[] MultiComments;
            /// <summary>The delimiter as given (for the control-character checks' texts).</summary>
            public string DelimiterText;
            /// <summary>The quote character as given.</summary>
            public string QuoteText;
            /// <summary>Lines skipped first.</summary>
            public int SkipRows;
            /// <summary>The maximum number of data rows (-1: all).</summary>
            public int MaxRows;
            /// <summary>The selected columns, or null for all.</summary>
            public int[] UseCols;
            /// <summary>The minimum result rank.</summary>
            public int Ndmin;
            /// <summary>The converters as given (resolved at the first data row, as NumPy resolves them).</summary>
            public object Converters;
            /// <summary>NumPy's <c>encoding='bytes'</c> (<c>python_byte_converters</c>): a converter receives the field's
            ///     Latin-1 form.</summary>
            public bool ByteConverters;
        }

        /// <summary>
        ///     Validates loadtxt's Python-level arguments in NumPy's order (<c>loadtxt</c> and <c>_read</c>, before the input
        ///     is opened) and resolves them.
        /// </summary>
        /// <param name="descr">The requested dtype, or null for float64.</param>
        /// <param name="comments">A string, a sequence (of strings, or of any Python objects), or null.</param>
        /// <param name="delimiter">The delimiter, or null for whitespace.</param>
        /// <param name="converters">The converters as given.</param>
        /// <param name="usecols">The selected columns, or null.</param>
        /// <param name="ndmin">The minimum result rank.</param>
        /// <param name="max_rows">The maximum number of data rows, or null.</param>
        /// <param name="quotechar">The quote character, or null.</param>
        /// <param name="skiprows">Leading lines to skip.</param>
        /// <param name="encoding">The encoding as given; only <c>"bytes"</c> matters here (NumPy's <c>_read</c> turns it
        ///     into byte converters and reads with the default encoding).</param>
        /// <returns>The configuration (the control characters are checked later, by <see cref="CheckLoadTxtControlCharacters"/>,
        ///     as NumPy's C reader checks them after the input is opened).</returns>
        /// <exception cref="ValueError">An invalid <paramref name="ndmin"/>, an empty comment, multiple / multi-character
        ///     comments with a quote character, a negative <paramref name="skiprows"/> / <paramref name="max_rows"/>.</exception>
        /// <exception cref="TypeError">Comments that are not iterable, or several comments one of which is the
        ///     delimiter.</exception>
        internal static LoadTxtConfig BuildLoadTxtConfig(DType descr, object comments, string delimiter,
            object converters, int[] usecols, int ndmin, int? max_rows, string quotechar, int skiprows, string encoding)
        {
            descr ??= DType.Double;
            var cfg = new LoadTxtConfig
            {
                Descr = descr,
                Tc = descr.GetTypeCode(),
                DelimiterText = delimiter,
                QuoteText = quotechar,
                SkipRows = skiprows,
                UseCols = usecols,
                Ndmin = ndmin,
                Converters = converters,
                ByteConverters = encoding == "bytes",
            };

            // loadtxt normalizes the comments before it calls _read, so a non-iterable argument is reported ahead of _read's
            // checks (comments=5 with ndmin=3 is "'int' object is not iterable", not the ndmin error).
            object[] list = ResolveLoadTxtComments(comments);

            // _ensure_ndmin_ndarray_check_param.
            if (ndmin != 0 && ndmin != 1 && ndmin != 2)
                throw new ValueError($"Illegal value of ndmin keyword: {ndmin}");

            // _read: an empty comment is refused ("" in comment — only a str equals ""), one one-character str goes to the C
            // tokenizer, anything else is stripped per line; several comments must not include the delimiter (Python's `in`
            // over the tuple: None is in it when an element is None), and stripped comments cannot meet quotes. Elements that
            // are neither str nor None are kept: NumPy's str.split raises for them at the first line it reads.
            if (list != null)
            {
                foreach (object c in list)
                    if (c is string { Length: 0 })
                        throw new ValueError("comments cannot be an empty string. Use comments=None to disable comments.");
                if (list.Length == 1 && list[0] is string { Length: 1 } one)
                    cfg.Comment = one[0];
                else if (list.Length > 0)
                {
                    if (list.Length > 1 && LoadTxtCommentsContain(list, delimiter))
                        throw new TypeError($"Comment characters '{PyTupleRepr(list)}' cannot include the delimiter '{delimiter ?? "None"}'");
                    cfg.MultiComments = list;
                }
                if (cfg.MultiComments != null && quotechar != null)
                    throw new ValueError("when multiple comments or a multi-character comment is given, quotes are not supported.  In this case quotechar must be set to None.");
            }

            // _check_nonneg_int(skiplines) / (max_rows).
            if (skiprows < 0)
                throw new ValueError("argument must be nonnegative");
            if (max_rows.HasValue && max_rows.Value < 0)
                throw new ValueError("argument must be nonnegative");
            cfg.MaxRows = max_rows ?? -1;
            return cfg;
        }

        /// <summary>
        ///     <c>loadtxt</c>'s comment normalization (<c>[x.decode('latin1') if isinstance(x, bytes) else x for x in
        ///     comment]</c> after a str / bytes became a one-element list): a string, a Latin-1 <c>byte[]</c> or a
        ///     <see cref="char"/> is a one-element list, a sequence is its elements with each <c>byte[]</c> decoded as Latin-1
        ///     and each <see cref="char"/> a one-character str, null is no comments.
        /// </summary>
        /// <param name="comments">The argument.</param>
        /// <returns>The comments as Python objects (a string, null for <c>None</c>, or any other object, which NumPy keeps and
        ///     whose use as a separator fails later), or null for no comments.</returns>
        /// <exception cref="TypeError">An argument that is not iterable (<c>'int' object is not iterable</c>, Python's text for
        ///     the comprehension over it).</exception>
        private static object[] ResolveLoadTxtComments(object comments)
        {
            switch (comments)
            {
                case null:
                    return null;
                case string s:
                    return new object[] { s };
                case char ch:
                    return new object[] { ch.ToString() };
                case byte[] b:
                    return new object[] { Encoding.Latin1.GetString(b) };
                case ITuple tuple:
                {
                    // A value tuple is a Python tuple, which is iterable (C#'s ValueTuple is not an IEnumerable).
                    var items = new object[tuple.Length];
                    for (int i = 0; i < items.Length; i++)
                        items[i] = AsLoadTxtComment(tuple[i]);
                    return items;
                }
                case IEnumerable seq:
                {
                    var list = new List<object>();
                    foreach (object x in seq)
                        list.Add(AsLoadTxtComment(x));
                    return list.ToArray();
                }
                default:
                    throw new TypeError($"'{NDPyObject.TypeName(comments)}' object is not iterable");
            }
        }

        /// <summary>One element of a comment sequence after <c>loadtxt</c>'s comprehension: a <c>byte[]</c> decoded as
        ///     Latin-1, a <see cref="char"/> as its one-character str, anything else as it is.</summary>
        /// <param name="x">The element.</param>
        /// <returns>The comment as a Python object.</returns>
        private static object AsLoadTxtComment(object x) => x switch
        {
            byte[] xb => Encoding.Latin1.GetString(xb),
            char xc => xc.ToString(),
            _ => x,
        };

        /// <summary>
        ///     Python's <c>delimiter in comments</c> over the comment tuple: an element that IS the delimiter or equals it —
        ///     None is in the tuple when an element is None (whitespace splitting has the delimiter None), a str delimiter when
        ///     an equal str is (an object of another type never equals a str).
        /// </summary>
        /// <param name="comments">The comments as Python objects.</param>
        /// <param name="delimiter">The delimiter, or null for None.</param>
        /// <returns>True when the delimiter is one of the comments.</returns>
        private static bool LoadTxtCommentsContain(object[] comments, string delimiter)
        {
            foreach (object c in comments)
                if (delimiter == null ? c == null : c is string s && s == delimiter)
                    return true;
            return false;
        }

        /// <summary>Python's <c>repr()</c> of the comment tuple (<c>('#', None)</c>), for NumPy's comment / delimiter text.</summary>
        /// <param name="items">The comments as Python objects.</param>
        /// <returns>The repr.</returns>
        private static string PyTupleRepr(object[] items)
        {
            var sb = new StringBuilder("(");
            for (int i = 0; i < items.Length; i++)
            {
                if (i > 0)
                    sb.Append(", ");
                sb.Append(NDPyObject.Repr(items[i]));
            }
            if (items.Length == 1)
                sb.Append(',');
            return sb.Append(')').ToString();
        }

        /// <summary>
        ///     The C reader's control-character checks (<c>parse_control_character</c> for the delimiter, the comment and the
        ///     quote, then <c>error_if_matching_control_characters</c>), raised after the input is opened, as NumPy raises them.
        /// </summary>
        /// <param name="cfg">The configuration; its <see cref="LoadTxtConfig.Delimiter"/> and <see cref="LoadTxtConfig.Quote"/>
        ///     are set here.</param>
        /// <exception cref="TypeError">A delimiter or quote that is not exactly one character (<c>Text reading control
        ///     character must be a single unicode character or None; but got: ', '</c>), a newline control character, or two
        ///     control characters that are the same (or a whitespace comment / quote with whitespace splitting).</exception>
        private static void CheckLoadTxtControlCharacters(LoadTxtConfig cfg)
        {
            cfg.Delimiter = ParseLoadTxtControlCharacter(cfg.DelimiterText);
            cfg.Quote = ParseLoadTxtControlCharacter(cfg.QuoteText);
            int delimiter = cfg.Delimiter, quote = cfg.Quote, comment = cfg.Comment;
            const int none = LoadTxtTokenizer.None;

            string first = null, second = null;
            if (comment != none)
            {
                if (comment == '\r' || comment == '\n')
                    first = "comment";
                else if (comment == quote)
                    (first, second) = ("comment", "quotechar");
                else if (comment == delimiter)
                    (first, second) = ("comment", "delimiter");
            }
            if (first == null && quote != none)
            {
                if (quote == '\r' || quote == '\n')
                    first = "quotechar";
                else if (quote == delimiter)
                    (first, second) = ("quotechar", "delimiter");
            }
            if (first == null && delimiter != none && (delimiter == '\r' || delimiter == '\n'))
                first = "delimiter";
            // The above doesn't work with delimiter=None, which means "whitespace".
            if (first == null && delimiter == none)
            {
                if (comment != none && LoadTxtTokenizer.IsSpace((char)comment))
                    (first, second) = ("delimiter", "comment");
                else if (quote != none && LoadTxtTokenizer.IsSpace((char)quote))
                    (first, second) = ("delimiter", "quotechar");
            }
            if (first == null)
                return;
            if (second != null)
                throw new TypeError($"The values for control characters '{first}' and '{second}' are incompatible");
            throw new TypeError($"control character '{first}' cannot be a newline (`\\r` or `\\n`).");
        }

        /// <summary><c>parse_control_character</c>: null is "not set", one character is itself.</summary>
        /// <param name="s">The control character as given.</param>
        /// <returns>The character, or <see cref="LoadTxtTokenizer.None"/>.</returns>
        /// <exception cref="TypeError">Not exactly one character (NumPy's text with the value's repr, cut to 100).</exception>
        private static int ParseLoadTxtControlCharacter(string s)
        {
            if (s is null)
                return LoadTxtTokenizer.None;
            if (s.Length != 1)
                throw new TypeError("Text reading control character must be a single unicode character or None; but got: " + Repr100(s));
            return s[0];
        }

        /// <summary>Python's <c>%.100R</c>: the repr of a text, cut to 100 code points (NumPy's error texts).</summary>
        /// <param name="s">The text.</param>
        /// <returns>The cut repr.</returns>
        private static string Repr100(string s)
        {
            string repr = Backends.Printing.PyRepr.Str(s);
            // Cut at 100 CODE POINTS (a surrogate pair is one), as PyUnicode_FromFormat's precision counts characters.
            int units = 0;
            for (int kept = 0; kept < 100 && units < repr.Length; kept++)
                units += char.IsHighSurrogate(repr[units]) && units + 1 < repr.Length && char.IsLowSurrogate(repr[units + 1]) ? 2 : 1;
            return units >= repr.Length ? repr : repr.Substring(0, units);
        }

        #endregion

        #region read_rows

        /// <summary>
        ///     NumPy's <c>read_rows</c> (<c>textreading/rows.c</c>): skips the leading lines, tokenizes row by row, resolves
        ///     the converters at the first data row, converts every selected field in order through the dtype's writer, and
        ///     finishes the 2-D result; then <c>_ensure_ndmin_ndarray</c> and <c>unpack</c>.
        /// </summary>
        /// <param name="stream">The input.</param>
        /// <param name="cfg">The configuration.</param>
        /// <param name="unpack">Transpose the result.</param>
        /// <returns>The result.</returns>
        /// <exception cref="ValueError">See <see cref="np.loadtxt(string,DType,object,string,object,int,int[],bool,int,string,int?,string)"/>.</exception>
        /// <exception cref="TypeError">See the same.</exception>
        /// <remarks>Scoped: the 2-D <c>read_rows</c> result is an intermediate whenever <c>ndmin</c> squeezes or
        ///     <paramref name="unpack"/> transposes it (the returned view keeps its buffer alive), and a converter's
        ///     transient arrays are reclaimed with it.</remarks>
        [NDScoped]
        internal static NDArray LoadTxtCore(LoadTxtStream stream, LoadTxtConfig cfg, bool unpack)
        {
            // An unsized string dtype ('U', 'S') is _read's chunked object read: `while max_rows != 0` reads chunks with the C
            // reader and casts each. With max_rows == 0 the loop never runs, so the C reader checks nothing and _read
            // concatenates an empty chunk list.
            if (cfg.MaxRows == 0 && cfg.Descr.Meta is StringDTypeMeta && cfg.Descr.itemsize == 0)
                throw new ValueError("need at least one array to concatenate");
            CheckLoadTxtControlCharacters(cfg);
            var tokenizer = new LoadTxtTokenizer(cfg.Delimiter, cfg.Quote, cfg.Comment);
            using LoadTxtSink sink = LoadTxtSink.Create(cfg.Descr, cfg.ByteConverters);

            // Fewer lines than skiplines is acceptable; a file that ends while skipping reads no rows.
            int tsResult = 0;
            for (int i = 0; i < cfg.SkipRows; i++)
            {
                tsResult = tokenizer.SkipLine(stream);
                if (tsResult != 0)
                    break;
            }

            int[] usecols = cfg.UseCols;
            int actualNumFields = usecols != null ? usecols.Length : -1;
            Func<string, object>[] convFuncs = null;
            bool started = false;
            long rowCount = 0;
            while ((cfg.MaxRows < 0 || rowCount < cfg.MaxRows) && tsResult == 0)
            {
                tsResult = tokenizer.Tokenize(stream);
                int currentNumFields = tokenizer.NumFields;
                if (currentNumFields == 0)
                    continue; // ignore an empty line (also not counted toward max_rows)

                if (!started)
                {
                    // Deferred to the first row: now the number of columns is known.
                    if (actualNumFields == -1)
                        actualNumFields = currentNumFields;
                    if (cfg.Converters != null)
                        convFuncs = CreateLoadTxtConverters(cfg.Converters, actualNumFields, usecols);
                    started = true;
                    // A source that bounds its remaining lines (an array or a list) bounds the rows: a sink of fixed-size
                    // elements then allocates the result once and fills it in place, where NumPy grows its array in blocks
                    // and shrinks it at the end (the values are the same either way).
                    long moreLines = stream.RemainingLinesHint;
                    if (moreLines >= 0 && actualNumFields > 0)
                    {
                        long bound = 1 + moreLines;
                        if (cfg.MaxRows >= 0 && cfg.MaxRows < bound)
                            bound = cfg.MaxRows;
                        sink.Reserve(bound, actualNumFields);
                    }
                }

                if (usecols == null && actualNumFields != currentNumFields)
                    throw new ValueError(
                        $"the number of columns changed from {actualNumFields} to {currentNumFields} at row {rowCount + 1}; " +
                        "use `usecols` to select a subset and avoid this error");

                if (usecols == null && convFuncs == null)
                {
                    // Every field of the row, in order, with no converter: the sink runs the row's loop itself (one call per
                    // row, the dtype's conversion inline), and reports the first field it could not store.
                    int stored = sink.StoreRow(tokenizer, actualNumFields, out Exception cause);
                    if (stored < actualNumFields)
                        throw LoadTxtConvertError(tokenizer.Field(stored), sink.DescrText, rowCount, stored, cause);
                    rowCount++;
                    continue;
                }

                for (int i = 0; i < actualNumFields; i++)
                {
                    int col = i;
                    if (usecols != null)
                    {
                        col = usecols[i];
                        // Python-like column indexing: k = -1 means the last column.
                        if (col < 0)
                            col += currentNumFields;
                        if (col < 0 || col >= currentNumFields)
                            throw new ValueError($"invalid column index {usecols[i]} at row {rowCount + 1} with {currentNumFields} columns");
                    }

                    // The main "conversion" step: parse the field's text (or run its converter) and store the result.
                    ReadOnlySpan<char> field = tokenizer.Field(col);
                    Func<string, object> conv = convFuncs?[i];
                    bool stored;
                    try
                    {
                        stored = sink.Store(field, conv);
                    }
                    catch (Exception e)
                    {
                        throw LoadTxtConvertError(field, sink.DescrText, rowCount, col, e);
                    }
                    if (!stored)
                        throw LoadTxtConvertError(field, sink.DescrText, rowCount, col, null);
                }
                rowCount++;
            }

            // No data row and no usecols: NumPy guesses one column ("we have no choice but to guess 1").
            NDArray arr = sink.Finish(rowCount, actualNumFields == -1 ? 1 : actualNumFields);
            NDArray shaped = EnsureNdmin(arr, cfg.Ndmin);
            return unpack ? shaped.T : shaped;
        }

        /// <summary>
        ///     NumPy's <c>create_conv_funcs</c>: the converter of each RESULT column. A single function serves every column;
        ///     a dictionary maps columns — without <paramref name="usecols"/> a key is a file column (negative from the end,
        ///     range-checked), with it a key must equal one of the <paramref name="usecols"/> entries AS GIVEN (the first one
        ///     that does; a key no entry equals is ignored, so <c>usecols=[-1]</c> with <c>{2: f}</c> applies nothing).
        /// </summary>
        /// <param name="converters">The converters as given.</param>
        /// <param name="numFields">The number of result columns.</param>
        /// <param name="usecols">The selected columns, or null.</param>
        /// <returns>One converter (or null) per result column.</returns>
        /// <exception cref="TypeError">Converters of another kind (<c>converters must be a dictionary mapping columns to
        ///     converter functions or a single callable.</c>), or a null converter in the dictionary.</exception>
        /// <exception cref="ValueError">A key outside the columns without <paramref name="usecols"/> (<c>converter specified
        ///     for column 5, which is invalid for the number of fields 2.</c>).</exception>
        private static Func<string, object>[] CreateLoadTxtConverters(object converters, int numFields, int[] usecols)
        {
            var conv = new Func<string, object>[numFields];
            if (converters is Func<string, object> all)
            {
                // A single converter used for all columns individually.
                for (int i = 0; i < numFields; i++)
                    conv[i] = all;
                return conv;
            }
            if (converters is not IDictionary<int, Func<string, object>> dict)
                throw new TypeError("converters must be a dictionary mapping columns to converter functions or a single callable.");

            foreach (var entry in dict)
            {
                int column = entry.Key;
                if (usecols != null)
                {
                    int i = 0;
                    for (; i < numFields; i++)
                    {
                        if (column == usecols[i])
                        {
                            column = i;
                            break;
                        }
                    }
                    if (i == numFields)
                        continue; // ignore an unused converter
                }
                else
                {
                    if (column < -numFields || column >= numFields)
                        throw new ValueError($"converter specified for column {column}, which is invalid for the number of fields {numFields}.");
                    if (column < 0)
                        column += numFields;
                }
                if (entry.Value is null)
                    throw new TypeError($"values of the converters dictionary must be callable, but the value associated with key {entry.Key} is not");
                conv[column] = entry.Value;
            }
            return conv;
        }

        /// <summary>
        ///     NumPy's conversion error (<c>rows.c</c>): <c>could not convert string %.100R to %S at row %zd, column %zd.</c> —
        ///     the field's repr cut to 100 characters, the dtype's <c>str()</c>, the 0-based data row and the 1-based column,
        ///     the original error as the cause.
        /// </summary>
        /// <param name="field">The field's text.</param>
        /// <param name="descrText">The dtype's <c>str()</c> (<c>float64</c>, <c>&lt;U5</c>, <c>StringDType()</c>).</param>
        /// <param name="row">The 0-based data row (blank and comment lines not counted).</param>
        /// <param name="col">The 0-based column in the file.</param>
        /// <param name="cause">The error the conversion raised, or null when the parser just refused the text.</param>
        /// <returns>The error to throw.</returns>
        private static ValueError LoadTxtConvertError(ReadOnlySpan<char> field, string descrText, long row, int col, Exception cause)
            => new ValueError($"could not convert string {Repr100(field.ToString())} to {descrText} at row {row}, column {col + 1}.", cause);

        /// <summary>
        ///     NumPy's <c>_ensure_ndmin_ndarray</c>: squeeze the size-1 axes of a result with more dimensions than
        ///     <paramref name="ndmin"/>, then expand one with fewer (<c>atleast_1d</c>; <c>atleast_2d(a).T</c>).
        /// </summary>
        /// <param name="a">The 2-D <c>read_rows</c> result.</param>
        /// <param name="ndmin">0, 1 or 2.</param>
        /// <returns><paramref name="a"/> ITSELF when nothing changes — NumPy's <c>PyArray_Squeeze</c> returns its
        ///     operand when no axis is 1, so a <c>(2, 3)</c> or <c>(0, 3)</c> result still owns its data — else a view.</returns>
        private static NDArray EnsureNdmin(NDArray a, int ndmin)
        {
            if (a.ndim > ndmin && Array.IndexOf(a.Shape.dimensions, 1L) >= 0)
                a = np.squeeze(a);
            if (a.ndim < ndmin)
            {
                if (ndmin == 1) a = np.atleast_1d(a);
                else if (ndmin == 2) a = np.atleast_2d(a).T;
            }
            return a;
        }

        #endregion
    }
}
