using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;

namespace NumSharp.Tests.Layering
{
    /// <summary>
    ///     The source scanner behind the two layering ratchets of <c>docs/plans/np-layer-dtype-logic.md</c>: it reads
    ///     <c>src/NumSharp.Core</c> as text, blanks out comments and literal contents, and counts per file the dtype
    ///     tests of the facade files (<see cref="FacadeDtypeRatchetTests"/>, R1) and the np.* calls of the engine files
    ///     (<see cref="EngineUpCallRatchetTests"/>, R2).
    /// </summary>
    /// <remarks>
    ///     <para><b>What it counts is a heuristic, on purpose.</b> A regular expression cannot tell every dtype decision
    ///     from every dtype use; the patterns aim at decisions (a comparison, a switch, a case arm, a dtype predicate, a
    ///     call into the text backend) and skip plain uses (<c>astype(NPTypeCode.Double)</c>,
    ///     <c>new NDArray(NPTypeCode.Complex, ...)</c>). A ratchet only needs the counting to be stable: a count may go
    ///     down as code moves out of the facade, and must never go up. A decision written in a form no pattern sees is a
    ///     hole in the gate, not a false failure.</para>
    ///     <para><b>Comments and literals are removed first</b>, so a doc comment that names <c>NDStringUfuncs</c> or an
    ///     error message that says <c>np.sum(...)</c> is not counted. Every newline is kept, so a match keeps its line
    ///     number for the failure message.</para>
    ///     <para><b>The scan needs the source tree.</b> It walks up from this file's own path; a run from a package without
    ///     sources finds nothing and the tests report Inconclusive instead of passing.</para>
    /// </remarks>
    internal static class LayeringScanner
    {
        /// <summary>
        ///     The environment variable that turns either ratchet into its allowlist writer for one run. The writer
        ///     refuses to raise a count or add a file, so it can only record progress.
        /// </summary>
        internal const string WriteAllowlistVariable = "NUMSHARP_WRITE_LAYERING_ALLOWLIST";

        /// <summary>
        ///     The np.* names R2 reports as structural (allocation, copies, broadcasting, dtype construction and
        ///     promotion helpers) rather than computations. The split only labels the counts; both kinds are ratcheted.
        /// </summary>
        private static readonly HashSet<string> StructuralNames = new(StringComparer.Ordinal)
        {
            "zeros", "empty", "ones", "full", "eye", "identity", "array", "asarray", "asanyarray", "ascontiguousarray",
            "asfortranarray", "copyto", "broadcast_to", "broadcast_shapes", "broadcast_arrays", "transpose", "moveaxis",
            "swapaxes", "reshape", "ravel", "expand_dims", "squeeze", "concatenate", "stack", "getbufsize", "setbufsize",
            "arange", "empty_like", "zeros_like", "ones_like", "full_like", "atleast_1d", "atleast_2d", "atleast_3d",
            "copy", "dtype", "float16", "float32", "float64", "str_", "bytes_", "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64", "bool_", "complex128", "intp", "frombuffer", "ndarray", "flip",
            "split", "array_split", "unstack", "take", "take_along_axis", "put", "put_along_axis", "repeat", "tile",
            "pad", "result_type", "promote_types", "can_cast", "_FindCommonType", "_FindCommonType_Array",
            "_FindCommonArrayType", "_FindCommonScalarType", "_FindCommonArrayScalarType", "AtLeastNdView",
            "PrefersWriteOnce", "FlatIterator", "dtypes.StringDType", "strings.OutputDescriptor",
        };

        /// <summary>
        ///     An np.* call: <c>np.</c>, any module path (<c>linalg.</c>, <c>@char.</c>, <c>strings.</c>, ...), the
        ///     function name, optional generic arguments and the opening parenthesis. Group 1 is the dotted name.
        /// </summary>
        private static readonly Regex NpCall = new(
            @"\bnp\s*\.\s*((?:@?[a-z_]\w*\s*\.\s*)*@?[a-zA-Z_]\w*)(?:\s*<[^<>()]*>)?\s*\(",
            RegexOptions.Compiled);

        /// <summary>
        ///     <c>using static NumSharp.np;</c> (or <c>using static np;</c>, <c>global::</c> allowed): a file that imports
        ///     np's statics calls them without the <c>np.</c> prefix, so <see cref="NpCall"/> cannot see those calls. R2
        ///     counts the directive itself as one computation in an engine file, so an engine file cannot hide its np.*
        ///     calls behind it (the partials beside the facades held it until P6 converted their calls).
        /// </summary>
        private static readonly Regex UsingStaticNp = new(
            @"^\s*using\s+static\s+(?:global\s*::\s*)?(?:NumSharp\s*\.\s*)?np\s*;",
            RegexOptions.Compiled);

        /// <summary>The dtype tests R1 counts as numeric decisions (comparisons, switches, case arms, predicates).</summary>
        private static readonly Regex NumericDecision = new(string.Join("|", new[]
        {
            // a typecode compared: x.typecode == ..., GetTypeCode != ..., tc is NPTypeCode.X or ...
            @"\b(?:typecode|GetTypeCode|TypeCode)\b(?:\s*\(\s*\))?\s*(?:==|!=|\bis\b)",
            @"(?:==|!=)\s*NPTypeCode\s*\.",
            @"\bNPTypeCode\s*\.\s*\w+\s*(?:==|!=)",
            @"\bis\s+(?:not\s+)?NPTypeCode\s*\.",
            // a switch over a typecode (statement or expression) and the arms keyed on NPTypeCode values
            @"\bswitch\s*\(\s*[\w.]*\b(?:typecode|GetTypeCode|TypeCode)\b",
            @"\b(?:typecode|GetTypeCode|TypeCode)\b(?:\s*\(\s*\))?\s*switch\b",
            @"\bcase\s+NPTypeCode\s*\.",
            @"\bNPTypeCode\s*\.\s*\w+\s*(?:=>|\bor\b|\band\b)",
            @"\bor\s+NPTypeCode\s*\.",
            // the dtype predicates
            @"\.\s*(?:IsInteger|IsFloatingPoint|IsUnsigned|IsSigned|IsRealNumber|IsNumerical|IsSimdCapable)\s*\(\s*\)",
            @"\bIsFloatTc\s*\(",
            @"\bIsComplex\w*\b",
            // a descriptor compared, or its kind / class tested
            @"\bdtype\b\s*(?:==|!=)",
            @"(?:==|!=)\s*np\s*\.\s*(?:bool_|int8|uint8|int16|uint16|int32|uint32|int64|uint64|float16|float32|float64|complex128|@decimal|intp)\b",
            @"\.\s*(?:kind|Kind)\b\s*(?:==|!=|\bis\b|switch)",
            @"\.\s*Meta\s*\bis\b",
        }), RegexOptions.Compiled);

        /// <summary>The text dtype tests that do not go through a backend type: <c>NPTypeCode.String</c> and the text DType classes.</summary>
        private static readonly Regex TextDecisionOther = new(
            @"\bNPTypeCode\s*\.\s*String\b|\b(?:StringDTypeMeta|VStringDTypeMeta)\b|\bBackends\s*\.\s*Strings\s*\.",
            RegexOptions.Compiled);

        /// <summary>
        ///     Finds <c>src/NumSharp.Core</c> by walking up from this source file.
        /// </summary>
        /// <param name="thisFile">Filled by the compiler with this file's path; never pass it.</param>
        /// <returns>The absolute path of <c>src/NumSharp.Core</c>, or null when the tests run without the source tree.</returns>
        internal static string FindCoreSourceDir([CallerFilePath] string thisFile = null)
        {
            for (var dir = Path.GetDirectoryName(thisFile); dir != null; dir = Path.GetDirectoryName(dir))
            {
                var candidate = Path.Combine(dir, "src", "NumSharp.Core");
                if (Directory.Exists(candidate))
                    return candidate;
            }
            return null;
        }

        /// <summary>
        ///     The directory of the ratchet sources, where the allowlists live next to the tests.
        /// </summary>
        /// <param name="thisFile">Filled by the compiler with this file's path; never pass it.</param>
        /// <returns>The absolute directory of this file.</returns>
        internal static string LayeringSourceDir([CallerFilePath] string thisFile = null) => Path.GetDirectoryName(thisFile);

        /// <summary>
        ///     Blanks out the comments of C# source and the contents of its string and char literals, keeping every
        ///     newline so line numbers hold.
        /// </summary>
        /// <param name="source">The file's text.</param>
        /// <returns>The same text with <c>//</c> and <c>/* */</c> comments replaced by spaces and literal contents
        ///     replaced by spaces (the quotes stay).</returns>
        /// <remarks>
        ///     Handles regular, verbatim (<c>@""</c>), interpolated (<c>$""</c>, <c>$@""</c>, <c>@$""</c>) and raw
        ///     (<c>"""</c>) strings and char literals. An interpolated string's holes are blanked with it: a call written
        ///     inside a hole is not counted, a deliberate simplification (holes in the scanned files hold no dtype tests).
        /// </remarks>
        internal static string Strip(string source)
        {
            var sb = new StringBuilder(source.Length);
            int i = 0, n = source.Length;
            while (i < n)
            {
                char c = source[i];
                char next = i + 1 < n ? source[i + 1] : '\0';
                if (c == '/' && next == '/')
                {
                    // a line comment ends before the newline, which is kept
                    int end = source.IndexOf('\n', i);
                    i = end < 0 ? n : end;
                    continue;
                }
                if (c == '/' && next == '*')
                {
                    int end = source.IndexOf("*/", i + 2, StringComparison.Ordinal);
                    end = end < 0 ? n : end + 2;
                    AppendBlank(sb, source, i, end);
                    i = end;
                    continue;
                }
                if (c == '"' && string.CompareOrdinal(source, i, "\"\"\"", 0, 3) == 0)
                {
                    // a raw string literal runs to the next triple quote
                    int end = source.IndexOf("\"\"\"", i + 3, StringComparison.Ordinal);
                    end = end < 0 ? n : end + 3;
                    sb.Append("\"\"");
                    AppendBlank(sb, source, Math.Min(i + 2, end), Math.Max(i + 2, end - 2));
                    sb.Append("\"\"");
                    i = end;
                    continue;
                }
                if (c == '"' || ((c == '@' || c == '$') && IsStringPrefix(source, i, out _)))
                {
                    int quote = i;
                    bool verbatim = false;
                    while (quote < n && (source[quote] == '@' || source[quote] == '$'))
                    {
                        verbatim |= source[quote] == '@';
                        quote++;
                    }
                    int j = quote + 1;
                    while (j < n)
                    {
                        if (verbatim)
                        {
                            if (source[j] == '"')
                            {
                                if (j + 1 < n && source[j + 1] == '"') { j += 2; continue; }
                                break;
                            }
                        }
                        else
                        {
                            if (source[j] == '\\') { j += 2; continue; }
                            if (source[j] == '"' || source[j] == '\n') break;
                        }
                        j++;
                    }
                    sb.Append(source, i, quote + 1 - i);
                    AppendBlank(sb, source, quote + 1, Math.Min(j, n));
                    sb.Append('"');
                    i = Math.Min(j + 1, n);
                    continue;
                }
                if (c == '\'')
                {
                    // a char literal: 'x', '\n', 'A', '\''
                    int j = i + 1;
                    if (j < n && source[j] == '\\')
                    {
                        j += 2;
                        while (j < n && source[j] != '\'' && j - i < 12) j++;
                    }
                    else j++;
                    if (j < n && source[j] == '\'')
                    {
                        sb.Append("' '");
                        i = j + 1;
                        continue;
                    }
                }
                sb.Append(c);
                i++;
            }
            return sb.ToString();
        }

        /// <summary>
        ///     Whether the <c>@</c> / <c>$</c> run at <paramref name="i"/> prefixes a string literal (rather than a verbatim
        ///     identifier such as <c>@char</c>).
        /// </summary>
        /// <param name="s">The source.</param>
        /// <param name="i">The index of the first prefix character.</param>
        /// <param name="quote">The index of the opening quote when it is a prefix.</param>
        /// <returns>True when the run of <c>@</c> / <c>$</c> ends at a double quote.</returns>
        private static bool IsStringPrefix(string s, int i, out int quote)
        {
            quote = i;
            while (quote < s.Length && (s[quote] == '@' || s[quote] == '$'))
                quote++;
            return quote > i && quote < s.Length && s[quote] == '"';
        }

        /// <summary>Appends spaces for <c>s[from..to)</c>, keeping its newlines.</summary>
        /// <param name="sb">The output.</param>
        /// <param name="s">The source.</param>
        /// <param name="from">The first index to blank.</param>
        /// <param name="to">One past the last index to blank.</param>
        private static void AppendBlank(StringBuilder sb, string s, int from, int to)
        {
            for (int k = from; k < to; k++)
                sb.Append(s[k] == '\n' ? '\n' : ' ');
        }

        /// <summary>
        ///     The array object's homes for NumPy's statements (Appendix D of <c>docs/plans/np-layer-dtype-logic.md</c>): the
        ///     <c>NDArray</c> partials named after the NumPy source of the functions they hold (<c>NDArray.Multiarray.cs</c> for
        ///     the C functions of the multiarray module, <c>NDArray.FunctionBase.cs</c> for <c>lib/_function_base_impl.py</c>,
        ///     ...). Their statements are NumPy's (the plan's kind A), not facade logic, so <see cref="FacadeFiles"/> leaves
        ///     them out, as the <c>Default.*.cs</c> partials beside the facades are left out by their names. A partial joins
        ///     the scan's blind spot only through this list: name a new home here when Appendix D gains one.
        /// </summary>
        internal static readonly HashSet<string> ArrayObjectHomes = new(StringComparer.Ordinal)
        {
            "NDArray.Multiarray.cs", "NDArray.FunctionBase.cs", "NDArray.Numeric.cs", "NDArray.ShapeBase.cs",
            "NDArray.TwoDimBase.cs", "NDArray.CoreShapeBase.cs", "NDArray.TypeCheck.cs", "NDArray.IndexTricks.cs",
            "NDArray.StrideTricks.cs", "NDArray.ArrayPad.cs", "NDArray.CoreAsArray.cs",
        };

        /// <summary>
        ///     The facade files R1 scans: every <c>np.*.cs</c> and <c>NDArray.*.cs</c> file outside <c>Backends/</c>, and
        ///     <c>Matrixlib/NDMatrix*.cs</c>, but the array object's homes (<see cref="ArrayObjectHomes"/>).
        /// </summary>
        /// <param name="core">The <c>src/NumSharp.Core</c> directory.</param>
        /// <returns>Paths relative to <paramref name="core"/>, with forward slashes, in ordinal order.</returns>
        /// <remarks>
        ///     Out of scope and not scanned, as the plan's §8 says: the text API itself (<c>Char/</c>,
        ///     <c>APIs/np.strings*</c>, <c>APIs/np.char*</c>), whose dtype tests are its own subject.
        /// </remarks>
        internal static IEnumerable<string> FacadeFiles(string core)
        {
            foreach (var path in Directory.EnumerateFiles(core, "*.cs", SearchOption.AllDirectories)
                         .Select(p => Path.GetRelativePath(core, p).Replace('\\', '/'))
                         .OrderBy(p => p, StringComparer.Ordinal))
            {
                if (path.StartsWith("Backends/", StringComparison.Ordinal) || path.StartsWith("Char/", StringComparison.Ordinal))
                    continue;
                if (path.StartsWith("APIs/np.strings", StringComparison.Ordinal) || path.StartsWith("APIs/np.char", StringComparison.Ordinal))
                    continue;
                string file = path.Substring(path.LastIndexOf('/') + 1);
                // An NDArray partial named after a NumPy source holds NumPy's statements: the array object, not a facade.
                if (ArrayObjectHomes.Contains(file))
                    continue;
                if (file.StartsWith("np.", StringComparison.Ordinal) || file.StartsWith("NDArray.", StringComparison.Ordinal)
                    || (path.StartsWith("Matrixlib/", StringComparison.Ordinal) && file.StartsWith("NDMatrix", StringComparison.Ordinal)))
                    yield return path;
            }
        }

        /// <summary>
        ///     The engine files R2 scans: every file under <c>Backends/</c>.
        /// </summary>
        /// <param name="core">The <c>src/NumSharp.Core</c> directory.</param>
        /// <returns>Paths relative to <paramref name="core"/>, with forward slashes, in ordinal order.</returns>
        internal static IEnumerable<string> EngineFiles(string core) =>
            Directory.EnumerateFiles(Path.Combine(core, "Backends"), "*.cs", SearchOption.AllDirectories)
                .Select(p => Path.GetRelativePath(core, p).Replace('\\', '/'))
                .OrderBy(p => p, StringComparer.Ordinal);

        /// <summary>
        ///     Builds the pattern for a reference to a text-backend type: every type declared in
        ///     <c>Backends/Strings</c> whose name is distinctive enough not to collide with other code (the <c>ND*</c>,
        ///     <c>VString*</c>, <c>Text*</c>, <c>PyUnicode*</c> and <c>Utf8*</c> names), followed by a member access.
        /// </summary>
        /// <param name="core">The <c>src/NumSharp.Core</c> directory.</param>
        /// <returns>A compiled pattern; it never matches when no such type exists.</returns>
        internal static Regex TextBackendReference(string core)
        {
            var names = new SortedSet<string>(StringComparer.Ordinal);
            var declaration = new Regex(@"\b(?:class|struct|enum|record|interface)\s+([A-Z]\w*)");
            foreach (var path in Directory.EnumerateFiles(Path.Combine(core, "Backends", "Strings"), "*.cs"))
                foreach (Match m in declaration.Matches(Strip(File.ReadAllText(path))))
                {
                    string name = m.Groups[1].Value;
                    if (Regex.IsMatch(name, @"^(?:ND|VString|Text|PyUnicode|Utf8)[A-Z0-9]"))
                        names.Add(name);
                }
            if (names.Count == 0)
                return new Regex(@"(?!)");
            return new Regex(@"\b(?:" + string.Join("|", names.OrderByDescending(s => s.Length)) + @")\s*\.", RegexOptions.Compiled);
        }

        /// <summary>
        ///     Counts the dtype-test lines of every facade file: text tests (calls into the text backend,
        ///     <c>NPTypeCode.String</c>, the text DType classes) and numeric tests (the other comparisons, switches, case
        ///     arms and predicates). A line with both counts as text.
        /// </summary>
        /// <param name="core">The <c>src/NumSharp.Core</c> directory.</param>
        /// <param name="lines">Receives, per file, the stripped text of each counted line with its kind and number,
        ///     for the failure message.</param>
        /// <returns>File → (text lines, numeric lines), only files with at least one counted line.</returns>
        internal static SortedDictionary<string, (int Text, int Numeric)> CountFacadeDtypeTests(
            string core, out Dictionary<string, List<string>> lines)
        {
            var textBackend = TextBackendReference(core);
            var counts = new SortedDictionary<string, (int, int)>(StringComparer.Ordinal);
            lines = new Dictionary<string, List<string>>(StringComparer.Ordinal);
            foreach (var file in FacadeFiles(core))
            {
                int text = 0, numeric = 0;
                var found = new List<string>();
                string[] stripped = Strip(File.ReadAllText(Path.Combine(core, file))).Split('\n');
                for (int k = 0; k < stripped.Length; k++)
                {
                    string line = stripped[k];
                    if (textBackend.IsMatch(line) || TextDecisionOther.IsMatch(line))
                    {
                        text++;
                        found.Add($"{file}:{k + 1}: text: {line.Trim()}");
                    }
                    else if (NumericDecision.IsMatch(line))
                    {
                        numeric++;
                        found.Add($"{file}:{k + 1}: numeric: {line.Trim()}");
                    }
                }
                if (text + numeric > 0)
                {
                    counts[file] = (text, numeric);
                    lines[file] = found;
                }
            }
            return counts;
        }

        /// <summary>
        ///     Counts the np.* calls of every engine file, split into computations and structural calls
        ///     (<see cref="StructuralNames"/>).
        /// </summary>
        /// <param name="core">The <c>src/NumSharp.Core</c> directory.</param>
        /// <param name="calls">Receives, per file, each counted call with its line number and kind, for the failure
        ///     message.</param>
        /// <returns>File → (computation calls, structural calls), only files with at least one call.</returns>
        internal static SortedDictionary<string, (int Computations, int Structural)> CountEngineUpCalls(
            string core, out Dictionary<string, List<string>> calls)
        {
            var counts = new SortedDictionary<string, (int, int)>(StringComparer.Ordinal);
            calls = new Dictionary<string, List<string>>(StringComparer.Ordinal);
            foreach (var file in EngineFiles(core))
            {
                int computations = 0, structural = 0;
                var found = new List<string>();
                string[] stripped = Strip(File.ReadAllText(Path.Combine(core, file))).Split('\n');
                for (int k = 0; k < stripped.Length; k++)
                {
                    // The directive that would hide the file's np.* calls counts as one computation of its own.
                    if (UsingStaticNp.IsMatch(stripped[k]))
                    {
                        computations++;
                        found.Add($"{file}:{k + 1}: computation: using static np (hides the file's np.* calls from this count)");
                    }
                    foreach (Match m in NpCall.Matches(stripped[k]))
                    {
                        string name = Regex.Replace(m.Groups[1].Value, @"[\s@]", "");
                        bool isStructural = StructuralNames.Contains(name);
                        if (isStructural) structural++; else computations++;
                        found.Add($"{file}:{k + 1}: {(isStructural ? "structural" : "computation")}: np.{name}");
                    }
                }
                if (computations + structural > 0)
                {
                    counts[file] = (computations, structural);
                    calls[file] = found;
                }
            }
            return counts;
        }

        /// <summary>
        ///     Reads an allowlist: <c>path TAB first TAB second [TAB # reason]</c> per line, <c>#</c> lines and blank lines
        ///     skipped.
        /// </summary>
        /// <param name="path">The allowlist file.</param>
        /// <param name="reasons">Receives the reason comment of each entry that has one (kept by the writer).</param>
        /// <returns>File → (first count, second count); empty when the allowlist does not exist.</returns>
        /// <exception cref="FormatException">A line that is not a comment does not hold a path and two integers.</exception>
        internal static SortedDictionary<string, (int, int)> ReadAllowlist(string path, out Dictionary<string, string> reasons)
        {
            var entries = new SortedDictionary<string, (int, int)>(StringComparer.Ordinal);
            reasons = new Dictionary<string, string>(StringComparer.Ordinal);
            if (!File.Exists(path))
                return entries;
            foreach (var raw in File.ReadAllLines(path))
            {
                string line = raw.TrimEnd();
                if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                    continue;
                string[] parts = line.Split('\t');
                if (parts.Length < 3 || !int.TryParse(parts[1], out int first) || !int.TryParse(parts[2], out int second))
                    throw new FormatException($"{Path.GetFileName(path)}: malformed allowlist line \"{raw}\" " +
                                              "(expected: path TAB count TAB count [TAB # reason])");
                entries[parts[0]] = (first, second);
                if (parts.Length > 3)
                    reasons[parts[0]] = string.Join("\t", parts.Skip(3));
            }
            return entries;
        }

        /// <summary>
        ///     Writes an allowlist in <see cref="ReadAllowlist"/>'s format, ordinal by path, with a header and the reasons
        ///     kept from the previous file.
        /// </summary>
        /// <param name="path">The allowlist file to overwrite.</param>
        /// <param name="header">The header lines, written as <c>#</c> comments.</param>
        /// <param name="entries">File → counts; zero entries are dropped.</param>
        /// <param name="reasons">The reason comment per file, written as a fourth column.</param>
        internal static void WriteAllowlist(string path, IEnumerable<string> header,
            IReadOnlyDictionary<string, (int, int)> entries, IReadOnlyDictionary<string, string> reasons)
        {
            var sb = new StringBuilder();
            foreach (var h in header)
                sb.Append("# ").Append(h).Append('\n');
            foreach (var kv in entries.OrderBy(e => e.Key, StringComparer.Ordinal))
            {
                if (kv.Value.Item1 == 0 && kv.Value.Item2 == 0)
                    continue;
                sb.Append(kv.Key).Append('\t').Append(kv.Value.Item1).Append('\t').Append(kv.Value.Item2);
                if (reasons.TryGetValue(kv.Key, out var reason))
                    sb.Append('\t').Append(reason);
                sb.Append('\n');
            }
            File.WriteAllText(path, sb.ToString());
        }

        /// <summary>
        ///     Compares a scan with its allowlist and either reports the differences or, in writer mode, records the scan.
        /// </summary>
        /// <param name="gate">The gate's name for messages (R1 / R2).</param>
        /// <param name="allowlistPath">The allowlist file.</param>
        /// <param name="header">The allowlist's header lines, used by the writer.</param>
        /// <param name="found">The scan: file → counts.</param>
        /// <param name="evidence">Per file, the counted lines, quoted when a count rises.</param>
        /// <param name="firstName">The first count's name in messages.</param>
        /// <param name="secondName">The second count's name in messages.</param>
        /// <returns>The problems; empty when the scan equals the allowlist (or the writer recorded it).</returns>
        /// <remarks>
        ///     <para>A file that is new, or whose count rose, is a problem, and so is a count that dropped below the
        ///     allowlist or an entry whose file no longer counts: the allowlist must be tightened in the same commit, so
        ///     no headroom builds up for a later addition to hide in.</para>
        ///     <para>The writer (<see cref="WriteAllowlistVariable"/> set to <c>1</c>) records the scan only when nothing
        ///     rose and no file is new; a rise must be written into the allowlist by hand, with a reason.</para>
        /// </remarks>
        internal static List<string> Compare(string gate, string allowlistPath, IEnumerable<string> header,
            SortedDictionary<string, (int, int)> found, Dictionary<string, List<string>> evidence,
            string firstName, string secondName)
        {
            var allowed = ReadAllowlist(allowlistPath, out var reasons);
            var rises = new List<string>();
            var drops = new List<string>();
            foreach (var kv in found)
            {
                if (!allowed.TryGetValue(kv.Key, out var pin))
                {
                    rises.Add($"NEW file {kv.Key}: {kv.Value.Item1} {firstName}, {kv.Value.Item2} {secondName}\n      "
                              + string.Join("\n      ", evidence[kv.Key]));
                    continue;
                }
                if (kv.Value.Item1 > pin.Item1 || kv.Value.Item2 > pin.Item2)
                    rises.Add($"{kv.Key}: {kv.Value.Item1} {firstName} / {kv.Value.Item2} {secondName}, the allowlist pins " +
                              $"{pin.Item1} / {pin.Item2}\n      " + string.Join("\n      ", evidence[kv.Key]));
                else if (kv.Value.Item1 < pin.Item1 || kv.Value.Item2 < pin.Item2)
                    drops.Add($"{kv.Key}: {kv.Value.Item1} {firstName} / {kv.Value.Item2} {secondName}, the allowlist pins " +
                              $"{pin.Item1} / {pin.Item2} (progress: tighten it)");
            }
            foreach (var kv in allowed)
                if (!found.ContainsKey(kv.Key))
                    drops.Add($"{kv.Key}: no longer counts, the allowlist pins {kv.Value.Item1} / {kv.Value.Item2} " +
                              "(progress: remove the entry)");

            bool writer = Environment.GetEnvironmentVariable(WriteAllowlistVariable) == "1";
            if (writer && (rises.Count == 0 || !File.Exists(allowlistPath)))
            {
                // A first write records everything; later writes only record progress (no rise, no new file).
                WriteAllowlist(allowlistPath, header, found.ToDictionary(e => e.Key, e => e.Value), reasons);
                return new List<string>();
            }

            var problems = new List<string>(rises);
            problems.AddRange(drops);
            if (problems.Count > 0)
                problems.Add($"{gate}: a rise means new dtype logic or a new up-call where the plan forbids it — move it " +
                             "down a layer, or pin it by hand in the allowlist with a reason. A drop means progress: rerun " +
                             $"with {WriteAllowlistVariable}=1 to tighten the allowlist, and commit it.");
            return problems;
        }
    }
}
