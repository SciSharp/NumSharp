using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The committed corpus reduced to the fields the COVERAGE gates read — one small header record per case —
    ///     scanned once per test process and shared by every gate that used to re-parse the whole corpus:
    ///     <see cref="OracleSurfaceCoverageTests"/>, <see cref="OracleCoverageStrengthTests"/>,
    ///     <see cref="Journey3TouchedOracleCoverageTests"/> and <see cref="OracleApplicabilityTests"/>.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why.</b> Those gates only ask which op keys, layouts, dtypes, params and outcome kinds the corpus
    ///     carries, yet each of the eight scans deserialized every line into a full <see cref="FuzzCorpus.Case"/>: ~950 MiB
    ///     of allocation per scan (hex payload strings, one <see cref="JsonDocument"/> per param value), held as a
    ///     whole-file list long enough to be promoted. In the Oracle test host that was ~7.8 s of the net10.0 run
    ///     (2026-09-24), ~65 % of it GC pauses. This survey reads each line with a <see cref="Utf8JsonReader"/>,
    ///     materializes only the header fields (payload hex strings are stepped over, never decoded), interns every
    ///     repeated string, and keeps ~220 K records of ~96 bytes (~21 MB) for the rest of the process — one ~0.5 s
    ///     pass instead of eight; the gates themselves now take ~20 ms each.</para>
    ///     <para><b>Scope.</b> Every <c>*.jsonl</c> under <c>Fuzz/corpus</c> EXCEPT the <c>*.host.jsonl</c> host pins (which
    ///     describe a binary, not cases — every consumer skipped them), in ordinal file-name order. Index (<c>index_*</c>)
    ///     and masked-array (<c>ma_*</c>) tiers are included, read through the ordinary case schema exactly as the gates'
    ///     <see cref="FuzzCorpus.Load"/> loops read them; each consumer keeps its own file filter.</para>
    ///     <para><b>Fidelity contract.</b> Each <see cref="SurveyCase"/> field is what the gates computed from the parsed
    ///     <see cref="FuzzCorpus.Case"/> of the same line, quirks included: property names match case-insensitively (the
    ///     serializer's <c>PropertyNameCaseInsensitive</c>) while param KEYS stay case-sensitive (a dictionary); a repeated
    ///     property or param key keeps its LAST value; <c>params: {}</c> signs as the empty string while a null or absent
    ///     <c>params</c> signs as <c>{}</c>; a param value's signature text is its exact source JSON (what
    ///     <see cref="JsonElement.GetRawText"/> returned). Header fields of the wrong JSON type fail the scan with a
    ///     <see cref="JsonException"/>, as the deserializer would; payload fields the survey steps over are not
    ///     type-checked here — the replay tiers still deserialize every line of every tier in full.
    ///     <see cref="CorpusSurveyTests"/> pins the agreement field for field against <see cref="FuzzCorpus.ParseLine"/>.</para>
    /// </remarks>
    internal static class CorpusSurvey
    {
        /// <summary>
        ///     The shared scan of the committed corpus, built on first use and kept for the process (thread-safe,
        ///     built once even when gates race to it).
        /// </summary>
        private static readonly Lazy<IReadOnlyList<SurveyFile>> Shared =
            new(() => ScanDirectory(Path.GetDirectoryName(FuzzCorpus.CorpusPath("unused"))),
                LazyThreadSafetyMode.ExecutionAndPublication);

        /// <summary>
        ///     Every non-host corpus file with its per-case header records, in ordinal file-name order.
        /// </summary>
        /// <exception cref="JsonException">A corpus line is malformed or has a header field of the wrong type (raised by
        /// the first access; <see cref="Lazy{T}"/> re-raises it on every later access).</exception>
        /// <exception cref="IOException">A corpus file cannot be read.</exception>
        internal static IReadOnlyList<SurveyFile> Files => Shared.Value;

        /// <summary>
        ///     Scans every non-host <c>*.jsonl</c> file of <paramref name="directory"/>.
        /// </summary>
        /// <param name="directory">The corpus directory.</param>
        /// <returns>One <see cref="SurveyFile"/> per file, ordinal by name.</returns>
        /// <exception cref="JsonException">A line is malformed or carries a header field of the wrong JSON type.</exception>
        /// <exception cref="IOException">A file cannot be read.</exception>
        internal static IReadOnlyList<SurveyFile> ScanDirectory(string directory)
        {
            string[] names = Directory.EnumerateFiles(directory, "*.jsonl")
                .Select(Path.GetFileName)
                .Where(name => !name.EndsWith(".host.jsonl", StringComparison.Ordinal))
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();

            // One scanner for the whole directory: its string pool is what lets 220 K records share a few thousand
            // distinct strings, so interning must span files (layouts and dtypes repeat across every tier).
            var scanner = new Scanner();
            var files = new SurveyFile[names.Length];
            for (int i = 0; i < names.Length; i++)
            {
                using var file = new CorpusFile(names[i], Path.Combine(directory, names[i]));
                files[i] = scanner.ScanFile(file);
            }
            return files;
        }

        /// <summary>
        ///     Scans in-memory JSONL content — the entry point <see cref="CorpusSurveyTests"/> uses to hold the scanner
        ///     against the case parser on synthetic lines. Line rules are <see cref="CorpusFile.TryReadLine"/>'s.
        /// </summary>
        /// <param name="name">The file name the records are filed under.</param>
        /// <param name="utf8">The UTF-8 content (a leading byte-order mark is NOT skipped here).</param>
        /// <returns>The scanned file.</returns>
        /// <exception cref="JsonException">A line is malformed or carries a header field of the wrong JSON type.</exception>
        internal static SurveyFile ScanContent(string name, ReadOnlySpan<byte> utf8)
        {
            var scanner = new Scanner();
            var cases = new List<SurveyCase>(CorpusFile.CountLines(utf8));
            foreach (ReadOnlySpan<byte> line in new CorpusFile.LineEnumerator(utf8))
                cases.Add(scanner.ScanLine(line));
            return new SurveyFile(name, cases.ToArray());
        }

        /// <summary>
        ///     The line scanner: a <see cref="Utf8JsonReader"/> walk that reads the header fields and steps over
        ///     everything else, plus the string pool its records share.
        /// </summary>
        private sealed class Scanner
        {
            private readonly Dictionary<string, string> _strings = new(StringComparer.Ordinal);
            private readonly Dictionary<string, string[]> _dtypeLists = new(StringComparer.Ordinal);
            private readonly List<KeyValuePair<string, string>> _params = new();
            private readonly List<string> _dtypes = new();

            /// <summary>Scans every case line of an opened corpus file.</summary>
            /// <param name="file">The opened file.</param>
            /// <returns>Its header records, in file order.</returns>
            /// <exception cref="JsonException">A line is malformed; the message names the file and 1-based case number.</exception>
            internal SurveyFile ScanFile(CorpusFile file)
            {
                var cases = new SurveyCase[file.Count];
                int n = 0;
                foreach (ReadOnlySpan<byte> line in file.EnumerateLines())
                {
                    try
                    {
                        cases[n] = ScanLine(line);
                    }
                    catch (JsonException e)
                    {
                        // The reader's own position is relative to the LINE; name the file and case so a bad
                        // regeneration is findable without re-running the scan under a debugger.
                        throw new JsonException($"{file.Name} case #{n + 1}: {e.Message}", e);
                    }
                    n++;
                }
                return new SurveyFile(file.Name, cases);
            }

            /// <summary>
            ///     Reads one case line into its header record.
            /// </summary>
            /// <param name="line">One JSONL line (UTF-8, terminator excluded).</param>
            /// <returns>The record.</returns>
            /// <exception cref="JsonException">The line is not a single JSON object, or a header field has the wrong type.</exception>
            internal SurveyCase ScanLine(ReadOnlySpan<byte> line)
            {
                // Default reader options are the serializer's defaults: no comments, no trailing commas, depth 64, and
                // exactly one top-level value (trailing content fails the final Read below, as it fails Deserialize).
                var reader = new Utf8JsonReader(line);
                Next(ref reader);
                if (reader.TokenType != JsonTokenType.StartObject)
                    throw new JsonException($"a corpus case must be a JSON object, found {reader.TokenType}");

                string op = null, layout = null, valueclass = null;
                string paramSignature = "{}", dist = null, ufunc = null, paramDtype = null;
                bool distPresent = false, hasAxes = false, expectsThrow = false, error = false, hasTruth = false, hasZeroD = false;
                string expectedKind = "array", expectedDtype = null;
                string[] dtypes = Array.Empty<string>();
                string dtypeSignature = "";

                while (true)
                {
                    Next(ref reader);
                    if (reader.TokenType == JsonTokenType.EndObject)
                        break;

                    // Each branch reassigns every field its property governs, so a repeated property keeps its
                    // LAST occurrence — the serializer's overwrite semantics.
                    if (NameIs(ref reader, "op"u8))
                    {
                        Next(ref reader);
                        op = ReadString(ref reader, "op");
                    }
                    else if (NameIs(ref reader, "layout"u8))
                    {
                        Next(ref reader);
                        layout = ReadString(ref reader, "layout");
                    }
                    else if (NameIs(ref reader, "valueclass"u8))
                    {
                        Next(ref reader);
                        valueclass = ReadString(ref reader, "valueclass");
                    }
                    else if (NameIs(ref reader, "params"u8))
                    {
                        Next(ref reader);
                        ReadParams(ref reader, line, out paramSignature, out dist, out distPresent, out ufunc,
                                   out paramDtype, out hasAxes);
                    }
                    else if (NameIs(ref reader, "operands"u8))
                    {
                        Next(ref reader);
                        ReadOperands(ref reader, out dtypes, out dtypeSignature, out hasZeroD);
                    }
                    else if (NameIs(ref reader, "expected"u8))
                    {
                        Next(ref reader);
                        ReadExpected(ref reader, out expectedKind, out expectedDtype, out hasTruth);
                    }
                    else if (NameIs(ref reader, "expects_throw"u8))
                    {
                        Next(ref reader);
                        expectsThrow = reader.TokenType switch
                        {
                            JsonTokenType.True => true,
                            JsonTokenType.False => false,
                            // A non-nullable bool: null (or anything else) is a deserialization error there too.
                            _ => throw new JsonException($"'expects_throw' must be a boolean, found {reader.TokenType}"),
                        };
                    }
                    else if (NameIs(ref reader, "error"u8))
                    {
                        Next(ref reader);
                        error = reader.TokenType switch
                        {
                            JsonTokenType.Null => false,
                            JsonTokenType.StartObject => true,
                            _ => throw new JsonException($"'error' must be an object or null, found {reader.TokenType}"),
                        };
                        reader.Skip();
                    }
                    else
                    {
                        // id, alias, and every key outside the case schema (the index tiers' base/tokens/np/...):
                        // step over the value — a container is skipped whole, never materialized.
                        Next(ref reader);
                        reader.Skip();
                    }
                }

                // One JSON value per line: anything but whitespace after the object makes Read throw.
                if (reader.Read())
                    throw new JsonException("a corpus line must hold exactly one JSON value");

                // A random-stream case names its sampler in params.dist; the gates read it with GetString(), which a
                // non-string would make throw (and a null would poison their dictionaries) — keep that loud.
                if (op == "rnd" && distPresent && dist == null)
                    throw new JsonException("an 'rnd' case's params.dist must be a string");

                return new SurveyCase(op, layout, valueclass, dtypes, dtypeSignature, hasZeroD, paramSignature,
                                      dist, ufunc, paramDtype, hasAxes, expectedKind, expectedDtype, hasTruth,
                                      error || expectsThrow);
            }

            /// <summary>
            ///     Reads a <c>params</c> value: its signature (<c>key=rawJson</c> pairs, ordinal by key, joined by
            ///     <c>|</c>) and the string-valued params the gates look up by name.
            /// </summary>
            /// <param name="reader">Positioned on the value's first token; left on its last.</param>
            /// <param name="line">The whole line, for slicing each value's source text.</param>
            /// <param name="signature">Receives the signature: <c>{}</c> for null, the empty string for an empty object.</param>
            /// <param name="dist">Receives <c>params.dist</c> when it is a JSON string, else null.</param>
            /// <param name="distPresent">Receives whether a key named exactly <c>dist</c> is present (any value).</param>
            /// <param name="ufunc">Receives <c>params.ufunc</c> when it is a JSON string, else null.</param>
            /// <param name="dtype">Receives <c>params.dtype</c> when it is a JSON string, else null.</param>
            /// <param name="hasAxes">Receives whether a key named exactly <c>axes</c> is present (any value, null included).</param>
            /// <exception cref="JsonException"><c>params</c> is neither an object nor null.</exception>
            private void ReadParams(ref Utf8JsonReader reader, ReadOnlySpan<byte> line, out string signature,
                                    out string dist, out bool distPresent, out string ufunc, out string dtype,
                                    out bool hasAxes)
            {
                dist = ufunc = dtype = null;
                distPresent = hasAxes = false;
                if (reader.TokenType == JsonTokenType.Null)
                {
                    // Params == null in the parsed case, which the strength gate signs as "{}".
                    signature = "{}";
                    return;
                }
                if (reader.TokenType != JsonTokenType.StartObject)
                    throw new JsonException($"'params' must be an object or null, found {reader.TokenType}");

                _params.Clear();
                while (true)
                {
                    Next(ref reader);
                    if (reader.TokenType == JsonTokenType.EndObject)
                        break;

                    // A dictionary key: the unescaped name, compared case-SENSITIVELY (the parsed case's
                    // Dictionary<string, JsonElement> uses the default comparer).
                    string key = reader.GetString();
                    Next(ref reader);
                    JsonTokenType kind = reader.TokenType;
                    string text = kind == JsonTokenType.String ? reader.GetString() : null;

                    // The value's exact source text, which is what JsonElement.GetRawText() returns: TokenStartIndex is the
                    // value's first byte (a string's opening quote), and after Skip() BytesConsumed is one past its last
                    // (a container's closing bracket; Skip leaves a primitive where it is).
                    int start = checked((int)reader.TokenStartIndex);
                    reader.Skip();
                    int end = checked((int)reader.BytesConsumed);
                    // Not pooled: only the joined signature below is kept by a record.
                    string raw = Encoding.UTF8.GetString(line[start..end]);

                    Upsert(_params, key, raw);
                    switch (key)
                    {
                        case "dist": dist = text; distPresent = true; break;
                        case "ufunc": ufunc = text; break;
                        case "dtype": dtype = text; break;
                        case "axes": hasAxes = true; break;
                    }
                }

                if (_params.Count == 0)
                {
                    signature = "";   // string.Join over an empty dictionary
                    return;
                }
                _params.Sort((a, b) => string.CompareOrdinal(a.Key, b.Key));
                var sb = new StringBuilder();
                for (int i = 0; i < _params.Count; i++)
                {
                    if (i > 0)
                        sb.Append('|');
                    sb.Append(_params[i].Key).Append('=').Append(_params[i].Value);
                }
                signature = Intern(sb.ToString());
            }

            /// <summary>
            ///     Reads an <c>operands</c> value: each operand's dtype, and whether any operand is 0-d.
            /// </summary>
            /// <param name="reader">Positioned on the value's first token; left on its last.</param>
            /// <param name="dtypes">Receives the per-operand dtypes (null where absent), shared between equal lists.</param>
            /// <param name="signature">Receives the dtypes joined by <c>,</c> with null as the empty string.</param>
            /// <param name="hasZeroD">Receives whether some operand's <c>shape</c> is an empty array (null does not count).</param>
            /// <exception cref="JsonException"><c>operands</c> is not an array of operand objects, or a header field has
            /// the wrong type.</exception>
            private void ReadOperands(ref Utf8JsonReader reader, out string[] dtypes, out string signature, out bool hasZeroD)
            {
                hasZeroD = false;
                if (reader.TokenType == JsonTokenType.Null)
                {
                    // Operands == null: the gates read it as no operands at all.
                    dtypes = Array.Empty<string>();
                    signature = "";
                    return;
                }
                if (reader.TokenType != JsonTokenType.StartArray)
                    throw new JsonException($"'operands' must be an array or null, found {reader.TokenType}");

                _dtypes.Clear();
                while (true)
                {
                    Next(ref reader);
                    if (reader.TokenType == JsonTokenType.EndArray)
                        break;
                    if (reader.TokenType != JsonTokenType.StartObject)
                        // A null element parses, but every gate dereferenced it — fail at the source instead.
                        throw new JsonException($"operands[{_dtypes.Count}] must be an object, found {reader.TokenType}");

                    string dtype = null;
                    int shapeRank = -1;   // -1: shape absent or null
                    while (true)
                    {
                        Next(ref reader);
                        if (reader.TokenType == JsonTokenType.EndObject)
                            break;
                        if (NameIs(ref reader, "dtype"u8))
                        {
                            Next(ref reader);
                            dtype = ReadString(ref reader, "operands[].dtype");
                        }
                        else if (NameIs(ref reader, "shape"u8))
                        {
                            Next(ref reader);
                            shapeRank = ReadShapeRank(ref reader);
                        }
                        else
                        {
                            // strides/offset/buffersize and the hex payloads (buffer, mask): never decoded.
                            Next(ref reader);
                            reader.Skip();
                        }
                    }
                    _dtypes.Add(dtype);
                    if (shapeRank == 0)
                        hasZeroD = true;
                }

                signature = Intern(string.Join(",", _dtypes.Select(d => d ?? "")));
                dtypes = ShareDtypeList(signature);
            }

            /// <summary>
            ///     Reads an <c>expected</c> value: the outcome kind, the result dtype, and whether a truth reference rides along.
            /// </summary>
            /// <param name="reader">Positioned on the value's first token; left on its last.</param>
            /// <param name="kind">Receives <see cref="FuzzCorpus.Expected.KindOrArray"/>, or <c>array</c> when expected is null.</param>
            /// <param name="dtype">Receives <c>expected.dtype</c> (null when absent).</param>
            /// <param name="hasTruth">Receives whether <c>expected.truth</c> is a string (not absent, not null).</param>
            /// <exception cref="JsonException"><c>expected</c> is neither an object nor null, or a header field has the wrong type.</exception>
            private void ReadExpected(ref Utf8JsonReader reader, out string kind, out string dtype, out bool hasTruth)
            {
                kind = "array";
                dtype = null;
                hasTruth = false;
                if (reader.TokenType == JsonTokenType.Null)
                    return;
                if (reader.TokenType != JsonTokenType.StartObject)
                    throw new JsonException($"'expected' must be an object or null, found {reader.TokenType}");

                string rawKind = null;
                while (true)
                {
                    Next(ref reader);
                    if (reader.TokenType == JsonTokenType.EndObject)
                        break;
                    if (NameIs(ref reader, "kind"u8))
                    {
                        Next(ref reader);
                        rawKind = ReadString(ref reader, "expected.kind");
                    }
                    else if (NameIs(ref reader, "dtype"u8))
                    {
                        Next(ref reader);
                        dtype = ReadString(ref reader, "expected.dtype");
                    }
                    else if (NameIs(ref reader, "truth"u8))
                    {
                        // Only presence matters, so the (large) hex string is validated as a string but not decoded.
                        Next(ref reader);
                        hasTruth = reader.TokenType switch
                        {
                            JsonTokenType.String => true,
                            JsonTokenType.Null => false,
                            _ => throw new JsonException($"'expected.truth' must be a string or null, found {reader.TokenType}"),
                        };
                    }
                    else
                    {
                        // shape, buffer, value, mask, slots: stepped over whole.
                        Next(ref reader);
                        reader.Skip();
                    }
                }
                kind = string.IsNullOrEmpty(rawKind) ? "array" : rawKind;
            }

            /// <summary>
            ///     Reads a <c>shape</c> value's rank, validating each dimension the way a <c>long[]</c> would.
            /// </summary>
            /// <param name="reader">Positioned on the value's first token; left on its last.</param>
            /// <returns>The number of dimensions, or -1 for a null shape.</returns>
            /// <exception cref="JsonException">The shape is not an array of integers, or null.</exception>
            private static int ReadShapeRank(ref Utf8JsonReader reader)
            {
                if (reader.TokenType == JsonTokenType.Null)
                    return -1;
                if (reader.TokenType != JsonTokenType.StartArray)
                    throw new JsonException($"'shape' must be an array or null, found {reader.TokenType}");
                int rank = 0;
                while (true)
                {
                    Next(ref reader);
                    if (reader.TokenType == JsonTokenType.EndArray)
                        return rank;
                    if (reader.TokenType != JsonTokenType.Number || !reader.TryGetInt64(out _))
                        throw new JsonException($"'shape' entries must be integers, found {reader.TokenType}");
                    rank++;
                }
            }

            /// <summary>
            ///     Reads a string-or-null header value, interned.
            /// </summary>
            /// <param name="reader">Positioned on the value.</param>
            /// <param name="what">The field, for the error message.</param>
            /// <returns>The pooled string, or null for JSON null.</returns>
            /// <exception cref="JsonException">The value is neither a string nor null (the deserializer rejects that too).</exception>
            private string ReadString(ref Utf8JsonReader reader, string what) => reader.TokenType switch
            {
                JsonTokenType.String => Intern(reader.GetString()),
                JsonTokenType.Null => null,
                _ => throw new JsonException($"'{what}' must be a string or null, found {reader.TokenType}"),
            };

            /// <summary>
            ///     Returns the pooled instance equal to <paramref name="value"/>, adding it on first sight.
            /// </summary>
            /// <param name="value">The string (null passes through).</param>
            /// <returns>The shared instance.</returns>
            private string Intern(string value)
            {
                if (value == null)
                    return null;
                if (_strings.TryGetValue(value, out string pooled))
                    return pooled;
                _strings.Add(value, value);
                return value;
            }

            /// <summary>
            ///     Returns one shared array per distinct operand-dtype list, so ~220 K records hold a few hundred arrays.
            /// </summary>
            /// <param name="signature">The list's joined signature (the lookup key).</param>
            /// <returns>The shared array; equal in content (null-ness included) to the list just read.</returns>
            private string[] ShareDtypeList(string signature)
            {
                // The signature maps null and "" to the same text, so confirm the element-wise match before sharing —
                // otherwise hand out a private copy rather than a list that differs in a null.
                if (_dtypeLists.TryGetValue(signature, out string[] shared) && shared.SequenceEqual(_dtypes))
                    return shared;
                string[] fresh = _dtypes.ToArray();
                _dtypeLists.TryAdd(signature, fresh);
                return fresh;
            }

            /// <summary>Adds or replaces a param entry, so a repeated key keeps its LAST value (the dictionary indexer).</summary>
            /// <param name="entries">The entries read so far.</param>
            /// <param name="key">The key.</param>
            /// <param name="value">The value's raw JSON text.</param>
            private static void Upsert(List<KeyValuePair<string, string>> entries, string key, string value)
            {
                for (int i = 0; i < entries.Count; i++)
                {
                    if (string.Equals(entries[i].Key, key, StringComparison.Ordinal))
                    {
                        entries[i] = new KeyValuePair<string, string>(key, value);
                        return;
                    }
                }
                entries.Add(new KeyValuePair<string, string>(key, value));
            }

            /// <summary>
            ///     Whether the current property name equals <paramref name="lowerAsciiName"/> ignoring ASCII case —
            ///     the serializer's case-insensitive property matching for the schema's all-ASCII names.
            /// </summary>
            /// <param name="reader">Positioned on a property name.</param>
            /// <param name="lowerAsciiName">The schema name, lower-case ASCII.</param>
            /// <returns>True on a match.</returns>
            private static bool NameIs(ref Utf8JsonReader reader, ReadOnlySpan<byte> lowerAsciiName)
            {
                // An escaped name (op) must be compared on its unescaped text; the corpus never writes one.
                if (reader.ValueIsEscaped)
                    return string.Equals(reader.GetString(), Encoding.ASCII.GetString(lowerAsciiName),
                                         StringComparison.OrdinalIgnoreCase);
                return Ascii.EqualsIgnoreCase(reader.ValueSpan, lowerAsciiName);
            }

            /// <summary>Advances to the next token, treating a premature end of the line as malformed JSON.</summary>
            /// <param name="reader">The reader.</param>
            /// <exception cref="JsonException">The line ended inside the object.</exception>
            private static void Next(ref Utf8JsonReader reader)
            {
                if (!reader.Read())
                    throw new JsonException("a corpus line ended inside its JSON object");
            }
        }
    }

    /// <summary>One scanned corpus file: its name and one header record per case, in file order.</summary>
    /// <param name="Name">The file name (e.g. <c>reduce.jsonl</c>), which the gates filter tiers by.</param>
    /// <param name="Cases">The header records; shared, never mutate.</param>
    internal sealed record SurveyFile(string Name, SurveyCase[] Cases);

    /// <summary>
    ///     The header fields of one corpus case — everything the coverage gates read, nothing they do not. Strings are
    ///     pooled across the whole survey; <see cref="OperandDtypes"/> arrays are shared between cases and must not be
    ///     mutated.
    /// </summary>
    /// <param name="Op">The op key (<see cref="FuzzCorpus.Case.Op"/>); null when absent.</param>
    /// <param name="Layout">The layout tag (<see cref="FuzzCorpus.Case.Layout"/>); null when absent.</param>
    /// <param name="Valueclass">The value-class tag (<see cref="FuzzCorpus.Case.Valueclass"/>); null when absent.</param>
    /// <param name="OperandDtypes">Each operand's dtype in order (null where absent); empty when operands are absent or null.</param>
    /// <param name="DtypeSignature"><see cref="OperandDtypes"/> joined by <c>,</c>, null as the empty string.</param>
    /// <param name="HasZeroDOperand">Some operand's shape is an empty array (a 0-d operand).</param>
    /// <param name="ParamSignature">The params as <c>key=rawJson</c> pairs ordinal by key joined by <c>|</c>; the empty
    /// string for <c>{}</c>, <c>{}</c> for null or absent params.</param>
    /// <param name="Dist"><c>params.dist</c> when it is a JSON string (the sampler of an <c>rnd</c> case), else null.</param>
    /// <param name="Ufunc"><c>params.ufunc</c> when it is a JSON string (the real ufunc behind an <c>out_*</c> vehicle), else null.</param>
    /// <param name="ParamDtype"><c>params.dtype</c> when it is a JSON string, else null.</param>
    /// <param name="HasAxesParam">A param keyed exactly <c>axes</c> is present, whatever its value.</param>
    /// <param name="ExpectedKind"><see cref="FuzzCorpus.Expected.KindOrArray"/>; <c>array</c> when expected is null or absent.</param>
    /// <param name="ExpectedDtype"><see cref="FuzzCorpus.Expected.Dtype"/>; null when absent (or expected is).</param>
    /// <param name="HasTruth">The expected result carries a correctly-rounded truth reference (the precision tier).</param>
    /// <param name="IsError">NumPy raised: an <c>error</c> object is recorded or <c>expects_throw</c> is true.</param>
    internal readonly record struct SurveyCase(
        string Op, string Layout, string Valueclass,
        string[] OperandDtypes, string DtypeSignature, bool HasZeroDOperand,
        string ParamSignature, string Dist, string Ufunc, string ParamDtype, bool HasAxesParam,
        string ExpectedKind, string ExpectedDtype, bool HasTruth,
        bool IsError);
}
