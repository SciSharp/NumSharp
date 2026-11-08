using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Tests.Fuzz;
using static NumSharp.Tests.TextIO.TextIOOracleCorpus;

namespace NumSharp.Tests.TextIO
{
    /// <summary>
    ///     Replays the NumPy 2.4.2 text I/O oracle (<c>test/oracle/gen_textio_oracle.py</c>, generated with Linux NumPy)
    ///     against NumSharp's <c>np.fromstring</c>, <c>np.fromfile</c>, <c>np.loadtxt</c>, <c>np.savetxt</c> and
    ///     <c>ndarray.tofile</c>, one test per corpus section, every case exactly: a result by its dtype, shape and bytes
    ///     (StringDType by its elements), a write by every byte or character the target received (also when the call
    ///     raised), a file read by the stream position afterwards, an error by NumPy's class and verbatim text.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Platform.</b> NumSharp's readers pin Linux / macOS NumPy (a 64-bit C <c>long</c>, glibc's <c>fscanf</c>, UTF-8
    ///     for a file opened without an encoding), so the corpus replays unchanged on every host but for one thing: a file
    ///     NAME written by <c>np.savetxt</c> goes through Python's text mode, which writes <c>os.linesep</c> for every
    ///     <c>\n</c>, and NumSharp follows the host the same way. On a host whose <see cref="Environment.NewLine"/> is
    ///     <c>\r\n</c> those cases are held to their <c>crlf</c> twin (the generator's text-mode wrapper with
    ///     <c>newline="\r\n"</c>, proven equal to the real file on the generating host with <c>"\n"</c>).
    ///     </para>
    ///     <para>
    ///     <b>Non-vacuity.</b> Each file's header record states its case count, which the replay must reach.
    ///     </para>
    ///     <para>
    ///     <b>Watchdog.</b> Every case runs under <see cref="CaseWatchdog"/>: a reader loop that stops advancing (a regressed
    ///     tokenizer state, a separator test that never consumes) ends the test host naming the case, instead of hanging
    ///     the run.
    ///     </para>
    /// </remarks>
    [TestClass]
    [TestCategory("TextIOOracle")]
    public class TextIOOracleTests
    {
        /// <summary>
        ///     The directory of this process' temporary files: one per process, so the two target frameworks' test hosts
        ///     never share a name. Each case deletes its own files; <see cref="Cleanup"/> removes the directory.
        /// </summary>
        /// <remarks>
        ///     Beside the test assembly rather than in <see cref="Path.GetTempPath"/>: about 13,000 cases write a file, and
        ///     on the development machine a file created, opened and deleted in the user's temp folder cost 2.4 ms (the
        ///     antivirus scans each new file on its first open) against 0.6 ms in the repository's build output, which put
        ///     a minute on the suite. The build output is ignored by git, and a crashed host's leftovers stay inside it.
        /// </remarks>
        private static readonly string TempDir =
            Path.Combine(AppContext.BaseDirectory, "TextIO", $"tmp_{Environment.ProcessId}");

        /// <summary>Whether a file NAME written in text mode receives <c>\r\n</c> for every <c>\n</c> on this host.</summary>
        private static readonly bool CrlfHost = Environment.NewLine == "\r\n";

        /// <summary>The number of failures a section reports in full; the rest are counted.</summary>
        private const int ReportedFailures = 30;

        /// <summary>Creates the temporary directory.</summary>
        /// <param name="context">The MSTest context (unused).</param>
        [ClassInitialize]
        public static void Initialize(TestContext context) => Directory.CreateDirectory(TempDir);

        /// <summary>Removes the temporary directory and anything a failed case left in it.</summary>
        [ClassCleanup]
        public static void Cleanup()
        {
            try
            {
                Directory.Delete(TempDir, recursive: true);
            }
            catch (IOException)
            {
                // A file still open in another test host of this process family: the OS temp cleaner takes it later.
            }
        }

        /// <summary>
        ///     Section fromtext: <c>np.fromstring</c> over a <c>str</c> and a <c>bytes</c> argument, and <c>np.fromfile</c>'s
        ///     text mode over a file name and an open file — NumPy's <c>array_from_text</c> with every separator form,
        ///     count and offset, and its C scanners over signs, digits, overflows, NaN / inf spellings, complex forms,
        ///     junk and long texts across the C <c>FILE</c> model's 64 KB chunks.
        /// </summary>
        [TestMethod]
        public void FromText() => Run("fromtext", ReplayFrom);

        /// <summary>
        ///     Section frombin: <c>np.fromfile</c>'s binary mode — counts, offsets (negative ones included), partial
        ///     trailing items, a <c>sep=None</c> call, every dtype, and the file's position afterwards.
        /// </summary>
        [TestMethod]
        public void FromBinary() => Run("frombin", ReplayFrom);

        /// <summary>
        ///     Section fromedges: a fixed grid for both text readers — every scanner token (integer magnitudes at each
        ///     width's limit in every sign, malformed integers, NaN / inf spellings, float limits, complex forms) alone,
        ///     after a clean element and before one; every separator in seven placements; bytes at or past 0x80 where
        ///     NumPy's signed <c>char</c> loses a 0xFF. A random text usually fails on an early token, so the rules for a
        ///     value that is read (an overflow's saturation, a lost byte) decide outcomes only here.
        /// </summary>
        [TestMethod]
        public void FromTextEdges() => Run("fromedges", ReplayFrom);

        /// <summary>
        ///     Section loadtxt: <c>np.loadtxt</c> with every option (delimiters, comments, quoting, <c>skiprows</c>,
        ///     <c>max_rows</c>, <c>usecols</c>, <c>ndmin</c>, <c>unpack</c>, encodings, converters) over a list of lines, a
        ///     text stream, a byte stream and a file name, and every dtype, StringDType included.
        /// </summary>
        [TestMethod]
        public void LoadTxt() => Run("loadtxt", ReplayLoadTxt);

        /// <summary>
        ///     Section ltfields: numeric fields — float16 through Python's <c>float()</c>, float32 / float64 / complex128
        ///     through NumPy's own parsers, the integer dtypes through <c>str_to_int64</c> / <c>str_to_uint64</c> — in
        ///     groups of 64 accepted fields, and every refused field alone.
        /// </summary>
        [TestMethod]
        public void LoadTxtFields() => Run("ltfields", ReplayLoadTxt);

        /// <summary>
        ///     Section lttokens: the tokenizer's edges read as text dtypes — Unicode whitespace, <c>U+001C..U+001F</c>,
        ///     newlines inside lines, comments, quotes, empty fields and fields past the 16-character scalar scan — over
        ///     the quote-free line path, the chunked state machine and a file.
        /// </summary>
        [TestMethod]
        public void LoadTxtTokens() => Run("lttokens", ReplayLoadTxt);

        /// <summary>
        ///     Section savetxt: <c>np.savetxt</c> with valid and broken <c>%</c> formats, delimiters, newlines, headers,
        ///     footers, comments and encodings, into a text stream, a byte stream and a file name, over every layout of
        ///     every dtype.
        /// </summary>
        [TestMethod]
        public void SaveTxt() => Run("savetxt", ReplayWriter);

        /// <summary>
        ///     Section tofile: <c>ndarray.tofile</c>'s binary and text modes (separators, formats, NUL and lone-surrogate
        ///     arguments that create no file) over every layout of every dtype.
        /// </summary>
        [TestMethod]
        public void ToFile() => Run("tofile", ReplayWriter);

        /// <summary>
        ///     Streams one section's corpus through a replay, one case at a time under the watchdog, and fails with the
        ///     first <see cref="ReportedFailures"/> divergences.
        /// </summary>
        /// <param name="section">The section name.</param>
        /// <param name="replay">Replays one case: null when NumSharp matches NumPy, else the divergence.</param>
        private static void Run(string section, Func<JsonElement, string> replay)
        {
            string path = PathOf(section);
            Assert.IsTrue(File.Exists(path), $"corpus file missing: {path} (run test/oracle/gen_textio_oracle.py, then build)");
            int declared = -1, seen = 0, failed = 0;
            var failures = new List<string>();
            using (var slot = CaseWatchdog.Watch(Path.GetFileName(path)))
            {
                foreach (string line in File.ReadLines(path))
                {
                    if (line.Length == 0)
                        continue;
                    using var doc = JsonDocument.Parse(line);
                    var c = doc.RootElement;
                    if (c.TryGetProperty("header", out var header))
                    {
                        declared = header.GetProperty("cases").GetInt32();
                        continue;
                    }
                    seen++;
                    string id = c.GetProperty("id").GetString();
                    slot.Enter(id);
                    string verdict;
                    try
                    {
                        verdict = replay(c);
                    }
                    catch (Exception e)
                    {
                        // A failure of the replay itself (a case it cannot rebuild) is a failure, never a skip.
                        verdict = $"replay threw {e.GetType().Name}: {e.Message}";
                    }
                    if (verdict == null)
                        continue;
                    failed++;
                    if (failures.Count < ReportedFailures)
                        failures.Add($"{id}: {verdict}");
                }
            }
            Assert.IsTrue(declared > 0, $"{Path.GetFileName(path)} has no header record");
            Assert.AreEqual(declared, seen, $"{Path.GetFileName(path)}: the header declares {declared} cases, {seen} were read");
            if (failed > 0)
                Assert.Fail($"{failed} of {seen} {section} cases diverge from NumPy 2.4.2:\n" + string.Join("\n", failures));
        }

        /// <summary>A temporary file for one case, written with the case's bytes.</summary>
        /// <param name="id">The case id (names the file).</param>
        /// <param name="data">The file's bytes.</param>
        /// <returns>The path.</returns>
        private static string WriteTemp(string id, byte[] data)
        {
            string path = Path.Combine(TempDir, id.Replace('/', '_') + ".dat");
            File.WriteAllBytes(path, data);
            return path;
        }

        /// <summary>A path for one case's output file, not yet created.</summary>
        /// <param name="id">The case id (names the file).</param>
        /// <returns>The path.</returns>
        private static string OutputPath(string id) => Path.Combine(TempDir, id.Replace('/', '_') + ".out");

        // ------------------------------------------------------------------------------------------------------------
        // np.fromstring / np.fromfile

        /// <summary>Replays one fromtext / frombin case.</summary>
        /// <param name="c">The case.</param>
        /// <returns>Null when NumSharp matches NumPy, else the divergence.</returns>
        private static string ReplayFrom(JsonElement c)
        {
            string id = c.GetProperty("id").GetString();
            string fn = c.GetProperty("fn").GetString();
            int count = c.GetProperty("count").GetInt32();
            string sep = Str(c.GetProperty("sep"));
            object got;
            long? tell = null;
            string path = null;
            try
            {
                DType dt = np.dtype(c.GetProperty("dtype").GetString());
                if (fn == "fromstring")
                {
                    got = c.TryGetProperty("bytes", out var b)
                        ? np.fromstring(Bytes(b), dt, count, sep)
                        : np.fromstring(Str(c.GetProperty("text")), dt, count, sep);
                }
                else
                {
                    path = WriteTemp(id, Bytes(c.GetProperty("data")));
                    long offset = c.GetProperty("offset").GetInt64();
                    if (c.GetProperty("mode").GetString() == "name")
                    {
                        // A null sep is NumPy's call without sep (its default "", the binary mode).
                        got = sep == null ? np.fromfile(path, dt, count, offset: offset) : np.fromfile(path, dt, count, sep, offset);
                    }
                    else
                    {
                        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
                        fs.Seek(c.GetProperty("skip").GetInt32(), SeekOrigin.Begin);
                        try
                        {
                            got = sep == null ? np.fromfile(fs, dt, count, offset: offset) : np.fromfile(fs, dt, count, sep, offset);
                        }
                        finally
                        {
                            // NumPy seeks the Python file to the C FILE's position after the read, and after an error too.
                            tell = fs.Position;
                        }
                    }
                }
            }
            catch (Exception e)
            {
                got = e;
            }
            finally
            {
                if (path != null)
                    File.Delete(path);
            }

            string verdict = CompareOutcome(c, got);
            if (verdict == null && c.TryGetProperty("tell", out var t) && tell != t.GetInt64())
                verdict = $"stream position: numpy {t.GetInt64()}, numsharp {tell}";
            return verdict;
        }

        /// <summary>
        ///     Compares a reader's outcome (a result or an exception) with the case's <c>expect</c> or <c>error</c>, and
        ///     disposes a returned array.
        /// </summary>
        /// <param name="c">The case.</param>
        /// <param name="got">NumSharp's result or exception.</param>
        /// <returns>Null when they match, else the divergence.</returns>
        private static string CompareOutcome(JsonElement c, object got)
        {
            if (c.TryGetProperty("error", out var error) && error.ValueKind != JsonValueKind.Null)
            {
                if (got is Exception ex)
                    return CompareError(error, ex);
                using var a = (NDArray)got;
                return $"numpy raised {error.GetProperty("type").GetString()}: {Shorten(Str(error.GetProperty("msg")))}, " +
                       $"numsharp returned {a.dtype}({string.Join(",", a.shape)})";
            }
            if (got is Exception e)
                return UnexpectedError(e);
            using var r = (NDArray)got;
            var expect = c.GetProperty("expect");
            return expect.TryGetProperty("cells", out _) ? CompareCells(expect, r) : CompareArray(expect, r);
        }

        // ------------------------------------------------------------------------------------------------------------
        // np.loadtxt

        /// <summary>Replays one loadtxt / ltfields / lttokens case.</summary>
        /// <param name="c">The case.</param>
        /// <returns>Null when NumSharp matches NumPy, else the divergence.</returns>
        private static string ReplayLoadTxt(JsonElement c)
        {
            string id = c.GetProperty("id").GetString();
            var p = c.GetProperty("params");
            object got;
            string path = null;
            try
            {
                DType dt = MakeDtype(c.GetProperty("dtype").GetString());
                string delimiter = p.TryGetProperty("delimiter", out var de) ? (string)Param(de) : null;
                // comments: absent is NumPy's default '#'; otherwise the Python object as given (None, a str, a number, or
                // a list of them, which loadtxt reads element by element as NumPy does).
                object comments = p.TryGetProperty("comments", out var ce) ? Param(ce) : "#";
                string quotechar = p.TryGetProperty("quotechar", out var qe) ? (string)Param(qe) : null;
                int skiprows = p.TryGetProperty("skiprows", out var se) ? se.GetInt32() : 0;
                int? maxRows = p.TryGetProperty("max_rows", out var me) && me.ValueKind == JsonValueKind.Number ? me.GetInt32() : null;
                // usecols: a single int is NumPy's one-element list (`list(usecols)` fails, `[usecols]` follows).
                int[] usecols = null;
                if (p.TryGetProperty("usecols", out var ue) && ue.ValueKind != JsonValueKind.Null)
                    usecols = ue.ValueKind == JsonValueKind.Array ? ue.EnumerateArray().Select(x => x.GetInt32()).ToArray()
                                                                  : new[] { ue.GetInt32() };
                int ndmin = p.TryGetProperty("ndmin", out var ne) ? ne.GetInt32() : 0;
                bool unpack = p.TryGetProperty("unpack", out var une) && une.GetBoolean();
                string encoding = p.TryGetProperty("encoding", out var ee) && ee.ValueKind == JsonValueKind.String ? ee.GetString() : null;
                object converters = null;
                if (p.TryGetProperty("converters", out var cv))
                {
                    var log = c.GetProperty("conv_calls");
                    if (cv.ValueKind == JsonValueKind.String)
                        converters = ConverterTwin(cv.GetString(), log);
                    else
                    {
                        var byColumn = new Dictionary<int, Func<string, object>>();
                        foreach (var column in cv.EnumerateObject())
                            byColumn[int.Parse(column.Name, System.Globalization.CultureInfo.InvariantCulture)] =
                                column.Value.ValueKind == JsonValueKind.Null ? null : ConverterTwin(column.Value.GetString(), log);
                        converters = byColumn;
                    }
                }
                switch (c.GetProperty("source").GetString())
                {
                    case "lines":
                    {
                        var lines = c.GetProperty("lines").EnumerateArray().Select(Str).ToList();
                        got = np.loadtxt(lines, dt, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding, maxRows, quotechar);
                        break;
                    }
                    case "reader":
                    {
                        using var reader = new StringReader(Str(c.GetProperty("text")));
                        got = np.loadtxt(reader, dt, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding, maxRows, quotechar);
                        break;
                    }
                    case "stream":
                    {
                        using var stream = new MemoryStream(Bytes(c.GetProperty("data")));
                        got = np.loadtxt(stream, dt, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding, maxRows, quotechar);
                        break;
                    }
                    case "file":
                        path = WriteTemp(id, Bytes(c.GetProperty("data")));
                        got = np.loadtxt(path, dt, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding, maxRows, quotechar);
                        break;
                    default:
                        throw new InvalidOperationException("unknown source " + c.GetProperty("source").GetString());
                }
            }
            catch (Exception e)
            {
                got = e;
            }
            finally
            {
                if (path != null)
                    File.Delete(path);
            }
            return CompareOutcome(c, got);
        }

        // ------------------------------------------------------------------------------------------------------------
        // np.savetxt / ndarray.tofile

        /// <summary>Rebuilds a case's base array (C order) from its bytes or StringDType cells.</summary>
        /// <param name="a">The recorded array (<c>dtype</c>, <c>shape</c>, <c>hex</c> or <c>cells</c> + <c>na</c>).</param>
        /// <returns>A fresh, owning array.</returns>
        private static NDArray BuildBase(JsonElement a)
        {
            string dt = a.GetProperty("dtype").GetString();
            long[] shape = a.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            if (dt == "T")
            {
                // The elements as the generator recorded them (TextIOOracleCorpus.CellKey): None, a NaN NA object, text.
                var values = a.GetProperty("cells").EnumerateArray().Select(x =>
                    x.ValueKind == JsonValueKind.Null ? null
                    : x.ValueKind == JsonValueKind.Object && x.TryGetProperty("nan", out _) ? double.NaN
                    : (object)Str(x)).ToArray();
                DType d = a.GetProperty("na").GetString() switch
                {
                    "nan" => np.dtypes.StringDType(double.NaN),
                    "strNA" => np.dtypes.StringDType("NA"),
                    _ => np.dtypes.StringDType(),
                };
                using var flat = np.array(values, dtype: d);
                using var shaped = flat.reshape(new Shape(shape));
                return shaped.copy();
            }
            byte[] bytes = Convert.FromHexString(a.GetProperty("hex").GetString());
            using var raw = np.frombuffer(bytes, np.dtype(dt));
            using var reshaped = raw.reshape(new Shape(shape));
            return reshaped.copy();
        }

        /// <summary>The written view of a base array, as the generator's recipe built it.</summary>
        /// <param name="b">The base array.</param>
        /// <param name="recipe">C, F, transposed, stepped, reversed, offset or broadcast.</param>
        /// <returns>The base itself for <c>c</c> (a 0-d base has no slice to take), else a new array object (a view, or for
        ///     <c>f</c> NumPy's F-ordered copy) that the caller disposes, which releases only its own reference on the base's
        ///     buffer.</returns>
        /// <exception cref="InvalidOperationException">An unknown recipe.</exception>
        private static NDArray View(NDArray b, string recipe)
        {
            switch (recipe)
            {
                case "c": return b;
                case "f": return np.asfortranarray(b);
                case "t": return b.T;
                case "s0": return b["::2"];
                case "s1": return b[":, ::2"];
                case "r0": return b["::-1"];
                case "r1": return b[":, ::-1"];
                case "o": return b["1:"];
                case "b":
                {
                    // The first row broadcast to three; the row view is released at once (the broadcast view holds its
                    // own reference on the buffer).
                    using var row = b["0:1"];
                    return np.broadcast_to(row, new Shape(new long[] { 3 }.Concat(b.shape.Skip(1).Select(x => (long)x)).ToArray()));
                }
                default:
                    throw new InvalidOperationException("unknown view recipe " + recipe);
            }
        }

        /// <summary>Replays one savetxt / tofile case: the call, then its error and the target's contents.</summary>
        /// <param name="c">The case.</param>
        /// <returns>Null when NumSharp matches NumPy, else the divergences (an error and an output can both differ).</returns>
        private static string ReplayWriter(JsonElement c)
        {
            string id = c.GetProperty("id").GetString();
            string target = c.GetProperty("target").GetString();
            var p = c.GetProperty("params");
            Exception err = null;
            string text = null;
            byte[] bytes = null;
            bool missing = false;
            string path = null;
            NDArray baseArray = null, view = null;
            try
            {
                baseArray = BuildBase(c.GetProperty("array"));
                view = View(baseArray, c.GetProperty("view").GetString());
                if (c.GetProperty("fn").GetString() == "savetxt")
                {
                    bool isList = c.GetProperty("fmt_is_list").GetBoolean();
                    var fe = p.GetProperty("fmt");
                    string fmt = isList ? null : Str(fe);
                    string[] fmts = isList ? fe.EnumerateArray().Select(Str).ToArray() : null;
                    string delimiter = p.TryGetProperty("delimiter", out var de) ? Str(de) : " ";
                    string newline = p.TryGetProperty("newline", out var nle) ? Str(nle) : "\n";
                    string header = p.TryGetProperty("header", out var he) ? Str(he) : "";
                    string footer = p.TryGetProperty("footer", out var foe) ? Str(foe) : "";
                    string comments = p.TryGetProperty("comments", out var ce) ? Str(ce) : "# ";
                    string encoding = p.TryGetProperty("encoding", out var ee) && ee.ValueKind == JsonValueKind.String ? ee.GetString() : null;
                    switch (target)
                    {
                        case "writer":
                        {
                            var sw = new StringWriter();
                            try
                            {
                                if (isList) np.savetxt(sw, view, fmts, delimiter, newline, header, footer, comments, encoding);
                                else np.savetxt(sw, view, fmt, delimiter, newline, header, footer, comments, encoding);
                            }
                            catch (Exception e)
                            {
                                err = e;
                            }
                            text = sw.ToString();
                            break;
                        }
                        case "stream":
                        {
                            var ms = new MemoryStream();
                            try
                            {
                                if (isList) np.savetxt(ms, view, fmts, delimiter, newline, header, footer, comments, encoding);
                                else np.savetxt(ms, view, fmt, delimiter, newline, header, footer, comments, encoding);
                            }
                            catch (Exception e)
                            {
                                err = e;
                            }
                            bytes = ms.ToArray();
                            break;
                        }
                        case "file":
                            path = OutputPath(id);
                            try
                            {
                                if (isList) np.savetxt(path, view, fmts, delimiter, newline, header, footer, comments, encoding);
                                else np.savetxt(path, view, fmt, delimiter, newline, header, footer, comments, encoding);
                            }
                            catch (Exception e)
                            {
                                err = e;
                            }
                            if (File.Exists(path)) bytes = File.ReadAllBytes(path);
                            else missing = true;
                            break;
                        default:
                            throw new InvalidOperationException("unknown target " + target);
                    }
                }
                else
                {
                    string sep = p.TryGetProperty("sep", out var se) ? Str(se) : "";
                    string format = p.TryGetProperty("format", out var fe) ? Str(fe) : "%s";
                    path = OutputPath(id);
                    try
                    {
                        view.tofile(path, sep, format);
                    }
                    catch (Exception e)
                    {
                        err = e;
                    }
                    if (File.Exists(path)) bytes = File.ReadAllBytes(path);
                    else missing = true;
                }
            }
            finally
            {
                if (!ReferenceEquals(view, baseArray))
                    view?.Dispose();
                baseArray?.Dispose();
                if (path != null && File.Exists(path))
                    File.Delete(path);
            }

            // A file NAME written in text mode: on a CRLF host, NumPy's (and NumSharp's) bytes and error are the twin's.
            var outcome = c;
            if (CrlfHost && target == "file" && c.GetProperty("fn").GetString() == "savetxt")
                outcome = c.GetProperty("crlf");
            var verdicts = new List<string>();
            var wantError = outcome.GetProperty("error");
            if (wantError.ValueKind == JsonValueKind.Null)
            {
                if (err != null)
                    verdicts.Add(UnexpectedError(err));
            }
            else if (err == null)
                verdicts.Add($"numpy raised {wantError.GetProperty("type").GetString()}: {Shorten(Str(wantError.GetProperty("msg")))}, numsharp did not");
            else if (CompareError(wantError, err) is string e2)
                verdicts.Add(e2);

            var wantOut = outcome.GetProperty("out");
            string outVerdict;
            if (wantOut.TryGetProperty("missing", out _))
                outVerdict = missing ? null : "numpy created no file, numsharp did";
            else if (missing)
                outVerdict = "numpy wrote a file, numsharp created none";
            else if (wantOut.TryGetProperty("text", out _) || wantOut.TryGetProperty("sha16", out _))
                outVerdict = CompareText(wantOut, text);
            else
                outVerdict = CompareBytes(wantOut, bytes);
            if (outVerdict != null)
                verdicts.Add(outVerdict);
            return verdicts.Count == 0 ? null : string.Join("\n    ", verdicts);
        }
    }
}
