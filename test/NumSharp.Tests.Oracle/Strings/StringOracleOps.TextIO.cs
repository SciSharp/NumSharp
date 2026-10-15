using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section tio (<c>gen_tio</c>, strings phase 1, P1.7c): text I/O over NEP 55
    ///     <c>StringDType</c> — <c>np.loadtxt</c> from the four sources NumPy reads (a list of lines, a text stream, a byte
    ///     stream, a file name), <c>np.savetxt</c> into a text or a byte stream, <c>ndarray.tofile</c>, and the refusals of
    ///     <c>np.fromfile</c> / <c>np.fromstring</c> / <c>np.frombuffer</c>.
    /// </summary>
    /// <remarks>
    ///     Every source and target is rebuilt as the C# object a ported NumPy line passes: a Python list of <c>str</c> is an
    ///     <see cref="IEnumerable{T}"/> of strings, <c>io.StringIO</c> a <see cref="StringReader"/> / <see cref="StringWriter"/>,
    ///     <c>io.BytesIO</c> a <see cref="MemoryStream"/>, and a file name a temporary file the case writes first (and deletes
    ///     afterwards). A converter travels as a NAME from a fixed menu that both sides map to the same computation
    ///     (<see cref="TioConverter"/>), because a Python function cannot travel in the corpus.
    /// </remarks>
    internal static partial class StringOracleOps
    {
        /// <summary>
        ///     The directory of this process' temporary tio files: one per process, so two test hosts (the two target
        ///     frameworks) never write the same name. Each case deletes its own files.
        /// </summary>
        private static readonly string TioTempDir =
            Path.Combine(Path.GetTempPath(), $"numsharp_tio_{Environment.ProcessId}");

        /// <summary>
        ///     Replays one section-tio op, or reports that <paramref name="op"/> is not one of them.
        /// </summary>
        /// <param name="op">The corpus op key (<c>t_loadtxt</c>, <c>t_savetxt</c>, …).</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="ops">The reconstructed operands (the array savetxt / tofile write).</param>
        /// <param name="result">The call's result: the loaded array or its cells, the text or bytes written.</param>
        /// <returns>True when <paramref name="op"/> is a section-tio op.</returns>
        /// <exception cref="InvalidOperationException">A parameter encoding or a converter name the replay does not
        ///     know.</exception>
        /// <remarks>Any exception the replayed call raises propagates: the runner compares it with NumPy's error.</remarks>
        private static bool TryInvokeTextIO(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            switch (op)
            {
                case "t_loadtxt":
                    result = LoadTxtCase(p);
                    return true;
                case "t_savetxt":
                    result = SaveTxtCase(p, ops[0], written: false);
                    return true;
                case "t_savetxt_written":
                    result = SaveTxtCase(p, ops[0], written: true);
                    return true;
                case "t_tofile":
                    result = ToFileCase(p, ops[0], written: false);
                    return true;
                case "t_tofile_written":
                    result = ToFileCase(p, ops[0], written: true);
                    return true;
                case "t_fromfile":
                {
                    string path = WriteTioFile(Convert.FromHexString(p["data"].GetProperty("b").GetString()));
                    try
                    {
                        result = np.fromfile(path, Dt(p, "dtype"), p["count"].GetInt32(),
                            (string)StringOracleCorpus.Decode(p["sep"]), p["offset"].GetInt64());
                    }
                    finally
                    {
                        File.Delete(path);
                    }
                    return true;
                }
                case "t_fromstring":
                    // A None sep is NumPy's call without sep (its binary mode); C#'s default sep is null, the same mode.
                    result = np.fromstring((string)StringOracleCorpus.Decode(p["text"]), Dt(p, "dtype"), -1,
                        (string)StringOracleCorpus.Decode(p["sep"]));
                    return true;
                // t_frombuffer is section tcreate's op (TryInvokeVString), which tio reuses with more buffers.
            }
            result = null;
            return false;
        }

        /// <summary>
        ///     One <c>np.loadtxt</c> case: the source rebuilt as its C# twin, every argument the generator passed (an omitted
        ///     one takes the C# default, which is NumPy's), and the array or its raw cells.
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <returns>The loaded array, or its <c>tobytes()</c> cells for <c>result == "cells"</c>.</returns>
        /// <exception cref="InvalidOperationException">An unknown source kind.</exception>
        private static object LoadTxtCase(Dictionary<string, JsonElement> p)
        {
            var dt = Dt(p, "dtype");
            // comments: absent is NumPy's default '#'; otherwise the Python object as given — None, a str, bytes, a number,
            // or a list / tuple of any of them (a C# object[], which loadtxt reads element by element as NumPy does).
            object comments = "#";
            if (p.TryGetValue("comments", out var ce))
            {
                var decoded = StringOracleCorpus.Decode(ce);
                comments = decoded is List<object> list ? list.ToArray() : decoded;
            }
            string delimiter = p.TryGetValue("delimiter", out var de) ? (string)StringOracleCorpus.Decode(de) : null;
            string quotechar = p.TryGetValue("quotechar", out var qe) ? (string)StringOracleCorpus.Decode(qe) : null;
            int skiprows = p.TryGetValue("skiprows", out var se) ? se.GetInt32() : 0;
            // usecols: a single int is NumPy's one-element list (`list(usecols)` fails, `[usecols]` follows).
            int[] usecols = null;
            if (p.TryGetValue("usecols", out var ue) && ue.ValueKind != JsonValueKind.Null)
                usecols = ue.ValueKind == JsonValueKind.Array ? ue.EnumerateArray().Select(x => x.GetInt32()).ToArray()
                                                              : new[] { ue.GetInt32() };
            bool unpack = p.TryGetValue("unpack", out var une) && une.GetBoolean();
            int ndmin = p.TryGetValue("ndmin", out var ne) ? ne.GetInt32() : 0;
            int? maxRows = p.TryGetValue("max_rows", out var me) && me.ValueKind == JsonValueKind.Number ? me.GetInt32() : null;
            string encoding = p.TryGetValue("encoding", out var ee) && ee.ValueKind == JsonValueKind.String ? ee.GetString() : null;
            object converters = p.TryGetValue("conv", out var cv) ? TioConverters(cv) : null;

            string source = Str(p, "source");
            NDArray arr;
            switch (source)
            {
                case "lines":
                {
                    var lines = ((List<object>)StringOracleCorpus.Decode(p["data"])).Cast<string>().ToList();
                    arr = np.loadtxt(lines, dt, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding,
                        maxRows, quotechar);
                    break;
                }
                case "reader":
                {
                    using var reader = new StringReader((string)StringOracleCorpus.Decode(p["data"]));
                    arr = np.loadtxt(reader, dt, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding,
                        maxRows, quotechar);
                    break;
                }
                case "stream":
                {
                    using var stream = new MemoryStream(TioBytes(p["data"]));
                    arr = np.loadtxt(stream, dt, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding,
                        maxRows, quotechar);
                    break;
                }
                case "file":
                {
                    string path = WriteTioFile(TioBytes(p["data"]));
                    try
                    {
                        arr = np.loadtxt(path, dt, comments, delimiter, converters, skiprows, usecols, unpack, ndmin,
                            encoding, maxRows, quotechar);
                    }
                    finally
                    {
                        File.Delete(path);
                    }
                    break;
                }
                default:
                    throw new InvalidOperationException($"unknown loadtxt source '{source}'");
            }
            // Both arms as object: a ?: of byte[] and NDArray is typed NDArray (the implicit Array -> NDArray conversion
            // would turn the cells into a uint8 array).
            return Str(p, "result") == "cells" ? (object)arr.tobytes() : arr;
        }

        /// <summary>
        ///     A stream / file source's bytes: the encoded bytes, or the generator's <c>{"repeat": [unit, count, tail]}</c>
        ///     recipe (<c>unit * count + tail</c>, both hex; a fourth item is a head before them) of a large file.
        /// </summary>
        /// <param name="e">The encoded source.</param>
        /// <returns>The bytes.</returns>
        private static byte[] TioBytes(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Object && e.TryGetProperty("repeat", out var r))
            {
                byte[] unit = Convert.FromHexString(r[0].GetString());
                int count = r[1].GetInt32();
                byte[] tail = Convert.FromHexString(r[2].GetString());
                // An optional fourth item is the bytes BEFORE the repeated run (an invalid byte the chunk starts with).
                byte[] head = r.GetArrayLength() > 3 ? Convert.FromHexString(r[3].GetString()) : Array.Empty<byte>();
                var bytes = new byte[head.Length + unit.Length * count + tail.Length];
                head.CopyTo(bytes, 0);
                for (int i = 0; i < count; i++)
                    unit.CopyTo(bytes, head.Length + i * unit.Length);
                tail.CopyTo(bytes, head.Length + unit.Length * count);
                return bytes;
            }
            return (byte[])StringOracleCorpus.Decode(e);
        }

        /// <summary>
        ///     One <c>np.savetxt</c> case: the array written into a <see cref="StringWriter"/> (NumPy's <c>io.StringIO</c>)
        ///     or a <see cref="MemoryStream"/> (<c>io.BytesIO</c>) with the case's arguments.
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="x">The array.</param>
        /// <param name="written">True for <c>t_savetxt_written</c>: the output is returned even when the call raised (what
        ///     reached the target before the error); false lets the error propagate.</param>
        /// <returns>The text (writer) or the bytes (stream) written.</returns>
        private static object SaveTxtCase(Dictionary<string, JsonElement> p, NDArray x, bool written)
        {
            object fmt = p.TryGetValue("fmt", out var fe) ? StringOracleCorpus.Decode(fe) : "%.18e";
            string delimiter = TioText(p, "delimiter", " ");
            string newline = TioText(p, "newline", "\n");
            string header = TioText(p, "header", "");
            string footer = TioText(p, "footer", "");
            string comments = TioText(p, "comments", "# ");
            string encoding = p.TryGetValue("encoding", out var ee) && ee.ValueKind == JsonValueKind.String ? ee.GetString() : null;
            // A list of specs is NumPy's sequence fmt (the string[] overloads); a str is the single / multi-% format.
            string[] fmts = fmt is List<object> list ? list.Cast<string>().ToArray() : null;
            if (Str(p, "target") == "writer")
            {
                var sw = new StringWriter();
                try
                {
                    if (fmts != null)
                        np.savetxt(sw, x, fmts, delimiter, newline, header, footer, comments, encoding);
                    else
                        np.savetxt(sw, x, (string)fmt, delimiter, newline, header, footer, comments, encoding);
                }
                catch (Exception) when (written)
                {
                    // NumPy's output before the error is the expectation; the error itself is a t_savetxt case.
                }
                return sw.ToString();
            }
            if (Str(p, "target") == "file")
            {
                // A fresh file name: np.savetxt creates it, encodes through the file's encoder; its bytes are the result
                // (the generator's file cases write no '\n', so they do not depend on the platform's line separator).
                Directory.CreateDirectory(TioTempDir);
                string path = Path.Combine(TioTempDir, $"save_{Guid.NewGuid():N}.txt");
                try
                {
                    try
                    {
                        if (fmts != null)
                            np.savetxt(path, x, fmts, delimiter, newline, header, footer, comments, encoding);
                        else
                            np.savetxt(path, x, (string)fmt, delimiter, newline, header, footer, comments, encoding);
                    }
                    catch (Exception) when (written)
                    {
                        // The file NumPy leaves behind: the writes before the error.
                    }
                    return File.ReadAllBytes(path);
                }
                finally
                {
                    File.Delete(path);
                }
            }
            var ms = new MemoryStream();
            try
            {
                if (fmts != null)
                    np.savetxt(ms, x, fmts, delimiter, newline, header, footer, comments, encoding);
                else
                    np.savetxt(ms, x, (string)fmt, delimiter, newline, header, footer, comments, encoding);
            }
            catch (Exception) when (written)
            {
                // As above: the bytes that reached the stream before the error.
            }
            return ms.ToArray();
        }

        /// <summary>
        ///     One <c>ndarray.tofile</c> case: the array written to a fresh temporary file name, whose bytes are the result.
        /// </summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="x">The array.</param>
        /// <param name="written">True for <c>t_tofile_written</c>: the file's bytes are returned even when the call raised
        ///     (NumPy opens the file — creating or truncating it — before it writes); false lets the error propagate.</param>
        /// <returns>The file's bytes.</returns>
        private static object ToFileCase(Dictionary<string, JsonElement> p, NDArray x, bool written)
        {
            Directory.CreateDirectory(TioTempDir);
            string path = Path.Combine(TioTempDir, Guid.NewGuid().ToString("N") + ".bin");
            try
            {
                try
                {
                    x.tofile(path, (string)StringOracleCorpus.Decode(p["sep"]), (string)StringOracleCorpus.Decode(p["format"]));
                }
                catch (Exception) when (written)
                {
                    // The file NumPy left behind is the expectation; the error itself is a t_tofile case.
                }
                return File.ReadAllBytes(path);
            }
            finally
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
        }

        /// <summary>Writes a case's bytes to a fresh temporary file (the "file" source of loadtxt, np.fromfile's input).</summary>
        /// <param name="data">The bytes.</param>
        /// <returns>The file's path; the caller deletes it.</returns>
        private static string WriteTioFile(byte[] data)
        {
            Directory.CreateDirectory(TioTempDir);
            string path = Path.Combine(TioTempDir, Guid.NewGuid().ToString("N") + ".txt");
            File.WriteAllBytes(path, data);
            return path;
        }

        /// <summary>A text parameter of savetxt, or its default when the generator left it out.</summary>
        /// <param name="p">The case parameters.</param>
        /// <param name="key">The parameter name.</param>
        /// <param name="default">NumPy's default.</param>
        /// <returns>The text.</returns>
        private static string TioText(Dictionary<string, JsonElement> p, string key, string @default)
            => p.TryGetValue(key, out var e) ? (string)StringOracleCorpus.Decode(e) : @default;

        /// <summary>
        ///     The <c>converters</c> argument of an encoded request: a converter name (one <see cref="Func{T, TResult}"/> for
        ///     every column), <c>{"col": name}</c> (a dictionary; a null name is a non-callable value), or
        ///     <c>"notcallable"</c> (NumPy's <c>converters=5</c>).
        /// </summary>
        /// <param name="e">The encoded request.</param>
        /// <returns>The C# argument.</returns>
        private static object TioConverters(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Null)
                return null;
            if (e.ValueKind == JsonValueKind.String)
                return e.GetString() == "notcallable" ? 5 : TioConverter(e.GetString());
            var dict = new Dictionary<int, Func<string, object>>();
            foreach (var prop in e.EnumerateObject())
                dict[int.Parse(prop.Name, CultureInfo.InvariantCulture)] =
                    prop.Value.ValueKind == JsonValueKind.Null ? null : TioConverter(prop.Value.GetString());
            return dict;
        }

        /// <summary>
        ///     The C# twin of one entry of the generator's <c>_TIO_CONVERTERS</c> menu: the same computation over the field's
        ///     text, returning the C# value that stands for NumPy's Python value (a <c>long</c> for an int, a <c>double</c> for
        ///     a float, <c>byte[]</c> for bytes, <c>object[]</c> for a list, a value tuple for a tuple, a dictionary for a
        ///     dict, an <see cref="NDArray"/> for an ndarray, a <see cref="Half"/> for an <c>np.float16</c>).
        /// </summary>
        /// <param name="name">The menu name.</param>
        /// <returns>The converter.</returns>
        /// <exception cref="InvalidOperationException">A name the menu does not have.</exception>
        private static Func<string, object> TioConverter(string name) => name switch
        {
            "upper" => s => s.ToUpperInvariant(),
            // Python's len() counts code points; the menu's fields are BMP text, where a code point is one char.
            "len" => s => (long)s.Length,
            "none" => _ => null,
            "nan" => _ => double.NaN,
            "na" => _ => "NA",
            "int" => s => long.Parse(s, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture),
            "float3" => s => double.Parse(s, CultureInfo.InvariantCulture) / 3,
            "bool" => s => s == "a",
            "complex" => _ => new Complex(1, 2),
            "utf8" => s => Encoding.UTF8.GetBytes(s),
            "badbytes" => _ => new byte[] { 0x80 },
            "list" => s => new object[] { s, s },
            "tuple" => s => ValueTuple.Create(s),
            "dict" => s => new Dictionary<string, long> { [s] = 1 },
            "arr0" => _ => np.array(5L),
            "arr0f" => _ => np.array(1.5),
            "arr1" => _ => np.array(new long[] { 5, 6 }),
            "raise" => _ => throw new DivideByZeroException("integer division or modulo by zero"),
            "surrogate" => _ => "\ud800",
            "same" => s => s,
            "half" => _ => (Half)0.5,
            _ => throw new InvalidOperationException($"unknown tio converter '{name}'"),
        };
    }
}
