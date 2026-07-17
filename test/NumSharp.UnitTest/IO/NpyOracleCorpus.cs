using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    ///     Loads the committed NumPy oracle (<c>IO/corpus/npy_oracle.zip</c>) and rebuilds arrays from it.
    /// </summary>
    /// <remarks>
    ///     The zip holds real NumPy 2.4.2 <c>np.save</c> / <c>np.savez</c> output plus a manifest saying,
    ///     for each case, what NumSharp must load, what bytes it must write, or which error it must
    ///     raise. Written by <c>test/oracle/gen_npy_oracle.py</c>; no Python runs at test time.
    /// </remarks>
    internal static class NpyOracleCorpus
    {
        private static readonly Lazy<(NpyCase[] Cases, string NumpyVersion)> _corpus = new(Load);

        public static NpyCase[] Cases => _corpus.Value.Cases;
        public static string NumpyVersion => _corpus.Value.NumpyVersion;

        public static IEnumerable<NpyCase> OfKind(params string[] kinds) =>
            Cases.Where(c => kinds.Contains(c.Kind, StringComparer.Ordinal));

        private static (NpyCase[], string) Load()
        {
            string path = Path.Combine(AppContext.BaseDirectory, "IO", "corpus", "npy_oracle.zip");
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"NumPy oracle corpus not found at '{path}'. It is committed; rebuild the test project to " +
                    "copy it, or regenerate it with: python test/oracle/gen_npy_oracle.py", path);

            // Buffer the whole zip: it is ~1 MB and every case needs random access to it.
            byte[] raw = File.ReadAllBytes(path);
            using var zip = new ZipArchive(new MemoryStream(raw), ZipArchiveMode.Read);

            string json;
            using (var reader = new StreamReader(zip.GetEntry("manifest.json")!.Open()))
                json = reader.ReadToEnd();

            using var doc = JsonDocument.Parse(json);
            JsonElement root = doc.RootElement;

            var cases = new List<NpyCase>();
            foreach (JsonElement e in root.GetProperty("cases").EnumerateArray())
                cases.Add(NpyCase.Parse(e, zip));

            return (cases.ToArray(), root.GetProperty("numpy_version").GetString());
        }

        /// <summary>Rebuild an array holding exactly <paramref name="data"/>, in the requested layout.</summary>
        /// <remarks>
        ///     <paramref name="data"/> is the manifest's canonical form: logical values, native byte order,
        ///     C-order. Writing it into a C-contiguous array and then copying to 'F' produces an
        ///     F-contiguous array with the same logical values — which is what a fortran_order case needs.
        /// </remarks>
        public static unsafe NDArray Build(NPTypeCode typeCode, long[] shape, byte[] data, bool fortranOrder)
        {
            var arr = new NDArray(typeCode, new Shape(shape));
            if (data.Length > 0)
                Marshal.Copy(data, 0, (IntPtr)arr.Storage.Address, data.Length);
            return fortranOrder ? arr.copy('F') : arr;
        }

        /// <summary>The array's logical values as raw C-order bytes — the form the manifest stores.</summary>
        public static unsafe byte[] RawBytes(NDArray a)
        {
            NDArray src = a.Shape.IsContiguous && a.Shape.offset == 0 ? a : a.copy('C');
            long len = src.size * src.dtypesize;
            var bytes = new byte[len];
            if (len > 0)
                Marshal.Copy((IntPtr)src.Storage.Address, bytes, 0, checked((int)len));
            GC.KeepAlive(src);
            return bytes;
        }
    }

    /// <summary>One oracle case. See <c>gen_npy_oracle.py</c>'s module docstring for the schema.</summary>
    internal sealed class NpyCase
    {
        public string Name { get; private init; }
        public string Kind { get; private init; }
        public string Note { get; private init; }

        /// <summary>The exact bytes NumPy produced.</summary>
        public byte[] Bytes { get; private init; }

        public NPTypeCode? NsDtype { get; private init; }
        public long[] Shape { get; private init; }
        public bool FortranOrder { get; private init; }
        public string Descr { get; private init; }
        public byte[] Version { get; private init; }

        /// <summary>Logical values, native byte order, C-order — what NumSharp must hold after loading.</summary>
        public byte[] NsBytes { get; private init; }

        /// <summary>Whether NumSharp's writer must reproduce <see cref="Bytes"/> exactly.</summary>
        public bool WriteExact { get; private init; }

        /// <summary>Text the load error must contain, or null if the case must load successfully.</summary>
        public string LoadError { get; private init; }

        /// <summary>Which entry point the error case exercises: <c>load</c> or <c>load_npy</c>.</summary>
        public string LoadVia { get; private init; }

        /// <summary>Per-case header-size cap, or null for the default.</summary>
        public long? MaxHeaderSize { get; private init; }

        /// <summary>For npz cases: member name → expected array.</summary>
        public Dictionary<string, NpyMember> Entries { get; private init; }

        /// <summary>For npz cases: member name → expected raw bytes (non-.npy members).</summary>
        public Dictionary<string, byte[]> RawEntries { get; private init; }

        /// <summary>For the multi-array stream case: the arrays in the order they were appended.</summary>
        public NpyMember[] Sequence { get; private init; }

        public bool Compressed { get; private init; }

        public override string ToString() => Name ?? "<unnamed>";

        public static NpyCase Parse(JsonElement e, ZipArchive zip)
        {
            byte[] ReadEntry(string entry)
            {
                using Stream s = zip.GetEntry(entry)!.Open();
                using var ms = new MemoryStream();
                s.CopyTo(ms);
                return ms.ToArray();
            }

            return new NpyCase
            {
                Name = e.GetProperty("name").GetString(),
                Kind = e.GetProperty("kind").GetString(),
                Note = Str(e, "note"),
                Bytes = ReadEntry(e.GetProperty("file").GetString()),
                NsDtype = Str(e, "ns_dtype") is { } d ? Enum.Parse<NPTypeCode>(d) : null,
                Shape = Longs(e, "shape"),
                FortranOrder = Bool(e, "fortran_order") ?? false,
                Descr = Str(e, "descr"),
                Version = Bytes8(e, "version"),
                NsBytes = Hex(Str(e, "ns_bytes")),
                WriteExact = Bool(e, "write_exact") ?? false,
                LoadError = Str(e, "load_error"),
                LoadVia = Str(e, "load_via") ?? "load",
                MaxHeaderSize = Long(e, "max_header_size"),
                Compressed = Bool(e, "compressed") ?? false,
                Entries = Members(e, "entries"),
                RawEntries = RawMembers(e, "raw_entries"),
                Sequence = e.TryGetProperty("sequence", out JsonElement s) && s.ValueKind == JsonValueKind.Array
                    ? s.EnumerateArray().Select(NpyMember.Parse).ToArray()
                    : null,
            };
        }

        private static string Str(JsonElement e, string name) =>
            e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

        private static bool? Bool(JsonElement e, string name) =>
            e.TryGetProperty(name, out JsonElement v) && (v.ValueKind == JsonValueKind.True || v.ValueKind == JsonValueKind.False)
                ? v.GetBoolean() : null;

        private static long? Long(JsonElement e, string name) =>
            e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt64() : null;

        private static long[] Longs(JsonElement e, string name) =>
            e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Array
                ? v.EnumerateArray().Select(x => x.GetInt64()).ToArray() : null;

        private static byte[] Bytes8(JsonElement e, string name) =>
            e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Array
                ? v.EnumerateArray().Select(x => (byte)x.GetInt32()).ToArray() : null;

        private static Dictionary<string, NpyMember> Members(JsonElement e, string name)
        {
            if (!e.TryGetProperty(name, out JsonElement v) || v.ValueKind != JsonValueKind.Object)
                return null;
            var d = new Dictionary<string, NpyMember>(StringComparer.Ordinal);
            foreach (JsonProperty p in v.EnumerateObject())
                d[p.Name] = NpyMember.Parse(p.Value);
            return d;
        }

        private static Dictionary<string, byte[]> RawMembers(JsonElement e, string name)
        {
            if (!e.TryGetProperty(name, out JsonElement v) || v.ValueKind != JsonValueKind.Object)
                return null;
            var d = new Dictionary<string, byte[]>(StringComparer.Ordinal);
            foreach (JsonProperty p in v.EnumerateObject())
                d[p.Name] = Hex(p.Value.GetString());
            return d;
        }

        internal static byte[] Hex(string hex)
        {
            if (hex == null) return null;
            return System.Convert.FromHexString(hex);
        }
    }

    /// <summary>One array inside an npz archive or an appended stream.</summary>
    internal sealed class NpyMember
    {
        public NPTypeCode NsDtype { get; private init; }
        public long[] Shape { get; private init; }
        public byte[] NsBytes { get; private init; }

        public static NpyMember Parse(JsonElement e) => new()
        {
            NsDtype = Enum.Parse<NPTypeCode>(e.GetProperty("ns_dtype").GetString()!),
            Shape = e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray(),
            NsBytes = NpyCase.Hex(e.GetProperty("ns_bytes").GetString()),
        };
    }
}
