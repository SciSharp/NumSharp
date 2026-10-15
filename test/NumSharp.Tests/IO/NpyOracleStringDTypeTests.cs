using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using NumSharp.IO;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     The <c>.npy</c> / <c>.npz</c> gate for NEP 55 <c>StringDType</c> arrays (the owner's decision D3): NumPy 2.4.2 writes
    ///     such an array as a protocol-4 PICKLE behind the header <c>{'descr': '|O', ...}</c>, and NumSharp must write the
    ///     same bytes, read NumPy's files (and bare pickle files) with <c>allow_pickle=True</c>, and refuse both directions
    ///     without it in NumPy's words — against the committed cases of <c>test/oracle/gen_npy_oracle.py</c> section 12.
    /// </summary>
    /// <remarks>
    ///     A loaded array is held to NumPy's descriptor (coerce and na_object), shape, layout, C-order values and its
    ///     MEMORY-ORDER cells: a freshly unpickled array holds only short cells and arena strings, so its cells are the
    ///     deterministic arena offsets NumPy's <c>_setlist_pkl</c> leaves (setitem in C logical order, F memory included).
    /// </remarks>
    [TestClass]
    public class NpyOracleStringDTypeTests
    {
        /// <summary>NumPy's message for a pickled file opened with a memory map.</summary>
        private const string MmapError = "Array can't be memory-mapped: Python objects in dtype.";

        /// <summary>
        ///     Every <c>tpickle</c> file and every bare pickle (<c>tpickle_raw</c>, protocols 2–5) loads with
        ///     <c>allow_pickle=True</c> to NumPy's array.
        /// </summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Read_NumPysPickle()
        {
            var diffs = new List<string>();
            foreach (NpyCase c in NpyOracleCorpus.OfKind("tpickle", "tpickle_raw"))
            {
                try
                {
                    using var loaded = (NDArray)np.load(c.Bytes, allow_pickle: true);
                    CheckArray(c.Name, loaded, NpyTNA.Descriptor(c.TSpec), c.Shape, c.FortranOrder, c.TValues, c.TCells, diffs);
                }
                catch (Exception e)
                {
                    diffs.Add($"{c.Name}: threw {e.GetType().Name}: {e.Message}");
                }
            }
            Report(diffs);
        }

        /// <summary>NumSharp's <c>np.save</c> of every <c>tpickle</c> array is byte-identical to NumPy's file.</summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Write_IsNumPysPickle()
        {
            var diffs = new List<string>();
            foreach (NpyCase c in NpyOracleCorpus.OfKind("tpickle"))
            {
                try
                {
                    using var arr = Build(c);
                    byte[] got;
                    if (c.TExplicitVersion != null)
                    {
                        using var ms = new MemoryStream();
                        np.save_version(ms, arr, new NpyFormat.FormatVersion(c.TExplicitVersion[0], c.TExplicitVersion[1]));
                        got = ms.ToArray();
                    }
                    else
                    {
                        got = np.save(arr);
                    }
                    if (!got.SequenceEqual(c.Bytes))
                        diffs.Add($"{c.Name}: {FirstDifference(c.Bytes, got)}");
                }
                catch (Exception e)
                {
                    diffs.Add($"{c.Name}: threw {e.GetType().Name}: {e.Message}");
                }
            }
            Report(diffs);
        }

        /// <summary>
        ///     Without <c>allow_pickle</c> NumPy refuses both directions: <c>np.load</c> raises before it reads a byte of the
        ///     pickle, and <c>np.save</c> raises only AFTER writing the header, leaving a header-only file — NumSharp must
        ///     raise the same texts and leave the same bytes.
        /// </summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void AllowPickleFalse_IsNumPysRefusal()
        {
            var diffs = new List<string>();
            foreach (NpyCase c in NpyOracleCorpus.OfKind("tpickle"))
            {
                try
                {
                    np.load(c.Bytes);
                    diffs.Add($"{c.Name}: np.load without allow_pickle succeeded");
                }
                catch (Exception e) when (e is not AssertFailedException)
                {
                    if (!e.Message.Contains(c.NoPickleLoadError, StringComparison.Ordinal))
                        diffs.Add($"{c.Name}: load error {e.GetType().Name}: {e.Message} (NumPy: {c.NoPickleLoadError})");
                }

                using var arr = Build(c);
                using var ms = new MemoryStream();
                try
                {
                    if (c.TExplicitVersion != null)
                        np.save_version(ms, arr, new NpyFormat.FormatVersion(c.TExplicitVersion[0], c.TExplicitVersion[1]),
                                        allow_pickle: false);
                    else
                        np.save(ms, arr, allow_pickle: false);
                    diffs.Add($"{c.Name}: np.save with allow_pickle=false succeeded");
                }
                catch (ValueError e)
                {
                    if (e.Message != c.NoPickleError)
                        diffs.Add($"{c.Name}: save error '{e.Message}' (NumPy: '{c.NoPickleError}')");
                    if (!ms.ToArray().SequenceEqual(c.NoPickleBytes))
                        diffs.Add($"{c.Name}: the partial file differs: {FirstDifference(c.NoPickleBytes, ms.ToArray())}");
                }
            }
            Report(diffs);
        }

        /// <summary>A pickled file cannot be memory-mapped, with or without <c>allow_pickle</c> (NumPy's text).</summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Mmap_IsRefused()
        {
            NpyCase c = NpyOracleCorpus.OfKind("tpickle").First(x => x.Name == "t_2d_f");
            string dir = Path.Combine(Path.GetTempPath(), "numsharp-npy-tpickle-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                string path = Path.Combine(dir, "t.npy");
                File.WriteAllBytes(path, c.Bytes);
                foreach (bool allow in new[] { false, true })
                {
                    Action map = () => np.load(path, mmap_mode: "r", allow_pickle: allow);
                    map.Should().Throw<ValueError>().WithMessage(MmapError);
                }
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        /// <summary>
        ///     An archive with a <c>StringDType</c> member: NumPy's archive loads with <c>allow_pickle=True</c> (the member
        ///     to NumPy's array, the numeric member as before), the member refuses without it, and NumSharp's own archive of
        ///     the same arrays holds members byte-identical to NumPy's (the zip framing around them differs, as for every npz).
        /// </summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Npz_StringDTypeMembers()
        {
            var diffs = new List<string>();
            foreach (NpyCase c in NpyOracleCorpus.OfKind("tnpz"))
            {
                var built = new Dictionary<string, NDArray>(StringComparer.Ordinal);
                try
                {
                    using (NpzFile z = np.load_npz(c.Bytes, allow_pickle: true))
                    {
                        foreach (KeyValuePair<string, JsonElement> kv in c.TEntries)
                        {
                            NDArray member = z[kv.Key];
                            if (kv.Value.TryGetProperty("t_spec", out JsonElement spec))
                            {
                                long[] shape = kv.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
                                CheckArray($"{c.Name}['{kv.Key}']", member, NpyTNA.Descriptor(spec), shape, false,
                                    NpyTNA.DecodeValues(kv.Value.GetProperty("t_values")),
                                    System.Convert.FromHexString(kv.Value.GetProperty("t_cells").GetString()), diffs);
                                built[kv.Key] = BuildC(NpyTNA.Descriptor(spec), shape, NpyTNA.DecodeValues(kv.Value.GetProperty("t_values")));
                            }
                            else
                            {
                                var m = NpyMember.Parse(kv.Value);
                                byte[] raw = NpyOracleCorpus.RawBytes(member);
                                if (!raw.SequenceEqual(m.NsBytes))
                                    diffs.Add($"{c.Name}['{kv.Key}']: numeric member bytes differ");
                                built[kv.Key] = NpyOracleCorpus.Build(m.NsDtype, m.NsDescr, m.Shape, m.NsBytes, false);
                            }
                        }
                    }

                    using (NpzFile z = np.load_npz(c.Bytes))
                    {
                        string tKey = c.TEntries.First(kv => kv.Value.TryGetProperty("t_spec", out _)).Key;
                        try
                        {
                            _ = z[tKey];
                            diffs.Add($"{c.Name}['{tKey}']: loaded without allow_pickle");
                        }
                        catch (Exception e) when (e is not AssertFailedException)
                        {
                            if (!e.Message.Contains(c.TMemberError, StringComparison.Ordinal))
                                diffs.Add($"{c.Name}['{tKey}']: {e.GetType().Name}: {e.Message} (NumPy: {c.TMemberError})");
                        }
                    }

                    byte[] archive = c.Compressed ? np.savez_compressed(built) : np.savez(built);
                    using var mine = new ZipArchive(new MemoryStream(archive), ZipArchiveMode.Read);
                    foreach (KeyValuePair<string, byte[]> member in c.TMembers)
                    {
                        ZipArchiveEntry entry = mine.GetEntry(member.Key);
                        if (entry == null)
                        {
                            diffs.Add($"{c.Name}: NumSharp's archive has no member {member.Key}");
                            continue;
                        }
                        using var es = entry.Open();
                        using var buf = new MemoryStream();
                        es.CopyTo(buf);
                        if (!buf.ToArray().SequenceEqual(member.Value))
                            diffs.Add($"{c.Name}: member {member.Key}: {FirstDifference(member.Value, buf.ToArray())}");
                    }
                }
                catch (Exception e)
                {
                    diffs.Add($"{c.Name}: threw {e.GetType().Name}: {e.Message}");
                }
                finally
                {
                    foreach (NDArray a in built.Values)
                        a.Dispose();
                }
            }
            Report(diffs);
        }

        /// <summary>The section's cases are present: every kind, every layout the writer must model, every na_object kind.</summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Corpus_HasTheStringDTypeCases()
        {
            NpyOracleCorpus.OfKind("tpickle").Count().Should().BeGreaterThanOrEqualTo(35);
            NpyOracleCorpus.OfKind("tpickle_raw").Count().Should().BeGreaterThanOrEqualTo(5);
            NpyOracleCorpus.OfKind("tnpz").Count().Should().BeGreaterThanOrEqualTo(2);
            NpyOracleCorpus.OfKind("tpickle").Select(c => c.TBuild).Distinct().Should().Contain(new[] { "C", "F", "strided", "reversed" });
            NpyOracleCorpus.OfKind("tpickle_raw").Select(c => c.Name).Should().Contain(new[] { "t_raw_p2", "t_raw_p3", "t_raw_p4", "t_raw_p5" });
            var naKinds = NpyOracleCorpus.OfKind("tpickle")
                .Select(c => c.TSpec.TryGetProperty("na", out JsonElement na) ? na.EnumerateObject().First().Name : "absent")
                .Distinct().ToList();
            naKinds.Should().Contain(new[] { "absent", "none", "float_bits", "str", "bool", "int" });
        }

        // --------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Rebuilds a <c>tpickle</c> case's array in the layout its <c>t_build</c> names: C, F (an F copy), strided (every
        ///     second element of a base twice as long) or reversed (a <c>::-1</c> view of the reversed values).
        /// </summary>
        /// <param name="c">The case.</param>
        /// <returns>The array (the caller disposes it).</returns>
        private static NDArray Build(NpyCase c)
        {
            DType descr = NpyTNA.Descriptor(c.TSpec);
            switch (c.TBuild)
            {
                case "F":
                {
                    using var cOrder = BuildC(descr, c.Shape, c.TValues);
                    return cOrder.copy('F');
                }
                case "strided":
                {
                    var doubled = c.TValues.SelectMany(v => new object[] { v, "filler" }).ToArray();
                    // The view keeps its own reference on the base's buffer, so the base can be disposed here.
                    using var b = BuildC(descr, new[] { (long)doubled.Length }, doubled);
                    return b["::2"];
                }
                case "reversed":
                {
                    var reversed = Enumerable.Reverse(c.TValues).ToArray();
                    using var b = BuildC(descr, new[] { (long)reversed.Length }, reversed);
                    return b["::-1"];
                }
                default:
                    return BuildC(descr, c.Shape, c.TValues);
            }
        }

        /// <summary>A C-ordered array of <paramref name="descr"/> holding <paramref name="values"/> (NA markers stored as the
        ///     descriptor's na_object, by setitem).</summary>
        /// <param name="descr">The descriptor.</param>
        /// <param name="shape">The shape.</param>
        /// <param name="values">The C-order values.</param>
        /// <returns>The array.</returns>
        private static NDArray BuildC(DType descr, long[] shape, object[] values)
        {
            var arr = np.empty(new Shape(shape), descr);
            ((VStringArraySlice)arr.Storage.InternalArray).SetRange(NpyTNA.StoreValues(values, descr));
            return arr;
        }

        /// <summary>
        ///     Compares a loaded array with NumPy's: a <c>StringDType</c> descriptor equal to <paramref name="descr"/> (NumPy's
        ///     equality: coerce and na_object), the shape, the F layout, the C-order values (an NA cell holds the na_object)
        ///     and the memory-order cells.
        /// </summary>
        /// <param name="name">The case, for the report.</param>
        /// <param name="loaded">The array NumSharp loaded.</param>
        /// <param name="descr">NumPy's descriptor.</param>
        /// <param name="shape">NumPy's shape.</param>
        /// <param name="fortran">Whether NumPy's array is F-contiguous and not C.</param>
        /// <param name="values">NumPy's C-order values.</param>
        /// <param name="cells">NumPy's memory-order cells.</param>
        /// <param name="diffs">Receives the divergences.</param>
        private static void CheckArray(string name, NDArray loaded, DType descr, long[] shape, bool fortran, object[] values,
                                       byte[] cells, List<string> diffs)
        {
            if (!loaded.Storage.IsVString || !loaded.dtype.Equals(descr))
            {
                diffs.Add($"{name}: dtype {loaded.dtype.ToString(true)}, NumPy {descr.ToString(true)}");
                return;
            }
            if (!loaded.shape.SequenceEqual(shape))
            {
                diffs.Add($"{name}: shape ({string.Join(", ", loaded.shape)}), NumPy ({string.Join(", ", shape)})");
                return;
            }
            bool isF = !loaded.Shape.IsContiguous && loaded.Shape.IsFContiguous;
            if (isF != fortran)
                diffs.Add($"{name}: F-ordered {isF}, NumPy {fortran}");
            using (var c = loaded.copy('C'))
            {
                for (int i = 0; i < values.Length; i++)
                {
                    object got = c.Storage.GetAtIndex(i);
                    object want = values[i] is NpyTNA ? descr.VString.na_object : values[i];
                    bool same = want is double w && got is double g ? (double.IsNaN(w) && double.IsNaN(g)) || w == g : Equals(want, got);
                    if (!same)
                    {
                        diffs.Add($"{name}: element {i} is {got ?? "None"}, NumPy {want ?? "None"}");
                        break;
                    }
                }
            }
            // The memory-order cells: an F array's transpose is C-contiguous over the same cells, so tobytes() reads them as
            // they lie (no copy, so the arena offsets are the loaded array's own).
            NDArray memory = fortran ? loaded.T : loaded;
            try
            {
                byte[] mine = memory.tobytes();
                if (!mine.SequenceEqual(cells))
                    diffs.Add($"{name}: memory-order cells differ: {FirstDifference(cells, mine)}");
            }
            finally
            {
                if (fortran)
                    memory.Dispose();
            }
        }

        /// <summary>Where two byte sequences first differ.</summary>
        /// <param name="want">NumPy's bytes.</param>
        /// <param name="got">NumSharp's bytes.</param>
        /// <returns>A short description.</returns>
        private static string FirstDifference(byte[] want, byte[] got)
        {
            int n = System.Math.Min(want.Length, got.Length);
            for (int i = 0; i < n; i++)
                if (want[i] != got[i])
                    return $"first difference at byte {i} of {want.Length} (NumPy {want[i]:X2}, NumSharp {got[i]:X2}; NumSharp length {got.Length})";
            return $"lengths differ: NumPy {want.Length}, NumSharp {got.Length}";
        }

        /// <summary>Fails with every divergence at once.</summary>
        /// <param name="diffs">The divergences.</param>
        private static void Report(List<string> diffs)
        {
            if (diffs.Count > 0)
                Assert.Fail($"{diffs.Count} divergence(s) from NumPy:\n  " + string.Join("\n  ", diffs.Take(40)));
        }
    }
}
