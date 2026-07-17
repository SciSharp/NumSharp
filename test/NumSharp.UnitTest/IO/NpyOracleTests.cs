using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.IO;

namespace NumSharp.UnitTest.IO
{
    /// <summary>
    ///     The .npy/.npz differential gate: replay every committed NumPy 2.4.2 case through NumSharp.
    /// </summary>
    /// <remarks>
    ///     Three independent claims per case, as recorded in the manifest:
    ///     <list type="bullet">
    ///       <item><b>read</b> — loading the file yields NumPy's dtype, shape and exact bytes.</item>
    ///       <item><b>write</b> — saving the same array reproduces NumPy's file BYTE FOR BYTE. This is the
    ///             strong claim: not "NumPy can read it" but "it is indistinguishable from NumPy's output".</item>
    ///       <item><b>error</b> — an unsupported or malformed file fails with NumPy's verbatim message.</item>
    ///     </list>
    ///     Each test reports every divergence at once rather than stopping at the first, so a regression
    ///     shows its full blast radius.
    /// </remarks>
    [TestClass]
    public class NpyOracleTests
    {
        /// <summary>Every .npy case loads to NumPy's dtype, shape and bytes.</summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Read_AllCases()
        {
            var f = new Failures("read");

            foreach (NpyCase c in NpyOracleCorpus.OfKind("npy").Where(c => c.LoadError == null))
            {
                try
                {
                    var loaded = (NDArray)np.load(c.Bytes);

                    if (loaded.typecode != c.NsDtype)
                        f.Add(c, $"dtype: expected {c.NsDtype}, got {loaded.typecode}");
                    else if (!loaded.shape.SequenceEqual(c.Shape))
                        f.Add(c, $"shape: expected ({string.Join(", ", c.Shape)}), got ({string.Join(", ", loaded.shape)})");
                    else
                    {
                        byte[] got = NpyOracleCorpus.RawBytes(loaded);
                        if (!got.SequenceEqual(c.NsBytes))
                            f.Add(c, "data: " + Diff(c.NsBytes, got));
                    }
                }
                catch (Exception e)
                {
                    f.Add(c, $"threw {e.GetType().Name}: {First(e.Message)}");
                }
            }

            f.Assert();
        }

        /// <summary>
        ///     Saving the array produces the exact bytes NumPy produced — header, padding, alignment and
        ///     data. Covers every supported dtype, C and Fortran order, all three format versions and the
        ///     0-d / empty / strided edges.
        /// </summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Write_IsByteIdenticalToNumPy()
        {
            var f = new Failures("write");

            foreach (NpyCase c in NpyOracleCorpus.OfKind("npy").Where(c => c.WriteExact))
            {
                try
                {
                    NDArray arr = NpyOracleCorpus.Build(c.NsDtype.Value, c.Shape, c.NsBytes, c.FortranOrder);

                    // The corpus records which version NumPy chose. Pass it through only when it is not
                    // the default, so the common path exercises NumSharp's own version auto-selection.
                    var version = new NpyFormat.FormatVersion(c.Version[0], c.Version[1]);
                    using var ms = new MemoryStream();
                    np.save_version(ms, arr, version == NpyFormat.FormatVersion.V1_0 ? null : version);

                    byte[] got = ms.ToArray();
                    if (!got.SequenceEqual(c.Bytes))
                        f.Add(c, Diff(c.Bytes, got));
                }
                catch (Exception e)
                {
                    f.Add(c, $"threw {e.GetType().Name}: {First(e.Message)}");
                }
            }

            f.Assert();
        }

        /// <summary>
        ///     Header-only cases: writer logic that no real array can reach.
        /// </summary>
        /// <remarks>
        ///     Two branches are only observable here. The growth padding
        ///     (<c>21 - len(repr(shape[0 or -1]))</c>) is normally invisible — shrink the body and the
        ///     alignment padding grows to compensate — so a wrong growth axis passes every ordinary
        ///     test; it only shows on shapes that tip the header across a 64-byte bucket, and those need
        ///     10^17 elements to allocate. Likewise the v1.0→v2.0 auto-selection boundary sits at ~21817
        ///     dimensions. Both are driven through the header dict directly, exactly as NumPy's
        ///     <c>_write_array_header</c> was to produce the expected bytes.
        /// </remarks>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void WriteHeader_MatchesNumPy()
        {
            var f = new Failures("header");

            foreach (NpyCase c in NpyOracleCorpus.OfKind("header"))
            {
                try
                {
                    var d = new Dictionary<string, object>
                    {
                        ["descr"] = c.Descr,
                        ["fortran_order"] = c.FortranOrder,
                        ["shape"] = c.Shape,
                    };

                    using var ms = new MemoryStream();
                    NpyFormat.WriteArrayHeader(ms, d); // no version => auto-select, as NumPy did

                    byte[] got = ms.ToArray();
                    if (!got.SequenceEqual(c.Bytes))
                        f.Add(c, Diff(c.Bytes, got));
                    else if (got[6] != c.Version[0])
                        f.Add(c, $"version: expected {c.Version[0]}.{c.Version[1]}, got {got[6]}.{got[7]}");
                }
                catch (Exception e)
                {
                    f.Add(c, $"threw {e.GetType().Name}: {First(e.Message)}");
                }
            }

            f.Assert();
        }

        /// <summary>
        ///     Saving a LIVE view — strided, reversed, offset, broadcast, transposed — produces NumPy's
        ///     bytes.
        /// </summary>
        /// <remarks>
        ///     <see cref="Write_IsByteIdenticalToNumPy"/> rebuilds each array from the manifest's
        ///     canonical C-order bytes, so the array it saves is always freshly contiguous (or an F-copy).
        ///     That never exercises the branch that matters most here: deciding what to do with an array
        ///     whose memory is not laid out the way the file needs. These recipes mirror the NumPy
        ///     expressions in <c>gen_npy_oracle.py</c> exactly, and are compared against the same files.
        /// </remarks>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Write_FromLiveViews_MatchesNumPy()
        {
            var f = new Failures("live-view write");

            // case name -> the NumSharp equivalent of the NumPy expression that produced it
            var recipes = new Dictionary<string, Func<NDArray>>
            {
                // np.arange(12, dtype=np.int32)[::2]
                ["strided_step2"] = () => Int32Range(12)["::2"],
                // np.arange(6, dtype=np.float64)[::-1]
                ["strided_reversed"] = () => np.arange(6).astype(typeof(double))["::-1"],
                // np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2]
                ["strided_2d_col"] = () => Int32Range(12).reshape(3, 4)[":, ::2"],
                // np.arange(12, dtype=np.int32).reshape(3, 4)[1:, :]
                ["strided_offset_row"] = () => Int32Range(12).reshape(3, 4)["1:, :"],
                // np.broadcast_to(np.arange(3, dtype=np.int32), (4, 3))
                ["broadcast_view"] = () => np.broadcast_to(Int32Range(3), new Shape(4, 3)),
                // np.arange(12, dtype=np.int32).reshape(3, 4).T   -> F-contiguous, fortran_order: True
                ["fortran_transposed_view"] = () => Int32Range(12).reshape(3, 4).T,
                // np.asfortranarray(np.arange(5, dtype=np.int32)) -> 1-D is C-contig first: False
                ["fortran_1d_is_c"] = () => Int32Range(5).copy('F'),
                // np.asfortranarray(np.array(7, dtype=np.int32)) -> promoted to shape (1,), fortran_order: False
                ["fortran_scalar_promoted_to_1d"] = () => np.array(new[] { 7 }),
            };

            foreach (KeyValuePair<string, Func<NDArray>> r in recipes)
            {
                NpyCase c = NpyOracleCorpus.Cases.SingleOrDefault(x => x.Name == r.Key);
                if (c == null)
                {
                    f.Add(new NpyCase(), $"corpus case '{r.Key}' is missing — the recipe has no oracle to check against");
                    continue;
                }

                try
                {
                    byte[] got = np.save(r.Value());
                    if (!got.SequenceEqual(c.Bytes))
                        f.Add(c, Diff(c.Bytes, got));
                }
                catch (Exception e)
                {
                    f.Add(c, $"threw {e.GetType().Name}: {First(e.Message)}");
                }
            }

            f.Assert();
        }

        // np.arange yields Int64 (matching NumPy 2.x); the oracle recipes are int32.
        private static NDArray Int32Range(int n) => np.arange(n).astype(typeof(int));

        /// <summary>Malformed and unsupported files fail with NumPy's message.</summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Errors_MatchNumPy()
        {
            var f = new Failures("error");

            foreach (NpyCase c in NpyOracleCorpus.Cases.Where(c => c.LoadError != null))
            {
                long max = c.MaxHeaderSize ?? NpyFormat.MaxHeaderSize;
                try
                {
                    if (c.LoadVia == "load_npy")
                        np.load_npy(c.Bytes, max_header_size: max);
                    else
                        np.load(c.Bytes, max_header_size: max);

                    f.Add(c, $"expected an error containing \"{c.LoadError}\", but the load succeeded");
                }
                catch (Exception e) when (!(e is AssertFailedException))
                {
                    if (!e.Message.Contains(c.LoadError, StringComparison.Ordinal))
                        f.Add(c, $"expected message to contain\n      \"{c.LoadError}\"\n    got {e.GetType().Name}:\n      \"{First(e.Message)}\"");
                }
            }

            f.Assert();
        }

        /// <summary>Every .npy case survives a NumSharp save/load round-trip unchanged.</summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void RoundTrip_PreservesEverything()
        {
            var f = new Failures("roundtrip");

            foreach (NpyCase c in NpyOracleCorpus.OfKind("npy").Where(c => c.LoadError == null))
            {
                try
                {
                    var original = (NDArray)np.load(c.Bytes);
                    var reloaded = (NDArray)np.load(np.save(original));

                    if (reloaded.typecode != original.typecode)
                        f.Add(c, $"dtype: {original.typecode} -> {reloaded.typecode}");
                    else if (!reloaded.shape.SequenceEqual(original.shape))
                        f.Add(c, $"shape: ({string.Join(", ", original.shape)}) -> ({string.Join(", ", reloaded.shape)})");
                    else if (!NpyOracleCorpus.RawBytes(reloaded).SequenceEqual(c.NsBytes))
                        f.Add(c, "data changed across the round-trip: " + Diff(c.NsBytes, NpyOracleCorpus.RawBytes(reloaded)));
                    // A Fortran-order file must still be Fortran-order after a round-trip: the layout is
                    // part of what the format preserves.
                    else if (c.FortranOrder && !(reloaded.Shape.IsFContiguous && !reloaded.Shape.IsContiguous))
                        f.Add(c, "fortran_order was not preserved across the round-trip");
                }
                catch (Exception e)
                {
                    f.Add(c, $"threw {e.GetType().Name}: {First(e.Message)}");
                }
            }

            f.Assert();
        }

        /// <summary>NumPy's .npz archives load: lazily, by either key spelling, with .files stripped.</summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Npz_ReadsNumPyArchives()
        {
            var f = new Failures("npz");

            foreach (NpyCase c in NpyOracleCorpus.OfKind("npz"))
            {
                try
                {
                    using NpzFile npz = np.load_npz(c.Bytes);

                    // .files normally follows from the members, except where the corpus pins it
                    // explicitly — a duplicated member name is listed twice but resolves once.
                    var expected = c.FilesOverride?.ToList()
                                   ?? new List<string>((c.Entries?.Keys ?? Enumerable.Empty<string>())
                                       .Concat(c.RawEntries?.Keys ?? Enumerable.Empty<string>()));
                    if (npz.Files.Count != expected.Count || expected.Any(k => !npz.Files.Contains(k)))
                        f.Add(c, $".files: expected [{string.Join(", ", expected)}], got [{string.Join(", ", npz.Files)}]");

                    foreach (KeyValuePair<string, NpyMember> kv in c.Entries ?? new Dictionary<string, NpyMember>())
                    {
                        NDArray arr = npz[kv.Key];

                        if (arr.typecode != kv.Value.NsDtype)
                            f.Add(c, $"['{kv.Key}'] dtype: expected {kv.Value.NsDtype}, got {arr.typecode}");
                        else if (!arr.shape.SequenceEqual(kv.Value.Shape))
                            f.Add(c, $"['{kv.Key}'] shape: expected ({string.Join(", ", kv.Value.Shape)}), got ({string.Join(", ", arr.shape)})");
                        else if (!NpyOracleCorpus.RawBytes(arr).SequenceEqual(kv.Value.NsBytes))
                            f.Add(c, $"['{kv.Key}'] data: " + Diff(kv.Value.NsBytes, NpyOracleCorpus.RawBytes(arr)));

                        // NumPy accepts the member's full name too, and caches: the same instance comes
                        // back. Archives with ambiguous names opt out — there '<key>' and '<key>.npy'
                        // deliberately resolve to DIFFERENT entries, which is the whole point of them.
                        if (c.SuffixAlias && !ReferenceEquals(npz[kv.Key + ".npy"], arr))
                            f.Add(c, $"['{kv.Key}.npy'] did not resolve to the same cached array as ['{kv.Key}']");
                    }

                    foreach (KeyValuePair<string, byte[]> kv in c.RawEntries ?? new Dictionary<string, byte[]>())
                    {
                        if (!npz.GetRawBytes(kv.Key).SequenceEqual(kv.Value))
                            f.Add(c, $"['{kv.Key}'] raw bytes differ");
                        if (npz.IsArray(kv.Key))
                            f.Add(c, $"['{kv.Key}'] is not a .npy member but IsArray said it was");
                    }
                }
                catch (Exception e)
                {
                    f.Add(c, $"threw {e.GetType().Name}: {First(e.Message)}");
                }
            }

            f.Assert();
        }

        /// <summary>
        ///     Arrays appended to one stream read back one at a time, in order, and the stream then
        ///     reports EOF — NumPy's <c>np.save(f, a); np.save(f, b)</c> idiom.
        /// </summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Stream_MultipleArraysPerFile()
        {
            NpyCase c = NpyOracleCorpus.OfKind("sequence").Single();
            var f = new Failures("sequence");

            using (var ms = new MemoryStream(c.Bytes))
            {
                for (int i = 0; i < c.Sequence.Length; i++)
                {
                    NDArray arr = np.load_npy(ms);
                    NpyMember want = c.Sequence[i];

                    if (arr.typecode != want.NsDtype)
                        f.Add(c, $"[{i}] dtype: expected {want.NsDtype}, got {arr.typecode}");
                    else if (!arr.shape.SequenceEqual(want.Shape))
                        f.Add(c, $"[{i}] shape: expected ({string.Join(", ", want.Shape)}), got ({string.Join(", ", arr.shape)})");
                    else if (!NpyOracleCorpus.RawBytes(arr).SequenceEqual(want.NsBytes))
                        f.Add(c, $"[{i}] data: " + Diff(want.NsBytes, NpyOracleCorpus.RawBytes(arr)));
                }

                Assert.ThrowsException<EndOfStreamException>(() => np.load_npy(ms),
                    "reading past the last array must report EOF, as NumPy does");
            }

            // ...and NumSharp writes that same multi-array stream byte-for-byte.
            using (var ms = new MemoryStream())
            {
                foreach (NpyMember m in c.Sequence)
                    np.save(ms, NpyOracleCorpus.Build(m.NsDtype, m.Shape, m.NsBytes, false));

                if (!ms.ToArray().SequenceEqual(c.Bytes))
                    f.Add(c, "writing the three arrays to one stream: " + Diff(c.Bytes, ms.ToArray()));
            }

            f.Assert();
        }

        /// <summary>
        ///     A hostile header-length field must not be trusted with the allocator: rejecting a 28-byte
        ///     file must not reserve the gigabytes it claims.
        /// </summary>
        /// <remarks>
        ///     The error-message cases in <see cref="Errors_MatchNumPy"/> pass either way — they would go
        ///     green even while allocating 1 GB per file — so the resource behaviour needs its own test.
        ///     NumPy handles this in ~2 KB because Python's <c>fp.read(n)</c> only allocates what it
        ///     returns. NumSharp gets the same property from the seekable stream's remaining length,
        ///     which is a hard bound on what any claim can deliver, and measures ~1.2 KB. The 64 KB
        ///     bound below leaves room for allocation noise while still being thousands of times below
        ///     any claim — enough to catch a regression to <c>new byte[claimedLength]</c>.
        /// </remarks>
        [TestMethod]
        [TestCategory("NpyOracle")]
        [DoNotParallelize]
        public void HostileHeaderLength_DoesNotAllocateTheClaim()
        {
            np.load_npy(NpyOracleCorpus.Cases.Single(x => x.Name == "int32_1d").Bytes); // warm the JIT

            foreach ((string name, long claim) in new[]
            {
                ("hostile_header_len_1gb", 1_000_000_000L),
                ("hostile_header_len_4gb", 4_294_967_280L),
                ("hostile_header_len_64k_v1", 65_535L),
            })
            {
                NpyCase c = NpyOracleCorpus.Cases.Single(x => x.Name == name);

                long before = GC.GetTotalAllocatedBytes(precise: true);
                Assert.ThrowsException<FormatException>(() => np.load_npy(c.Bytes), name);
                long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;

                Assert.IsTrue(allocated < 64 * 1024,
                    $"rejecting '{name}' — a {c.Bytes.Length}-byte file claiming a {claim:N0}-byte header — " +
                    $"allocated {allocated:N0} bytes. The header length comes straight from the file and must " +
                    "never size an allocation: read incrementally, bounded by what the stream actually holds.");
            }
        }

        /// <summary>The corpus is present and non-vacuous — a silent zero-case run would prove nothing.</summary>
        [TestMethod]
        [TestCategory("NpyOracle")]
        public void Corpus_IsLoadedAndNonVacuous()
        {
            Assert.AreEqual("2.4.2", NpyOracleCorpus.NumpyVersion, "corpus should be generated by NumPy 2.4.2");
            Assert.IsTrue(NpyOracleCorpus.Cases.Length >= 200, $"expected 200+ cases, got {NpyOracleCorpus.Cases.Length}");
            Assert.IsTrue(NpyOracleCorpus.Cases.Count(c => c.WriteExact) >= 150, "expected 150+ byte-exact write cases");
            Assert.IsTrue(NpyOracleCorpus.Cases.Count(c => c.LoadError != null) >= 25, "expected 25+ error cases");
            Assert.IsTrue(NpyOracleCorpus.OfKind("npz").Count() >= 5, "expected 5+ npz cases");

            // Every NumSharp dtype that can round-trip must actually appear in the corpus, or a dtype
            // could silently lose coverage.
            var covered = NpyOracleCorpus.OfKind("npy").Where(c => c.LoadError == null)
                                         .Select(c => c.NsDtype.Value).Distinct().ToHashSet();
            var expected = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex,
            };
            CollectionAssert.AreEquivalent(expected, covered.ToArray(),
                "every NumPy-representable NumSharp dtype must be exercised by the corpus " +
                "(Decimal is excluded: it has no NumPy dtype)");
        }

        #region helpers

        private static string First(string message) => message.Split('\n')[0].TrimEnd('\r');

        // Locate the first differing byte and show a window around it — a raw dump of two 800 KB
        // buffers would bury the signal.
        private static string Diff(byte[] expected, byte[] actual)
        {
            int at = 0;
            int min = Math.Min(expected.Length, actual.Length);
            while (at < min && expected[at] == actual[at])
                at++;

            var sb = new StringBuilder();
            if (expected.Length != actual.Length)
                sb.Append($"length {actual.Length} != expected {expected.Length}; ");
            sb.Append($"first difference at byte {at}");

            int from = Math.Max(0, at - 8);
            sb.Append($"\n      expected[{from}..] {Window(expected, from)}");
            sb.Append($"\n      actual  [{from}..] {Window(actual, from)}");
            return sb.ToString();
        }

        private static string Window(byte[] b, int from)
        {
            if (from >= b.Length) return "<past end>";
            byte[] slice = b.Skip(from).Take(24).ToArray();
            return System.Convert.ToHexString(slice) + $"   {Printable(slice)}";
        }

        private static string Printable(byte[] b) =>
            new string(b.Select(x => x >= 0x20 && x < 0x7F ? (char)x : '.').ToArray());

        /// <summary>Collects every divergence so one run shows the whole picture, not just the first.</summary>
        private sealed class Failures
        {
            private readonly List<string> _items = new();
            private readonly string _what;

            public Failures(string what) => _what = what;

            public void Add(NpyCase c, string detail) =>
                _items.Add($"  {c.Name}\n    {detail}\n    (case: {c.Note})");

            public void Assert()
            {
                if (_items.Count == 0)
                    return;

                throw new AssertFailedException(
                    $"{_items.Count} {_what} divergence(s) from NumPy {NpyOracleCorpus.NumpyVersion}:\n\n" +
                    string.Join("\n\n", _items.Take(25)) +
                    (_items.Count > 25 ? $"\n\n  ... and {_items.Count - 25} more" : ""));
            }
        }

        #endregion
    }
}
