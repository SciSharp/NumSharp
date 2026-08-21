using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     Builds the array each <c>flags_oracle.jsonl</c> recipe token names — a 1:1 twin of
    ///     <c>test/oracle/gen_flags_oracle.py::build</c>. Keep the two in lockstep: every branch here
    ///     mirrors one Python line, so a corpus case replays against the IDENTICAL construction.
    /// </summary>
    internal static class FlagsOracleRecipes
    {
        public static NPTypeCode Dtype(string token) => token switch
        {
            "bool" => NPTypeCode.Boolean,
            "int8" => NPTypeCode.SByte,
            "uint8" => NPTypeCode.Byte,
            "int16" => NPTypeCode.Int16,
            "uint16" => NPTypeCode.UInt16,
            "int32" => NPTypeCode.Int32,
            "uint32" => NPTypeCode.UInt32,
            "int64" => NPTypeCode.Int64,
            "uint64" => NPTypeCode.UInt64,
            "float16" => NPTypeCode.Half,
            "float32" => NPTypeCode.Single,
            "float64" => NPTypeCode.Double,
            "complex128" => NPTypeCode.Complex,
            _ => throw new ArgumentException($"unknown dtype token '{token}'"),
        };

        /// <summary>
        ///     Build the recipe's array. <paramref name="mmapDir"/> hosts the per-case .npy files the
        ///     memmap recipes map (a FRESH file per call — a live 'r' mapping locks the file, so cases
        ///     must not share one). Callers dispose memmap results to release the mapping.
        /// </summary>
        public static NDArray Build(string recipe, string dtype, string mmapDir)
        {
            var tc = Dtype(dtype);
            switch (recipe)
            {
                case "c1d": return np.arange(6).astype(tc);
                case "c2d_view": return np.arange(12).astype(tc).reshape(3, 4);
                case "c2d_owned": return np.zeros(new Shape(3, 4), tc);
                case "f2d": return np.asfortranarray(np.arange(12).astype(tc).reshape(3, 4));
                case "f1d": return np.asfortranarray(np.arange(6).astype(tc));
                case "c3d": return np.arange(24).astype(tc).reshape(2, 3, 4);
                case "rank5": return np.zeros(new Shape(2, 1, 3, 1, 4), tc);
                case "singleton_mid": return np.zeros(new Shape(3, 1, 4), tc);
                case "zerod": return np.array(5).astype(tc);
                case "zerod_view": return np.arange(6).astype(tc)[":1"].reshape(new Shape());
                case "onelem": return np.zeros(new Shape(1), tc);
                case "empty2d": return np.zeros(new Shape(0, 3), tc);
                case "empty_sliced": return np.zeros(new Shape(0, 3), tc)["::2, :"];
                case "t2d": return np.arange(12).astype(tc).reshape(3, 4).T;
                case "t3d": return np.transpose(np.arange(24).astype(tc).reshape(2, 3, 4), new int[] { 2, 0, 1 });
                case "strided": return np.arange(12).astype(tc).reshape(3, 4)["...,::2"];
                case "negstride": return np.arange(6).astype(tc)["::-1"];
                case "neg2d": return np.arange(12).astype(tc).reshape(3, 4)["::-1"];
                case "slice_offset": return np.arange(10).astype(tc)["2:7"];
                case "slice_step": return np.arange(10).astype(tc)["1:9:2"];
                case "slice_composed": return np.arange(24).astype(tc).reshape(4, 6)["1:3"].T;
                case "row": return np.arange(12).astype(tc).reshape(3, 4)["1"];
                case "col": return np.arange(12).astype(tc).reshape(3, 4)[":,1"];
                case "newaxis": return np.arange(6).astype(tc)[Slice.NewAxis, Slice.All];
                case "bcast_full": return np.broadcast_to(np.arange(3).astype(tc), new Shape(4, 3));
                case "bcast_same": return np.broadcast_to(np.arange(3).astype(tc), new Shape(3));
                case "bcast_scalar": return np.broadcast_to(np.array(5).astype(tc), new Shape(2, 3));
                case "bcast_partial": return np.broadcast_to(np.arange(6).astype(tc).reshape(1, 6), new Shape(4, 6));
                case "bcast_arrays0":
                {
                    var (l, _) = np.broadcast_arrays(np.arange(3).astype(tc), np.arange(3).astype(tc).reshape(3, 1));
                    return l;
                }
                case "fancy": return np.arange(12).astype(tc).reshape(3, 4)[new int[] { 0, 2 }];
                case "fancy1d": return np.arange(6).astype(tc)[new int[] { 0, 2, 4 }];
                case "bmask":
                {
                    var a = np.arange(12).astype(tc).reshape(3, 4);
                    var mask = np.arange(12).reshape(3, 4) % 2 == 0;
                    return a[mask];
                }
                case "reshape_view": return np.arange(12).astype(tc).reshape(3, 4).reshape(12);
                case "reshape_copy": return np.arange(12).astype(tc).reshape(3, 4).T.reshape(12);
                case "ravel_c": return np.arange(12).astype(tc).reshape(3, 4).ravel();
                case "ravel_t": return np.arange(12).astype(tc).reshape(3, 4).T.ravel();
                case "view_same": return np.arange(6).view(np.float64);
                case "view_diff": return np.arange(6).view(np.int32);
                case "diag2d": return np.diag(np.arange(12).astype(tc).reshape(3, 4));
                case "diagonal_m": return np.arange(12).astype(tc).reshape(3, 4).diagonal();
                case "imag_real": return np.imag(np.arange(6.0));
                case "real_complex": return np.real(np.arange(4).astype(NPTypeCode.Complex));
                case "astype": return np.arange(6).astype(np.int32);
                case "copy_c": return np.arange(12).astype(tc).reshape(3, 4).copy();
                case "copy_f": return np.arange(12).astype(tc).reshape(3, 4).copy('F');
                case "eye3": return np.eye(3);
                case "frombuffer_ro":
                {
                    var src = np.arange(4).astype(np.uint8);
                    src.setflags(write: false);
                    return np.frombuffer(src.data, typeof(byte));
                }
                case "frombuffer_rw": return np.frombuffer(np.arange(4).astype(np.uint8).data, typeof(byte));
                // --- wave 2: producers found by scanning NumPy's own flags usages ---------------
                case "squeeze_c": return np.squeeze(np.zeros(new Shape(3, 1, 4), tc));
                case "squeeze_f": return np.squeeze(np.asfortranarray(np.zeros(new Shape(3, 1, 4), tc)));
                case "swapaxes3d": return np.swapaxes(np.arange(24).astype(tc).reshape(2, 3, 4), 0, 2);
                case "expand_dims0": return np.expand_dims(np.arange(6).astype(tc), 0);
                case "atleast2d_1d": return np.atleast_2d(np.arange(6).astype(tc))[0];
                case "rot90_2d": return np.rot90(np.arange(12).astype(tc).reshape(3, 4));
                case "flip0": return np.flip(np.arange(12).astype(tc).reshape(3, 4), 0);
                case "mt2d": return np.arange(12).astype(tc).reshape(3, 4).mT;
                case "split0": return np.split(np.arange(12).astype(tc), 3)[0];
                case "unstack0": return np.unstack(np.arange(6).astype(tc).reshape(2, 3))[0];
                case "getfield_i32": return np.arange(6).getfield(np.int32, 0);
                case "pad_c": return np.pad(np.arange(12).astype(tc).reshape(3, 4), 1);
                case "pad_f": return np.pad(np.asfortranarray(np.arange(12).astype(tc).reshape(3, 4)), 1);
                case "delete_f": return np.delete(np.asfortranarray(np.arange(12).astype(tc).reshape(3, 4)), 1, axis: 0);
                case "insert_f": return np.insert(np.asfortranarray(np.arange(12).astype(tc).reshape(3, 4)), 1, 0, axis: 0);
                case "concat_cc":
                {
                    var c = np.arange(6).astype(tc).reshape(2, 3);
                    return np.concatenate(new[] { c, c });
                }
                case "concat_ff":
                {
                    var f = np.asfortranarray(np.arange(6).astype(tc).reshape(2, 3));
                    return np.concatenate(new[] { f, f });
                }
                case "zeros_like_f": return np.zeros_like(np.asfortranarray(np.arange(6).astype(tc).reshape(2, 3)));
                case "ones_like_t": return np.ones_like(np.arange(12).astype(tc).reshape(3, 4).T);
                case "empty_like_strided": return np.empty_like(np.arange(12).astype(tc).reshape(3, 4)[":, ::2"]);
                case "meshgrid_nocopy": return np.meshgrid(np.arange(3).astype(tc), np.arange(2).astype(tc), copy: false)[0];
                case "ro_view":
                {
                    var x = np.arange(6).astype(tc);
                    x.setflags(write: false);
                    return x["1:"];
                }
                case "ro_view_dtype":
                {
                    var x = np.arange(6);
                    x.setflags(write: false);
                    return x.view(np.int32);
                }
                case "mmap_r": return Mmap(mmapDir, np.arange(5.0), "r");
                case "mmap_rp": return Mmap(mmapDir, np.arange(5.0), "r+");
                case "mmap_c": return Mmap(mmapDir, np.arange(5.0), "c");
                case "mmap_r_f": return Mmap(mmapDir, np.asfortranarray(np.arange(6.0).reshape(2, 3)), "r");
                case "mmap_empty_r": return Mmap(mmapDir, np.zeros(new Shape(0, 3)), "r");
                default: throw new ArgumentException($"unknown recipe '{recipe}'");
            }
        }

        public static bool IsMmap(string recipe) => recipe.StartsWith("mmap", StringComparison.Ordinal);

        /// <summary>
        ///     Identity-vs-copy consumer cases: builds (source, result) for the asarray family and
        ///     reports whether the result SHARES the source's memory — NumPy's <c>np.shares_memory</c>,
        ///     detected here by a write-probe (set a source element, observe the result, restore).
        /// </summary>
        public static (NDArray result, bool shared) BuildConsumer(string recipe)
        {
            switch (recipe)
            {
                case "ascontig_c":
                {
                    var a = np.arange(12).reshape(3, 4);
                    var r = np.ascontiguousarray(a);
                    return (r, SharesMemory(a, r));
                }
                case "ascontig_f":
                {
                    var a = np.asfortranarray(np.arange(6).reshape(2, 3));
                    var r = np.ascontiguousarray(a);
                    return (r, SharesMemory(a, r));
                }
                case "ascontig_strided":
                {
                    var a = np.arange(12).reshape(3, 4)[":, ::2"];
                    var r = np.ascontiguousarray(a);
                    return (r, SharesMemory(a, r));
                }
                case "asfortran_f":
                {
                    var a = np.asfortranarray(np.arange(6).reshape(2, 3));
                    var r = np.asfortranarray(a);
                    return (r, SharesMemory(a, r));
                }
                case "asfortran_c":
                {
                    var a = np.arange(12).reshape(3, 4);
                    var r = np.asfortranarray(a);
                    return (r, SharesMemory(a, r));
                }
                case "ravel_c1d":
                {
                    var a = np.arange(6);
                    var r = np.ravel(a);
                    return (r, SharesMemory(a, r));
                }
                default: throw new ArgumentException($"unknown consumer recipe '{recipe}'");
            }
        }

        // Write-probe shares_memory for the int64 consumer sources: flip the source's first logical
        // element, observe the result's, restore. (GetAtIndex/SetAtIndex address logical flat
        // positions through any view's strides; all consumer sources are writeable int64.)
        private static bool SharesMemory(NDArray src, NDArray result)
        {
            long orig = Convert.ToInt64(src.GetAtIndex(0));
            long marker = orig == long.MaxValue ? long.MinValue : long.MaxValue;
            src.SetAtIndex(marker, 0);
            bool shared = Convert.ToInt64(result.GetAtIndex(0)) == marker;
            src.SetAtIndex(orig, 0);
            return shared;
        }

        private static int _mmapSeq;

        private static NDArray Mmap(string dir, NDArray contents, string mode)
        {
            string p = Path.Combine(dir, $"flags_oracle_{System.Threading.Interlocked.Increment(ref _mmapSeq)}.npy");
            np.save(p, contents);
            // Windows can transiently lock a freshly-written file (AV scan) — retry briefly.
            for (int i = 0; ; i++)
            {
                try { return (NDArray)np.load(p, mmap_mode: mode); }
                catch (IOException) when (i < 25) { System.Threading.Thread.Sleep(100); }
            }
        }

        /// <summary>Apply the corpus op tokens, stopping at the first error (as the generator does).</summary>
        public static (string type, string message)? ApplyOps(NDArray a, IEnumerable<string> ops)
        {
            foreach (var op in ops)
            {
                try
                {
                    switch (op)
                    {
                        case "w0": a.setflags(write: false); break;
                        case "w1": a.setflags(write: true); break;
                        case "a0": a.setflags(align: false); break;
                        case "a1": a.setflags(align: true); break;
                        case "u0": a.setflags(uic: false); break;
                        case "u1": a.setflags(uic: true); break;
                        case "w0a0": a.setflags(write: false, align: false); break;
                        case "a0u1": a.setflags(align: false, uic: true); break;
                        case "a0w1": a.setflags(align: false, write: true); break;
                        default: throw new ArgumentException($"unknown op token '{op}'");
                    }
                }
                catch (ValueError e)
                {
                    return (e.GetType().Name, e.Message);
                }
            }

            return null;
        }
    }

    /// <summary>
    ///     The <c>ndarray.flags</c> / <c>ndarray.setflags</c> differential gate: replays
    ///     <c>Backends/corpus/flags_oracle.jsonl</c> — 615 cases of REAL NumPy 2.4.2 output
    ///     (52 layout/producer recipes × base record + verbatim repr + ~9 setflags transition
    ///     scenarios each incl. error messages and post-error rollback states + a 13-dtype ×
    ///     6-layout independence sweep) — and asserts NumSharp's ENTIRE flags record matches
    ///     bit-for-bit. No Python at test time; regenerate with
    ///     <c>python test/oracle/gen_flags_oracle.py</c>.
    ///
    ///     <para>Plus NumPy-free heavy sweeps: every bracket key against its dotted twin across all
    ///     52 recipes, a 14-API write-guard enforcement sweep across the setflags toggle, flags
    ///     equality-law checks across all recipe pairs, and np.require flag interplay.</para>
    /// </summary>
    [TestClass]
    public class FlagsOracleTests
    {
        private sealed record OracleCase(
            string Id, string Recipe, string Dtype, string[] Ops,
            (string type, string message)? Err,
            Dictionary<string, long> Flags, string Str, bool? Shared);

        private static List<OracleCase> _cases;
        private static string _mmapDir;

        [ClassInitialize]
        public static void LoadCorpus(TestContext _)
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Backends", "corpus", "flags_oracle.jsonl");
            if (!File.Exists(path))
                Assert.Fail($"flags oracle corpus missing: {path} — the csproj must copy Backends\\corpus\\flags_oracle.jsonl to the output (regenerate with python test/oracle/gen_flags_oracle.py).");

            _cases = new List<OracleCase>();
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;

                (string, string)? err = null;
                if (root.TryGetProperty("err", out var e) && e.ValueKind == JsonValueKind.Object)
                    err = (e.GetProperty("t").GetString(), e.GetProperty("m").GetString());

                var flags = new Dictionary<string, long>();
                foreach (var p in root.GetProperty("f").EnumerateObject())
                    flags[p.Name] = p.Value.GetInt64();

                _cases.Add(new OracleCase(
                    root.GetProperty("id").GetString(),
                    root.GetProperty("recipe").GetString(),
                    root.GetProperty("dtype").GetString(),
                    root.GetProperty("ops").EnumerateArray().Select(o => o.GetString()).ToArray(),
                    err,
                    flags,
                    root.TryGetProperty("str", out var s) ? s.GetString() : null,
                    root.TryGetProperty("shared", out var sh) ? sh.GetBoolean() : null));
            }

            _mmapDir = Path.Combine(Path.GetTempPath(), "ns_flags_oracle_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_mmapDir);
        }

        [ClassCleanup]
        public static void Cleanup()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            try { Directory.Delete(_mmapDir, recursive: true); } catch { /* best-effort */ }
        }

        // ---- the comparator ---------------------------------------------------------------

        private static Dictionary<string, long> Record(NDArray a)
        {
            var f = a.flags;
            long b(bool v) => v ? 1 : 0;
            return new Dictionary<string, long>
            {
                ["C"] = b(f.c_contiguous), ["F"] = b(f.f_contiguous), ["O"] = b(f.owndata),
                ["W"] = b(f.writeable), ["A"] = b(f.aligned), ["X"] = b(f.writebackifcopy),
                ["fnc"] = b(f.fnc), ["forc"] = b(f.forc), ["behaved"] = b(f.behaved),
                ["carray"] = b(f.carray), ["farray"] = b(f.farray), ["num"] = f.num,
            };
        }

        private static void CompareCase(OracleCase c, List<string> failures)
        {
            NDArray a = null;
            try
            {
                if (c.Shared.HasValue)
                {
                    // Identity-vs-copy consumer case: flags of the result + shares-memory verdict.
                    var (result, shared) = FlagsOracleRecipes.BuildConsumer(c.Recipe);
                    a = result;
                    if (shared != c.Shared.Value)
                        failures.Add($"{c.Id}: shares-memory — numpy={c.Shared.Value} numsharp={shared}");
                    var consumerActual = Record(a);
                    foreach (var kv in c.Flags)
                    {
                        if (consumerActual[kv.Key] != kv.Value)
                            failures.Add($"{c.Id}: flags.{kv.Key} — numpy={kv.Value} numsharp={consumerActual[kv.Key]}");
                    }
                    return;
                }

                a = FlagsOracleRecipes.Build(c.Recipe, c.Dtype, _mmapDir);

                var actualErr = FlagsOracleRecipes.ApplyOps(a, c.Ops);
                if (c.Err.HasValue != actualErr.HasValue
                    || (c.Err.HasValue && (c.Err.Value.type != actualErr.Value.type || c.Err.Value.message != actualErr.Value.message)))
                {
                    failures.Add($"{c.Id}: error mismatch — numpy={FmtErr(c.Err)} numsharp={FmtErr(actualErr)}");
                    return;
                }

                var actual = Record(a);
                foreach (var kv in c.Flags)
                {
                    if (actual[kv.Key] != kv.Value)
                        failures.Add($"{c.Id}: flags.{kv.Key} — numpy={kv.Value} numsharp={actual[kv.Key]}");
                }

                if (c.Str != null)
                {
                    var s = a.flags.ToString();
                    if (s != c.Str)
                        failures.Add($"{c.Id}: str(flags) mismatch —\nnumpy:\n{c.Str}\nnumsharp:\n{s}");
                }
            }
            catch (Exception ex)
            {
                failures.Add($"{c.Id}: replay threw {ex.GetType().Name}: {ex.Message}");
            }
            finally
            {
                if (a is not null && FlagsOracleRecipes.IsMmap(c.Recipe))
                    a.Dispose(); // release the mapping so the temp dir can be deleted
            }
        }

        private static string FmtErr((string type, string message)? e)
            => e.HasValue ? $"{e.Value.type}('{e.Value.message}')" : "none";

        private static void RunGroup(Func<OracleCase, bool> pick, int floor)
        {
            var group = _cases.Where(pick).ToList();
            Assert.IsTrue(group.Count >= floor, $"corpus group shrank below its floor: {group.Count} < {floor}");

            var failures = new List<string>();
            foreach (var c in group)
                CompareCase(c, failures);

            if (failures.Count > 0)
            {
                var sb = new StringBuilder();
                sb.AppendLine($"{failures.Count} divergences out of {group.Count} cases:");
                foreach (var f in failures.Take(60))
                    sb.AppendLine("  " + f);
                if (failures.Count > 60)
                    sb.AppendLine($"  ... and {failures.Count - 60} more");
                Assert.Fail(sb.ToString());
            }
        }

        // ---- the differential gates --------------------------------------------------------

        [TestMethod]
        public void Corpus_BaseRecords_AndVerbatimRepr_MatchNumpy()
            => RunGroup(c => c.Ops.Length == 0 && c.Dtype == "int64" && !c.Shared.HasValue && !c.Id.Contains("/dtype."), floor: 74);

        [TestMethod]
        public void Corpus_SetflagsTransitions_MatchNumpy()
            => RunGroup(c => c.Ops.Length > 0, floor: 690);

        [TestMethod]
        public void Corpus_DtypeIndependenceSweep_MatchNumpy()
            => RunGroup(c => c.Id.Contains("/dtype."), floor: 76);

        [TestMethod]
        public void Corpus_IdentityVsCopyConsumers_MatchNumpy()
            => RunGroup(c => c.Shared.HasValue, floor: 6);

        [TestMethod]
        public void Corpus_Floors_AllRecipesPresent_ManyErrorCases()
        {
            Assert.IsTrue(_cases.Count >= 850, $"corpus shrank: {_cases.Count} < 850");
            Assert.AreEqual(76, _cases.Where(c => !c.Shared.HasValue).Select(c => c.Recipe).Distinct().Count(), "recipe catalog changed size");
            Assert.IsTrue(_cases.Count(c => c.Err.HasValue) >= 150, "error-case floor");
            // every scenario token family is represented
            foreach (var op in new[] { "w0", "w1", "a0", "a1", "u1", "w0a0", "a0u1", "a0w1", "u0" })
                Assert.IsTrue(_cases.Any(c => c.Ops.Contains(op)), $"no case exercises op '{op}'");
        }

        [TestMethod]
        public void Comparator_HasTeeth()
        {
            // The comparator must DETECT a divergence — perturb a known case and require a failure.
            var c = _cases.First(x => x.Id == "c1d/base");
            var failures = new List<string>();
            var perturbed = new OracleCase(c.Id, c.Recipe, c.Dtype, c.Ops, c.Err,
                new Dictionary<string, long>(c.Flags) { ["W"] = 0, ["num"] = 263 }, c.Str, null);
            CompareCase(perturbed, failures);
            Assert.IsTrue(failures.Count >= 2, "a perturbed expectation must be detected (W + num)");

            var badErr = new OracleCase(c.Id, c.Recipe, c.Dtype, new[] { "w0" },
                ("ValueError", "phantom"), c.Flags, null, null);
            failures.Clear();
            CompareCase(badErr, failures);
            Assert.IsTrue(failures.Count >= 1, "a phantom expected error must be detected");

            var badShared = new OracleCase("consumer/ascontig_c", "ascontig_c", "int64",
                Array.Empty<string>(), null,
                _cases.First(x => x.Id == "consumer/ascontig_c").Flags, null, false);
            failures.Clear();
            CompareCase(badShared, failures);
            Assert.IsTrue(failures.Count >= 1, "a wrong shares-memory expectation must be detected");
        }

        // ---- NumPy-free heavy sweeps -------------------------------------------------------

        private static readonly (string key, Func<NDArrayFlags, bool> dotted)[] BracketKeys =
        {
            ("C", f => f.c_contiguous), ("CONTIGUOUS", f => f.contiguous), ("C_CONTIGUOUS", f => f.c_contiguous),
            ("F", f => f.f_contiguous), ("FORTRAN", f => f.fortran), ("F_CONTIGUOUS", f => f.f_contiguous),
            ("W", f => f.writeable), ("WRITEABLE", f => f.writeable),
            ("B", f => f.behaved), ("BEHAVED", f => f.behaved),
            ("O", f => f.owndata), ("OWNDATA", f => f.owndata),
            ("A", f => f.aligned), ("ALIGNED", f => f.aligned),
            ("X", f => f.writebackifcopy), ("WRITEBACKIFCOPY", f => f.writebackifcopy),
            ("CA", f => f.carray), ("CARRAY", f => f.carray),
            ("FA", f => f.farray), ("FARRAY", f => f.farray),
            ("FNC", f => f.fnc), ("FORC", f => f.forc),
        };

        [TestMethod]
        public void BracketKeys_EqualDottedProperties_AcrossEveryRecipe()
        {
            // 52 recipes × 22 keys — the subscript surface can never drift from the attributes.
            foreach (var recipe in _cases.Where(c => !c.Shared.HasValue).Select(c => c.Recipe).Distinct())
            {
                var a = FlagsOracleRecipes.Build(recipe, "int64", _mmapDir);
                try
                {
                    var f = a.flags;
                    foreach (var (key, dotted) in BracketKeys)
                        Assert.AreEqual(dotted(f), f[key], $"{recipe}: flags[\"{key}\"]");

                    ((Action)(() => { var _ = f["ZZZ"]; })).Should().Throw<KeyError>().WithMessage("*Unknown flag*");
                    ((Action)(() => f["ZZZ"] = true)).Should().Throw<KeyError>().WithMessage("*Unknown flag*");
                }
                finally
                {
                    if (FlagsOracleRecipes.IsMmap(recipe))
                        a.Dispose();
                }
            }
        }

        [TestMethod]
        public void FlagsEquality_IsExactlyNumEquality_AcrossEveryRecipePair()
        {
            // NumPy's arrayflags __eq__ compares the flags int; ours compares num. The law must hold
            // over every pair of layouts (52² comparisons), plus hashcode consistency.
            var arrays = _cases.Where(c => !c.Shared.HasValue).Select(c => c.Recipe).Distinct()
                .Select(r => (r, a: FlagsOracleRecipes.Build(r, "int64", _mmapDir))).ToList();
            try
            {
                foreach (var (r1, a1) in arrays)
                foreach (var (r2, a2) in arrays)
                {
                    bool numEqual = a1.flags.num == a2.flags.num;
                    Assert.AreEqual(numEqual, a1.flags == a2.flags, $"({r1}) == ({r2})");
                    Assert.AreEqual(!numEqual, a1.flags != a2.flags, $"({r1}) != ({r2})");
                    if (numEqual)
                        Assert.AreEqual(a1.flags.GetHashCode(), a2.flags.GetHashCode(), $"hash({r1}) vs hash({r2})");
                }
            }
            finally
            {
                foreach (var (r, a) in arrays)
                    if (FlagsOracleRecipes.IsMmap(r))
                        a.Dispose();
            }
        }

        // ---- write-guard enforcement across the setflags toggle ----------------------------

        // Most write paths refuse through NumSharpException ("… is read-only", the house mapping of
        // NumPy's ValueError); sort/partition/byteswap raise NumSharp's ValueError type with NumPy
        // 2.4.2's per-API texts verbatim ("sort array is read-only" / "partition array is read-only" /
        // "array to be byte-swapped is read-only" — probed).
        private static readonly (string name, Action<NDArray> write, Type exception)[] WriteApis =
        {
            ("indexer int set", a => a[0] = (NDArray)9L, typeof(NumSharpException)),
            ("indexer slice set", a => a["1:3"] = (NDArray)9L, typeof(NumSharpException)),
            ("fancy set", a => a[new int[] { 0, 2 }] = (NDArray)9L, typeof(NumSharpException)),
            ("mask set", a => a[a > 2] = (NDArray)9L, typeof(NumSharpException)),
            ("fill", a => a.fill(5L), typeof(NumSharpException)),
            ("copyto", a => np.copyto(a, np.zeros(new Shape(6), NPTypeCode.Int64)), typeof(NumSharpException)),
            ("ufunc add out=", a => np.add(a, a, a), typeof(NumSharpException)),
            ("ufunc negative out=", a => np.negative(np.ones(new Shape(6), NPTypeCode.Int64), a), typeof(NumSharpException)),
            ("put", a => np.put(a, (NDArray)0L, (NDArray)9L), typeof(NumSharpException)),
            ("place", a => np.place(a, np.ones(new Shape(6), NPTypeCode.Boolean), new long[] { 7 }), typeof(NumSharpException)),
            ("sort() in-place", a => a.sort(), typeof(ValueError)),
            ("partition() in-place", a => a.partition(2), typeof(ValueError)),
            ("byteswap(inplace)", a => a.byteswap(inplace: true), typeof(ValueError)),
            ("random.shuffle", a => np.random.shuffle(a), typeof(NumSharpException)),
            ("flatiter set", a => a.flatiter[0] = 5L, typeof(NumSharpException)),
        };

        [TestMethod]
        public void WriteGuards_EnforceAfterSetflagsFalse_ReleaseAfterSetflagsTrue()
        {
            foreach (var (name, write, exceptionType) in WriteApis)
            {
                var a = np.arange(6);

                a.setflags(write: false);
                try
                {
                    write(a);
                    Assert.Fail($"'{name}' must refuse a setflags(write:false) target");
                }
                catch (AssertFailedException) { throw; }
                catch (Exception e)
                {
                    Assert.AreEqual(exceptionType, e.GetType(), $"'{name}' exception type ({e.Message})");
                    StringAssert.Contains(e.Message, "read-only", $"'{name}' message: {e.Message}");
                }

                a.setflags(write: true);
                write(a); // must succeed after re-enable — throws = test fails
            }
        }

        [TestMethod]
        public void WriteGuards_EnforceOnReEnabledBroadcast_WritesAliasAfter()
        {
            // A re-enabled broadcast view accepts writes (NumPy parity), and they alias.
            var src = np.arange(3);
            var bc = np.broadcast_to(src, new Shape(4, 3));
            ((Action)(() => bc.fill(7L))).Should().Throw<NumSharpException>().WithMessage("*read-only*");
            bc.setflags(write: true);
            bc.fill(7L);
            src.GetInt64(0).Should().Be(7);
            src.GetInt64(2).Should().Be(7);
        }

        // ---- np.require reads the same flags -----------------------------------------------

        [TestMethod]
        public void Require_O_CopiesMemmap_AndViews()
        {
            string p = Path.Combine(_mmapDir, "req.npy");
            np.save(p, np.arange(5.0));
            var m = (NDArray)np.load(p, mmap_mode: "r");
            try
            {
                m.flags.owndata.Should().BeFalse("memmap memory belongs to the file mapping (NumPy: base is the mmap)");
                var owned = np.require(m, requirements: new[] { "O" });
                owned.flags.owndata.Should().BeTrue("require('O') must copy an externally-based array");
                owned.flags.writeable.Should().BeTrue();

                var view = np.arange(6).reshape(2, 3);
                np.require(view, requirements: new[] { "O" }).flags.owndata.Should().BeTrue();
            }
            finally
            {
                m.Dispose();
            }
        }

        [TestMethod]
        public void Require_W_CopiesReadOnly_PreservesWriteable()
        {
            var ro = np.broadcast_to(np.arange(3), new Shape(4, 3));
            var w = np.require(ro, requirements: new[] { "W" });
            w.flags.writeable.Should().BeTrue("require('W') must copy a read-only array");
            w.flags.owndata.Should().BeTrue();

            var already = np.arange(6);
            ReferenceEquals(np.require(already, requirements: new[] { "W" }), already).Should().BeTrue("already-satisfied W must not copy");
        }

        // ---- flag-consumer laws (NumPy semantics expressed over the corpus-pinned fields) ---

        [TestMethod]
        public void Isfortran_Law_EqualsFnc_AcrossEveryRecipe()
        {
            // np.isfortran IS a.flags.fnc (numpy/_core/numeric.py) — hold the law over every
            // corpus base record, whose fnc field is NumPy-pinned.
            foreach (var c in _cases.Where(x => x.Ops.Length == 0 && x.Dtype == "int64"
                                                && !x.Shared.HasValue && !x.Id.Contains("/dtype.")))
            {
                var a = FlagsOracleRecipes.Build(c.Recipe, c.Dtype, _mmapDir);
                try
                {
                    Assert.AreEqual(c.Flags["fnc"] == 1, np.isfortran(a), $"{c.Recipe}: np.isfortran");
                }
                finally
                {
                    if (FlagsOracleRecipes.IsMmap(c.Recipe))
                        a.Dispose();
                }
            }
        }

        [TestMethod]
        public void DataReadonly_Law_EqualsNotWriteable_AcrossRecipes()
        {
            // NumPy: memoryview(a).readonly ⇔ not a.flags.writeable (buffer.c exports the WRITEABLE
            // flag through the buffer protocol). a.data requires a C-contiguous export here, so the
            // law is asserted for every recipe whose result is C-contiguous — with a floor so the
            // sweep can never silently go vacuous.
            int asserted = 0;
            foreach (var c in _cases.Where(x => x.Ops.Length == 0 && x.Dtype == "int64"
                                                && !x.Shared.HasValue && !x.Id.Contains("/dtype.")))
            {
                var a = FlagsOracleRecipes.Build(c.Recipe, c.Dtype, _mmapDir);
                try
                {
                    if (!a.flags.c_contiguous)
                        continue;
                    Assert.AreEqual(!a.flags.writeable, a.data.@readonly, $"{c.Recipe}: data.readonly");
                    asserted++;
                }
                finally
                {
                    if (FlagsOracleRecipes.IsMmap(c.Recipe))
                        a.Dispose();
                }
            }

            Assert.IsTrue(asserted >= 30, $"data.readonly law went near-vacuous: only {asserted} recipes asserted");
        }

        [TestMethod]
        public void RealImagSetters_RefuseReadOnly()
        {
            // getset.c:807 — the .real/.imag setters run PyArray_FailUnlessWriteable("assignment
            // destination"); message identical here (probed on 2.4.2 for complex AND real targets).
            var z = np.arange(4).astype(NPTypeCode.Complex);
            z.setflags(write: false);
            ((Action)(() => z.real = (NDArray)1.0)).Should().Throw<NumSharpException>().WithMessage("assignment destination is read-only");
            ((Action)(() => z.imag = (NDArray)1.0)).Should().Throw<NumSharpException>().WithMessage("assignment destination is read-only");

            var r = np.arange(4.0);
            r.setflags(write: false);
            ((Action)(() => r.real = (NDArray)1.0)).Should().Throw<NumSharpException>().WithMessage("assignment destination is read-only");
        }

        [TestMethod]
        public void Nditer_WriteFlag_OnReadOnlyOperand_Refuses()
        {
            // nditer_constr.c:1068 — a readwrite/writeonly op-flag on a read-only operand refuses at
            // CONSTRUCTION with NumPy's verbatim message; a plain (readonly) iteration is fine.
            var q = np.arange(6);
            q.setflags(write: false);

            ((Action)(() => { using var it = np.nditer(q, op_flags: new[] { "readwrite" }); }))
                .Should().Throw<ValueError>().WithMessage("operand array with iterator write flag set is read-only");
            ((Action)(() => { using var it = np.nditer(q, op_flags: new[] { "writeonly" }); }))
                .Should().Throw<ValueError>().WithMessage("operand array with iterator write flag set is read-only");

            using (var it = np.nditer(q)) { } // readonly iteration of a read-only array is legal
        }

        [TestMethod]
        public void Resize_ChecksOwnershipNotWriteability()
        {
            // PyArray_Resize checks OWNERSHIP (and refs), never WRITEABLE (probed on 2.4.2):
            // a memmap and a view refuse with the same text; a READ-ONLY OWNER resizes fine.
            string p = Path.Combine(_mmapDir, "resize_probe.npy");
            np.save(p, np.arange(5.0));
            var m = (NDArray)np.load(p, mmap_mode: "r+");
            try
            {
                ((Action)(() => m.resize(new Shape(3)))).Should()
                    .Throw<IncorrectShapeException>().WithMessage("cannot resize this array: it does not own its data");
            }
            finally
            {
                m.Dispose();
            }

            var v = np.arange(6)["1:"];
            ((Action)(() => v.resize(new Shape(3)))).Should()
                .Throw<IncorrectShapeException>().WithMessage("cannot resize this array: it does not own its data");

            var ro = np.arange(6);
            ro.setflags(write: false);
            ro.resize(new Shape(3)); // ownership check only — NumPy allows this too
            ro.size.Should().Be(3);
        }

        // ---- documented divergences ([Misaligned]) ----------------------------------------

        [TestMethod]
        [Misaligned]
        public void SliceOfWriteableBroadcast_RevertsToReadOnly()
        {
            // DIVERGENCE: NumPy views INHERIT the writeable flag, so a slice of a setflags-writeable
            // broadcast stays writeable; NumSharp recomputes broadcast shapes as read-only on every
            // fresh view (its deliberate broadcast-write-protection invariant) and only ever CLEARS
            // on inheritance — deliberate-clear producers like np.diagonal depend on clear-only
            // propagation, so a blanket NumPy-style inherit would silently undo their contracts.
            var bc = np.broadcast_to(np.arange(3), new Shape(4, 3));
            bc.setflags(write: true);
            bc.flags.writeable.Should().BeTrue();
            bc["1:3"].flags.writeable.Should().BeFalse("NumSharp recomputes broadcast W on fresh views (NumPy would inherit True)");
        }
    }
}
