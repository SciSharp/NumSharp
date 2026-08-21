using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     Differential replay of <c>test/oracle/gen_layout_parity_oracle.py</c> (NumPy 2.4.2):
    ///     the 210-case corpus pinning the modelled numpy-internal representations — reshape's
    ///     nocopy-views/view-of-copy, sort/partition/copy/astype KEEPORDER layouts, the
    ///     concatenate/stack stride vote, nonzero's shared multi-index buffer, linspace's
    ///     owndata, the read-only reduction SCALARS, and broadcast writeable-override
    ///     inheritance. Unlike the flags oracle (flags records only), every case here compares
    ///     the result's SHAPE, byte STRIDES, flags.num/OWNDATA/WRITEABLE, an exact
    ///     shares-memory verdict against the case's source, and the VALUES bit-for-bit
    ///     (base64 of NumPy's <c>tobytes(order='C')</c> vs NumSharp's <c>tobytes('C')</c>) —
    ///     plus C#-side write-through/independence probes the corpus record cannot express.
    /// </summary>
    [TestClass]
    public class LayoutParityOracleTests
    {
        // ---------------------------------------------------------------- corpus loading

        private sealed class Case
        {
            public string Id, Fam, Key, Dtype, Vals, KthVals, RDtype;
            public long[] Shape, Strides;
            public int? Num, Own, W, Shares, SameBase;
        }

        private static readonly List<Case> _cases = Load();

        private static List<Case> Load()
        {
            var path = Path.Combine(AppContext.BaseDirectory, "Backends", "corpus", "layout_parity_oracle.jsonl");
            var list = new List<Case>();
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                using var doc = JsonDocument.Parse(line);
                var r = doc.RootElement;
                long[] Arr(string name) => r.TryGetProperty(name, out var p) && p.ValueKind == JsonValueKind.Array
                    ? p.EnumerateArray().Select(x => x.GetInt64()).ToArray()
                    : null;
                int? I(string name) => r.TryGetProperty(name, out var p) && p.ValueKind == JsonValueKind.Number
                    ? p.GetInt32()
                    : (int?)null;
                string S(string name) => r.TryGetProperty(name, out var p) && p.ValueKind == JsonValueKind.String
                    ? p.GetString()
                    : null;
                list.Add(new Case
                {
                    Id = S("id"), Fam = S("fam"), Key = S("key"), Dtype = S("dtype"),
                    Shape = Arr("shape"), Strides = Arr("strides"),
                    Num = I("num"), Own = I("own"), W = I("w"), Shares = I("shares"), SameBase = I("samebase"),
                    Vals = S("vals"), KthVals = S("kthvals"), RDtype = S("rdtype"),
                });
            }

            return list;
        }

        // ---------------------------------------------------------------- twin builders
        // Keys are IDENTICAL to gen_layout_parity_oracle.py; every builder returns the case's
        // (source, result) pair so shares-memory and the write-through probes have the operand.

        private static NDArray Src(string name, NPTypeCode tc)
        {
            switch (name)
            {
                case "c2d": return np.arange(12).astype(tc).reshape(3, 4);
                case "f2d": return np.asfortranarray(np.arange(12).astype(tc).reshape(3, 4));
                case "t2d": return np.arange(12).astype(tc).reshape(3, 4).T;
                case "st": return np.arange(24).astype(tc).reshape(3, 8)[":, ::2"];
                case "neg1d": return np.arange(12).astype(tc)["::-1"];
                case "negrow": return np.arange(12).astype(tc).reshape(3, 4)["::-1"];
                case "bc": return np.broadcast_to(np.arange(3).astype(tc), new Shape(4, 3));
                case "off": return np.arange(20).astype(tc)["4:16"].reshape(3, 4);
                case "t3d": return np.transpose(np.arange(24).astype(tc).reshape(2, 3, 4), new int[] { 2, 0, 1 });
                case "hi5d": return np.arange(32).astype(tc).reshape(2, 2, 2, 2, 2);
                case "e03": return np.zeros(new Shape(0, 3), tc);
                case "one": return np.arange(1).astype(tc).reshape(1, 1);
                case "sc": return np.arange(6).astype(tc)[":1"].reshape(new Shape());
                // numpy: (arange(12).astype(dt).reshape(3,4) % 3).astype(dt). For bool the
                // astype-first collapses to truthiness (0,1,1,…); for every other dtype the
                // small-int lattice makes it equal (arange % 3).astype(dt).
                case "c2z":
                    return tc == NPTypeCode.Boolean
                        ? np.arange(12).astype(NPTypeCode.Boolean).reshape(3, 4)
                        : (np.arange(12).reshape(3, 4) % 3).astype(tc);
                case "f2z": return np.asfortranarray((np.arange(12) % 3).astype(tc).reshape(3, 4));
                case "z1d": return np.array(new long[] { 0, 1, 2, 0, 3 }).astype(tc);
                case "z3d": return (np.arange(8) % 2).astype(tc).reshape(2, 2, 2);
                case "zall": return np.zeros(new Shape(3, 4), tc);
                case "zneg": return (np.arange(12) % 3).astype(tc).reshape(3, 4)["::-1"];
                case "zbc": return np.broadcast_to(np.array(new long[] { 0, 1, 0 }).astype(tc), new Shape(4, 3));
                default: throw new ArgumentException(name);
            }
        }

        private static Shape Tgt(string token) => token switch
        {
            "12" => new Shape(12), "26" => new Shape(2, 6), "62" => new Shape(6, 2),
            "43" => new Shape(4, 3), "34" => new Shape(3, 4), "223" => new Shape(2, 2, 3),
            "232" => new Shape(2, 3, 2), "m1" => new Shape(-1), "46" => new Shape(4, 6),
            "2223" => new Shape(2, 2, 2, 3), "48" => new Shape(4, 8), "32" => new Shape(32),
            "0" => new Shape(0), "30" => new Shape(3, 0), "scalar" => new Shape(),
            "1" => new Shape(1), "11" => new Shape(1, 1), "111" => new Shape(1, 1, 1),
            "24" => new Shape(24),
            _ => throw new ArgumentException(token),
        };

        private static NDArray BCW()
        {
            var b = np.broadcast_to(np.arange(3).astype(NPTypeCode.Int64), new Shape(4, 3));
            b.setflags(write: true);
            return b;
        }

        private static (NDArray src, NDArray res) Build(Case c)
        {
            var i64 = NPTypeCode.Int64;
            switch (c.Fam)
            {
                case "reshape":
                {
                    // key = {src}_{tgt}_{order}
                    int last = c.Key.LastIndexOf('_');
                    int mid = c.Key.LastIndexOf('_', last - 1);
                    var src = Src(c.Key.Substring(0, mid), FlagsOracleRecipes.Dtype(c.Dtype));
                    var tgt = Tgt(c.Key.Substring(mid + 1, last - mid - 1));
                    char order = c.Key[last + 1];
                    return (src, src.reshape(tgt, order));
                }
                case "sort":
                {
                    if (c.Key == "nan_f")
                    {
                        // Values are corpus-excluded for this cell (NumPy canonicalizes sorted
                        // NaN to positive-quiet 0x7ff8…, NumSharp's radix to .NET's negative
                        // 0xfff8… — the documented set-ops-era divergence). The replay pins the
                        // K-layout; NaN-last ordering is asserted positionally below.
                        var nan = np.asfortranarray(np.array(new double[,]
                        {
                            { 3.0, double.NaN, 1.0, 2.0 }, { 3.0, double.NaN, 1.0, 2.0 }, { 3.0, double.NaN, 1.0, 2.0 },
                        }));
                        var sorted = np.sort(nan, axis: -1);
                        for (int row = 0; row < 3; row++)
                        {
                            Assert.IsTrue(double.IsNaN(sorted.GetDouble(row, 3)), "NaN must sort last");
                            Assert.AreEqual(1.0, sorted.GetDouble(row, 0));
                            Assert.AreEqual(3.0, sorted.GetDouble(row, 2));
                        }

                        return (nan, sorted);
                    }

                    int cut = c.Key.LastIndexOf('_');
                    var src = Src(c.Key.Substring(0, cut), FlagsOracleRecipes.Dtype(c.Dtype));
                    string ax = c.Key.Substring(cut + 1);
                    return (src, ax == "flat" ? np.sort(src, axis: null) : np.sort(src, axis: int.Parse(ax.Substring(2))));
                }
                case "argsort":
                {
                    int cut = c.Key.LastIndexOf('_');
                    var src = Src(c.Key.Substring(0, cut), i64);
                    return (src, np.argsort(src, axis: int.Parse(c.Key.Substring(cut + 3))));
                }
                case "partition":
                {
                    // key = {src}_k{kth}_ax{axis}
                    var parts = c.Key.Split('_');
                    var src = Src(parts[0], i64);
                    int kth = int.Parse(parts[1].Substring(1));
                    int axis = int.Parse(parts[2].Substring(2));
                    return (src, np.partition(src, kth, axis: axis));
                }
                case "stack":
                {
                    int cut = c.Key.LastIndexOf('_');
                    string s = c.Key.Substring(0, cut) switch
                    {
                        "f" => "f2d", "c" => "c2d", "t" => "t2d", "st" => "st", "n1" => "neg1d", "t3" => "t3d",
                        _ => throw new ArgumentException(c.Key),
                    };
                    int axis = int.Parse(c.Key.Substring(cut + 3));
                    return (null, np.stack(new[] { Src(s, i64), Src(s, i64) }, axis: axis));
                }
                case "concat":
                    return (null, c.Key switch
                    {
                        "ff_ax0" => np.concatenate(new[] { Src("f2d", i64), Src("f2d", i64) }, 0),
                        "ff_ax1" => np.concatenate(new[] { Src("f2d", i64), Src("f2d", i64) }, 1),
                        "cc_ax0" => np.concatenate(new[] { Src("c2d", i64), Src("c2d", i64) }, 0),
                        "cf_ax0" => np.concatenate(new[] { Src("c2d", i64), Src("f2d", i64) }, 0),
                        "tt_ax0" => np.concatenate(new[] { Src("t2d", i64), Src("t2d", i64) }, 0),
                        "stst_ax1" => np.concatenate(new[] { Src("st", i64), Src("st", i64) }, 1),
                        "negneg_ax0" => np.concatenate(new[] { Src("negrow", i64), Src("negrow", i64) }, 0),
                        "col_ax1" => np.concatenate(new[]
                        {
                            np.arange(3).astype(i64).reshape(3, 1), np.arange(3).astype(i64).reshape(3, 1),
                        }, 1),
                        "e_ax0" => np.concatenate(new[] { Src("e03", i64), Src("e03", i64) }, 0),
                        "three_ax0" => np.concatenate(new[] { Src("f2d", i64), Src("f2d", i64), Src("f2d", i64) }, 0),
                        "bcbc_ax0" => np.concatenate(new[] { Src("bc", i64), Src("bc", i64) }, 0),
                        "mixdt_ax0" => np.concatenate(new[] { np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4), Src("c2d", i64) }, 0),
                        "flatnone" => np.concatenate(new[] { Src("f2d", i64), Src("st", i64) }, axis: null),
                        _ => throw new ArgumentException(c.Key),
                    });
                case "nonzero":
                {
                    // key = {src}_{entry}; dtype "src_<dt>" carries the source dtype sweep
                    int cut = c.Key.LastIndexOf('_');
                    var stc = c.Dtype.StartsWith("src_")
                        ? FlagsOracleRecipes.Dtype(c.Dtype.Substring(4))
                        : i64;
                    var src = Src(c.Key.Substring(0, cut), stc);
                    var nz = np.nonzero(src);
                    return (nz[0], nz[int.Parse(c.Key.Substring(cut + 1))]);
                }
                case "where1":
                {
                    var w = np.where(Src("c2z", i64) > 0);
                    return (null, w[int.Parse(c.Key.Substring(c.Key.Length - 1))]);
                }
                case "argwhere": return (null, np.argwhere(Src("c2z", i64)));
                case "flatnonzero": return (null, np.flatnonzero(Src("c2z", i64)));
                case "copyk":
                {
                    int cut = c.Key.LastIndexOf('_');
                    var src = Src(c.Key.Substring(0, cut), i64);
                    return (src, c.Key.EndsWith("_copy") ? src.copy('K') : src.astype(NPTypeCode.Double, copy: true, order: 'K'));
                }
                case "linspace":
                    return (null, c.Key switch
                    {
                        "f64_5" => np.linspace(0.0, 1.0, 5),
                        "f64_2_3" => np.linspace(2.0, 3.0, 5),
                        "f64_11" => np.linspace(0.0, 10.0, 11),
                        "f64_num1" => np.linspace(0.0, 1.0, 1),
                        "f64_num1_noep" => np.linspace(0.0, 1.0, 1, endpoint: false),
                        "f64_num0" => np.linspace(0.0, 1.0, 0),
                        "f64_num2" => np.linspace(0.0, 1.0, 2),
                        "f64_noep" => np.linspace(0.0, 1.0, 4, endpoint: false),
                        "f32_5" => np.linspace(0.0, 1.0, 5, true, NPTypeCode.Single),
                        "i64_5" => np.linspace(0.0, 10.0, 5, true, NPTypeCode.Int64),
                        _ => throw new ArgumentException(c.Key),
                    });
                case "reduce":
                {
                    var tc = FlagsOracleRecipes.Dtype(c.Dtype == "bool" ? "int64" : c.Dtype);
                    NDArray A() => np.arange(12).astype(tc).reshape(3, 4);
                    NDArray A1() => np.arange(3).astype(tc);
                    NDArray NanA() => np.array(new double[,] { { 1.0, double.NaN, 3.0 }, { 4.0, 5.0, double.NaN } });
                    return (null, c.Key switch
                    {
                        "sum_flat" => np.sum(A()),
                        "prod_flat" => np.prod(A1()),
                        "mean_flat" => np.mean(A()),
                        "std_flat" => np.std(A()),
                        "var_flat" => np.@var(A()),
                        "amin_flat" => np.amin(A()),
                        "amax_flat" => np.amax(A()),
                        "median_flat" => np.median(A()),
                        "ptp_flat" => np.ptp(A()),
                        "trace_flat" => np.trace(A()),
                        // the flat NDArray-returning argmax surfaces through the engine
                        // (np.argmax(a) itself returns a C# long)
                        "argmax_flat" => A().TensorEngine.ReduceArgMax(A(), null),
                        "sum_ax0_1d" => np.sum(A1(), 0),
                        "argmax_ax0_1d" => np.argmax(A1(), 0),
                        "percentile50" => np.percentile(A(), 50.0),
                        "quantile50" => np.quantile(A(), 0.5),
                        "nansum_flat" => np.nansum(NanA()),
                        "nanmean_flat" => np.nanmean(NanA()),
                        "nanmax_flat" => np.nanmax(NanA()),
                        "nanmedian_flat" => np.nanmedian(NanA()),
                        "sum_keepdims" => np.sum(np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4), true),
                        "sum_keepdims_0d" => np.sum(NDArray.Scalar(5L), true),
                        "sum_out0d" => SumIntoOut(),
                        "all_flat" => np.all(np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4) > -1, (int?)null),
                        "any_flat" => np.any(np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4) > 5, (int?)null),
                        "all_ax0_1d" => np.all(np.arange(3).astype(NPTypeCode.Int64) > -1, 0),
                        "any_ax0_1d" => np.any(np.arange(3).astype(NPTypeCode.Int64) > 1, 0),
                        _ => throw new ArgumentException(c.Key),
                    });

                    static NDArray SumIntoOut()
                    {
                        var o = NDArray.Scalar(0L);
                        var a = np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4);
                        return a.TensorEngine.ReduceAdd(a, null, false, null, o);
                    }
                }
                case "bcastw":
                    return (null, c.Key switch
                    {
                        "slice" => BCW()["1:3"],
                        "T" => BCW().T,
                        "step" => BCW()[":, ::2"],
                        "squeeze" => np.squeeze(BCW()["np.newaxis"]),
                        "row" => BCW()["0"],
                        "chain" => BCW()["1:3"]["0:1"],
                        "plain_slice" => Src("bc", i64)["1:3"],
                        "plain_row" => Src("bc", i64)["0"],
                        "rebroadcast" => np.broadcast_to(BCW(), new Shape(2, 4, 3)),
                        _ => throw new ArgumentException(c.Key),
                    });
                default: throw new ArgumentException(c.Fam);
            }
        }

        // ---------------------------------------------------------------- the comparator

        private static void RunFamily(string fam, int floor)
        {
            var group = _cases.Where(c => c.Fam == fam).ToList();
            Assert.IsTrue(group.Count >= floor, $"corpus family '{fam}' shrank: {group.Count} < {floor}");

            var failures = new List<string>();
            foreach (var c in group)
            {
                try
                {
                    var (src, res) = Build(c);
                    Verify(c, src, res);
                }
                catch (Exception ex) when (ex is not AssertFailedException)
                {
                    failures.Add($"{c.Id}: {ex.GetType().Name}: {ex.Message}");
                    continue;
                }
            }

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count} case(s) threw:\n" + string.Join("\n", failures.Take(10)));
        }

        private static void Verify(Case c, NDArray src, NDArray res)
        {
            // shape (always)
            CollectionAssert.AreEqual(c.Shape, res.Shape.dimensions.ToArray(), $"{c.Id}: shape");

            long size = res.size;

            // byte strides — construction-path-dependent inside NumPy itself for size-0 (its
            // reshape fills max(dim,1) products where zeros() fills 0s), so size-0 skips.
            if (c.Strides != null && size > 0)
            {
                var bytes = res.Shape.strides.Select((s, i) => s * res.dtypesize).ToArray();
                CollectionAssert.AreEqual(c.Strides, bytes, $"{c.Id}: byte strides");
            }

            if (c.Num.HasValue)
                Assert.AreEqual(c.Num.Value, res.flags.num, $"{c.Id}: flags.num");
            if (c.Own.HasValue)
                Assert.AreEqual(c.Own.Value == 1, res.flags.owndata, $"{c.Id}: owndata");
            if (c.W.HasValue)
                Assert.AreEqual(c.W.Value == 1, res.flags.writeable, $"{c.Id}: writeable");

            // exact shares-memory verdict vs the case's source (size-0 shares nothing in NumPy)
            if (c.Shares.HasValue && src is not null && size > 0)
            {
                bool shares = NDMemOverlap.StoragesMayShareMemory(res.Storage, src.Storage);
                Assert.AreEqual(c.Shares.Value == 1, shares, $"{c.Id}: shares-memory");
            }

            // shared multi-index base (nonzero tuple entries alias ONE buffer)
            if (c.SameBase.HasValue && src is not null)
                Assert.AreEqual(c.SameBase.Value == 1,
                    ReferenceEquals(res.Storage.InternalArray, src.Storage.InternalArray),
                    $"{c.Id}: samebase");

            // values, bit-for-bit in logical C order
            if (c.Vals != null)
            {
                string expectDtype = c.RDtype ?? c.Dtype;
                if (c.Fam == "nonzero" || c.Fam == "where1" || c.Fam == "argwhere" || c.Fam == "flatnonzero")
                    expectDtype = "int64";
                Assert.AreEqual(FlagsOracleRecipes.Dtype(expectDtype), res.typecode, $"{c.Id}: result dtype");
                Assert.AreEqual(c.Vals, Convert.ToBase64String(res.tobytes('C')), $"{c.Id}: values");
            }

            // partition: layout is contractual, between-anchor arrangement is not — pin the
            // kth slice plus the two-sided invariant.
            if (c.KthVals != null)
            {
                var parts = c.Key.Split('_');
                int kth = int.Parse(parts[1].Substring(1));
                int axis = int.Parse(parts[2].Substring(2));
                var slice = np.take(res, (long)kth, axis: axis < 0 ? res.ndim + axis : axis);
                Assert.AreEqual(c.KthVals, Convert.ToBase64String(slice.tobytes('C')), $"{c.Id}: kth values");
                AssertPartitionInvariant(res, kth, axis < 0 ? res.ndim + axis : axis, c.Id);
            }

            // write-through / independence probes (int64 rows only — the plumbing is
            // dtype-independent, and int64 keeps the sentinel cast-exact)
            if (src is not null && c.Shares.HasValue && size > 0 && res.flags.writeable
                && res.typecode == NPTypeCode.Int64 && src.typecode == NPTypeCode.Int64
                && src.flags.writeable)
            {
                var before = src.tobytes('C');
                res.SetAtIndex(unchecked((long)0x5DEADBEEFL), 0);
                var after = src.tobytes('C');
                if (c.Shares.Value == 1)
                    CollectionAssert.AreNotEqual(before, after, $"{c.Id}: a shared view must write through to its source");
                else
                    CollectionAssert.AreEqual(before, after, $"{c.Id}: an owned/copy-backed result must not touch its source");
            }
        }

        private static void AssertPartitionInvariant(NDArray res, int kth, int axis, string id)
        {
            // every lane: max(left-of-kth) <= res[kth] <= min(right-of-kth)
            var moved = axis == res.ndim - 1 ? res : np.swapaxes(res, axis, res.ndim - 1);
            long lanes = moved.size / moved.shape[moved.ndim - 1];
            long n = moved.shape[moved.ndim - 1];
            var flat = moved.reshape(lanes, n);
            for (long l = 0; l < lanes; l++)
            {
                long pivot = flat.GetInt64(l, kth);
                for (long j = 0; j < n; j++)
                {
                    long v = flat.GetInt64(l, j);
                    if (j < kth)
                        Assert.IsTrue(v <= pivot, $"{id}: lane {l} left value {v} > pivot {pivot}");
                    else if (j > kth)
                        Assert.IsTrue(v >= pivot, $"{id}: lane {l} right value {v} < pivot {pivot}");
                }
            }
        }

        // ---------------------------------------------------------------- family gates

        [TestMethod]
        public void Reshape_MatchesNumpy() => RunFamily("reshape", floor: 60);

        [TestMethod]
        public void Sort_MatchesNumpy() => RunFamily("sort", floor: 20);

        [TestMethod]
        public void ArgSort_MatchesNumpy() => RunFamily("argsort", floor: 4);

        [TestMethod]
        public void Partition_MatchesNumpy() => RunFamily("partition", floor: 4);

        [TestMethod]
        public void Stack_MatchesNumpy() => RunFamily("stack", floor: 9);

        [TestMethod]
        public void Concatenate_MatchesNumpy() => RunFamily("concat", floor: 12);

        [TestMethod]
        public void NonZero_MatchesNumpy() => RunFamily("nonzero", floor: 18);

        [TestMethod]
        public void WhereOneArg_MatchesNumpy() => RunFamily("where1", floor: 2);

        [TestMethod]
        public void ArgwhereValues_MatchNumpy() => RunFamily("argwhere", floor: 1);

        [TestMethod]
        public void FlatNonzeroValues_MatchNumpy() => RunFamily("flatnonzero", floor: 1);

        [TestMethod]
        public void CopyAstypeKeepOrder_MatchesNumpy() => RunFamily("copyk", floor: 10);

        [TestMethod]
        public void Linspace_MatchesNumpy() => RunFamily("linspace", floor: 10);

        [TestMethod]
        public void ReductionScalars_MatchNumpy() => RunFamily("reduce", floor: 40);

        [TestMethod]
        public void BroadcastWriteableViews_MatchNumpy() => RunFamily("bcastw", floor: 9);

        [TestMethod]
        public void Corpus_Floor_AndEveryFamilyPresent()
        {
            Assert.IsTrue(_cases.Count >= 205, $"corpus shrank: {_cases.Count} < 205");
            var fams = _cases.Select(c => c.Fam).Distinct().ToArray();
            foreach (var f in new[]
                     {
                         "reshape", "sort", "argsort", "partition", "stack", "concat", "nonzero",
                         "where1", "argwhere", "flatnonzero", "copyk", "linspace", "reduce", "bcastw",
                     })
                Assert.IsTrue(fams.Contains(f), $"family '{f}' missing from corpus");
        }
    }
}
