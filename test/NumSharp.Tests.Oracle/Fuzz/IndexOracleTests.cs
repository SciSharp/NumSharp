using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Differential GETTER/SETTER index gate. NumPy 2.4.2 is the oracle: the committed corpus
    ///     (test/oracle/gen_index_oracle.py) records, per case, the base recipe, a portable TOKEN
    ///     index, and NumPy's result (shape + int64 values) or the exception it raised. This test
    ///     rebuilds the SAME base + index in NumSharp, runs get/set, and bit-compares shape, values,
    ///     and which-side-raised. No Python at test time.
    ///
    ///     <para>Token encoding (mirrors the generator):
    ///     ["int",n] ["slice",start,stop,step] ["new"] ["ell"] ["arr",flat,shape] ["barr",flat,shape]
    ///     ["b0",bool] ["a0",n]; value: ["scalar",n] | ["arr",flat,shape].</para>
    ///
    ///     <para>Three corpora: <c>index_curated</c> (deterministic matrix — the CI gate, must stay
    ///     0 divergences), <c>index_dtype</c> (forms × 13 dtypes — CI gate), and
    ///     <c>index_random_&lt;seed&gt;</c> (seeded fuzz — the target for the full mapping.c port;
    ///     marked [OpenBugs] until the combinatorial advanced-index work lands).</para>
    /// </summary>
    [TestClass]
    public class IndexOracleTests
    {
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Index_Curated() => RunIndexCorpus("index_curated.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Index_Dtype() => RunDtypeCorpus("index_dtype.jsonl");

        // G15: CROSS-DTYPE setters — the assigned value's dtype differs from the base's
        // (float->int truncation, int->bool coercion, unsigned modular wrap of np-scalar values).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Index_SetterDtype() => RunSetterDtypeCorpus("index_setter_dtype.jsonl");

        // Seeded random fuzz over the whole index space — **0 divergences** across every measurable
        // window (all five mapping.c-parity buckets fixed; the forms are pinned independently by
        // Indexing.CombinatorialParity, a CI [FuzzMatrix] gate). R3 — the flaky teardown SEGFAULT that
        // kept this [OpenBugs] — is now FIXED: it was an out-of-bounds heap write in the boolean-mask
        // gather/scatter when the trailing block is EMPTY (blockSize == 0, e.g. arr[:, 6:-4][mask] or
        // E03[True] = v), which allocated a zero-length result/selection buffer and then wrote an
        // element into it, corrupting the native heap and crashing a later GC. Guarded in
        // Default.BooleanMask.cs (BooleanMask + BooleanMaskSet); the full 10 000-case single-process
        // run now completes cleanly, so this is a live CI gate again.
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Index_Random() => RunIndexCorpus("index_random_20240626.jsonl");

        // ───────── base reconstruction (mirrors gen_index_oracle.make_base) ─────────
        private static NDArray A() => np.arange(12L).reshape(3, 4);

        /// <summary>
        ///     Rebuilds a named index-oracle BASE array (mirrors <c>gen_index_oracle.make_base</c>): the
        ///     0-d scalar, the 1-D <c>V*</c> vectors, the 3x4 <c>A</c> family in every view layout the oracle
        ///     indexes (transposed, row/col-strided, negative-stride, offset, stride-0 broadcast), the 3-D
        ///     <c>B</c>/<c>BT</c> pair and the empty-dimension bases. Shared with the leak sweep
        ///     (<see cref="UndisposedIntermediateTests"/>), which replays the SAME index cases with every
        ///     result disposed — the reason it is <c>internal</c> rather than private.
        /// </summary>
        /// <remarks>View bases alias a freshly built parent (<c>A()</c>/<c>np.arange</c>) that nothing else
        /// references; a caller that must not litter wraps the call in an <see cref="NDScope"/> and yields
        /// the returned view, so the parent is released and ARC keeps the shared buffer alive.</remarks>
        /// <param name="n">The base token recorded in the case (<c>S</c>, <c>V0</c>, <c>A</c>, <c>AT</c>, …).</param>
        /// <returns>A fresh array (or view of a fresh parent) equal to NumPy's base for that token.</returns>
        /// <exception cref="ArgumentException"><paramref name="n"/> is not a known base token (a corpus/harness drift).</exception>
        internal static NDArray Base(string n)
        {
            switch (n)
            {
                case "S":   return (NDArray)5L;                                 // 0-d scalar
                case "V0":  return np.arange(0L);
                case "V1":  return np.arange(1L);
                case "V6":  return np.arange(6L);
                case "A":   return A();
                case "AT":  return A().T;                                       // (4,3)
                case "ARS": return A()["::2"];                                  // (2,4)
                case "ACS": return A()[":", "::2"];                             // (3,2)
                case "ANR": return A()["::-1"];                                 // (3,4)
                case "ANC": return A()[":", "::-1"];                            // (3,4)
                case "ASO": return A()["1:"];                                   // (2,4)
                case "ABC": return np.broadcast_to(np.arange(4L), new Shape(3, 4)); // (3,4)
                case "B":   return np.arange(24L).reshape(2, 3, 4);
                case "BT":  return np.arange(24L).reshape(2, 3, 4).T;           // (4,3,2)
                case "E03": return np.zeros(new Shape(0, 3), dtype: np.int64);  // empty (0,3) — zero LEADING
                case "E30": return np.zeros(new Shape(3, 0), dtype: np.int64);  // empty (3,0) — zero TRAILING
                case "E230": return np.zeros(new Shape(2, 3, 0), dtype: np.int64); // empty (2,3,0)
                default: throw new ArgumentException("base? " + n);
            }
        }

        /// <summary>
        ///     Rebuilds the dtype-tier base (<c>arange(12).reshape(3,4)</c> cast to <paramref name="dt"/>,
        ///     or its parity pattern for bool) that <c>index_dtype</c>/<c>index_setter_dtype</c> cases index.
        ///     Shared with the leak sweep, hence <c>internal</c>.
        /// </summary>
        /// <param name="dt">The NumPy dtype name recorded in the case.</param>
        /// <returns>A fresh 3x4 array of the requested dtype.</returns>
        /// <exception cref="ArgumentException"><paramref name="dt"/> is not one of the 12 dtype tokens the tier uses.</exception>
        internal static NDArray DtypeBase(string dt)
        {
            var b = np.arange(12L).reshape(3, 4);
            if (dt == "bool") return (b % 2L).astype(NPTypeCode.Boolean);
            return b.astype(dt switch
            {
                "uint8" => NPTypeCode.Byte, "int8" => NPTypeCode.SByte, "int16" => NPTypeCode.Int16,
                "uint16" => NPTypeCode.UInt16, "int32" => NPTypeCode.Int32, "uint32" => NPTypeCode.UInt32,
                "int64" => NPTypeCode.Int64, "uint64" => NPTypeCode.UInt64, "float16" => NPTypeCode.Half,
                "float32" => NPTypeCode.Single, "float64" => NPTypeCode.Double, "complex128" => NPTypeCode.Complex,
                _ => throw new ArgumentException("dt? " + dt)
            });
        }

        // ───────── token -> NumSharp index object (mirrors gen_index_oracle.tok_to_np) ─────────
        private static long? OptL(JsonElement e) => e.ValueKind == JsonValueKind.Null ? (long?)null : e.GetInt64();

        private static object Tok(JsonElement t)
        {
            string k = t[0].GetString();
            switch (k)
            {
                case "int":   return (int)t[1].GetInt64();
                case "slice": return new Slice(OptL(t[1]), OptL(t[2]), t[3].ValueKind == JsonValueKind.Null ? 1 : t[3].GetInt64());
                case "new":   return Slice.NewAxis;
                case "ell":   return Slice.Ellipsis;
                case "arr":   return np.array(t[1].EnumerateArray().Select(x => x.GetInt64()).ToArray())
                                        .reshape(t[2].EnumerateArray().Select(x => (int)x.GetInt64()).ToArray());
                case "barr":  return np.array(t[1].EnumerateArray().Select(x => x.GetBoolean()).ToArray())
                                        .reshape(t[2].EnumerateArray().Select(x => (int)x.GetInt64()).ToArray());
                case "b0":    return (NDArray)t[1].GetBoolean();
                case "a0":    return (NDArray)t[1].GetInt64();
                default: throw new ArgumentException("tok? " + k);
            }
        }

        /// <summary>
        ///     Converts a case's serialized index tokens into the NumSharp index objects the indexer takes
        ///     (ints, <see cref="Slice"/>s, newaxis/ellipsis, integer/boolean index ARRAYS and 0-d arrays) —
        ///     mirrors <c>gen_index_oracle.tok_to_np</c>. Shared with the leak sweep, hence <c>internal</c>.
        /// </summary>
        /// <remarks>Array tokens are freshly built <see cref="NDArray"/>s the caller owns (dispose them, or
        /// build under an <see cref="NDScope"/> and yield them).</remarks>
        /// <param name="tokens">The case's <c>tokens</c> JSON array.</param>
        /// <returns>The index objects in token order, ready for <c>base[idx]</c>.</returns>
        /// <exception cref="ArgumentException">A token kind is unknown (a corpus/harness drift).</exception>
        internal static object[] BuildIndex(JsonElement tokens) => tokens.EnumerateArray().Select(Tok).ToArray();

        /// <summary>
        ///     Builds a SETTER case's int64 right-hand side: a 0-d scalar or an int64 array of the recorded
        ///     shape. Shared with the leak sweep, hence <c>internal</c>.
        /// </summary>
        /// <param name="v">The case's <c>value</c> JSON (<c>["scalar", n]</c> or <c>["arr", flat, shape]</c>).</param>
        /// <returns>A fresh array the caller owns.</returns>
        internal static NDArray BuildValue(JsonElement v)
        {
            if (v[0].GetString() == "scalar") return (NDArray)v[1].GetInt64();
            return np.array(v[1].EnumerateArray().Select(x => x.GetInt64()).ToArray())
                     .reshape(v[2].EnumerateArray().Select(x => (int)x.GetInt64()).ToArray());
        }

        // G15 value forms: ["scalar",n] int64 | ["fscalar",x] float64 | ["farr",flat,shape] float64
        // (mirrors gen_index_oracle.setter_val_to_np — np-typed scalars, so uint8 = -1 WRAPS).
        /// <summary>
        ///     Builds a CROSS-DTYPE setter's right-hand side (the G15 value forms: int64 scalar, float64
        ///     scalar, or float64 array). Shared with the leak sweep, hence <c>internal</c>.
        /// </summary>
        /// <param name="v">The case's <c>value</c> JSON.</param>
        /// <returns>A fresh array the caller owns.</returns>
        /// <exception cref="ArgumentException">The value form is unknown (a corpus/harness drift).</exception>
        internal static NDArray BuildTypedValue(JsonElement v)
        {
            switch (v[0].GetString())
            {
                case "scalar":  return (NDArray)v[1].GetInt64();
                case "fscalar": return (NDArray)v[1].GetDouble();
                case "farr":
                    return np.array(v[1].EnumerateArray().Select(x => x.GetDouble()).ToArray())
                             .reshape(v[2].EnumerateArray().Select(x => (int)x.GetInt64()).ToArray());
                default: throw new ArgumentException("val? " + v[0].GetString());
            }
        }

        private static long[] Ravel(NDArray a) => a.size == 0 ? Array.Empty<long>() : a.ravel().ToArray<long>();

        private static long[] NpVals(JsonElement np) => np.GetProperty("vals").EnumerateArray().Select(x => x.GetInt64()).ToArray();
        private static long[] NpShape(JsonElement np) => np.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();

        /// <summary>
        ///     Streams an index-oracle corpus file as parsed JSON cases (blank lines skipped). The index tiers
        ///     use their own schema (base/tokens/value/np), not <see cref="FuzzCorpus.Case"/>, which is why
        ///     they have a dedicated loader. Shared with the leak sweep, hence <c>internal</c>.
        /// </summary>
        /// <param name="file">Corpus file name under <c>Fuzz/corpus/</c>.</param>
        /// <returns>One cloned root element per case line, lazily.</returns>
        /// <exception cref="System.IO.FileNotFoundException">The corpus file was not copied next to the test assembly.</exception>
        internal static IEnumerable<JsonElement> LoadLines(string file)
        {
            var path = FuzzCorpus.CorpusPath(file);
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                yield return JsonDocument.Parse(line).RootElement.Clone();
            }
        }

        private static void RunIndexCorpus(string file)
        {
            var cases = LoadLines(file).ToList();
            Assert.IsTrue(cases.Count > 0, $"index corpus '{file}' has no cases (was it generated/copied?)");

            var failures = new List<string>();
            int pass = 0;

            foreach (var c in cases)
            {
                string op = c.GetProperty("op").GetString();
                string baseN = c.GetProperty("base").GetString();
                var npEl = c.GetProperty("np");
                bool npOk = npEl.GetProperty("ok").GetBoolean();
                string id = c.GetProperty("id").GetString();

                bool nsOk;
                long[] nsShape = null, nsVals = null;
                string nsErr = null;
                try
                {
                    var b = Base(baseN);
                    var idx = BuildIndex(c.GetProperty("tokens"));
                    if (op == "get")
                    {
                        var r = b[idx];
                        nsShape = r.shape.Select(x => (long)x).ToArray();
                        nsVals = Ravel(r);
                    }
                    else
                    {
                        // order='K' (from the case) preserves a view base's F-contig / strided
                        // memory layout, so the SET runs into a genuinely non-contiguous
                        // destination (the SetIndicesNDNonLinear scatter path); default 'C'.
                        char ord = c.TryGetProperty("order", out var oe) ? oe.GetString()[0] : 'C';
                        b = b.copy(ord);                       // setter writes an independent copy
                        b[idx] = BuildValue(c.GetProperty("value"));
                        nsShape = b.shape.Select(x => (long)x).ToArray();
                        nsVals = Ravel(b);
                    }
                    nsOk = true;
                }
                catch (Exception e) { nsOk = false; nsErr = e.GetType().Name; }

                if (npOk && nsOk)
                {
                    var es = NpShape(npEl);
                    var ev = NpVals(npEl);
                    bool sOk = es.SequenceEqual(nsShape);
                    bool vOk = ev.SequenceEqual(nsVals);
                    if (sOk && vOk) pass++;
                    else failures.Add($"{id}: np shape=[{string.Join(",", es)}] vals=[{Trunc(ev)}] | " +
                                      $"ns shape=[{string.Join(",", nsShape)}] vals=[{Trunc(nsVals)}]");
                }
                else if (!npOk && !nsOk) pass++;
                else if (npOk && !nsOk)
                    failures.Add($"{id}: NumPy OK (shape [{string.Join(",", NpShape(npEl))}]) but NumSharp threw {nsErr}");
                else
                    failures.Add($"{id}: NumPy raised {npEl.GetProperty("err").GetString()} but NumSharp returned shape [{string.Join(",", nsShape)}]");
            }

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count}/{cases.Count} index cases diverged from NumPy (pass={pass}):\n  " +
                            string.Join("\n  ", failures.Take(60)));
        }

        private static void RunDtypeCorpus(string file)
        {
            var cases = LoadLines(file).ToList();
            Assert.IsTrue(cases.Count > 0, $"dtype corpus '{file}' has no cases");

            var failures = new List<string>();
            int pass = 0;

            foreach (var c in cases)
            {
                string dt = c.GetProperty("dtype").GetString();
                var npEl = c.GetProperty("np");
                bool npOk = npEl.GetProperty("ok").GetBoolean();
                string id = c.GetProperty("id").GetString();

                bool nsOk;
                long[] nsShape = null;
                List<long> nsVals = null;
                string nsErr = null;
                try
                {
                    var b = DtypeBase(dt);
                    var r = b[BuildIndex(c.GetProperty("tokens"))];
                    nsShape = r.shape.Select(x => (long)x).ToArray();
                    nsVals = new List<long>();
                    var f = r.ravel();
                    for (long i = 0; i < r.size; i++)
                    {
                        object v = f.GetValue(i);
                        if (dt == "complex128") { var z = (Complex)v; nsVals.Add((long)z.Real); nsVals.Add((long)z.Imaginary); }
                        else if (dt == "bool") nsVals.Add(((bool)v) ? 1 : 0);
                        else nsVals.Add((long)(v is Half h ? (double)h : Convert.ToDouble(v)));
                    }
                    nsOk = true;
                }
                catch (Exception e) { nsOk = false; nsErr = e.GetType().Name; }

                if (npOk && nsOk)
                {
                    var es = NpShape(npEl);
                    var ev = NpVals(npEl);
                    if (es.SequenceEqual(nsShape) && ev.SequenceEqual(nsVals.ToArray())) pass++;
                    else failures.Add($"{id}: np shape=[{string.Join(",", es)}] vals=[{Trunc(ev)}] | " +
                                      $"ns shape=[{string.Join(",", nsShape)}] vals=[{Trunc(nsVals.ToArray())}]");
                }
                else if (!npOk && !nsOk) pass++;
                else failures.Add($"{id}: npOk={npOk} nsOk={nsOk} {nsErr}");
            }

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count}/{cases.Count} dtype-index cases diverged (pass={pass}):\n  " +
                            string.Join("\n  ", failures.Take(60)));
        }

        // G15: replay a cross-dtype SET on the dtype base and compare the mutated array
        // (shape + values re-encoded per the base dtype) against NumPy's record.
        private static void RunSetterDtypeCorpus(string file)
        {
            var cases = LoadLines(file).ToList();
            Assert.IsTrue(cases.Count > 0, $"setter-dtype corpus '{file}' has no cases");

            var failures = new List<string>();
            int pass = 0;

            foreach (var c in cases)
            {
                string dt = c.GetProperty("dtype").GetString();
                var npEl = c.GetProperty("np");
                bool npOk = npEl.GetProperty("ok").GetBoolean();
                string id = c.GetProperty("id").GetString();

                bool nsOk;
                long[] nsShape = null;
                List<long> nsVals = null;
                string nsErr = null;
                try
                {
                    var b = DtypeBase(dt).copy();
                    b[BuildIndex(c.GetProperty("tokens"))] = BuildTypedValue(c.GetProperty("value"));
                    nsShape = b.shape.Select(x => (long)x).ToArray();
                    nsVals = new List<long>();
                    var f = b.ravel();
                    for (long i = 0; i < b.size; i++)
                    {
                        object v = f.GetValue(i);
                        if (dt == "bool") nsVals.Add(((bool)v) ? 1 : 0);
                        else nsVals.Add((long)Convert.ToDouble(v));
                    }
                    nsOk = true;
                }
                catch (Exception e) { nsOk = false; nsErr = e.GetType().Name; }

                if (npOk && nsOk)
                {
                    var es = NpShape(npEl);
                    var ev = NpVals(npEl);
                    if (es.SequenceEqual(nsShape) && ev.SequenceEqual(nsVals.ToArray())) pass++;
                    else failures.Add($"{id}: np shape=[{string.Join(",", es)}] vals=[{Trunc(ev)}] | " +
                                      $"ns shape=[{string.Join(",", nsShape)}] vals=[{Trunc(nsVals.ToArray())}]");
                }
                else if (!npOk && !nsOk) pass++;
                else failures.Add($"{id}: npOk={npOk} nsOk={nsOk} {nsErr}");
            }

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count}/{cases.Count} setter-dtype cases diverged (pass={pass}):\n  " +
                            string.Join("\n  ", failures.Take(60)));
        }

        private static string Trunc(long[] v)
        {
            if (v.Length <= 16) return string.Join(",", v);
            return string.Join(",", v.Take(16)) + ",…(+" + (v.Length - 16) + ")";
        }
    }
}
