using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
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

        // Seeded random fuzz over the whole index space. As of commit 7e968f5e it is **0 divergences**
        // across every measurable window (all five mapping.c-parity buckets fixed; the now-passing
        // forms are pinned independently by Indexing.CombinatorialParity, a CI [FuzzMatrix] gate).
        // It stays [OpenBugs] ONLY because the full-corpus single-process run still SEGFAULTs at a
        // flaky, pre-existing teardown OOB (handover R3) — a memory-safety bug unrelated to indexing
        // correctness. Un-mark once R3 is fixed (the full run completes without an AccessViolation).
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        [OpenBugs]
        public void Index_Random() => RunIndexCorpus("index_random_20240626.jsonl");

        // ───────── base reconstruction (mirrors gen_index_oracle.make_base) ─────────
        private static NDArray A() => np.arange(12L).reshape(3, 4);

        private static NDArray Base(string n)
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
                case "E03": return np.zeros(new Shape(0, 3), dtype: np.int64);  // empty (0,3)
                default: throw new ArgumentException("base? " + n);
            }
        }

        private static NDArray DtypeBase(string dt)
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

        private static object[] BuildIndex(JsonElement tokens) => tokens.EnumerateArray().Select(Tok).ToArray();

        private static NDArray BuildValue(JsonElement v)
        {
            if (v[0].GetString() == "scalar") return (NDArray)v[1].GetInt64();
            return np.array(v[1].EnumerateArray().Select(x => x.GetInt64()).ToArray())
                     .reshape(v[2].EnumerateArray().Select(x => (int)x.GetInt64()).ToArray());
        }

        // G15 value forms: ["scalar",n] int64 | ["fscalar",x] float64 | ["farr",flat,shape] float64
        // (mirrors gen_index_oracle.setter_val_to_np — np-typed scalars, so uint8 = -1 WRAPS).
        private static NDArray BuildTypedValue(JsonElement v)
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

        private static IEnumerable<JsonElement> LoadLines(string file)
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
                        b = b.copy();                          // setter writes an independent copy
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
