using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Math
{
    /// <summary>
    /// Boolean argmax/argmin (find-first scans), comparisons (normalized truth-value byte lanes),
    /// sum (popcount route) and bool-source casts (normalize-then-widen kernels). All expected
    /// values are probed NumPy 2.4.2 output; non-canonical cases use frombuffer raw bytes, where
    /// NumPy judges every comparison and cast by TRUTH VALUE (any nonzero byte is True).
    /// </summary>
    [TestClass]
    public class BoolReductionCompareCastTests
    {
        /// <summary>
        /// NumPy BOOL_argmax = index of the FIRST True (early exit; all-False -> 0);
        /// BOOL_argmin = memchr for the first zero byte (all-True -> 0). The block-scan
        /// sizes cross the 128-byte SIMD window at every boundary.
        /// </summary>
        [TestMethod]
        public void ArgMaxArgMin_Bool_FindFirst_AllBoundaries()
        {
            var a = np.array(new[] { false, false, true, false, true });
            Assert.AreEqual(2L, np.argmax(a));
            Assert.AreEqual(0L, np.argmin(a));

            Assert.AreEqual(0L, np.argmax(np.zeros(new Shape(1000), NPTypeCode.Boolean)));
            Assert.AreEqual(0L, np.argmin(np.ones(new Shape(1000), NPTypeCode.Boolean)));

            foreach (int n in new[] { 1, 127, 128, 129, 500, 10_000 })
            {
                foreach (int pos in new[] { 0, n - 1, n / 2 })
                {
                    var z = new bool[n];
                    z[pos] = true;
                    Assert.AreEqual((long)pos, np.argmax(np.array(z)), $"argmax n={n} pos={pos}");

                    var o = Enumerable.Repeat(true, n).ToArray();
                    o[pos] = false;
                    Assert.AreEqual((long)pos, np.argmin(np.array(o)), $"argmin n={n} pos={pos}");
                }
            }
        }

        [TestMethod]
        public void ArgMaxArgMin_Bool_NonCanonicalAndAxis()
        {
            // frombuffer 0x80 is True (probed: argmax == 1); argmin finds the first ZERO byte.
            NDArray nc = np.frombuffer(new byte[] { 0x00, 0x80, 0x01 }, np.@bool);
            Assert.AreEqual(1L, np.argmax(nc));
            NDArray nc2 = np.frombuffer(new byte[] { 0x02, 0x80, 0x00 }, np.@bool);
            Assert.AreEqual(2L, np.argmin(nc2));

            var m = np.array(new[,] { { false, true }, { true, false }, { false, false } });
            CollectionAssert.AreEqual(new long[] { 1, 0 }, np.argmax(m, 0).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 1, 0, 0 }, np.argmax(m, 1).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 0, 1, 0 }, np.argmin(m, 1).ToArray<long>());

            Assert.ThrowsException<ArgumentException>(() => np.argmax(np.array(new bool[0])));
        }

        /// <summary>
        /// Bool comparisons over byte lanes with NumPy's normalized semantics
        /// (loops_comparison b8: VOP over the ==0 masks) — sizes cross the 32-lane vector and
        /// 128-lane unroll boundaries so body, remainder and scalar tail all execute.
        /// </summary>
        [TestMethod]
        public void Comparisons_Bool_AllKernelStages()
        {
            foreach (int n in new[] { 1, 31, 32, 33, 127, 129, 1000, 10_007 })
            {
                var rnd = new Random(3 + n);
                var ab = new bool[n];
                var bb = new bool[n];
                for (int i = 0; i < n; i++)
                {
                    ab[i] = rnd.Next(2) == 1;
                    bb[i] = rnd.Next(2) == 1;
                }

                NDArray a = np.array(ab);
                NDArray b = np.array(bb);

                CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x == y).ToArray(), (a == b).ToArray<bool>(), $"eq n={n}");
                CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x != y).ToArray(), (a != b).ToArray<bool>(), $"ne n={n}");
                CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => !x && y).ToArray(), (a < b).ToArray<bool>(), $"lt n={n}");
                CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => !x || y).ToArray(), (a <= b).ToArray<bool>(), $"le n={n}");
                CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x && !y).ToArray(), (a > b).ToArray<bool>(), $"gt n={n}");
                CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x || !y).ToArray(), (a >= b).ToArray<bool>(), $"ge n={n}");
            }
        }

        /// <summary>
        /// Non-canonical bytes compare by truth value (probed: 0x01 == 0x80 is True), and the
        /// scalar-broadcast / strided routes agree with the SIMD route.
        /// </summary>
        [TestMethod]
        public void Comparisons_Bool_NonCanonical_Strided_ScalarRhs()
        {
            NDArray n1 = np.frombuffer(new byte[] { 0x01, 0x80, 0x00, 0x02 }, np.@bool);
            NDArray n2 = np.frombuffer(new byte[] { 0x80, 0x00, 0x00, 0x01 }, np.@bool);
            CollectionAssert.AreEqual(new[] { true, false, true, true }, (n1 == n2).ToArray<bool>());
            CollectionAssert.AreEqual(new[] { false, true, false, false }, (n1 > n2).ToArray<bool>());
            CollectionAssert.AreEqual(new[] { true, true, true, true }, (n1 >= n2).ToArray<bool>());

            var ab = Enumerable.Range(0, 100).Select(i => i % 3 == 0).ToArray();
            NDArray big = np.array(ab);
            CollectionAssert.AreEqual(
                Enumerable.Range(0, 50).Select(i => ab[i * 2] == ab[i * 2 + 1]).ToArray(),
                (big["::2"] == big["1::2"]).ToArray<bool>());

            NDArray a = np.array(new[] { true, false, true });
            NDArray t = NDArray.Scalar(true);
            CollectionAssert.AreEqual(new[] { true, false, true }, (a == t).ToArray<bool>());
            CollectionAssert.AreEqual(new[] { false, true, false }, (a < t).ToArray<bool>());
        }

        /// <summary>
        /// Bool sum is a POPCOUNT of truth values (probed: sum(frombuffer([0x80,0x01,0x00,0x02]))
        /// == 3, dtype int64). The count route serves integer accumulators; axis/strided/float
        /// dtype= keep their existing paths.
        /// </summary>
        [TestMethod]
        public void Sum_Bool_CountsTruthValues()
        {
            var rnd = new Random(11);
            var ab = new bool[10_001];
            long want = 0;
            for (int i = 0; i < ab.Length; i++)
            {
                ab[i] = rnd.Next(2) == 1;
                if (ab[i]) want++;
            }

            NDArray a = np.array(ab);
            var s = np.sum(a);
            Assert.AreEqual(NPTypeCode.Int64, s.typecode);
            Assert.AreEqual(want, (long)s);

            NDArray nc = np.frombuffer(new byte[] { 0x80, 0x01, 0x00, 0x02 }, np.@bool);
            Assert.AreEqual(3L, (long)np.sum(nc));
            Assert.AreEqual(0.75, (double)np.mean(nc), 1e-15);

            var s32 = np.sum(a, dtype: np.int32);
            Assert.AreEqual(NPTypeCode.Int32, s32.typecode);
            Assert.AreEqual((int)want, (int)s32);

            CollectionAssert.AreEqual(new long[] { 2, 1 },
                np.sum(np.array(new[,] { { true, false }, { true, true } }), 0).ToArray<long>());

            long wantStrided = 0;
            for (int i = 0; i < ab.Length; i += 2)
                if (ab[i]) wantStrided++;
            Assert.AreEqual(wantStrided, (long)np.sum(a["::2"]));

            Assert.AreEqual(0L, (long)np.sum(np.array(new bool[0])));
        }

        /// <summary>
        /// Bool-source casts NORMALIZE (probed: frombuffer 0x80 astype(int8) == 1) — the
        /// normalize-then-widen kernels must agree with the scalar tail on every byte value,
        /// across every destination width.
        /// </summary>
        [TestMethod]
        public void Cast_BoolSource_NormalizesAcrossAllWidths()
        {
            foreach (int n in new[] { 1, 31, 33, 100, 1000 })
            {
                var rnd = new Random(n);
                var raw = new byte[n];
                for (int i = 0; i < n; i++)
                    raw[i] = (byte)(rnd.Next(4) == 0 ? 0 : rnd.Next(1, 256));

                NDArray a = np.frombuffer(raw, np.@bool);
                var want = raw.Select(x => x != 0 ? 1 : 0).ToArray();

                CollectionAssert.AreEqual(want.Select(x => (byte)x).ToArray(), a.astype(NPTypeCode.Byte).ToArray<byte>(), $"u8 n={n}");
                CollectionAssert.AreEqual(want.Select(x => (sbyte)x).ToArray(), a.astype(NPTypeCode.SByte).ToArray<sbyte>(), $"i8 n={n}");
                CollectionAssert.AreEqual(want.Select(x => (short)x).ToArray(), a.astype(NPTypeCode.Int16).ToArray<short>(), $"i16 n={n}");
                CollectionAssert.AreEqual(want.Select(x => (ushort)x).ToArray(), a.astype(NPTypeCode.UInt16).ToArray<ushort>(), $"u16 n={n}");
                CollectionAssert.AreEqual(want, a.astype(NPTypeCode.Int32).ToArray<int>(), $"i32 n={n}");
                CollectionAssert.AreEqual(want.Select(x => (uint)x).ToArray(), a.astype(NPTypeCode.UInt32).ToArray<uint>(), $"u32 n={n}");
                CollectionAssert.AreEqual(want.Select(x => (long)x).ToArray(), a.astype(NPTypeCode.Int64).ToArray<long>(), $"i64 n={n}");
                CollectionAssert.AreEqual(want.Select(x => (ulong)x).ToArray(), a.astype(NPTypeCode.UInt64).ToArray<ulong>(), $"u64 n={n}");
                CollectionAssert.AreEqual(want.Select(x => (float)x).ToArray(), a.astype(NPTypeCode.Single).ToArray<float>(), $"f32 n={n}");
                CollectionAssert.AreEqual(want.Select(x => (double)x).ToArray(), a.astype(NPTypeCode.Double).ToArray<double>(), $"f64 n={n}");
            }

            // char destination + strided source (scalar normalize path, unchanged).
            NDArray b = np.array(new[] { true, false, true });
            CollectionAssert.AreEqual(new[] { (char)1, (char)0, (char)1 }, b.astype(NPTypeCode.Char).ToArray<char>());
            var ab2 = Enumerable.Range(0, 100).Select(i => i % 3 == 0).ToArray();
            CollectionAssert.AreEqual(
                Enumerable.Range(0, 50).Select(i => ab2[i * 2] ? 1 : 0).ToArray(),
                np.array(ab2)["::2"].astype(NPTypeCode.Int32).ToArray<int>());
        }
    }
}
