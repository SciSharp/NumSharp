using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Math
{
    /// <summary>
    /// Boolean bitwise/logical kernel family — the byte-lane SIMD ride with NumPy's NORMALIZED
    /// semantics. NumPy aliases the bool bitwise ufuncs onto its logical loops (loops.h.src:
    /// BOOL_bitwise_and -> BOOL_logical_and, _or -> logical_or, _xor -> not_equal,
    /// invert -> logical_not), so any nonzero byte counts as True and every output byte is
    /// canonical 0/1 — a plain bytewise AND would call 0x01 &amp; 0x80 False where NumPy says True.
    /// All expected values here are probed NumPy 2.4.2 output.
    /// </summary>
    [TestClass]
    public class BoolBitwiseSimdTests
    {
        private static (bool[] a, bool[] b) RandomBools(int n, int seed)
        {
            var rnd = new Random(seed);
            var a = new bool[n];
            var b = new bool[n];
            for (int i = 0; i < n; i++)
            {
                a[i] = rnd.Next(2) == 1;
                b[i] = rnd.Next(2) == 1;
            }

            return (a, b);
        }

        /// <summary>
        /// Sizes crossing every kernel stage boundary: scalar tail only, one vector, 4x-unrolled
        /// body + remainder + tail (V256 = 32 byte lanes, unroll step 128).
        /// </summary>
        private static readonly int[] StageSizes = { 1, 31, 32, 33, 127, 128, 129, 1000, 100_003 };

        [TestMethod]
        public void BitwiseAndOrXor_AllKernelStages_MatchScalarSemantics()
        {
            foreach (int n in StageSizes)
            {
                var (ab, bb) = RandomBools(n, 7 + n);
                NDArray a = np.array(ab);
                NDArray b = np.array(bb);

                var and = np.bitwise_and(a, b);
                Assert.AreEqual(NPTypeCode.Boolean, and.typecode);
                CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x & y).ToArray(), and.ToArray<bool>(), $"and n={n}");

                var or = np.bitwise_or(a, b);
                CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x | y).ToArray(), or.ToArray<bool>(), $"or n={n}");

                var xor = np.bitwise_xor(a, b);
                CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x ^ y).ToArray(), xor.ToArray<bool>(), $"xor n={n}");

                var inv = np.invert(a);
                CollectionAssert.AreEqual(ab.Select(x => !x).ToArray(), inv.ToArray<bool>(), $"invert n={n}");
            }
        }

        [TestMethod]
        public void BoolOps_ScalarBroadcast_BothSides()
        {
            NDArray a = np.array(new[] { true, false, true, false });
            NDArray t = NDArray.Scalar(true);
            NDArray f = NDArray.Scalar(false);

            CollectionAssert.AreEqual(new[] { true, false, true, false }, np.bitwise_and(a, t).ToArray<bool>());
            CollectionAssert.AreEqual(new[] { false, false, false, false }, np.bitwise_and(a, f).ToArray<bool>());
            CollectionAssert.AreEqual(new[] { true, false, true, false }, np.bitwise_and(t, a).ToArray<bool>());
            CollectionAssert.AreEqual(new[] { true, true, true, true }, np.bitwise_or(a, t).ToArray<bool>());
            CollectionAssert.AreEqual(new[] { false, true, false, true }, np.bitwise_xor(a, t).ToArray<bool>());
        }

        [TestMethod]
        public void BoolOps_StridedReversedBroadcastViews()
        {
            var (ab, bb) = RandomBools(200, 99);
            NDArray a = np.array(ab);
            NDArray b = np.array(bb);

            var stridedAnd = np.bitwise_and(a["::2"], b["::2"]);
            CollectionAssert.AreEqual(
                Enumerable.Range(0, 100).Select(i => ab[i * 2] & bb[i * 2]).ToArray(),
                stridedAnd.ToArray<bool>());

            var reversedXor = np.bitwise_xor(a["::-1"], b["::-1"]);
            CollectionAssert.AreEqual(
                Enumerable.Range(0, 200).Select(i => ab[199 - i] ^ bb[199 - i]).ToArray(),
                reversedXor.ToArray<bool>());

            var invStrided = np.invert(a["::2"]);
            CollectionAssert.AreEqual(
                Enumerable.Range(0, 100).Select(i => !ab[i * 2]).ToArray(),
                invStrided.ToArray<bool>());

            // (1,N) & (N,1) broadcast — spot-check the outer grid.
            var big = np.bitwise_or(np.array(ab).reshape(1, 200), np.array(bb).reshape(200, 1));
            for (int i = 0; i < 200; i += 37)
                for (int j = 0; j < 200; j += 41)
                    Assert.AreEqual(bb[i] | ab[j], (bool)big[i, j]);
        }

        /// <summary>
        /// Non-canonical bool bytes (frombuffer) follow NumPy's normalized logical semantics —
        /// probed 2.4.2: nc1=[0x01,0x80,0x02,0x00], nc2=[0x80,0x01,0x03,0x00] give
        /// and=[1,1,1,0], or=[1,1,1,0], xor=[0,0,0,0], invert(nc1)=[0,0,0,1]. A bytewise AND
        /// would answer 0x01&amp;0x80=0 — the exact divergence NumPy's loops_logical guards against.
        /// SIMD body and scalar tail must agree, so the 1000-element case covers both.
        /// </summary>
        [TestMethod]
        public unsafe void NonCanonicalBools_NormalizedLikeNumPy()
        {
            NDArray nc1 = np.frombuffer(new byte[] { 0x01, 0x80, 0x02, 0x00 }, np.@bool);
            NDArray nc2 = np.frombuffer(new byte[] { 0x80, 0x01, 0x03, 0x00 }, np.@bool);

            static byte[] RawBytes(NDArray nd, int count)
            {
                var p = (byte*)nd.Unsafe.Address;
                var r = new byte[count];
                for (int i = 0; i < count; i++) r[i] = p[i];
                return r;
            }

            CollectionAssert.AreEqual(new byte[] { 1, 1, 1, 0 }, RawBytes(np.bitwise_and(nc1, nc2), 4));
            CollectionAssert.AreEqual(new byte[] { 1, 1, 1, 0 }, RawBytes(np.bitwise_or(nc1, nc2), 4));
            CollectionAssert.AreEqual(new byte[] { 0, 0, 0, 0 }, RawBytes(np.bitwise_xor(nc1, nc2), 4));
            CollectionAssert.AreEqual(new byte[] { 0, 0, 0, 1 }, RawBytes(np.invert(nc1), 4));

            // Large non-canonical input drives the SIMD body, not just the scalar tail.
            var rnd = new Random(3);
            var raw1 = new byte[1000];
            var raw2 = new byte[1000];
            rnd.NextBytes(raw1);
            rnd.NextBytes(raw2);
            NDArray n1 = np.frombuffer(raw1, np.@bool);
            NDArray n2 = np.frombuffer(raw2, np.@bool);

            CollectionAssert.AreEqual(
                raw1.Zip(raw2, (x, y) => (byte)((x != 0 && y != 0) ? 1 : 0)).ToArray(),
                RawBytes(np.bitwise_and(n1, n2), 1000));
            CollectionAssert.AreEqual(
                raw1.Zip(raw2, (x, y) => (byte)((x != 0) ^ (y != 0) ? 1 : 0)).ToArray(),
                RawBytes(np.bitwise_xor(n1, n2), 1000));
            CollectionAssert.AreEqual(
                raw1.Select(x => (byte)(x == 0 ? 1 : 0)).ToArray(),
                RawBytes(np.invert(n1), 1000));
        }

        /// <summary>
        /// bool array ⊕ C# int scalar promotes to int64 — the C# `int` literal is the weak Python
        /// int literal, and NumPy 2.4.2 promotes bool + weak int to the DEFAULT integer (int is a
        /// different kind than bool). Probed: (np.array([True]) + 2).dtype == int64, same for
        /// *, &, <<, >>. The narrower strong spellings keep their own dtype.
        /// </summary>
        [TestMethod]
        public void BoolArray_WeakIntScalar_PromotesToInt64()
        {
            NDArray a = np.array(new[] { true, false });

            Assert.AreEqual(NPTypeCode.Int64, (a + 2).typecode);
            CollectionAssert.AreEqual(new long[] { 3, 2 }, (a + 2).ToArray<long>());
            Assert.AreEqual(NPTypeCode.Int64, (a * 2).typecode);
            Assert.AreEqual(NPTypeCode.Int64, np.bitwise_and(a, np.asanyarray(2)).typecode);
            CollectionAssert.AreEqual(new long[] { 0, 0 }, np.bitwise_and(a, np.asanyarray(2)).ToArray<long>());
            Assert.AreEqual(NPTypeCode.Int64, (a + 2L).typecode);

            // Strong narrower scalar spellings are unchanged (np.int16(2) analog).
            Assert.AreEqual(NPTypeCode.Int16, (a + NDArray.Scalar((short)2)).typecode);
            // bool + bool stays bool (add -> logical or).
            NDArray b = np.array(new[] { true, true });
            Assert.AreEqual(NPTypeCode.Boolean, (a + b).typecode);
            CollectionAssert.AreEqual(new[] { true, true }, (a + b).ToArray<bool>());
            // float scalar unaffected.
            Assert.AreEqual(NPTypeCode.Double, (a + 2.5).typecode);
        }

        /// <summary>
        /// Shift dtype parity, probed NumPy 2.4.2: np.left_shift(bool_arr, 2) -> int64 (weak int),
        /// np.left_shift(bool_arr, bool_arr) -> int8 (no bool loop; smallest int loop).
        /// </summary>
        [TestMethod]
        public void BoolShift_DtypeAndValues_MatchNumPy()
        {
            NDArray a = np.array(new[] { true, false, true, true });

            var ls = np.left_shift(a, 2);
            Assert.AreEqual(NPTypeCode.Int64, ls.typecode);
            CollectionAssert.AreEqual(new long[] { 4, 0, 4, 4 }, ls.ToArray<long>());

            var rs = np.right_shift(a, 2);
            Assert.AreEqual(NPTypeCode.Int64, rs.typecode);
            CollectionAssert.AreEqual(new long[] { 0, 0, 0, 0 }, rs.ToArray<long>());

            // count == 0 is a widening copy, not the zeros shortcut.
            CollectionAssert.AreEqual(new long[] { 1, 0, 1, 1 }, np.right_shift(a, 0).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 1, 0, 1, 1 }, np.left_shift(a, 0).ToArray<long>());

            // Overflow / negative counts: left fills 0; right of a {0,1} value is also 0.
            CollectionAssert.AreEqual(new long[] { 0, 0, 0, 0 }, np.left_shift(a, 64).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 0, 0, 0, 0 }, np.right_shift(a, 64).ToArray<long>());
            CollectionAssert.AreEqual(new long[] { 0, 0, 0, 0 }, np.right_shift(a, -1).ToArray<long>());

            NDArray b = np.array(new[] { true, true, false, false });
            var lsb = np.left_shift(a, b);
            Assert.AreEqual(NPTypeCode.SByte, lsb.typecode);
            CollectionAssert.AreEqual(new sbyte[] { 2, 0, 1, 1 }, lsb.ToArray<sbyte>());

            // Strided bool value through the fused kernel.
            var big = np.array(Enumerable.Range(0, 100).Select(i => i % 2 == 0).ToArray());
            var lsS = np.left_shift(big["::2"], 3);
            Assert.AreEqual(NPTypeCode.Int64, lsS.typecode);
            CollectionAssert.AreEqual(Enumerable.Repeat(8L, 50).ToArray(), lsS.ToArray<long>());

            // Integer arrays keep their own dtype (same-type loops).
            Assert.AreEqual(NPTypeCode.Int32, np.left_shift(np.array(new[] { 1, 2 }), 2).typecode);
            Assert.AreEqual(NPTypeCode.Byte, np.left_shift(np.array(new byte[] { 1, 2 }), 1).typecode);
        }

        /// <summary>
        /// The ufunc out=/where= path over the bool loops — probed NumPy 2.4.2 (masked-off slots
        /// keep prior out contents; bool result casts same_kind into an int out; a strided out
        /// writes through its base).
        /// </summary>
        [TestMethod]
        public void BoolOps_OutWhere_MatchNumPy()
        {
            NDArray a = np.array(new[] { true, false, true, false });
            NDArray b = np.array(new[] { true, true, false, false });
            NDArray m = np.array(new[] { true, false, true, true });

            var outw = np.ones(new Shape(4), NPTypeCode.Boolean);
            var r = np.bitwise_and(a, b, @out: outw, where: m);
            Assert.AreSame(outw, r);
            CollectionAssert.AreEqual(new[] { true, true, false, false }, r.ToArray<bool>());

            var outw2 = np.ones(new Shape(4), NPTypeCode.Boolean);
            CollectionAssert.AreEqual(new[] { false, true, false, true },
                np.invert(a, @out: outw2, where: m).ToArray<bool>());

            var outw3 = np.full(new Shape(4), 7, NPTypeCode.Int32);
            var r3 = np.bitwise_xor(a, b, @out: outw3, where: m);
            Assert.AreEqual(NPTypeCode.Int32, r3.typecode);
            CollectionAssert.AreEqual(new[] { 0, 7, 1, 0 }, r3.ToArray<int>());

            var buf = np.zeros(new Shape(8), NPTypeCode.Boolean);
            np.bitwise_or(a, b, @out: buf["::2"]);
            CollectionAssert.AreEqual(new[] { true, false, true, false, true, false, false, false },
                buf.ToArray<bool>());
        }

        /// <summary>
        /// logical_and/or/xor/not and maximum/minimum on bool ride the same bitwise engine ops
        /// (ExecuteBinaryOp remaps bool max->or, min->and; np.logical_* route directly).
        /// </summary>
        [TestMethod]
        public void BoolLogicalAliases_StayConsistent()
        {
            var (ab, bb) = RandomBools(1000, 11);
            NDArray a = np.array(ab);
            NDArray b = np.array(bb);

            CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x && y).ToArray(), np.logical_and(a, b).ToArray<bool>());
            CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x || y).ToArray(), np.logical_or(a, b).ToArray<bool>());
            CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x ^ y).ToArray(), np.logical_xor(a, b).ToArray<bool>());
            CollectionAssert.AreEqual(ab.Select(x => !x).ToArray(), np.logical_not(a).ToArray<bool>());
            CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x | y).ToArray(), np.maximum(a, b).ToArray<bool>());
            CollectionAssert.AreEqual(ab.Zip(bb, (x, y) => x & y).ToArray(), np.minimum(a, b).ToArray<bool>());
        }
    }
}
