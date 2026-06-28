using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.MathSuite
{
    /// <summary>
    /// out=/where= plan slice 1 (docs/OUT_WHERE_NPYITER_FAMILIES_PLAN.md §4.2):
    /// np.bitwise_and / np.bitwise_or / np.bitwise_xor created with ufunc
    /// out=/where= from day one, plus the NumPy no-loop TypeError for
    /// non-integer inputs.
    ///
    /// Every expectation is pinned to a NumPy 2.4.2 probe (plan §2.6/§2.7,
    /// texts verbatim — including the doubled quotes in ''safe'' and the
    /// trailing space after the shape list in broadcast errors).
    ///
    /// Probed validation ORDER: ① where must be bool → ② loop resolution
    /// (no-loop TypeError) → ③ out same_kind cast → ④ broadcast/shape.
    /// A float-input bitwise call with a bad out still reports ② — pinned
    /// below in NoLoop_BeatsBadOut tests.
    /// </summary>
    [TestClass]
    public class UfuncBitwiseOutWhereTests
    {
        // i4a=[0b1100,0b1010,0b1111,0b0001]; i4b=[0b1010,0b1100,0b0101,0b0011]
        private static NDArray Ai() => np.array(new[] { 12, 10, 15, 1 }).astype(np.int32);
        private static NDArray Bi() => np.array(new[] { 10, 12, 5, 3 }).astype(np.int32);
        private static NDArray Mask4() => np.array(new[] { true, false, true, false });

        // =====================================================================
        // out= basics (probe B1)
        // =====================================================================

        [TestMethod]
        public void And_Out_SameDtype_IdentityAndValues()
        {
            // NumPy: bitwise_and(i4, i4, out=i4) -> [8, 8, 5, 1], returns out.
            var o = np.empty(new Shape(4), np.int32);
            var r = np.bitwise_and(Ai(), Bi(), o);

            Assert.IsTrue(ReferenceEquals(r, o), "out= must return the provided instance");
            var expected = new[] { 8, 8, 5, 1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetInt32(i));
        }

        [TestMethod]
        public void Or_Xor_Out_Values()
        {
            // NumPy: bitwise_or -> [14,14,15,3]; bitwise_xor -> [6,6,10,2].
            var oOr = np.empty(new Shape(4), np.int32);
            var rOr = np.bitwise_or(Ai(), Bi(), oOr);
            Assert.IsTrue(ReferenceEquals(rOr, oOr));
            var expectedOr = new[] { 14, 14, 15, 3 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedOr[i], oOr.GetInt32(i));

            var oXor = np.empty(new Shape(4), np.int32);
            var rXor = np.bitwise_xor(Ai(), Bi(), oXor);
            Assert.IsTrue(ReferenceEquals(rXor, oXor));
            var expectedXor = new[] { 6, 6, 10, 2 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedXor[i], oXor.GetInt32(i));
        }

        [TestMethod]
        public void Out_SameKindCasts_FromInt32Loop()
        {
            // NumPy: out=int16 (same_kind narrowing), int64, float32, float64
            // all accept the int32 loop result [8, 8, 5, 1].
            var o16 = np.empty(new Shape(4), np.int16);
            np.bitwise_and(Ai(), Bi(), o16);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((short)new[] { 8, 8, 5, 1 }[i], o16.GetInt16(i));

            var o64 = np.empty(new Shape(4), np.int64);
            np.bitwise_and(Ai(), Bi(), o64);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((long)new[] { 8, 8, 5, 1 }[i], o64.GetInt64(i));

            var oF32 = np.empty(new Shape(4), np.float32);
            np.bitwise_and(Ai(), Bi(), oF32);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((float)new[] { 8, 8, 5, 1 }[i], oF32.GetSingle(i));

            var oF64 = np.empty(new Shape(4), np.float64);
            np.bitwise_and(Ai(), Bi(), oF64);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((double)new[] { 8, 8, 5, 1 }[i], oF64.GetDouble(i));
        }

        [TestMethod]
        public void Out_SignedToUnsigned_ThrowsSameKindText()
        {
            // NumPy: UFuncTypeError — signed → unsigned is NOT same_kind.
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.bitwise_and(Ai(), Bi(), np.empty(new Shape(4), np.uint32)));

            Assert.AreEqual(
                "Cannot cast ufunc 'bitwise_and' output from dtype('int32') to " +
                "dtype('uint32') with casting rule 'same_kind'",
                ex.Message);
        }

        [TestMethod]
        public void Out_IntToBool_ThrowsSameKindText()
        {
            // NumPy: int → bool is NOT same_kind (ufunc name per op pinned too).
            var exAnd = Assert.ThrowsException<ArgumentException>(() =>
                np.bitwise_and(Ai(), Bi(), np.empty(new Shape(4), np.@bool)));
            Assert.AreEqual(
                "Cannot cast ufunc 'bitwise_and' output from dtype('int32') to " +
                "dtype('bool') with casting rule 'same_kind'",
                exAnd.Message);

            var exOr = Assert.ThrowsException<ArgumentException>(() =>
                np.bitwise_or(Ai(), Bi(), np.empty(new Shape(4), np.uint32)));
            StringAssert.Contains(exOr.Message, "ufunc 'bitwise_or'");
        }

        // =====================================================================
        // dtype matrix (probes B2, B3, D1)
        // =====================================================================

        [TestMethod]
        public void MixedInputs_Int32WithInt64_LoopIsInt64_NarrowingOutOk()
        {
            // NumPy: bitwise_and(i4, i8) runs the int64 loop; without out the
            // result dtype is int64, and out=i4 same_kind-narrows to [8,8,5,1].
            var b64 = Bi().astype(np.int64);

            var plain = np.bitwise_and(Ai(), b64);
            Assert.AreEqual(NPTypeCode.Int64, plain.typecode);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((long)new[] { 8, 8, 5, 1 }[i], plain.GetInt64(i));

            var o = np.empty(new Shape(4), np.int32);
            np.bitwise_and(Ai(), b64, o);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(new[] { 8, 8, 5, 1 }[i], o.GetInt32(i));
        }

        [TestMethod]
        public void UnsignedInputs_OutUnsignedWiderSignedAndFloat()
        {
            // NumPy: bitwise_and(u4, u4) with out=u4 / i8 / f8 all OK
            // (unsigned → wider signed and int → float are same_kind).
            var ua = Ai().astype(np.uint32);
            var ub = Bi().astype(np.uint32);

            var oU = np.empty(new Shape(4), np.uint32);
            np.bitwise_and(ua, ub, oU);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((uint)new[] { 8, 8, 5, 1 }[i], oU.GetUInt32(i));

            var oI64 = np.empty(new Shape(4), np.int64);
            np.bitwise_and(ua, ub, oI64);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((long)new[] { 8, 8, 5, 1 }[i], oI64.GetInt64(i));

            var oF64 = np.empty(new Shape(4), np.float64);
            np.bitwise_and(ua, ub, oF64);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((double)new[] { 8, 8, 5, 1 }[i], oF64.GetDouble(i));
        }

        [TestMethod]
        public void BoolInputs_BoolLoop_IntAndFloatOut()
        {
            // NumPy: bitwise_and(bool, bool) -> bool loop [T,F,F,F];
            // out=i4 -> [1,0,0,0]; out=f8 -> [1.,0.,0.,0.].
            var ba = np.array(new[] { true, true, false, false });
            var bb = np.array(new[] { true, false, true, false });

            var plain = np.bitwise_and(ba, bb);
            Assert.AreEqual(NPTypeCode.Boolean, plain.typecode);
            var expectedBool = new[] { true, false, false, false };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedBool[i], plain.GetBoolean(i));

            var oI = np.empty(new Shape(4), np.int32);
            np.bitwise_and(ba, bb, oI);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedBool[i] ? 1 : 0, oI.GetInt32(i));

            var oF = np.empty(new Shape(4), np.float64);
            np.bitwise_and(ba, bb, oF);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedBool[i] ? 1.0 : 0.0, oF.GetDouble(i));
        }

        // =====================================================================
        // out= shape rules (probe B4 — texts identical to the binary core)
        // =====================================================================

        [TestMethod]
        public void Out_WrongShape_ThrowsBroadcastText()
        {
            // NumPy lists every operand shape, trailing space included.
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.bitwise_and(Ai(), Bi(), np.empty(new Shape(5), np.int32)));

            Assert.AreEqual(
                "operands could not be broadcast together with shapes (4,) (4,) (5,) ",
                ex.Message);
        }

        [TestMethod]
        public void Out_WouldStretch_ThrowsNonBroadcastableText()
        {
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.bitwise_and(Ai(), Bi(), np.empty(new Shape(1), np.int32)));

            Assert.AreEqual(
                "non-broadcastable output operand with shape (1,) doesn't match the " +
                "broadcast shape (4,)",
                ex.Message);
        }

        [TestMethod]
        public void Out_Larger_InputsBroadcastUp()
        {
            // NumPy: (4,)&(4,) into out=(2,4) — rows repeat [8,8,5,1].
            var o = np.empty(new Shape(2, 4), np.int32);
            np.bitwise_and(Ai(), Bi(), o);

            var flat = np.ravel(o);
            var expected = new[] { 8, 8, 5, 1, 8, 8, 5, 1 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], flat.GetInt32(i));
        }

        // =====================================================================
        // aliasing + views (COPY_IF_OVERLAP / strided write-through)
        // =====================================================================

        [TestMethod]
        public void Out_FullAliasInput_InPlace()
        {
            // NumPy: bitwise_and(a, b, out=a) — the in-place `a &= b` shape.
            var a = Ai();
            var r = np.bitwise_and(a, Bi(), a);

            Assert.IsTrue(ReferenceEquals(r, a));
            var expected = new[] { 8, 8, 5, 1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], a.GetInt32(i));
        }

        [TestMethod]
        public void Out_PartialOverlap_CopyIfOverlap()
        {
            // NumPy: bitwise_and(x[0:4], x[1:5], out=x[0:4]) reads the ORIGINAL
            // operand values (overlap forces a temp): [12&10,10&15,15&1,1&3]
            // = [8,10,1,1] -> x = [8,10,1,1,3].
            var x = np.array(new[] { 12, 10, 15, 1, 3 }).astype(np.int32);
            np.bitwise_and(x["0:4"], x["1:5"], x["0:4"]);

            var expected = new[] { 8, 10, 1, 1, 3 };
            for (int i = 0; i < 5; i++)
                Assert.AreEqual(expected[i], x.GetInt32(i));
        }

        [TestMethod]
        public void Out_StridedView_WritesThroughStrides()
        {
            // NumPy: out=big[::2] -> [8,0,8,0,5,0,1,0].
            var big = np.zeros(new Shape(8), np.int32);
            np.bitwise_and(Ai(), Bi(), big["::2"]);

            var expected = new[] { 8, 0, 8, 0, 5, 0, 1, 0 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], big.GetInt32(i));
        }

        // =====================================================================
        // where= (probe B5)
        // =====================================================================

        [TestMethod]
        public void Where_WithOut_MaskedOffKeepPrior()
        {
            // NumPy: bitwise_and(i4, i4, out prior=-1, where=[T,F,T,F])
            // -> [8, -1, 5, -1].
            var o = np.full(new Shape(4), -1, np.int32);
            np.bitwise_and(Ai(), Bi(), o, Mask4());

            var expected = new[] { 8, -1, 5, -1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetInt32(i));
        }

        [TestMethod]
        public void Where_WithoutOut_MaskedOnSlotsOnly()
        {
            // NumPy: masked-OFF slots are uninitialized garbage (probe B5 saw
            // stale float bit patterns 1090519040 / 1065353216) — assert the
            // masked-ON slots only. Result dtype follows the loop (int32).
            var r = np.bitwise_and(Ai(), Bi(), where: Mask4());

            Assert.AreEqual(NPTypeCode.Int32, r.typecode);
            Assert.AreEqual(8, r.GetInt32(0));
            Assert.AreEqual(5, r.GetInt32(2));
        }

        [TestMethod]
        public void Where_JoinsOutputShape_MaskedOnOnly()
        {
            // NumPy: (4,)&(4,) with a (2,4) mask -> shape (2,4); only
            // masked-on slots are defined.
            var mask = np.array(new[,] { { true, false, true, false }, { false, true, false, true } });
            var r = np.bitwise_and(Ai(), Bi(), where: mask);

            Assert.AreEqual(2, (int)r.shape[0]);
            Assert.AreEqual(4, (int)r.shape[1]);
            Assert.AreEqual(8, r.GetInt32(0, 0));
            Assert.AreEqual(5, r.GetInt32(0, 2));
            Assert.AreEqual(8, r.GetInt32(1, 1));
            Assert.AreEqual(1, r.GetInt32(1, 3));
        }

        [TestMethod]
        public void Where_NonBool_ThrowsSafeText()
        {
            // NumPy: the mask converter casts with 'safe' — only bool passes.
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.bitwise_and(Ai(), Bi(), where: np.array(new long[] { 1, 0, 1, 0 })));

            Assert.AreEqual(
                "Cannot cast array data from dtype('int64') to dtype('bool') " +
                "according to the rule 'safe'",
                ex.Message);
        }

        [TestMethod]
        public void Where_OutCast_Composed()
        {
            // NumPy: int32 loop, f8 out prior=-1, where=[T,F,T,F]
            // -> [8., -1., 5., -1.] (masked windowed flush).
            var o = np.full(new Shape(4), -1.0, np.float64);
            np.bitwise_and(Ai(), Bi(), o, Mask4());

            var expected = new[] { 8.0, -1, 5, -1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i));
        }

        // =====================================================================
        // no-loop TypeError + validation order (probes B4/B6, plan §2.7)
        // =====================================================================

        [TestMethod]
        public void NoLoop_FloatInputs_ThrowsNumPyText()
        {
            var f = np.arange(4).astype(np.float64);

            var exAnd = Assert.ThrowsException<TypeError>(() => np.bitwise_and(f, f));
            Assert.AreEqual(
                "ufunc 'bitwise_and' not supported for the input types, and the inputs " +
                "could not be safely coerced to any supported types according to the casting rule ''safe''",
                exAnd.Message);

            var exOr = Assert.ThrowsException<TypeError>(() => np.bitwise_or(f, f));
            StringAssert.Contains(exOr.Message, "ufunc 'bitwise_or'");

            var exXor = Assert.ThrowsException<TypeError>(() => np.bitwise_xor(f, f));
            StringAssert.Contains(exXor.Message, "ufunc 'bitwise_xor'");

            // Half inputs hit the same no-loop rule.
            var h = np.arange(4).astype(np.float16);
            Assert.ThrowsException<TypeError>(() => np.bitwise_and(h, h));

            // Decimal is a NumSharp-only dtype: it follows the same float-kind
            // no-loop rule (unreachable in NumPy, pinned as NumSharp behavior).
            var d = np.array(new decimal[] { 1, 2, 3, 4 });
            Assert.ThrowsException<TypeError>(() => np.bitwise_and(d, d));
        }

        [TestMethod]
        public void NoLoop_BeatsBadOutCast_AndBadOutShape()
        {
            // NumPy order: loop resolution fires BEFORE out validation —
            // f8&f8 with out=i4 AND with out=(5,) both report the no-loop text.
            var f = np.arange(4).astype(np.float64);

            var exCast = Assert.ThrowsException<TypeError>(() =>
                np.bitwise_and(f, f, np.empty(new Shape(4), np.int32)));
            StringAssert.Contains(exCast.Message, "ufunc 'bitwise_and' not supported");

            var exShape = Assert.ThrowsException<TypeError>(() =>
                np.bitwise_and(f, f, np.empty(new Shape(5), np.float64)));
            StringAssert.Contains(exShape.Message, "ufunc 'bitwise_and' not supported");
        }

        [TestMethod]
        public void BadWhere_BeatsNoLoop()
        {
            // NumPy order: the where bool check is argument PARSING — it fires
            // before loop resolution even for float inputs.
            var f = np.arange(4).astype(np.float64);

            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.bitwise_and(f, f, where: np.array(new long[] { 1, 0, 1, 0 })));

            Assert.AreEqual(
                "Cannot cast array data from dtype('int64') to dtype('bool') " +
                "according to the rule 'safe'",
                ex.Message);
        }

        // =====================================================================
        // multi-window cast flush (>8192 elements), 0-d, empty
        // =====================================================================

        [TestMethod]
        public void Out_CastMultiWindow_20005Elements()
        {
            // 20 005 elements force 3 buffered windows (8192 default) for the
            // int32-loop → float64-out flush; values are i & 63.
            const int n = 20_005;
            var a = np.arange(n).astype(np.int32);
            var b = np.full(new Shape(n), 63, np.int32);
            var o = np.empty(new Shape(n), np.float64);

            var r = np.bitwise_and(a, b, o);

            Assert.IsTrue(ReferenceEquals(r, o));
            foreach (var i in new[] { 0, 5000, 8191, 8192, 12000, 16384, 20004 })
                Assert.AreEqual((double)(i & 63), o.GetDouble(i), $"index {i}");
        }

        [TestMethod]
        public void ZeroD_Scalars_WithOut()
        {
            // NumPy: bitwise_and(0-d, 0-d, out=0-d) -> 8, reference identity.
            var o = np.empty(new Shape(), np.int32);
            var r = np.bitwise_and(NDArray.Scalar(12), NDArray.Scalar(10), o);

            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual(8, o.GetInt32());
        }

        [TestMethod]
        public void Empty_WithOut_ReturnsOut()
        {
            var a = np.empty(new Shape(0), np.int32);
            var o = np.empty(new Shape(0), np.int32);

            var r = np.bitwise_and(a, np.empty(new Shape(0), np.int32), o);
            Assert.IsTrue(ReferenceEquals(r, o));
        }
    }
}
