using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// Wave 2.1 (roadmap): ufunc <c>out=</c> / <c>where=</c> parameters.
    ///
    /// Every expectation is pinned to NumPy 2.4.2 output (probed via
    /// python_run; texts verbatim including the trailing space NumPy leaves
    /// after the shape list in the could-not-broadcast message).
    ///
    /// Key probed semantics:
    ///  • out joins the broadcast — inputs broadcast UP to a larger out
    ///    (add((4,),(4,),out=(2,4)) repeats rows); out itself never stretches.
    ///  • the loop dtype comes from the INPUTS; out only needs a same_kind
    ///    cast from it (f64 result → i32 out raises; → f32/i16 fine).
    ///  • the out instance is returned (reference identity).
    ///  • where must be bool; it broadcasts and joins the output shape;
    ///    mask-false slots keep prior out contents.
    ///  • out aliasing an input is well-defined (COPY_IF_OVERLAP).
    /// </summary>
    [TestClass]
    public class UfuncOutWhereTests
    {
        private static NDArray Af() => np.arange(4).astype(np.float64);
        private static NDArray Bf() => np.arange(4).astype(np.float64) + 0.5;
        private static NDArray Mask4() => np.array(new[] { true, false, true, false });

        // =====================================================================
        // out= basics
        // =====================================================================

        [TestMethod]
        public void Out_ReturnsSameInstance_WithValues()
        {
            // NumPy: np.add(af, bf, out=o) returns o; [0.5, 2.5, 4.5, 6.5]
            var o = np.empty(new Shape(4), np.float64);
            var r = np.add(Af(), Bf(), o);

            Assert.IsTrue(ReferenceEquals(r, o), "out= must return the provided instance");
            var expected = new[] { 0.5, 2.5, 4.5, 6.5 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i));
        }

        [TestMethod]
        public void Out_SafeCastUp_Int32InputsToFloat64Out()
        {
            // NumPy: add(i32, i32, out=f64) — loop is int32, result cast f64.
            var o = np.empty(new Shape(4), np.float64);
            np.add(np.arange(4).astype(np.int32), np.arange(4).astype(np.int32) + 10, o);

            for (int i = 0; i < 4; i++)
                Assert.AreEqual(10.0 + 2 * i, o.GetDouble(i));
        }

        [TestMethod]
        public void Out_SameKindCastDown_Int16AndFloat32()
        {
            // NumPy: int→int and float→float are same_kind: i32+i32→i16 out and
            // f64+f64→f32 out both work.
            var o16 = np.empty(new Shape(4), np.int16);
            np.add(np.arange(4).astype(np.int32), np.arange(4).astype(np.int32) + 10, o16);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((short)(10 + 2 * i), o16.GetInt16(i));

            var o32 = np.empty(new Shape(4), np.float32);
            np.add(Af(), Bf(), o32);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(0.5f + 2 * i, o32.GetSingle(i));
        }

        [TestMethod]
        public void Out_FloatToInt_ThrowsNumPyText()
        {
            // NumPy: UFuncTypeError, exact text.
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.add(Af(), Bf(), np.empty(new Shape(4), np.int32)));

            Assert.AreEqual(
                "Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32') " +
                "with casting rule 'same_kind'",
                ex.Message);
        }

        [TestMethod]
        public void Out_UfuncNamesInCastErrors()
        {
            // NumPy ufunc names: np.mod is 'remainder', np.sqrt is 'sqrt'.
            var exMod = Assert.ThrowsException<ArgumentException>(() =>
                np.mod(Af(), Bf(), np.empty(new Shape(4), np.int32)));
            StringAssert.Contains(exMod.Message, "ufunc 'remainder'");

            var exSqrt = Assert.ThrowsException<ArgumentException>(() =>
                np.sqrt(Af(), np.empty(new Shape(4), np.int32)));
            StringAssert.Contains(exSqrt.Message, "ufunc 'sqrt'");
        }

        // =====================================================================
        // out= shape rules
        // =====================================================================

        [TestMethod]
        public void Out_IncompatibleShape_ThrowsNumPyText()
        {
            // NumPy lists every operand shape, trailing space included.
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.add(Af(), Bf(), np.empty(new Shape(5), np.float64)));

            Assert.AreEqual(
                "operands could not be broadcast together with shapes (4,) (4,) (5,) ",
                ex.Message);
        }

        [TestMethod]
        public void Out_WouldStretch_ThrowsNumPyText()
        {
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.add(Af(), Bf(), np.empty(new Shape(1), np.float64)));

            Assert.AreEqual(
                "non-broadcastable output operand with shape (1,) doesn't match the " +
                "broadcast shape (4,)",
                ex.Message);

            var ex2 = Assert.ThrowsException<ArgumentException>(() =>
                np.add(np.zeros(new Shape(2, 4), np.float64), Bf(), np.empty(new Shape(4), np.float64)));

            Assert.AreEqual(
                "non-broadcastable output operand with shape (4,) doesn't match the " +
                "broadcast shape (2,4)",
                ex2.Message);
        }

        [TestMethod]
        public void Out_InputsBroadcastUpToLargerOut()
        {
            // NumPy: add((4,),(4,),out=(2,4)) — inputs broadcast UP, rows repeat.
            var o = np.empty(new Shape(2, 4), np.float64);
            np.add(Af(), Bf(), o);

            var flat = np.ravel(o);
            var expected = new[] { 0.5, 2.5, 4.5, 6.5, 0.5, 2.5, 4.5, 6.5 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], flat.GetDouble(i));
        }

        [TestMethod]
        public void Out_AliasingInput_IsWellDefined()
        {
            // NumPy: np.add(x[:-1], x[:-1], out=x[1:]) -> [0,0,2,4,6,8,10,12]
            // (COPY_IF_OVERLAP forces a temporary; without it the propagating
            // write corrupts).
            var x = np.arange(8).astype(np.float64);
            np.add(x["0:7"], x["0:7"], x["1:8"]);

            var expected = new[] { 0.0, 0, 2, 4, 6, 8, 10, 12 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], x.GetDouble(i));
        }

        [TestMethod]
        public void Out_StridedView_WritesThroughStrides()
        {
            // NumPy: out=big[::2] -> [0.5, 0, 2.5, 0, 4.5, 0, 6.5, 0]
            var big = np.zeros(new Shape(8), np.float64);
            np.add(Af(), Bf(), big["::2"]);

            var expected = new[] { 0.5, 0, 2.5, 0, 4.5, 0, 6.5, 0 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], big.GetDouble(i));
        }

        // =====================================================================
        // where=
        // =====================================================================

        [TestMethod]
        public void Where_WithOut_FalseSlotsKeepPrior()
        {
            // NumPy: [0.5, -1, 4.5, -1]
            var o = np.full(new Shape(4), -1.0, np.float64);
            np.add(Af(), Bf(), o, Mask4());

            var expected = new[] { 0.5, -1, 4.5, -1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i));
        }

        [TestMethod]
        public void Where_BroadcastsOverOutput()
        {
            // NumPy: row mask (4,) over (2,4) -> [2,-1,2,-1,2,-1,2,-1]
            var o = np.full(new Shape(2, 4), -1.0, np.float64);
            np.add(np.ones(new Shape(2, 4), np.float64), np.ones(new Shape(4), np.float64), o, Mask4());

            var flat = np.ravel(o);
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(i % 2 == 0 ? 2.0 : -1.0, flat.GetDouble(i), $"flat {i}");
        }

        [TestMethod]
        public void Where_NonBoolMask_ThrowsNumPyText()
        {
            // NumPy: TypeError — the wheremask converter casts with 'safe'.
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.add(Af(), Bf(), np.full(new Shape(4), -1.0, np.float64), np.arange(4)));

            Assert.AreEqual(
                "Cannot cast array data from dtype('int64') to dtype('bool') " +
                "according to the rule 'safe'",
                ex.Message);
        }

        [TestMethod]
        public void Where_WithoutOut_MaskedSlotsComputed()
        {
            // NumPy: unmasked slots are uninitialized (unobservable); the
            // masked slots must hold the computed values.
            var r = np.add(Af(), Bf(), null, Mask4());

            Assert.AreEqual(0.5, r.GetDouble(0));
            Assert.AreEqual(4.5, r.GetDouble(2));
            Assert.AreEqual(4, (int)r.size);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
        }

        [TestMethod]
        public void Where_MixedDtype_CastOut_Masked_Combo()
        {
            // The full composition: i32 + f64 inputs (fused converts), f32 out
            // (windowed buffered cast), bool mask (masked inner loop + masked
            // flush). NumPy: [0.5, -1, 4.5, -1].
            var o = np.full(new Shape(4), -1.0f, np.float32);
            np.add(np.arange(4).astype(np.int32), Bf(), o, Mask4());

            var expected = new[] { 0.5f, -1f, 4.5f, -1f };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetSingle(i));
        }

        [TestMethod]
        public void Where_MultiWindow_CastOut_MaskHoldsAcrossWindows()
        {
            // N=20005 -> 3 buffer windows; mask every 3rd element; i32+f64
            // inputs into f32 out. 6669 written, 13336 kept, 0 wrong.
            const int N = 20_005;
            var x = np.arange(N).astype(np.int32);
            var y = np.ones(new Shape(N), np.float64);
            var o = np.full(new Shape(N), -1.0f, np.float32);
            var m = np.equal(np.mod(np.arange(N), np.array(3)), np.array(0));

            np.add(x, y, o, m);

            long wrote = 0, kept = 0;
            for (int i = 0; i < N; i++)
            {
                float v = o.GetSingle(i);
                if (i % 3 == 0)
                {
                    if (v == i + 1f) wrote++;
                    else Assert.Fail($"element {i}: {v}, expected {i + 1f}");
                }
                else
                {
                    if (v == -1f) kept++;
                    else Assert.Fail($"element {i}: {v}, expected untouched -1");
                }
            }
            Assert.AreEqual(6669L, wrote);
            Assert.AreEqual(13336L, kept);
        }

        // =====================================================================
        // unary out=/where=
        // =====================================================================

        [TestMethod]
        public void Unary_Sqrt_OutIdentity_PromotingInput_Where()
        {
            // sqrt(f64, out) identity + values.
            var o = np.empty(new Shape(4), np.float64);
            var r = np.sqrt(Af(), o);
            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual(System.Math.Sqrt(2), o.GetDouble(2), 1e-15);

            // sqrt(i32) -> f64 out (promoting input, buffered-cast route).
            var o2 = np.empty(new Shape(4), np.float64);
            np.sqrt(np.arange(4).astype(np.int32), o2);
            Assert.AreEqual(System.Math.Sqrt(3), o2.GetDouble(3), 1e-15);

            // sqrt with where: [2, -1, 4, -1] (NumPy O18).
            var o3 = np.full(new Shape(4), -1.0, np.float64);
            np.sqrt(np.array(new[] { 4.0, 9.0, 16.0, 25.0 }), o3, Mask4());
            var expected = new[] { 2.0, -1, 4, -1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o3.GetDouble(i));
        }

        [TestMethod]
        public void Ufunc_Family_OutSmoke()
        {
            // One value-pinned call per shipped out=-capable ufunc — now checking EVERY element
            // (not just [3]), the out= reference identity (the provided instance is returned), and
            // that the loop wrote the float64 out. Full arrays probed against NumPy 2.4.2.
            var af = Af();  // [0,1,2,3]
            var bf = Bf();  // [0.5,1.5,2.5,3.5]
            var o = np.empty(new Shape(4), np.float64);

            // Reference identity is the out= contract; AssertRow re-checks size/dtype/every element.
            Assert.IsTrue(ReferenceEquals(np.subtract(bf, af, o), o));
            AssertRow(o, 0.5, 0.5, 0.5, 0.5);

            Assert.IsTrue(ReferenceEquals(np.multiply(af, bf, o), o));
            AssertRow(o, 0.0, 1.5, 5.0, 10.5);

            Assert.IsTrue(ReferenceEquals(np.divide(bf, af + 1.0, o), o));
            AssertRow(o, 0.5, 0.75, 0.8333333333333334, 0.875);

            Assert.IsTrue(ReferenceEquals(np.mod(bf, af + 1.0, o), o));
            AssertRow(o, 0.5, 1.5, 2.5, 3.5);

            Assert.IsTrue(ReferenceEquals(np.power(af, np.full(new Shape(4), 2.0, np.float64), o), o));
            AssertRow(o, 0.0, 1.0, 4.0, 9.0);

            Assert.IsTrue(ReferenceEquals(np.floor_divide(bf, af + 1.0, o), o));
            AssertRow(o, 0.0, 0.0, 0.0, 0.0);

            Assert.IsTrue(ReferenceEquals(np.negative(af, o), o));
            AssertRow(o, -0.0, -1.0, -2.0, -3.0);

            Assert.IsTrue(ReferenceEquals(np.abs(np.negative(af), o), o));
            AssertRow(o, 0.0, 1.0, 2.0, 3.0);

            Assert.IsTrue(ReferenceEquals(np.absolute(np.negative(af), o), o));
            AssertRow(o, 0.0, 1.0, 2.0, 3.0);

            Assert.IsTrue(ReferenceEquals(np.exp(np.zeros(new Shape(4), np.float64), o), o));
            AssertRow(o, 1.0, 1.0, 1.0, 1.0);

            Assert.IsTrue(ReferenceEquals(np.log(np.ones(new Shape(4), np.float64), o), o));
            AssertRow(o, 0.0, 0.0, 0.0, 0.0);

            Assert.IsTrue(ReferenceEquals(np.sin(np.zeros(new Shape(4), np.float64), o), o));
            AssertRow(o, 0.0, 0.0, 0.0, 0.0);

            Assert.IsTrue(ReferenceEquals(np.cos(np.zeros(new Shape(4), np.float64), o), o));
            AssertRow(o, 1.0, 1.0, 1.0, 1.0);

            Assert.IsTrue(ReferenceEquals(np.tan(np.zeros(new Shape(4), np.float64), o), o));
            AssertRow(o, 0.0, 0.0, 0.0, 0.0);

            Assert.IsTrue(ReferenceEquals(np.square(af, o), o));
            AssertRow(o, 0.0, 1.0, 4.0, 9.0);

            Assert.IsTrue(ReferenceEquals(np.sqrt(np.array(new[] { 0.0, 1, 4, 9 }), o), o));
            AssertRow(o, 0.0, 1.0, 2.0, 3.0);
        }

        /// <summary>
        /// Assert a float64 <paramref name="o"/> has exactly <paramref name="expected"/>.Length elements,
        /// is float64 dtype, and matches every element within a tight tolerance — the shared checker that
        /// turns the family smoke test from a single-element pin into a full-row assertion.
        /// </summary>
        /// <param name="o">The out array to check.</param>
        /// <param name="expected">The full expected row (probed against NumPy 2.4.2).</param>
        private static void AssertRow(NDArray o, params double[] expected)
        {
            Assert.AreEqual(expected.Length, (int)o.size, "size");
            Assert.AreEqual(NPTypeCode.Double, o.typecode, "dtype");
            for (int i = 0; i < expected.Length; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i), 1e-12, $"element {i}");
        }

        [TestMethod]
        public void Out_ZeroD_Scalars()
        {
            // NumPy O17: add(scalar, scalar, out=0-d) works.
            var o = np.empty(new Shape(), np.float64);
            var r = np.add(NDArray.Scalar(2.0), NDArray.Scalar(3.0), o);
            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual(5.0, o.GetDouble());
        }
    }
}
