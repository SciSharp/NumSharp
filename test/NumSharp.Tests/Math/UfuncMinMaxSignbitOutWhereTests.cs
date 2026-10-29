using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// out=/where= coverage for the families whose np.* overloads gained the NumPy-shaped
    /// <c>f(x1[, x2], out=null, where=null, dtype=null)</c> surface in the 2026-09-18 coverage pass:
    /// the min/max family (<c>maximum</c>/<c>minimum</c>/<c>fmax</c>/<c>fmin</c> — the engine already
    /// routed out=/where=, only the public overloads exposed <c>dtype</c>), the sign-bit predicate
    /// (<c>signbit</c>, previously out=/where=-capable but entirely untested), and the two-output
    /// <c>divmod</c> (its <c>out=(q,r)</c>/<c>where=</c> path was implemented but had zero tests).
    ///
    /// Every expectation is pinned to a NumPy 2.4.2 probe. Load-bearing probed semantics:
    ///  • maximum/minimum PROPAGATE NaN; fmax/fmin IGNORE it (return the non-NaN operand).
    ///  • the loop dtype comes from the INPUTS; a same_kind-castable out is accepted and returned as-is.
    ///  • signbit reads the raw sign bit: -0.0 → True, a signed integer follows x&lt;0, and bool casts
    ///    same_kind into any numeric out (True→1).
    ///  • divmod(x1,x2) == (floor_divide, remainder); out=(q,r) writes both and returns them as-is.
    ///  • the ±0 SIGN of a min/max result is non-contractual (NumPy varies it by SIMD lane), so it is
    ///    deliberately NOT asserted here (see MisalignedRegistry K13).
    /// </summary>
    [TestClass]
    public class UfuncMinMaxSignbitOutWhereTests
    {
        private static NDArray A() => np.array(new[] { 1.0, 5.0, 3.0, 2.0 });
        private static NDArray B() => np.array(new[] { 4.0, 2.0, 6.0, 2.0 });
        private static NDArray Mask4() => np.array(new[] { true, false, true, false });

        // =====================================================================
        // maximum / minimum — out=, where=, cast, reference identity
        // =====================================================================

        [TestMethod]
        public void Maximum_Minimum_Out_ReturnsSameInstance_WithValues()
        {
            // NumPy: maximum([1,5,3,2],[4,2,6,2]) -> [4,5,6,2]; minimum -> [1,2,3,2].
            var oMax = np.empty(new Shape(4), np.float64);
            var rMax = np.maximum(A(), B(), oMax);
            Assert.IsTrue(ReferenceEquals(rMax, oMax), "out= must return the provided instance");
            var expMax = new[] { 4.0, 5.0, 6.0, 2.0 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expMax[i], oMax.GetDouble(i));

            var oMin = np.empty(new Shape(4), np.float64);
            np.minimum(A(), B(), oMin);
            var expMin = new[] { 1.0, 2.0, 3.0, 2.0 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expMin[i], oMin.GetDouble(i));
        }

        [TestMethod]
        public void Maximum_Where_MaskedOffKeepPrior()
        {
            // NumPy: maximum(a, b, out=prior=-9, where=[T,F,T,F]) -> [4,-9,6,-9].
            var o = np.full(new Shape(4), -9.0, np.float64);
            np.maximum(A(), B(), o, Mask4());
            var expected = new[] { 4.0, -9.0, 6.0, -9.0 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i));
        }

        [TestMethod]
        public void Maximum_OutCastDown_F64ToF32()
        {
            // NumPy: maximum(f64, f64, out=f32) — same_kind cast, [4,5,6,2].
            var o = np.empty(new Shape(4), np.float32);
            np.maximum(A(), B(), o);
            var expected = new[] { 4f, 5f, 6f, 2f };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetSingle(i));
        }

        [TestMethod]
        public void Maximum_OutCastError_NamesUfunc()
        {
            // NumPy: UFuncTypeError naming 'maximum' (f64 -> i32 is not same_kind).
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.maximum(A(), B(), np.empty(new Shape(4), np.int32)));
            Assert.AreEqual(
                "Cannot cast ufunc 'maximum' output from dtype('float64') to dtype('int32') " +
                "with casting rule 'same_kind'", ex.Message);
        }

        // =====================================================================
        // fmax / fmin — NaN handling contrast (probed)
        // =====================================================================

        [TestMethod]
        public void FMax_FMin_IgnoreNaN_While_MaxMin_Propagate()
        {
            var a = np.array(new[] { 1.0, double.NaN, 3.0 });
            var b = np.array(new[] { double.NaN, 2.0, double.NaN });

            // NumPy: fmax ignores NaN -> [1,2,3]; maximum propagates -> [nan,nan,nan].
            var oFmax = np.empty(new Shape(3), np.float64);
            np.fmax(a, b, oFmax);
            Assert.AreEqual(1.0, oFmax.GetDouble(0));
            Assert.AreEqual(2.0, oFmax.GetDouble(1));
            Assert.AreEqual(3.0, oFmax.GetDouble(2));

            var oMax = np.empty(new Shape(3), np.float64);
            np.maximum(a, b, oMax);
            for (int i = 0; i < 3; i++)
                Assert.IsTrue(double.IsNaN(oMax.GetDouble(i)), $"maximum propagates NaN at {i}");

            // fmin/minimum symmetric.
            var oFmin = np.empty(new Shape(3), np.float64);
            np.fmin(a, b, oFmin);
            Assert.AreEqual(1.0, oFmin.GetDouble(0));
            Assert.AreEqual(2.0, oFmin.GetDouble(1));
            Assert.AreEqual(3.0, oFmin.GetDouble(2));
        }

        [TestMethod]
        public void FMax_Where_And_Dtype_Compose()
        {
            // where= masks; dtype= selects the loop (validate + compute at that precision).
            var o = np.full(new Shape(4), -1.0, np.float64);
            np.fmax(A(), B(), o, Mask4());
            var expected = new[] { 4.0, -1.0, 6.0, -1.0 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i));

            // dtype: float32 loop, values stored into an f64 out at f32 precision.
            var o2 = np.empty(new Shape(4), np.float64);
            np.fmax(A(), B(), o2, dtype: NPTypeCode.Single);
            Assert.AreEqual(4.0, o2.GetDouble(0));
        }

        // =====================================================================
        // signbit — out=, where=, numeric out, integer input, strided out
        // =====================================================================

        [TestMethod]
        public void SignBit_Out_Bool_And_Numeric()
        {
            // NumPy: signbit([1,-0.0,-3,2]) -> [F,T,T,F]; -0.0 sets the sign bit.
            var s = np.array(new[] { 1.0, -0.0, -3.0, 2.0 });

            var ob = np.empty(new Shape(4), np.@bool);
            var r = np.signbit(s, ob);
            Assert.IsTrue(ReferenceEquals(r, ob));
            Assert.AreEqual(false, ob.GetBoolean(0));
            Assert.AreEqual(true, ob.GetBoolean(1));
            Assert.AreEqual(true, ob.GetBoolean(2));
            Assert.AreEqual(false, ob.GetBoolean(3));

            // NumPy: bool casts same_kind into an int out (True->1).
            var oi = np.empty(new Shape(4), np.int32);
            np.signbit(s, oi);
            var expected = new[] { 0, 1, 1, 0 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], oi.GetInt32(i));
        }

        [TestMethod]
        public void SignBit_IntInput_FollowsLessThanZero()
        {
            // NumPy: signbit(int32 [1,-2,3,-4]) -> [F,T,F,T] (two's-complement MSB).
            var si = np.array(new[] { 1, -2, 3, -4 }).astype(np.int32);
            var o = np.empty(new Shape(4), np.@bool);
            np.signbit(si, o);
            var expected = new[] { false, true, false, true };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetBoolean(i));
        }

        [TestMethod]
        public void SignBit_Where_And_StridedOut()
        {
            // where= masks; masked-off out slots keep prior.
            var s = np.array(new[] { 1.0, -0.0, -3.0, 2.0 });
            var o = np.full(new Shape(4), -1, np.int32);
            np.signbit(s, o, Mask4());
            var expected = new[] { 0, -1, 1, -1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetInt32(i));

            // strided out writes through the view (1-D strided is bit-exact vs NumPy).
            var big = np.zeros(new Shape(8), np.@bool);
            np.signbit(np.array(new[] { -1.0, 2.0, -3.0, 4.0 }), big["::2"]);
            var expectedBig = new[] { true, false, false, false, true, false, false, false };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expectedBig[i], big.GetBoolean(i), $"index {i}");
        }

        [TestMethod]
        public void SignBit_Dtype_ValidateOnly_NonBoolRaises()
        {
            // NumPy: signbit has bool loops only — a non-bool dtype= raises the no-loop TypeError.
            var s = np.array(new[] { 1.0, -1.0 });
            Assert.IsTrue(np.signbit(s, dtype: NPTypeCode.Boolean).GetBoolean(1));
            var ex = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.signbit(s, dtype: NPTypeCode.Double));
            Assert.AreEqual(
                "No loop matching the specified signature and casting was found for ufunc signbit",
                ex.Message);
        }

        // =====================================================================
        // divmod — out=(q,r) and where= (the two-output ufunc)
        // =====================================================================

        [TestMethod]
        public void DivMod_OutTuple_ReturnsProvidedArrays_WithValues()
        {
            // NumPy: divmod([7,-7,8,-8],[3,3,3,3]) -> q=[2,-3,2,-3], r=[1,2,2,1].
            var x = np.array(new[] { 7.0, -7.0, 8.0, -8.0 });
            var y = np.array(new[] { 3.0, 3.0, 3.0, 3.0 });
            var q = np.full(new Shape(4), -1.0, np.float64);
            var r = np.full(new Shape(4), -1.0, np.float64);

            var res = np.divmod(x, y, (q, r));
            Assert.IsTrue(ReferenceEquals(res.Quotient, q), "out1 returned as-is");
            Assert.IsTrue(ReferenceEquals(res.Remainder, r), "out2 returned as-is");

            var expQ = new[] { 2.0, -3.0, 2.0, -3.0 };
            var expR = new[] { 1.0, 2.0, 2.0, 1.0 };
            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(expQ[i], q.GetDouble(i), $"q[{i}]");
                Assert.AreEqual(expR[i], r.GetDouble(i), $"r[{i}]");
            }
        }

        [TestMethod]
        public void DivMod_Where_MaskedOffKeepPrior_BothOutputs()
        {
            // NumPy: divmod(x,y,out=(q,r) prior=-1, where=[T,F,T,F])
            //   -> q=[2,-1,2,-1], r=[1,-1,2,-1].
            var x = np.array(new[] { 7.0, -7.0, 8.0, -8.0 });
            var y = np.array(new[] { 3.0, 3.0, 3.0, 3.0 });
            var q = np.full(new Shape(4), -1.0, np.float64);
            var r = np.full(new Shape(4), -1.0, np.float64);

            np.divmod(x, y, (q, r), Mask4());

            var expQ = new[] { 2.0, -1.0, 2.0, -1.0 };
            var expR = new[] { 1.0, -1.0, 2.0, -1.0 };
            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(expQ[i], q.GetDouble(i), $"q[{i}]");
                Assert.AreEqual(expR[i], r.GetDouble(i), $"r[{i}]");
            }
        }
    }
}
