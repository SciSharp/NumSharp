using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.MathSuite
{
    /// <summary>
    /// out=/where= plan slice 2 (docs/OUT_WHERE_NPYITER_FAMILIES_PLAN.md §4.3):
    /// the remaining unary-math batch + invert + arctan2 gain the merged
    /// NumPy-shaped overload <c>f(x[, x2], out=null, where=null, dtype=null)</c>
    /// (round_/around: out= only — np.round is a function, not a ufunc).
    ///
    /// Every expectation pinned to a NumPy 2.4.2 probe (texts verbatim).
    /// Load-bearing probed semantics:
    ///  • floor/ceil/trunc have IDENTITY loops for every bool/int dtype
    ///    ('?->?','b->b',…,'Q->Q'); np.round's int path is an identity copy.
    ///  • the loop dtype comes from the INPUT tier; out only constrains the
    ///    write-back cast (sinh(i1, out=f8) stores float16-precision values).
    ///  • dtype= selects the LOOP: the input must reach it via same_kind
    ///    (input-cast UFuncTypeError), float-only ufuncs raise "No loop
    ///    matching…", invert raises it for float dtype= over an int input.
    ///  • validation order: where parse → loop resolution → out cast → shape.
    /// </summary>
    [TestClass]
    public class UfuncUnaryBatchOutWhereTests
    {
        private static NDArray I4() => np.arange(4).astype(np.int32);
        private static NDArray F8() => np.arange(4).astype(np.float64);
        private static NDArray Mask4() => np.array(new[] { true, false, true, false });

        // =====================================================================
        // out= basics: identity + values (S1)
        // =====================================================================

        [TestMethod]
        public void Floor_Out_SameDtype_IdentityAndValues()
        {
            var x = np.array(new[] { 0.5, 1.5, 2.5, 3.5 });
            var o = np.empty(new Shape(4), np.float64);
            var r = np.floor(x, o);

            Assert.IsTrue(ReferenceEquals(r, o));
            var expected = new[] { 0.0, 1, 2, 3 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i));
        }

        [TestMethod]
        public void Invert_Out_Values_AndUnsignedToSignedOut()
        {
            // NumPy: invert([12,10,15,1] i4) -> [-13,-11,-16,-2]; out=i8 casts.
            var x = np.array(new[] { 12, 10, 15, 1 }).astype(np.int32);
            var o = np.empty(new Shape(4), np.int64);
            np.invert(x, o);
            var expected = new long[] { -13, -11, -16, -2 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetInt64(i));

            // NumPy: invert(u4 [1,2,3,4], out=i4) -> [-2,-3,-4,-5] (u→i same_kind).
            var u = np.array(new[] { 1, 2, 3, 4 }).astype(np.uint32);
            var oi = np.empty(new Shape(4), np.int32);
            np.invert(u, oi);
            var expectedI = new[] { -2, -3, -4, -5 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedI[i], oi.GetInt32(i));

            // NumPy: invert(bool) is logical not; out=i4 -> [0,0,1,1].
            var b = np.array(new[] { true, true, false, false });
            var ob = np.empty(new Shape(4), np.int32);
            np.invert(b, ob);
            var expectedB = new[] { 0, 0, 1, 1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedB[i], ob.GetInt32(i));
        }

        [TestMethod]
        public void ArcTan2_Out_Values_And0d()
        {
            // NumPy: arctan2(arange(4), arange(4)+1, out=f4)
            // -> [0., 0.4636476, 0.5880026, 0.6435011] float32.
            var y = I4();
            var x = I4() + 1;
            var o = np.empty(new Shape(4), np.float32);
            var r = np.arctan2(y, x, o);

            Assert.IsTrue(ReferenceEquals(r, o));
            var expected = new[] { 0f, 0.4636476f, 0.5880026f, 0.6435011f };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetSingle(i), 1e-6f);

            // 0-d everything (probed OK).
            var o0 = np.empty(new Shape(), np.float64);
            var r0 = np.arctan2(NDArray.Scalar(1.0), NDArray.Scalar(1.0), o0);
            Assert.IsTrue(ReferenceEquals(r0, o0));
            Assert.AreEqual(Math.PI / 4, o0.GetDouble(), 1e-12);
        }

        // =====================================================================
        // THE loop-dtype-from-inputs pin (S29, probe C1)
        // =====================================================================

        [TestMethod]
        public void Sinh_Int8Input_Float64Out_StoresFloat16Precision()
        {
            // NumPy: sinh(i1, out=f8) runs the FLOAT16 loop (input tier) and
            // casts up -> [0., 1.17480469, 3.62695312, 10.015625]. The out
            // dtype does not promote the loop.
            var x = np.arange(4).astype(np.@byte); // np.@byte == sbyte (i1)
            var o = np.empty(new Shape(4), np.float64);
            np.sinh(x, o);

            var expected = new[] { 0.0, 1.17480469, 3.62695312, 10.015625 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i), 1e-7, $"index {i}");
        }

        // =====================================================================
        // out-cast errors carry each op's NumPy ufunc name (S4)
        // =====================================================================

        [TestMethod]
        public void OutCastErrors_UfuncNamesVerbatim()
        {
            var f = F8();
            var i4out = np.empty(new Shape(4), np.int32);

            var exFloor = Assert.ThrowsException<ArgumentException>(() => np.floor(f, i4out));
            Assert.AreEqual(
                "Cannot cast ufunc 'floor' output from dtype('float64') to dtype('int32') " +
                "with casting rule 'same_kind'", exFloor.Message);

            var exTrunc = Assert.ThrowsException<ArgumentException>(() => np.trunc(f, i4out));
            StringAssert.Contains(exTrunc.Message, "ufunc 'trunc'");

            var exRint = Assert.ThrowsException<ArgumentException>(() => np.round_(f, @out: i4out));
            StringAssert.Contains(exRint.Message, "ufunc 'rint'");

            var exSinh = Assert.ThrowsException<ArgumentException>(() => np.sinh(f, i4out));
            StringAssert.Contains(exSinh.Message, "ufunc 'sinh'");

            var exATan = Assert.ThrowsException<ArgumentException>(() => np.arctan(f, i4out));
            StringAssert.Contains(exATan.Message, "ufunc 'arctan'");

            var exSign = Assert.ThrowsException<ArgumentException>(() => np.sign(f, i4out));
            StringAssert.Contains(exSign.Message, "ufunc 'sign'");

            var exAt2 = Assert.ThrowsException<ArgumentException>(() => np.arctan2(f, f, i4out));
            Assert.AreEqual(
                "Cannot cast ufunc 'arctan2' output from dtype('float64') to dtype('int32') " +
                "with casting rule 'same_kind'", exAt2.Message);
        }

        // =====================================================================
        // dtype= — loop signature rule (probed matrix)
        // =====================================================================

        [TestMethod]
        public void Dtype_IdentityLoops_SelectAndCast()
        {
            // NumPy: floor(i4, dtype=i8) -> int64 identity; dtype=f4 -> float32.
            var r64 = np.floor(I4(), dtype: NPTypeCode.Int64);
            Assert.AreEqual(NPTypeCode.Int64, r64.typecode);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((long)i, r64.GetInt64(i));

            var rf = np.floor(I4(), dtype: NPTypeCode.Single);
            Assert.AreEqual(NPTypeCode.Single, rf.typecode);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((float)i, rf.GetSingle(i));

            // NumPy: sign(i4, dtype=f8) -> [0., 1., 1., 1.].
            var rs = np.sign(I4(), dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, rs.typecode);
            Assert.AreEqual(0.0, rs.GetDouble(0));
            Assert.AreEqual(1.0, rs.GetDouble(3));

            // NumPy: reciprocal(i4 [1,2,3,4], dtype=f8) -> [1., .5, .333…, .25].
            var rr = np.reciprocal(np.array(new[] { 1, 2, 3, 4 }).astype(np.int32), dtype: NPTypeCode.Double);
            Assert.AreEqual(1.0, rr.GetDouble(0));
            Assert.AreEqual(0.5, rr.GetDouble(1));
            Assert.AreEqual(1.0 / 3.0, rr.GetDouble(2), 1e-12);
            Assert.AreEqual(0.25, rr.GetDouble(3));
        }

        [TestMethod]
        public void Dtype_InputCastErrors_Verbatim()
        {
            // NumPy: the INPUT must reach the dtype-selected loop via same_kind.
            var exFloor = Assert.ThrowsException<ArgumentException>(() =>
                np.floor(F8(), dtype: NPTypeCode.Int32));
            Assert.AreEqual(
                "Cannot cast ufunc 'floor' input from dtype('float64') to dtype('int32') " +
                "with casting rule 'same_kind'", exFloor.Message);

            var exU = Assert.ThrowsException<ArgumentException>(() =>
                np.floor(I4(), dtype: NPTypeCode.UInt32));
            Assert.AreEqual(
                "Cannot cast ufunc 'floor' input from dtype('int32') to dtype('uint32') " +
                "with casting rule 'same_kind'", exU.Message);

            var exSign = Assert.ThrowsException<ArgumentException>(() =>
                np.sign(F8(), dtype: NPTypeCode.Int32));
            StringAssert.Contains(exSign.Message, "ufunc 'sign' input");

            var exInv = Assert.ThrowsException<ArgumentException>(() =>
                np.invert(F8(), dtype: NPTypeCode.Int32));
            StringAssert.Contains(exInv.Message, "ufunc 'invert' input");
        }

        [TestMethod]
        public void Dtype_FloatOnlyOps_RejectIntRequests_NoLoopText()
        {
            // NumPy: "No loop matching the specified signature and casting was
            // found for ufunc <name>" — each op names ITSELF.
            var exSinh = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.sinh(F8(), dtype: NPTypeCode.Int32));
            Assert.AreEqual(
                "No loop matching the specified signature and casting was found for ufunc sinh",
                exSinh.Message);

            var exLog2 = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.log2(F8(), dtype: NPTypeCode.Int32));
            StringAssert.Contains(exLog2.Message, "ufunc log2");

            var exAt2 = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.arctan2(F8(), F8(), dtype: NPTypeCode.Int32));
            StringAssert.Contains(exAt2.Message, "ufunc arctan2");

            // invert: float dtype over an int input is also a no-loop signature.
            var exInv = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.invert(I4(), dtype: NPTypeCode.Double));
            Assert.AreEqual(
                "No loop matching the specified signature and casting was found for ufunc invert",
                exInv.Message);
        }

        [TestMethod]
        public void Dtype_SelectsLoopPrecision_AndComposes()
        {
            // NumPy: sinh(i4, dtype=f4) -> float32 [0., 1.1752012, 3.6268604, 10.017875].
            var r = np.sinh(I4(), dtype: NPTypeCode.Single);
            Assert.AreEqual(NPTypeCode.Single, r.typecode);
            var expected = new[] { 0f, 1.1752012f, 3.6268604f, 10.017875f };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], r.GetSingle(i), 1e-5f);

            // NumPy: invert(i4, dtype=i8) -> int64 [-1,-2,-3,-4];
            // invert(bool, dtype=i4) runs the INT loop on bool values -> [-2,-1,-2,-1].
            var ri = np.invert(I4(), dtype: NPTypeCode.Int64);
            Assert.AreEqual(NPTypeCode.Int64, ri.typecode);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((long)(~i), ri.GetInt64(i));

            var rb = np.invert(Mask4(), dtype: NPTypeCode.Int32);
            Assert.AreEqual(NPTypeCode.Int32, rb.typecode);
            var expectedB = new[] { -2, -1, -2, -1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedB[i], rb.GetInt32(i));

            // dtype composes with out: arctan2(i4, i4, dtype=f4) into an f8 out
            // stores the float32-precision values.
            var o = np.empty(new Shape(4), np.float64);
            np.arctan2(I4(), I4() + 1, o, dtype: NPTypeCode.Single);
            Assert.AreEqual((double)0.4636476f, o.GetDouble(1), 1e-7);
        }

        // =====================================================================
        // invert / float no-loop ordering (probes §2.7)
        // =====================================================================

        [TestMethod]
        public void Invert_FloatInputs_NotSupportedText_AndOrder()
        {
            var f = F8();

            var ex = Assert.ThrowsException<TypeError>(() => np.invert(f));
            Assert.AreEqual(
                "ufunc 'invert' not supported for the input types, and the inputs " +
                "could not be safely coerced to any supported types according to the casting rule ''safe''",
                ex.Message);

            // no-loop beats bad out.
            var exOut = Assert.ThrowsException<TypeError>(() =>
                np.invert(f, np.empty(new Shape(4), np.int32)));
            StringAssert.Contains(exOut.Message, "ufunc 'invert' not supported");

            // bad where beats no-loop (where is argument parsing).
            var exWhere = Assert.ThrowsException<ArgumentException>(() =>
                np.invert(f, where: np.array(new long[] { 1, 0, 1, 0 })));
            Assert.AreEqual(
                "Cannot cast array data from dtype('int64') to dtype('bool') " +
                "according to the rule 'safe'", exWhere.Message);
        }

        [TestMethod]
        public void Where_BeatsDtypeNoLoop_OnFloatTierOps()
        {
            // NumPy order ①②: where parse precedes loop resolution — a bad
            // where wins over the dtype= no-loop raise.
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.sinh(F8(), where: np.array(new long[] { 1, 0, 1, 0 }), dtype: NPTypeCode.Int32));
            StringAssert.Contains(ex.Message, "according to the rule 'safe'");
        }

        // =====================================================================
        // integer identity loops (S26, probes §2.9/§2.11)
        // =====================================================================

        [TestMethod]
        public void IdentityLoops_IntAndBool_DtypePreserved()
        {
            // floor/ceil/trunc(i4) -> int32 (identity); previously trunc/round_
            // promoted to f64 — NumPy preserves.
            Assert.AreEqual(NPTypeCode.Int32, np.floor(I4()).typecode);
            Assert.AreEqual(NPTypeCode.Int32, np.ceil(I4()).typecode);
            Assert.AreEqual(NPTypeCode.Int32, np.trunc(I4()).typecode);
            Assert.AreEqual(NPTypeCode.Int32, np.round_(I4()).typecode);

            for (int i = 0; i < 4; i++)
                Assert.AreEqual(i, np.trunc(I4()).GetInt32(i));

            // floor/ceil/trunc(bool) -> bool identity (probed '?->?' loops).
            var b = Mask4();
            foreach (var r in new[] { np.floor(b), np.ceil(b), np.trunc(b) })
            {
                Assert.AreEqual(NPTypeCode.Boolean, r.typecode);
                Assert.AreEqual(true, r.GetBoolean(0));
                Assert.AreEqual(false, r.GetBoolean(1));
            }
        }

        [TestMethod]
        public void Floor_Int_OutAndWhere_IdentityLoop()
        {
            // NumPy: floor(i4, out prior=-9, where=[T,F,T,F]) -> [0,-9,2,-9].
            var o = np.full(new Shape(4), -9, np.int32);
            np.floor(I4(), o, Mask4());
            var expected = new[] { 0, -9, 2, -9 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetInt32(i));

            // floor(i4, out=f8) -> identity ints cast to f8.
            var of = np.empty(new Shape(4), np.float64);
            np.floor(I4(), of);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((double)i, of.GetDouble(i));
        }

        // =====================================================================
        // round_ / around — out= only (np.round is not a ufunc)
        // =====================================================================

        [TestMethod]
        public void Round_Decimals_OutIdentity_AndValues()
        {
            // NumPy: round([1.44,1.55,2.5,-2.5], 1, out) -> [1.4, 1.6, 2.5, -2.5], returns out.
            var x = np.array(new[] { 1.44, 1.55, 2.5, -2.5 });
            var o = np.empty(new Shape(4), np.float64);
            var r = np.round_(x, 1, o);

            Assert.IsTrue(ReferenceEquals(r, o));
            var expected = new[] { 1.4, 1.6, 2.5, -2.5 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], o.GetDouble(i), 1e-12);

            // decimals=0 route: banker's rounding [0., 2., 2., -2.].
            var o2 = np.empty(new Shape(4), np.float64);
            np.around(np.array(new[] { 0.5, 1.5, 2.5, -2.5 }), @out: o2);
            var expected2 = new[] { 0.0, 2, 2, -2 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected2[i], o2.GetDouble(i));

            // round(f8, 1, out=f4) -> float32 values.
            var o3 = np.empty(new Shape(4), np.float32);
            np.round_(x, 1, o3);
            Assert.AreEqual(1.6f, o3.GetSingle(1), 1e-6f);

            // round(i4, 1, out=i4) -> identity ints (probed).
            var oi = np.empty(new Shape(4), np.int32);
            np.round_(I4(), 1, oi);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(i, oi.GetInt32(i));
        }

        [TestMethod]
        public void Round_OutCast_NamesRintAndMultiply()
        {
            var x = np.array(new[] { 1.44, 1.55, 2.5, -2.5 });

            // decimals==0 -> the rint ufunc. (np.round's 2nd positional is
            // decimals per NumPy — out rides the 3rd slot or the name.)
            var exRint = Assert.ThrowsException<ArgumentException>(() =>
                np.round_(x, @out: np.empty(new Shape(4), np.int32)));
            Assert.AreEqual(
                "Cannot cast ufunc 'rint' output from dtype('float64') to dtype('int32') " +
                "with casting rule 'same_kind'", exRint.Message);

            // decimals!=0 is NumPy's multiply→rint→divide composition: the cast
            // error names ufunc 'multiply' (probed).
            var exMul = Assert.ThrowsException<ArgumentException>(() =>
                np.round_(x, 1, np.empty(new Shape(4), np.int32)));
            Assert.AreEqual(
                "Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int32') " +
                "with casting rule 'same_kind'", exMul.Message);
        }

        // =====================================================================
        // reciprocal — int semantics fixed to NumPy (probe C4)
        // =====================================================================

        [TestMethod]
        public void Reciprocal_Int_ZeroGivesMinValue_NumPyParity()
        {
            // NumPy 2.4.2: reciprocal(i4 [1,2,-3,0]) -> [1, 0, 0, -2147483648]
            // (RuntimeWarning; previously NumSharp returned 0 for 1/0).
            var r = np.reciprocal(np.array(new[] { 1, 2, -3, 0 }).astype(np.int32));
            Assert.AreEqual(NPTypeCode.Int32, r.typecode);
            Assert.AreEqual(1, r.GetInt32(0));
            Assert.AreEqual(0, r.GetInt32(1));
            Assert.AreEqual(0, r.GetInt32(2));
            Assert.AreEqual(int.MinValue, r.GetInt32(3));
        }

        [TestMethod]
        public void Reciprocal_Int_StridedView_NoLongerThrows()
        {
            // ReciprocalInteger walked _Unsafe.Address linearly and THREW for
            // sliced views (pre-existing defect, plan §2.10) — views now work.
            var x = np.array(new[] { 1, 9, 2, 9, 4, 9 }).astype(np.int32);
            var r = np.reciprocal(x["::2"]);

            Assert.AreEqual(1, r.GetInt32(0));
            Assert.AreEqual(0, r.GetInt32(1));
            Assert.AreEqual(0, r.GetInt32(2));
        }

        [TestMethod]
        public void Reciprocal_OutAndWhere()
        {
            // float route with out: [1, .5, .25, .125].
            var o = np.empty(new Shape(4), np.float64);
            var r = np.reciprocal(np.array(new[] { 1.0, 2, 4, 8 }), o);
            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual(0.125, o.GetDouble(3));

            // int route with out+where: masked-off keep prior.
            var oi = np.full(new Shape(4), -7, np.int32);
            np.reciprocal(np.array(new[] { 1, 2, 1, 2 }).astype(np.int32), oi, Mask4());
            var expected = new[] { 1, -7, 1, -7 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expected[i], oi.GetInt32(i));
        }

        // =====================================================================
        // where= across the families (S13/S16)
        // =====================================================================

        [TestMethod]
        public void Where_WithOut_MaskedOffKeepPrior_PerFamily()
        {
            // invert (probed): [-13, -1, -16, -1].
            var inv = np.full(new Shape(4), -1, np.int32);
            np.invert(np.array(new[] { 12, 10, 15, 1 }).astype(np.int32), inv, Mask4());
            var expectedInv = new[] { -13, -1, -16, -1 };
            for (int i = 0; i < 4; i++)
                Assert.AreEqual(expectedInv[i], inv.GetInt32(i));

            // arctan2 (probed): y=[0,1,-1,1], x=[1,0,0,1] -> [0, -1, -π/2, -1].
            var at2 = np.full(new Shape(4), -1.0, np.float64);
            np.arctan2(np.array(new[] { 0.0, 1, -1, 1 }), np.array(new[] { 1.0, 0, 0, 1 }), at2, Mask4());
            Assert.AreEqual(0.0, at2.GetDouble(0));
            Assert.AreEqual(-1.0, at2.GetDouble(1));
            Assert.AreEqual(-Math.PI / 2, at2.GetDouble(2), 1e-12);
            Assert.AreEqual(-1.0, at2.GetDouble(3));

            // sinh float-tier with out-cast composed (f8 loop → f4 out, masked).
            var sh = np.full(new Shape(4), -1.0f, np.float32);
            np.sinh(F8(), sh, Mask4());
            Assert.AreEqual((float)Math.Sinh(0), sh.GetSingle(0), 1e-6f);
            Assert.AreEqual(-1.0f, sh.GetSingle(1));
            Assert.AreEqual((float)Math.Sinh(2), sh.GetSingle(2), 1e-5f);
            Assert.AreEqual(-1.0f, sh.GetSingle(3));
        }

        [TestMethod]
        public void Where_NonBool_SafeText_PerFamily()
        {
            var intMask = np.array(new long[] { 1, 0, 1, 0 });

            foreach (var act in new Action[]
                     {
                         () => np.floor(F8(), where: intMask),
                         () => np.sinh(F8(), where: intMask),
                         () => np.sign(I4(), where: intMask),
                         () => np.arctan2(F8(), F8(), where: intMask),
                     })
            {
                var ex = Assert.ThrowsException<ArgumentException>(act);
                Assert.AreEqual(
                    "Cannot cast array data from dtype('int64') to dtype('bool') " +
                    "according to the rule 'safe'", ex.Message);
            }
        }

        // =====================================================================
        // sign / positive bool loop guards (probes §2.13 + .types)
        // =====================================================================

        [TestMethod]
        public void Sign_Bool_NoLoop_Text()
        {
            var ex = Assert.ThrowsException<TypeError>(() => np.sign(Mask4()));
            Assert.AreEqual(
                "ufunc 'sign' did not contain a loop with signature matching types " +
                "<class 'numpy.dtypes.BoolDType'> -> None", ex.Message);
        }

        [TestMethod]
        public void Positive_Bool_NoLoop_Text()
        {
            var ex = Assert.ThrowsException<TypeError>(() => np.positive(Mask4()));
            Assert.AreEqual(
                "ufunc 'positive' did not contain a loop with signature matching types " +
                "<class 'numpy.dtypes.BoolDType'> -> None", ex.Message);
        }

        // =====================================================================
        // shape rules — unary error form lists (input) (out)  (S5/S6)
        // =====================================================================

        [TestMethod]
        public void Out_WrongShape_UnaryTexts()
        {
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.sinh(F8(), np.empty(new Shape(5), np.float64)));
            Assert.AreEqual(
                "operands could not be broadcast together with shapes (4,) (5,) ",
                ex.Message);

            var ex2 = Assert.ThrowsException<ArgumentException>(() =>
                np.floor(F8(), np.empty(new Shape(1), np.float64)));
            Assert.AreEqual(
                "non-broadcastable output operand with shape (1,) doesn't match the " +
                "broadcast shape (4,)", ex2.Message);
        }

        // =====================================================================
        // strided out / multi-window / aliasing / empty (S3/S8/S9/S12)
        // =====================================================================

        [TestMethod]
        public void Out_StridedView_WritesThroughStrides()
        {
            var big = np.zeros(new Shape(8), np.float64);
            np.floor(np.array(new[] { 0.5, 1.5, 2.5, 3.5 }), big["::2"]);

            var expected = new[] { 0.0, 0, 1, 0, 2, 0, 3, 0 };
            for (int i = 0; i < 8; i++)
                Assert.AreEqual(expected[i], big.GetDouble(i));
        }

        [TestMethod]
        public void Out_FullAlias_InPlaceFloor()
        {
            // floor(x, out: x) — one pass, in place.
            var x = np.array(new[] { 0.5, 1.5, 2.5, 3.5 });
            var r = np.floor(x, x);

            Assert.IsTrue(ReferenceEquals(r, x));
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((double)i, x.GetDouble(i));
        }

        [TestMethod]
        public void Out_CastMultiWindow_20005Elements()
        {
            // Identity int loop -> f8 out across 3 buffered flush windows.
            const int n = 20_005;
            var a = np.arange(n).astype(np.int32);
            var o = np.empty(new Shape(n), np.float64);

            np.floor(a, o);
            foreach (var i in new[] { 0, 5000, 8191, 8192, 16384, 20004 })
                Assert.AreEqual((double)i, o.GetDouble(i), $"index {i}");
        }

        [TestMethod]
        public void Empty_And0d_WithOut()
        {
            var oEmpty = np.empty(new Shape(0), np.float64);
            var rEmpty = np.sinh(np.empty(new Shape(0), np.float64), oEmpty);
            Assert.IsTrue(ReferenceEquals(rEmpty, oEmpty));

            var o0 = np.empty(new Shape(), np.float64);
            var r0 = np.sinh(NDArray.Scalar(1.0), o0);
            Assert.IsTrue(ReferenceEquals(r0, o0));
            Assert.AreEqual(Math.Sinh(1.0), o0.GetDouble(), 1e-12);
        }

        // =====================================================================
        // family completeness smoke — one value-pinned out= call per op
        // =====================================================================

        [TestMethod]
        public void UnaryBatch_Family_OutSmoke()
        {
            var o = np.empty(new Shape(4), np.float64);

            np.log2(np.array(new[] { 1.0, 2, 4, 8 }), o);
            Assert.AreEqual(3.0, o.GetDouble(3));

            np.log10(np.array(new[] { 1.0, 10, 100, 1000 }), o);
            Assert.AreEqual(3.0, o.GetDouble(3));

            np.log1p(np.zeros(new Shape(4), np.float64), o);
            Assert.AreEqual(0.0, o.GetDouble(3));

            np.exp2(np.array(new[] { 0.0, 1, 2, 3 }), o);
            Assert.AreEqual(8.0, o.GetDouble(3));

            np.expm1(np.zeros(new Shape(4), np.float64), o);
            Assert.AreEqual(0.0, o.GetDouble(3));

            np.cbrt(np.array(new[] { 0.0, 1, 8, 27 }), o);
            Assert.AreEqual(3.0, o.GetDouble(3), 1e-12);

            np.cosh(np.zeros(new Shape(4), np.float64), o);
            Assert.AreEqual(1.0, o.GetDouble(3));

            np.tanh(np.zeros(new Shape(4), np.float64), o);
            Assert.AreEqual(0.0, o.GetDouble(3));

            np.arcsin(np.array(new[] { 0.0, 0, 0, 1 }), o);
            Assert.AreEqual(Math.PI / 2, o.GetDouble(3), 1e-12);

            np.arccos(np.array(new[] { 1.0, 1, 1, 0 }), o);
            Assert.AreEqual(Math.PI / 2, o.GetDouble(3), 1e-12);

            np.arctan(np.array(new[] { 0.0, 0, 0, 1 }), o);
            Assert.AreEqual(Math.PI / 4, o.GetDouble(3), 1e-12);

            np.deg2rad(np.array(new[] { 0.0, 90, 180, 360 }), o);
            Assert.AreEqual(2 * Math.PI, o.GetDouble(3), 1e-12);

            np.rad2deg(np.array(new[] { 0.0, Math.PI / 2, Math.PI, 2 * Math.PI }), o);
            Assert.AreEqual(360.0, o.GetDouble(3), 1e-12);

            np.ceil(np.array(new[] { 0.5, 1.5, 2.5, 3.5 }), o);
            Assert.AreEqual(4.0, o.GetDouble(3));

            np.trunc(np.array(new[] { 0.5, 1.5, 2.5, -3.5 }), o);
            Assert.AreEqual(-3.0, o.GetDouble(3));

            // aliases ride the same merged surface.
            np.radians(np.array(new[] { 0.0, 90, 180, 360 }), o);
            Assert.AreEqual(2 * Math.PI, o.GetDouble(3), 1e-12);

            np.degrees(np.array(new[] { 0.0, Math.PI / 2, Math.PI, 2 * Math.PI }), o);
            Assert.AreEqual(360.0, o.GetDouble(3), 1e-12);

            var oi = np.empty(new Shape(4), np.int32);
            np.bitwise_not(np.array(new[] { 12, 10, 15, 1 }).astype(np.int32), oi);
            Assert.AreEqual(-13, oi.GetInt32(0));
        }

        // =====================================================================
        // legacy positional-dtype overloads stay source-compatible
        // =====================================================================

        [TestMethod]
        public void LegacyPositionalDtype_Overloads_StillBind()
        {
            // Demoted (x, NPTypeCode) and (x, Type) forms keep compiling and
            // now honor the dtype (the log2/log10/log1p Type overloads
            // previously DROPPED it — fixed in this slice).
            var r1 = np.floor(np.array(new[] { 0.5, 1.5 }), NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, r1.typecode);

            var r2 = np.log2(np.array(new[] { 1.0, 2 }), np.float32);
            Assert.AreEqual(NPTypeCode.Single, r2.typecode);

            var r3 = np.log10(np.array(new[] { 1.0, 10 }), np.float32);
            Assert.AreEqual(NPTypeCode.Single, r3.typecode);

            var r4 = np.sinh(np.array(new[] { 0.0, 1 }), NPTypeCode.Single);
            Assert.AreEqual(NPTypeCode.Single, r4.typecode);
        }
    }
}
