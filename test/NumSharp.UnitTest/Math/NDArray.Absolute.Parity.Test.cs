using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.MathSuite
{
    /// <summary>
    ///     Exhaustive NumPy-2.4.2 parity coverage for <c>np.abs</c>/<c>np.absolute</c>
    ///     (the <c>absolute</c> ufunc). Every expectation below was probed against
    ///     real <c>numpy==2.4.2</c>. Focus areas — all previously divergent:
    ///     <list type="bullet">
    ///       <item><b>dtype= loop selection</b> — result dtype == dtype=, gated by a
    ///         same_kind input cast (asymmetric: uint8→int8 ok, int8→uint8 not).</item>
    ///       <item><b>unsigned narrowing</b> — abs(uint8 200, dtype=int8) casts BEFORE
    ///         the loop (200→-56) then takes abs → 56, NOT a bare 200/-56 copy.</item>
    ///       <item><b>complex magnitude</b> — only the D->d loop exists: dtype=complex128
    ///         → "No loop matching…"; dtype=float32/int/… → "Cannot cast…"; real input +
    ///         dtype=complex128 → "did not contain a loop … -> Complex128DType".</item>
    ///       <item><b>overflow / IEEE edges</b> — signed MIN abs wraps to MIN; -0.0→+0.0;
    ///         NaN/±inf; npy_cabs magnitude (inf if any inf component, else NaN).</item>
    ///       <item><b>layout</b> — F-contig preserved, strided/reversed → contiguous.</item>
    ///       <item><b>out=/where=</b> — out returned as-is; masked-off slots untouched.</item>
    ///     </list>
    /// </summary>
    [TestClass]
    public class NDArrayAbsoluteParityTests
    {
        // =====================================================================
        // Result dtype with no dtype= : abs preserves input dtype (complex→float64)
        // =====================================================================
        [TestMethod]
        public void NoDtype_ResultDtype_PreservesInput()
        {
            var pairs = new (NPTypeCode input, NPTypeCode expect)[]
            {
                (NPTypeCode.Boolean, NPTypeCode.Boolean),
                (NPTypeCode.SByte,   NPTypeCode.SByte),
                (NPTypeCode.Byte,    NPTypeCode.Byte),
                (NPTypeCode.Int16,   NPTypeCode.Int16),
                (NPTypeCode.UInt16,  NPTypeCode.UInt16),
                (NPTypeCode.Int32,   NPTypeCode.Int32),
                (NPTypeCode.UInt32,  NPTypeCode.UInt32),
                (NPTypeCode.Int64,   NPTypeCode.Int64),
                (NPTypeCode.UInt64,  NPTypeCode.UInt64),
                (NPTypeCode.Char,    NPTypeCode.Char),
                (NPTypeCode.Half,    NPTypeCode.Half),
                (NPTypeCode.Single,  NPTypeCode.Single),
                (NPTypeCode.Double,  NPTypeCode.Double),
                (NPTypeCode.Decimal, NPTypeCode.Decimal),
                (NPTypeCode.Complex, NPTypeCode.Double),   // |z| is float64
            };
            foreach (var (input, expect) in pairs)
            {
                var a = np.array(new double[] { 1, 2, 3 }).astype(input);
                Assert.AreEqual(expect, np.abs(a).typecode, $"abs({input}) dtype");
                Assert.AreEqual(expect, np.absolute(a).typecode, $"absolute({input}) dtype");
            }
        }

        // =====================================================================
        // dtype= widening (value + dtype): cast happens, no overflow
        // =====================================================================
        [TestMethod]
        public void Dtype_Widening_ValueAndDtype()
        {
            // int8 -128, dtype=int32 → cast to int32 first, then abs → 128 (no overflow)
            var r = np.abs(np.array(new sbyte[] { -128 }), dtype: NPTypeCode.Int32);
            Assert.AreEqual(NPTypeCode.Int32, r.typecode);
            Assert.AreEqual(128L, r.GetInt32(0));

            // int8 → float64
            var f = np.abs(np.array(new sbyte[] { -5 }), dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, f.typecode);
            Assert.AreEqual(5.0, f.GetDouble(0), 0);

            // bool → int8
            var b = np.abs(np.array(new bool[] { true, false }), dtype: NPTypeCode.SByte);
            Assert.AreEqual(NPTypeCode.SByte, b.typecode);
            Assert.AreEqual(1, b.GetSByte(0));
            Assert.AreEqual(0, b.GetSByte(1));

            // uint8 → int32 (widen, identity value)
            var u = np.abs(np.array(new byte[] { 200 }), dtype: NPTypeCode.Int32);
            Assert.AreEqual(NPTypeCode.Int32, u.typecode);
            Assert.AreEqual(200, u.GetInt32(0));
        }

        // =====================================================================
        // dtype= unsigned NARROWING/equal signed: cast-then-abs (regression bug)
        //   abs(uint8 200, dtype=int8) : 200 → int8(-56) → abs → 56  (NOT -56, NOT 200)
        // =====================================================================
        [TestMethod]
        public void Dtype_UnsignedNarrowingSigned_CastThenAbs()
        {
            var r1 = np.abs(np.array(new byte[] { 200 }), dtype: NPTypeCode.SByte);
            Assert.AreEqual(NPTypeCode.SByte, r1.typecode);
            Assert.AreEqual((sbyte)56, r1.GetSByte(0));

            var r2 = np.abs(np.array(new byte[] { 130 }), dtype: NPTypeCode.SByte);
            Assert.AreEqual((sbyte)126, r2.GetSByte(0));   // 130→-126→126

            // uint16 40000 → int16(-25536) → abs → 25536
            var r3 = np.abs(np.array(new ushort[] { 40000 }), dtype: NPTypeCode.Int16);
            Assert.AreEqual(NPTypeCode.Int16, r3.typecode);
            Assert.AreEqual((short)25536, r3.GetInt16(0));

            // uint16 200 → int8(-56) → abs → 56
            var r4 = np.abs(np.array(new ushort[] { 200 }), dtype: NPTypeCode.SByte);
            Assert.AreEqual((sbyte)56, r4.GetSByte(0));
        }

        // =====================================================================
        // dtype= identity for unsigned with no/same dtype (fast-path stays correct)
        // =====================================================================
        [TestMethod]
        public void Dtype_UnsignedIdentity_Unchanged()
        {
            var r = np.abs(np.array(new byte[] { 200, 0, 255 }), dtype: NPTypeCode.Byte);
            Assert.AreEqual(NPTypeCode.Byte, r.typecode);
            Assert.AreEqual((byte)200, r.GetByte(0));
            Assert.AreEqual((byte)0, r.GetByte(1));
            Assert.AreEqual((byte)255, r.GetByte(2));

            var r2 = np.abs(np.array(new byte[] { 7 }));   // no dtype
            Assert.AreEqual(NPTypeCode.Byte, r2.typecode);
            Assert.AreEqual((byte)7, r2.GetByte(0));
        }

        // =====================================================================
        // dtype= same_kind FAILURES → ArgumentException with NumPy's exact text
        // =====================================================================
        [TestMethod]
        public void Dtype_SameKindFailures_ThrowCannotCast()
        {
            void AssertCannotCast(Func<NDArray> act, string from, string to)
            {
                var ex = Assert.ThrowsException<ArgumentException>(() => act());
                Assert.AreEqual(
                    $"Cannot cast ufunc 'absolute' input from dtype('{from}') to dtype('{to}') with casting rule 'same_kind'",
                    ex.Message);
            }

            // signed → unsigned is NOT same_kind
            AssertCannotCast(() => np.abs(np.array(new sbyte[] { -1 }), dtype: NPTypeCode.Byte), "int8", "uint8");
            AssertCannotCast(() => np.abs(np.array(new int[] { -1 }), dtype: NPTypeCode.UInt32), "int32", "uint32");
            // any → bool is NOT same_kind
            AssertCannotCast(() => np.abs(np.array(new sbyte[] { -1 }), dtype: NPTypeCode.Boolean), "int8", "bool");
            // float → int is NOT same_kind
            AssertCannotCast(() => np.abs(np.array(new double[] { -1.5 }), dtype: NPTypeCode.Int32), "float64", "int32");
            AssertCannotCast(() => np.abs(np.array(new float[] { -1.5f }), dtype: NPTypeCode.Int64), "float32", "int64");
        }

        // =====================================================================
        // dtype=complex128 on REAL input → "did not contain a loop … -> Complex128DType"
        //   (absolute has NO complex-output loop; real→complex IS same_kind so the
        //    generic cast check would wrongly pass — this is the distinct guard)
        // =====================================================================
        [TestMethod]
        public void Dtype_Complex128_OnRealInput_NoLoopForSignature()
        {
            void AssertNoLoop(Func<NDArray> act, string fromClass)
            {
                var ex = Assert.ThrowsException<TypeError>(() => act());
                Assert.AreEqual(
                    "ufunc 'absolute' did not contain a loop with signature matching types " +
                    $"<class 'numpy.dtypes.{fromClass}'> -> <class 'numpy.dtypes.Complex128DType'>",
                    ex.Message);
            }

            AssertNoLoop(() => np.abs(np.array(new bool[] { true }), dtype: NPTypeCode.Complex), "BoolDType");
            AssertNoLoop(() => np.abs(np.array(new int[] { -1 }), dtype: NPTypeCode.Complex), "Int32DType");
            AssertNoLoop(() => np.abs(np.array(new double[] { -1.5 }), dtype: NPTypeCode.Complex), "Float64DType");
        }

        // =====================================================================
        // COMPLEX input dtype= matrix
        // =====================================================================
        [TestMethod]
        public void Complex_Dtype_OnlyFloat64Loop()
        {
            var z = np.array(new Complex[] { new Complex(3, 4) });

            // dtype=None and dtype=float64 → the D->d magnitude loop
            Assert.AreEqual(NPTypeCode.Double, np.abs(z).typecode);
            Assert.AreEqual(5.0, np.abs(z).GetDouble(0), 1e-12);
            var d = np.abs(z, dtype: NPTypeCode.Double);
            Assert.AreEqual(NPTypeCode.Double, d.typecode);
            Assert.AreEqual(5.0, d.GetDouble(0), 1e-12);

            // dtype=complex128 → no D->D loop → "No loop matching the specified signature"
            var exC = Assert.ThrowsException<IncorrectTypeException>(() => np.abs(z, dtype: NPTypeCode.Complex));
            Assert.AreEqual(
                "No loop matching the specified signature and casting was found for ufunc absolute",
                exC.Message);

            // dtype= any other real/int/bool → complex→X is never same_kind → "Cannot cast…"
            void AssertCannotCast(NPTypeCode dt, string to)
            {
                var ex = Assert.ThrowsException<ArgumentException>(() => np.abs(z, dtype: dt));
                Assert.AreEqual(
                    $"Cannot cast ufunc 'absolute' input from dtype('complex128') to dtype('{to}') with casting rule 'same_kind'",
                    ex.Message);
            }
            AssertCannotCast(NPTypeCode.Single, "float32");
            AssertCannotCast(NPTypeCode.Half, "float16");
            AssertCannotCast(NPTypeCode.Int32, "int32");
            AssertCannotCast(NPTypeCode.Boolean, "bool");
        }

        // =====================================================================
        // Signed-integer MIN abs wraps to MIN (two's-complement overflow, like NumPy)
        // =====================================================================
        [TestMethod]
        public void SignedMin_Overflow_WrapsToMin()
        {
            Assert.AreEqual((sbyte)-128, np.abs(np.array(new sbyte[] { sbyte.MinValue })).GetSByte(0));
            Assert.AreEqual(short.MinValue, np.abs(np.array(new short[] { short.MinValue })).GetInt16(0));
            Assert.AreEqual(int.MinValue, np.abs(np.array(new int[] { int.MinValue })).GetInt32(0));
            Assert.AreEqual(long.MinValue, np.abs(np.array(new long[] { long.MinValue })).GetInt64(0));

            // with a widening dtype= the cast precedes the loop → no overflow
            Assert.AreEqual(128L, np.abs(np.array(new sbyte[] { sbyte.MinValue }), dtype: NPTypeCode.Int32).GetInt32(0));
            Assert.AreEqual(2147483648L, np.abs(np.array(new int[] { int.MinValue }), dtype: NPTypeCode.Int64).GetInt64(0));
        }

        // =====================================================================
        // Float IEEE edges: -0.0 → +0.0, NaN → NaN, ±inf → +inf
        // =====================================================================
        [TestMethod]
        public void Float_SpecialValues()
        {
            var a = np.array(new double[]
            {
                -0.0, 0.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, -1.5
            });
            var r = np.abs(a);
            Assert.IsFalse(double.IsNegative(r.GetDouble(0)), "-0.0 must become +0.0");
            Assert.AreEqual(0.0, r.GetDouble(0), 0);
            Assert.IsFalse(double.IsNegative(r.GetDouble(1)));
            Assert.IsTrue(double.IsNaN(r.GetDouble(2)));
            Assert.IsTrue(double.IsPositiveInfinity(r.GetDouble(3)));
            Assert.IsTrue(double.IsPositiveInfinity(r.GetDouble(4)));
            Assert.AreEqual(1.5, r.GetDouble(5), 0);

            // float32 too
            var rf = np.abs(np.array(new float[] { -0.0f, float.NegativeInfinity, float.NaN }));
            Assert.IsFalse(float.IsNegative(rf.GetSingle(0)));
            Assert.IsTrue(float.IsPositiveInfinity(rf.GetSingle(1)));
            Assert.IsTrue(float.IsNaN(rf.GetSingle(2)));
        }

        // =====================================================================
        // Complex magnitude (npy_cabs): inf if any component inf, else NaN if any NaN
        // =====================================================================
        [TestMethod]
        public void Complex_Magnitude_SpecialValues()
        {
            var ca = np.array(new Complex[]
            {
                new Complex(3, 4),
                new Complex(-3, -4),
                new Complex(double.PositiveInfinity, 1),
                new Complex(1, double.PositiveInfinity),
                new Complex(double.NaN, 1),
                new Complex(double.PositiveInfinity, double.NaN),   // inf dominates NaN
            });
            var r = np.abs(ca);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(5.0, r.GetDouble(0), 1e-12);
            Assert.AreEqual(5.0, r.GetDouble(1), 1e-12);
            Assert.IsTrue(double.IsPositiveInfinity(r.GetDouble(2)));
            Assert.IsTrue(double.IsPositiveInfinity(r.GetDouble(3)));
            Assert.IsTrue(double.IsNaN(r.GetDouble(4)));
            Assert.IsTrue(double.IsPositiveInfinity(r.GetDouble(5)));
        }

        // =====================================================================
        // Layout: F-contig preserved; strided / negative-stride → contiguous copy
        // =====================================================================
        [TestMethod]
        public void Layout_Preservation()
        {
            // F-contiguous unsigned input → F-contiguous result (matches NumPy)
            var f = np.asfortranarray(np.array(new uint[,] { { 1, 2, 3 }, { 4, 5, 6 } }));
            var rf = np.abs(f);
            Assert.IsTrue(rf.Shape.IsFContiguous, "F-contig input → F-contig abs");
            Assert.AreEqual(5u, rf.GetUInt32(1, 1));

            // strided view abs → contiguous result with right values
            var strided = np.array(new ushort[] { 10, 20, 30, 40, 50 })["::2"];
            var rs = np.abs(strided);
            Assert.IsTrue(rs.Shape.IsContiguous);
            Assert.AreEqual((ushort)10, rs.GetUInt16(0));
            Assert.AreEqual((ushort)30, rs.GetUInt16(1));
            Assert.AreEqual((ushort)50, rs.GetUInt16(2));

            // negative-stride (reversed) signed view abs → contiguous, reversed order
            var rev = np.array(new int[] { -1, -2, -3, -4, -5 })["::-1"];
            var rr = np.abs(rev);
            Assert.IsTrue(rr.Shape.IsContiguous);
            Assert.AreEqual(5, rr.GetInt32(0));
            Assert.AreEqual(1, rr.GetInt32(4));
        }

        // =====================================================================
        // out= : returns the SAME instance, writes computed values
        // =====================================================================
        [TestMethod]
        public void Out_ReturnsSameInstance_WritesValues()
        {
            var o = np.zeros(new Shape(3), np.int64);
            var r = np.abs(np.array(new long[] { -1, -2, -3 }), o);
            Assert.IsTrue(ReferenceEquals(r, o));
            Assert.AreEqual(1L, o.GetInt64(0));
            Assert.AreEqual(2L, o.GetInt64(1));
            Assert.AreEqual(3L, o.GetInt64(2));

            // complex magnitude into a float64 out
            var of = np.zeros(new Shape(1), np.float64);
            np.abs(np.array(new Complex[] { new Complex(3, 4) }), of);
            Assert.AreEqual(5.0, of.GetDouble(0), 1e-12);
        }

        // =====================================================================
        // where= : masked-off output slots keep their prior contents
        // =====================================================================
        [TestMethod]
        public void Where_MaskedOff_KeepsPriorContents()
        {
            var o = np.array(new int[] { 9, 9, 9, 9 });
            var mask = np.array(new bool[] { true, false, true, false });
            np.abs(np.array(new int[] { -1, -2, -3, -4 }), o, where: mask);
            Assert.AreEqual(1, o.GetInt32(0));
            Assert.AreEqual(9, o.GetInt32(1));   // untouched
            Assert.AreEqual(3, o.GetInt32(2));
            Assert.AreEqual(9, o.GetInt32(3));   // untouched
        }

        // =====================================================================
        // Empty arrays and 0-d scalars
        // =====================================================================
        [TestMethod]
        public void Empty_And_Scalar()
        {
            var e = np.abs(np.zeros(new Shape(0, 3), np.float32));
            Assert.AreEqual(NPTypeCode.Single, e.typecode);
            Assert.AreEqual(0L, e.size);   // size is long
            Assert.AreEqual(2, e.ndim);
            Assert.AreEqual(0L, e.shape[0]);   // nd.shape is long[]; (0,3) preserved like NumPy
            Assert.AreEqual(3L, e.shape[1]);

            var s = np.abs(NDArray.Scalar(-3.5));
            Assert.AreEqual(0, s.ndim);
            Assert.AreEqual(3.5, s.GetDouble(), 0);
        }
    }
}
