using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    /// NumPy 2.4.2 parity for the fused 1-D inner product (numpy.dot vector·vector).
    /// Values verified against actual numpy output. Covers dtype preservation,
    /// integer wrap, bool/Complex/Decimal semantics, empty, strided views, and
    /// mixed-type promotion.
    /// </summary>
    [TestClass]
    public class np_dot_fused_test : TestClass
    {
        // numpy: same-type 1-D dot PRESERVES the input dtype (it does NOT widen like sum).
        [TestMethod]
        public void Dot1D_Int32_PreservesDtype()
        {
            var r = np.dot(np.array(new int[] { 1, 2, 3, 4 }), np.array(new int[] { 1, 2, 3, 4 }));
            Assert.AreEqual(NPTypeCode.Int32, r.typecode);   // not Int64
            Assert.AreEqual(30, r.GetAtIndex<int>(0));
        }

        [TestMethod]
        public void Dot1D_Double()
        {
            var r = np.dot(np.array(new double[] { 1, 2, 3, 4 }), np.array(new double[] { 1, 2, 3, 4 }));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(30.0, r.GetAtIndex<double>(0), 1e-12);
        }

        [TestMethod]
        public void Dot1D_Single()
        {
            var r = np.dot(np.array(new float[] { 1, 2, 3, 4 }), np.array(new float[] { 1, 2, 3, 4 }));
            Assert.AreEqual(NPTypeCode.Single, r.typecode);
            Assert.AreEqual(30f, r.GetAtIndex<float>(0), 1e-4f);
        }

        // numpy: int8 products wrap in int8 BEFORE accumulating: [100,100]·[100,100] -> 32.
        [TestMethod]
        public void Dot1D_SByte_WrapsInDtype()
        {
            var r = np.dot(np.array(new sbyte[] { 100, 100 }), np.array(new sbyte[] { 100, 100 }));
            Assert.AreEqual(NPTypeCode.SByte, r.typecode);
            Assert.AreEqual((sbyte)32, r.GetAtIndex<sbyte>(0));
        }

        // numpy: bool dot = OR over k of (a[k] AND b[k]) -> bool.
        [TestMethod]
        public void Dot1D_Bool_OrOfAnds()
        {
            var t = np.dot(np.array(new[] { true, true, false }), np.array(new[] { true, false, true }));
            Assert.AreEqual(NPTypeCode.Boolean, t.typecode);
            Assert.IsTrue(t.GetAtIndex<bool>(0));

            var f = np.dot(np.array(new[] { false, true }), np.array(new[] { true, false }));
            Assert.IsFalse(f.GetAtIndex<bool>(0));
        }

        // numpy: complex dot has NO conjugation: (1+1i)(1)+(2)(1+1i) = 3+3i.
        [TestMethod]
        public void Dot1D_Complex_NoConjugation()
        {
            var r = np.dot(np.array(new Complex[] { new(1, 1), new(2, 0) }),
                           np.array(new Complex[] { new(1, 0), new(1, 1) }));
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
            Assert.AreEqual(new Complex(3, 3), r.GetAtIndex<Complex>(0));
        }

        [TestMethod]
        public void Dot1D_Decimal()
        {
            var r = np.dot(np.array(new decimal[] { 1.5m, 2.5m }), np.array(new decimal[] { 2m, 4m }));
            Assert.AreEqual(13m, r.GetAtIndex<decimal>(0));
        }

        // numpy: empty dot -> scalar 0 of the INPUT dtype (int32 stays int32, not widened).
        [TestMethod]
        public void Dot1D_Empty_PreservesDtype()
        {
            var rd = np.dot(np.array(new double[] { }), np.array(new double[] { }));
            Assert.AreEqual(NPTypeCode.Double, rd.typecode);
            Assert.AreEqual(0.0, rd.GetAtIndex<double>(0), 0);

            var ri = np.dot(np.array(new int[] { }), np.array(new int[] { }));
            Assert.AreEqual(NPTypeCode.Int32, ri.typecode);
        }

        // Stride-aware: sliced/reversed 1-D views are consumed in place (no copy).
        [TestMethod]
        public void Dot1D_StridedAndReversed()
        {
            var a = np.arange(10.0);
            Assert.AreEqual(120.0, np.dot(a["::2"], a["::2"]).GetAtIndex<double>(0), 1e-9);  // 0+4+16+36+64
            Assert.AreEqual(120.0, np.dot(a["::-1"], a).GetAtIndex<double>(0), 1e-9);
        }

        // Mixed dtype -> NEP50 promotion (fallback path), result dtype = promoted.
        [TestMethod]
        public void Dot1D_MixedType_Promotes()
        {
            var r1 = np.dot(np.array(new int[] { 1, 2, 3, 4 }), np.array(new long[] { 1, 2, 3, 4 }));
            Assert.AreEqual(NPTypeCode.Int64, r1.typecode);
            Assert.AreEqual(30L, r1.GetAtIndex<long>(0));

            var r2 = np.dot(np.array(new int[] { 1, 2, 3, 4 }), np.array(new double[] { 1, 2, 3, 4 }));
            Assert.AreEqual(NPTypeCode.Double, r2.typecode);
        }

        // numpy: shape mismatch -> error with "not aligned" message.
        [TestMethod]
        public void Dot1D_ShapeMismatch_Throws()
        {
            try
            {
                np.dot(np.array(new double[] { 1, 2, 3 }), np.array(new double[] { 1, 2 }));
                Assert.Fail("expected a shape-mismatch exception");
            }
            catch (AssertFailedException) { throw; }
            catch (Exception e)
            {
                StringAssert.Contains(e.Message, "not aligned");
            }
        }
    }
}
