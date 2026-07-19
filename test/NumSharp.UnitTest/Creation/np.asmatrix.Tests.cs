using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Tests for np.asmatrix — verified 1-to-1 against NumPy 2.4.2. Coerces the input to a 2-D array
    /// (NumSharp has no matrix subclass): 0-D → (1,1), 1-D of length N → (1,N), 2-D unchanged, and
    /// &gt;2-D drops length-1 axes and must land on exactly 2-D else ValueError. Coercion is a
    /// stride-preserving view (no copy) unless a dtype change forces a cast. Also parses matrix
    /// strings ("1 2; 3 4").
    /// </summary>
    [TestClass]
    public class np_asmatrix_Tests
    {
        // ─── dimensional coercion ───────────────────────────────────────────

        [TestMethod]
        public void Scalar_Becomes_1x1()
        {
            var m = np.asmatrix(NDArray.Scalar(5.0));
            m.shape.Should().Equal(new long[] { 1, 1 });
            m.GetValue<double>(0, 0).Should().Be(5.0);
        }

        [TestMethod]
        public void OneD_Becomes_1xN()
        {
            var m = np.asmatrix(np.array(new[] { 1L, 2L, 3L }));
            m.shape.Should().Equal(new long[] { 1, 3 });
            m.typecode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void TwoD_Unchanged()
        {
            var m = np.asmatrix(np.array(new[,] { { 1L, 2L }, { 3L, 4L } }));
            m.shape.Should().Equal(new long[] { 2, 2 });
        }

        [TestMethod]
        public void ThreeD_AllAxesGreaterThanOne_Raises() =>
            ((Action)(() => np.asmatrix(np.zeros(new Shape(2, 2, 2), NPTypeCode.Double))))
                .Should().Throw<ValueError>().WithMessage("shape too large to be a matrix.*");

        [TestMethod]
        public void ThreeD_LeadingSingleton_Squeezes()
        {
            np.asmatrix(np.zeros(new Shape(1, 2, 3), NPTypeCode.Double)).shape.Should().Equal(new long[] { 2, 3 });
        }

        [TestMethod]
        public void ThreeD_TrailingSingleton_Squeezes()
        {
            np.asmatrix(np.zeros(new Shape(2, 3, 1), NPTypeCode.Double)).shape.Should().Equal(new long[] { 2, 3 });
        }

        [TestMethod]
        public void ThreeD_TwoSingletons_BecomesRowVector()
        {
            np.asmatrix(np.zeros(new Shape(1, 1, 5), NPTypeCode.Double)).shape.Should().Equal(new long[] { 1, 5 });
        }

        [TestMethod]
        public void FourD_OuterAndInnerSingletons_Squeezes()
        {
            np.asmatrix(np.zeros(new Shape(1, 2, 2, 1), NPTypeCode.Double)).shape.Should().Equal(new long[] { 2, 2 });
        }

        // ─── empty arrays ───────────────────────────────────────────────────

        [TestMethod]
        public void Empty_2D_Preserved()
        {
            np.asmatrix(np.zeros(new Shape(0, 3), NPTypeCode.Double)).shape.Should().Equal(new long[] { 0, 3 });
        }

        [TestMethod]
        public void Empty_1D_Becomes_1x0()
        {
            np.asmatrix(np.zeros(new Shape(0), NPTypeCode.Double)).shape.Should().Equal(new long[] { 1, 0 });
        }

        // ─── view (no-copy) semantics ───────────────────────────────────────

        [TestMethod]
        public void TwoD_SharesMemory()
        {
            var x = np.array(new[,] { { 1L, 2L }, { 3L, 4L } });
            var m = np.asmatrix(x);
            x[0, 0] = 99L;
            m.GetValue<long>(0, 0).Should().Be(99L); // shared memory
        }

        [TestMethod]
        public void Transposed_SharesMemory_AndKeepsShape()
        {
            var v = np.arange(12).reshape(3, 4); // arange is Int64 in NumSharp
            var m = np.asmatrix(v.T);
            m.shape.Should().Equal(new long[] { 4, 3 });
            m[0, 0] = -5L;
            v.GetValue<long>(0, 0).Should().Be(-5L); // write propagates through the view
        }

        [TestMethod]
        public void Strided1D_SharesMemory()
        {
            var w = np.arange(10)["::2"];
            var m = np.asmatrix(w);
            m.shape.Should().Equal(new long[] { 1, 5 });
            m[0, 0] = -7L;
            w.GetValue<long>(0).Should().Be(-7L);
        }

        [TestMethod]
        public void Reversed1D_SharesMemory_AndOrder()
        {
            var r = np.arange(6)["::-1"]; // 5,4,3,2,1,0
            var m = np.asmatrix(r);
            m.shape.Should().Equal(new long[] { 1, 6 });
            m.GetValue<long>(0, 0).Should().Be(5L); // first element of the reversed view
        }

        // ─── dtype ──────────────────────────────────────────────────────────

        [TestMethod]
        public void Dtype_Casts_AndCoerces()
        {
            var m = np.asmatrix(np.array(new[] { 1, 2, 3 }), typeof(float));
            m.dtype.Should().Be(typeof(float));
            m.shape.Should().Equal(new long[] { 1, 3 });
        }

        [TestMethod]
        public void Dtype_NPTypeCode_Overload()
        {
            np.asmatrix(np.array(new[] { 1, 2, 3 }), NPTypeCode.Double).dtype.Should().Be(typeof(double));
        }

        // ─── string matrix parser ───────────────────────────────────────────

        [TestMethod]
        public void String_IntegerMatrix()
        {
            var m = np.asmatrix("1 2; 3 4");
            m.shape.Should().Equal(new long[] { 2, 2 });
            m.typecode.Should().Be(NPTypeCode.Int64);
            m.GetValue<long>(0, 0).Should().Be(1L);
            m.GetValue<long>(1, 1).Should().Be(4L);
        }

        [TestMethod]
        public void String_SingleRow_CommaSeparated()
        {
            np.asmatrix("1,2,3").shape.Should().Equal(new long[] { 1, 3 });
        }

        [TestMethod]
        public void String_FloatMatrix_InfersDouble()
        {
            var m = np.asmatrix("1.5 2");
            m.typecode.Should().Be(NPTypeCode.Double);
            m.GetValue<double>(0, 0).Should().Be(1.5);
        }

        [TestMethod]
        public void String_BracketsStripped()
        {
            np.asmatrix("[1 2; 3 4]").shape.Should().Equal(new long[] { 2, 2 });
        }

        [TestMethod]
        public void String_RaggedRows_Raises() =>
            ((Action)(() => np.asmatrix("1 2; 3 4 5")))
                .Should().Throw<ValueError>().WithMessage("Rows not the same size.*");

        [TestMethod]
        public void String_WithDtype_Casts()
        {
            np.asmatrix("1 2; 3 4", typeof(float)).dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void NullNDArrayInput_Throws() =>
            ((Action)(() => np.asmatrix((NDArray)null))).Should().Throw<ArgumentNullException>();

        [TestMethod]
        public void NullStringInput_Throws() =>
            ((Action)(() => np.asmatrix((string)null))).Should().Throw<ArgumentNullException>();

        // ─── second-pass edge cases (verified against NumPy) ────────────────

        [TestMethod]
        public void String_Empty_InfersDouble()
        {
            // NumPy's np.array([[]]) defaults an element-free matrix to float64.
            var m = np.asmatrix("");
            m.shape.Should().Equal(new long[] { 1, 0 });
            m.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void String_WhitespaceOnly_InfersDouble() =>
            np.asmatrix("   ").typecode.Should().Be(NPTypeCode.Double);

        [TestMethod]
        public void String_ScientificAndSigns()
        {
            np.asmatrix("1e3 2").typecode.Should().Be(NPTypeCode.Double);       // exponent ⇒ float
            np.asmatrix(".5 1").typecode.Should().Be(NPTypeCode.Double);        // leading dot ⇒ float
            var signed = np.asmatrix("+5 -3");                                  // signed integers stay int
            signed.typecode.Should().Be(NPTypeCode.Int64);
            signed.GetValue<long>(0, 0).Should().Be(5L);
            signed.GetValue<long>(0, 1).Should().Be(-3L);
        }

        [TestMethod]
        public void String_MultiRow_ThreeByTwo()
        {
            np.asmatrix("1 2;3 4;5 6").shape.Should().Equal(new long[] { 3, 2 });
        }

        [TestMethod]
        public void Degenerate_EmptyThreeD_Raises() =>
            // (0,3,1) squeezes to (3,) but the element count (0) can't fill it — NumPy raises a reshape error.
            ((Action)(() => np.asmatrix(np.zeros(new Shape(0, 3, 1), NPTypeCode.Double))))
                .Should().Throw<ValueError>();

        [TestMethod]
        public void MultipleSingletonAxes_Squeeze()
        {
            np.asmatrix(np.zeros(new Shape(1, 3, 1, 4, 1), NPTypeCode.Double)).shape.Should().Equal(new long[] { 3, 4 });
        }

        [TestMethod]
        public void OneElement1D_Becomes_1x1()
        {
            np.asmatrix(np.zeros(new Shape(1), NPTypeCode.Double)).shape.Should().Equal(new long[] { 1, 1 });
        }

        [TestMethod]
        public void ZeroDFromIntegerIndex_PreservesOffset()
        {
            var baseA = np.array(new[,] { { 10.0, 20.0 }, { 30.0, 40.0 } });
            var m = np.asmatrix(baseA[1, 1]); // 0-D view at offset pointing to 40
            m.shape.Should().Equal(new long[] { 1, 1 });
            m.GetValue<double>(0, 0).Should().Be(40.0);
        }

        [TestMethod]
        public void NonContiguous2D_SharesMemory()
        {
            var a = np.arange(16).reshape(4, 4);
            var m = np.asmatrix(a["::2, ::2"]); // strided in both axes, neither C nor F
            m.shape.Should().Equal(new long[] { 2, 2 });
            m[0, 0] = -99L;
            a.GetValue<long>(0, 0).Should().Be(-99L);
        }

        [TestMethod]
        public void Broadcast1D_MatrixIsNonWriteable()
        {
            var bc = np.broadcast_to(NDArray.Scalar(5.0), new Shape(4)); // non-writeable
            var m = np.asmatrix(bc);
            m.shape.Should().Equal(new long[] { 1, 4 });
            m.Shape.IsWriteable.Should().BeFalse();
        }
    }
}
