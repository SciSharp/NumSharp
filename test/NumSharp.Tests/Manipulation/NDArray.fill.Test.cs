using System;
using System.Numerics;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     <see cref="NDArray.fill"/> — in-place scalar fill. Behavior probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class NDArray_fill_Test
    {
        // ---- basic value + dtype coverage -------------------------------------------------

        [TestMethod]
        public void Fill_Int32_Contiguous()
        {
            var a = np.zeros(new Shape(5), NPTypeCode.Int32);
            a.fill(7);
            for (int i = 0; i < 5; i++) a.GetInt32(i).Should().Be(7);
        }

        [TestMethod]
        public void Fill_AllDtypes_Basic()
        {
            foreach (var tc in new[]
            {
                NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
                NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double,
                NPTypeCode.Decimal, NPTypeCode.Complex
            })
            {
                var a = np.zeros(new Shape(4), tc);
                a.fill(3);
                // Compare dtype-agnostically against a value-3 array (Half/Complex are not IConvertible).
                np.array_equal(a, np.full(new Shape(4), 3, tc)).Should().BeTrue($"dtype {tc} fill(3)");
            }
        }

        [TestMethod]
        public void Fill_Bool_NonzeroIsTrue()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.Boolean);
            a.fill(5);
            for (int i = 0; i < 3; i++) a.GetBoolean(i).Should().BeTrue();
            a.fill(0);
            for (int i = 0; i < 3; i++) a.GetBoolean(i).Should().BeFalse();
        }

        [TestMethod]
        public void Fill_Complex_FromScalarAndComplex()
        {
            var a = np.zeros(new Shape(2), NPTypeCode.Complex);
            a.fill(2);
            a.GetAtIndex(0).Should().Be(new Complex(2, 0));
            a.fill(new Complex(2, 3));
            a.GetAtIndex(1).Should().Be(new Complex(2, 3));
        }

        // ---- casting: NEP50 weak-scalar rules ---------------------------------------------

        [TestMethod]
        public void Fill_FloatIntoInt_TruncatesTowardZero()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.Int32);
            a.fill(3.9);
            a.GetInt32(0).Should().Be(3);       // 3.9 -> 3, not rounded to 4
            a.fill(-3.9);
            a.GetInt32(0).Should().Be(-3);      // -3.9 -> -3
        }

        [TestMethod]
        public void Fill_IntOutOfRange_RaisesOverflow_Int8()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.SByte);
            Action act = () => a.fill(300);
            act.Should().Throw<OverflowException>().WithMessage("*Python integer 300 out of bounds for int8*");
        }

        [TestMethod]
        public void Fill_NegativeIntoUnsigned_RaisesOverflow()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.Byte);
            Action act = () => a.fill(-1);
            act.Should().Throw<OverflowException>().WithMessage("*Python integer -1 out of bounds for uint8*");
        }

        [TestMethod]
        public void Fill_FloatOutOfRangeIntoInt_TruncatesThenRaises()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.SByte);
            Action act = () => a.fill(300.0);
            act.Should().Throw<OverflowException>().WithMessage("*300 out of bounds for int8*");
        }

        [TestMethod]
        public void Fill_NonFiniteFloatIntoInt_Raises()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.Int32);
            ((Action)(() => a.fill(double.NaN))).Should().Throw<OverflowException>();
            ((Action)(() => a.fill(double.PositiveInfinity))).Should().Throw<OverflowException>();
        }

        [TestMethod]
        public void Fill_FloatSaturatesToInfinity()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.Single);
            a.fill(1e300);
            float.IsPositiveInfinity(a.GetSingle(0)).Should().BeTrue();

            var h = np.zeros(new Shape(3), NPTypeCode.Half);
            h.fill(70000.0);
            Half.IsPositiveInfinity((Half)h.GetAtIndex(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Fill_Float32_ExactAndSpecials()
        {
            var a = np.zeros(new Shape(2), NPTypeCode.Single);
            a.fill(1.5);
            a.GetSingle(0).Should().Be(1.5f);
            a.fill(float.NaN);
            float.IsNaN(a.GetSingle(0)).Should().BeTrue();
        }

        // ---- strong scalar / sequence / null ----------------------------------------------

        [TestMethod]
        public void Fill_StrongScalar0dNDArray_Wraps()
        {
            // A 0-d NDArray is a STRONG scalar: it WRAPS on cast (like np.int64(-1) -> uint8 -> 255),
            // unlike a weak C# -1 which raises.
            var a = np.zeros(new Shape(2), NPTypeCode.Byte);
            a.fill(NDArray.Scalar((long)-1));
            a.GetByte(0).Should().Be(255);
            a.GetByte(1).Should().Be(255);
        }

        [TestMethod]
        public void Fill_ScalarNDArray0d_Works()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.Int32);
            a.fill(np.array(7));
            for (int i = 0; i < 3; i++) a.GetInt32(i).Should().Be(7);
        }

        [TestMethod]
        public void Fill_MultiElementArray_IsSequenceError()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.Int32);
            Action act = () => a.fill(np.array(new[] { 1, 2, 3 }));
            act.Should().Throw<ValueError>().WithMessage("*setting an array element with a sequence*");
        }

        [TestMethod]
        public void Fill_Null_Throws()
        {
            var a = np.zeros(new Shape(3), NPTypeCode.Int32);
            Action act = () => a.fill(null);
            act.Should().Throw<ArgumentNullException>();
        }

        // ---- write-through across memory layouts ------------------------------------------

        [TestMethod]
        public void Fill_StridedRowsView_WritesThrough()
        {
            var m = np.arange(9).astype(np.int32).reshape(3, 3);
            m["::2"].fill(5);
            // rows 0 and 2 become 5; row 1 unchanged
            m.GetInt32(0, 0).Should().Be(5); m.GetInt32(0, 2).Should().Be(5);
            m.GetInt32(1, 0).Should().Be(3);
            m.GetInt32(2, 1).Should().Be(5);
        }

        [TestMethod]
        public void Fill_Transposed_WritesThrough()
        {
            var m = np.arange(9).astype(np.int32).reshape(3, 3);
            m.T.fill(7);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    m.GetInt32(i, j).Should().Be(7);
        }

        [TestMethod]
        public void Fill_ColumnView_WritesThrough()
        {
            var m = np.arange(9).astype(np.int32).reshape(3, 3);
            m[":, 1"].fill(-1);
            m.GetInt32(0, 1).Should().Be(-1);
            m.GetInt32(1, 1).Should().Be(-1);
            m.GetInt32(2, 1).Should().Be(-1);
            m.GetInt32(0, 0).Should().Be(0);   // neighbors untouched
        }

        [TestMethod]
        public void Fill_NegativeStride_WritesThrough()
        {
            var r = np.arange(6).astype(np.int32);
            r["::-1"].fill(3);
            for (int i = 0; i < 6; i++) r.GetInt32(i).Should().Be(3);
        }

        [TestMethod]
        public void Fill_FortranContiguous_WritesThrough()
        {
            var f = np.asfortranarray(np.arange(6).astype(np.int32).reshape(2, 3));
            f.fill(9);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    f.GetInt32(i, j).Should().Be(9);
        }

        [TestMethod]
        public void Fill_OffsetContiguousRow_DoesNotCorruptNeighbors()
        {
            var parent = np.arange(12).astype(np.int32).reshape(3, 4);
            parent["1"].fill(99);   // contiguous view at offset 4
            for (int j = 0; j < 4; j++) parent.GetInt32(1, j).Should().Be(99);
            parent.GetInt32(0, 3).Should().Be(3);   // row above untouched
            parent.GetInt32(2, 0).Should().Be(8);   // row below untouched
        }

        // ---- ordering + edge cases --------------------------------------------------------

        [TestMethod]
        public void Fill_ReadOnly_RaisesBeforeAnythingElse()
        {
            var b = np.broadcast_to(np.array(new[] { 0 }).reshape(1, 1), new Shape(3, 3));
            Action act = () => b.fill(300);   // bad value too, but read-only fires first
            act.Should().Throw<NumSharpException>().WithMessage("*read-only*");
        }

        [TestMethod]
        public void Fill_Empty_OutOfRangeValue_StillRaises()
        {
            // NumPy packs the scalar before checking size, so an empty array still validates the value.
            var a = np.zeros(new Shape(0, 3), NPTypeCode.SByte);
            Action act = () => a.fill(300);
            act.Should().Throw<OverflowException>().WithMessage("*out of bounds for int8*");
        }

        [TestMethod]
        public void Fill_Empty_ValidValue_NoOp()
        {
            var a = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
            a.fill(5);
            a.size.Should().Be(0);
        }

        [TestMethod]
        public void Fill_ScalarZeroD_WritesThrough()
        {
            var a = np.array(9).astype(np.int32);   // 0-d
            a.fill(4);
            a.GetInt32(0).Should().Be(4);
        }
    }
}
