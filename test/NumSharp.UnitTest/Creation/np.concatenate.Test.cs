using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class np_concatenate_test : TestClass
    {
        [TestMethod]
        public void Case1_Axis1()
        {
            var a = np.full(new Shape(3, 1, 3), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 2, 3), 2, NPTypeCode.Int32);
            var c = np.concatenate((a, b), 1);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, 0, :"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c[":, 1, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c[":, 2, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case1_Axis1_Cast()
        {
            var a = np.full(new Shape(3, 1, 3), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 2, 3), 2, NPTypeCode.Double);
            var c = np.concatenate((a, b), 1);

            c.dtype.Should().Be<double>();
            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, 0, :"].flat.Cast<double>().Should().AllBeEquivalentTo(1);
            c[":, 1, :"].flat.Cast<double>().Should().AllBeEquivalentTo(2);
            c[":, 2, :"].flat.Cast<double>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case1_Axis0()
        {
            var a = np.full(new Shape(1, 3, 3), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(2, 3, 3), 2, NPTypeCode.Int32);
            var c = np.concatenate((a, b), 0);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c["0, :, :"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c["1, :, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c["2, :, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case1_Axis2()
        {
            var a = np.full(new Shape(3, 3, 1), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 3, 2), 2, NPTypeCode.Int32);
            var c = np.concatenate((a, b), 2);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, :, 0"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c[":, :, 1"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c[":, :, 2"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case1_Axis_minus1()
        {
            var a = np.full(new Shape(3, 3, 1), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 3, 2), 2, NPTypeCode.Int32);
            var c = np.concatenate((a, b), -1);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, :, 0"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c[":, :, 1"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c[":, :, 2"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [TestMethod]
        public void Case2_Axis1_3Arrays_Cast()
        {
            var a = np.full(new Shape(3, 1, 3), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 2, 3), 2, NPTypeCode.Decimal);
            var c = np.full(new Shape(3, 1, 3), 2, NPTypeCode.Byte);
            var d = np.concatenate((a, b, c), 1);
            d.dtype.Should().Be<decimal>();
            d.shape.Should().HaveCount(3).And.ContainInOrder(3, 4, 3);
            d.size.Should().Be(36);
            d[":, 0, :"].flat.Cast<decimal>().Should().AllBeEquivalentTo(1);
            d[":, 1, :"].flat.Cast<decimal>().Should().AllBeEquivalentTo(2);
            d[":, 2, :"].flat.Cast<decimal>().Should().AllBeEquivalentTo(2);
        }

        /// <summary>
        ///     hstack with column-broadcast arrays correctly preserves row data.
        ///     Verifies: broadcast_to column vectors, then hstack horizontally.
        ///     NumPy: np.hstack([np.broadcast_to([[1],[2]],(2,2)), np.broadcast_to([[3],[4]],(2,3))])
        ///            → [[1,1,3,3,3], [2,2,4,4,4]]
        /// </summary>
        [TestMethod]
        public void Hstack_BroadcastColumnVectors()
        {
            var a = np.broadcast_to(np.array(new int[,] { { 1 }, { 2 } }), new Shape(2, 2));
            var b = np.broadcast_to(np.array(new int[,] { { 3 }, { 4 } }), new Shape(2, 3));
            var r = np.hstack(a, b);

            r.shape.Should().BeEquivalentTo(new long[] { 2, 5 });

            var expected = np.array(new int[,] { { 1, 1, 3, 3, 3 }, { 2, 2, 4, 4, 4 } });
            np.array_equal(r, expected).Should().BeTrue(
                "hstack with broadcast arrays should preserve row data correctly");
        }

        /// <summary>
        ///     vstack with sliced+broadcast array correctly reads all rows.
        ///     Verifies: slice column, broadcast, then vstack vertically.
        ///     NumPy: x = np.arange(6).reshape(3,2)
        ///            np.vstack([np.broadcast_to(x[:,0:1],(3,3)), [[10,20,30]]])
        ///            → [[0,0,0], [2,2,2], [4,4,4], [10,20,30]]
        /// </summary>
        [TestMethod]
        public void Vstack_SlicedBroadcast()
        {
            var x = np.arange(6).reshape(3, 2);
            var col = x[":, 0:1"]; // (3,1): [[0],[2],[4]]
            var bcol = np.broadcast_to(col, new Shape(3, 3));
            var other = np.array(new int[,] { { 10, 20, 30 } });

            var r = np.vstack(bcol, other);
            r.shape.Should().BeEquivalentTo(new long[] { 4, 3 });

            r.GetInt64(0, 0).Should().Be(0L, "Row 0 should be [0,0,0]");
            r.GetInt64(1, 0).Should().Be(2L, "Row 1 should be [2,2,2]");
            r.GetInt64(2, 0).Should().Be(4L, "Row 2 should be [4,4,4]");
            r.GetInt64(3, 0).Should().Be(10L, "Row 3 should be [10,20,30]");
        }

        /// <summary>
        ///     concatenate axis=0 with sliced+broadcast correctly reads all data.
        ///     Verifies: slice non-first column, broadcast, then concatenate.
        ///     NumPy: x = np.arange(12).reshape(3,4)
        ///            np.concatenate([np.broadcast_to(x[:,1:2],(3,3)), np.ones((1,3),dtype=int)])
        ///            → [[1,1,1], [5,5,5], [9,9,9], [1,1,1]]
        /// </summary>
        [TestMethod]
        public void Concatenate_SlicedBroadcast_Axis0()
        {
            var x = np.arange(12).reshape(3, 4);
            var col = x[":, 1:2"]; // (3,1): [[1],[5],[9]]
            var bcol = np.broadcast_to(col, new Shape(3, 3));
            var other = np.ones(new Shape(1, 3), np.int32);

            var r = np.concatenate(new NDArray[] { bcol, other }, axis: 0);
            r.shape.Should().BeEquivalentTo(new long[] { 4, 3 });

            r.GetInt64(0, 0).Should().Be(1L, "Row 0 should be [1,1,1]");
            r.GetInt64(1, 0).Should().Be(5L, "Row 1 should be [5,5,5]");
            r.GetInt64(2, 0).Should().Be(9L, "Row 2 should be [9,9,9]");
            r.GetInt64(3, 0).Should().Be(1L, "Row 3 should be [1,1,1]");
        }

        // ================================================================
        // NumPy 2.x parity: NEP50 promotion, out=, dtype=, casting=, axis=None
        // ================================================================

        // -- NEP50 promotion (T1.8) --

        [TestMethod]
        public void NEP50_Float32_Int64_PromotesToFloat64()
        {
            // python: np.concatenate([np.float32([1]), np.int64([2])]).dtype == float64
            var r = np.concatenate(new[] { np.array(new float[] { 1f }), np.array(new long[] { 2L }) });
            r.typecode.Should().Be(NPTypeCode.Double);
            r.Data<double>().Should().Equal(1.0, 2.0);
        }

        [TestMethod]
        public void NEP50_Int8_UInt8_PromotesToInt16()
        {
            // python: np.concatenate([np.int8([1]), np.uint8([2])]).dtype == int16
            var r = np.concatenate(new[] { np.array(new sbyte[] { 1 }), np.array(new byte[] { 2 }) });
            r.typecode.Should().Be(NPTypeCode.Int16);
        }

        [TestMethod]
        public void NEP50_Half_Single_PromotesToSingle()
        {
            // python: np.concatenate([np.float16([1]), np.float32([2])]).dtype == float32
            var r = np.concatenate(new[] { np.array(new Half[] { (Half)1f }), np.array(new float[] { 2f }) });
            r.typecode.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void NEP50_Complex_Double_PromotesToComplex()
        {
            var c = np.array(new System.Numerics.Complex[] { new(1, 0) });
            var d = np.array(new double[] { 2.0 });
            var r = np.concatenate(new[] { c, d });
            r.typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void NEP50_Mixed_SByte_Half_Complex_PromotesToComplex()
        {
            // T1.9 regression: previously crashed for mixed dtypes including SByte/Half/Complex.
            var a = np.array(new sbyte[] { 1 });
            var b = np.array(new Half[] { (Half)2f });
            var c = np.array(new System.Numerics.Complex[] { new(3, 0) });
            var r = np.concatenate(new[] { a, b, c });
            r.typecode.Should().Be(NPTypeCode.Complex);
        }

        // -- axis=None (flatten) --

        [TestMethod]
        public void AxisNone_Flattens2DInputs()
        {
            // python: np.concatenate([[[1,2],[3,4]], [[5,6]]], axis=None) → [1,2,3,4,5,6]
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var b = np.array(new int[,] { { 5, 6 } });
            var r = np.concatenate(new[] { a, b }, axis: null);
            r.ndim.Should().Be(1);
            r.shape[0].Should().Be(6);
            r.Data<int>().Should().Equal(1, 2, 3, 4, 5, 6);
        }

        [TestMethod]
        public void AxisNone_SingleArrayReturnsFlatCopy()
        {
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var r = np.concatenate(new[] { a }, axis: null);
            r.ndim.Should().Be(1);
            r.Data<int>().Should().Equal(1, 2, 3, 4);
        }

        // -- dtype= override --

        [TestMethod]
        public void Dtype_OverridesPromotion()
        {
            var r = np.concatenate(
                new[] { np.array(new int[] { 1, 2 }), np.array(new int[] { 3, 4 }) },
                dtype: NPTypeCode.Double);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.Data<double>().Should().Equal(1.0, 2.0, 3.0, 4.0);
        }

        [TestMethod]
        public void Dtype_DownCastUnsafe()
        {
            // int64 → int32 needs unsafe casting.
            var r = np.concatenate(
                new[] { np.array(new long[] { 1L, 2L }), np.array(new long[] { 3L, 4L }) },
                dtype: NPTypeCode.Int32,
                casting: "unsafe");
            r.typecode.Should().Be(NPTypeCode.Int32);
            r.Data<int>().Should().Equal(1, 2, 3, 4);
        }

        // -- out= --

        [TestMethod]
        public void Out_WritesIntoProvidedBuffer()
        {
            var dst = np.zeros(new Shape(4), NPTypeCode.Int32);
            var r = np.concatenate(
                new[] { np.array(new int[] { 1, 2 }), np.array(new int[] { 3, 4 }) },
                @out: dst);
            r.Should().BeSameAs(dst);
            dst.Data<int>().Should().Equal(1, 2, 3, 4);
        }

        [TestMethod]
        public void Out_WrongShape_Throws()
        {
            var dst = np.zeros(new Shape(5), NPTypeCode.Int32); // wrong size
            Action act = () => np.concatenate(
                new[] { np.array(new int[] { 1, 2 }), np.array(new int[] { 3, 4 }) },
                @out: dst);
            act.Should().Throw<IncorrectShapeException>().WithMessage("*wrong shape*");
        }

        [TestMethod]
        public void OutPlusDtype_Throws()
        {
            var dst = np.zeros(new Shape(2), NPTypeCode.Int32);
            Action act = () => np.concatenate(
                new[] { np.array(new int[] { 1, 2 }) },
                @out: dst,
                dtype: NPTypeCode.Int32);
            act.Should().Throw<ArgumentException>().WithMessage("*only takes*out*dtype*");
        }

        // -- casting= --

        [TestMethod]
        public void Casting_DefaultSameKind_BlocksFloatToInt()
        {
            // NumPy: TypeError under default same_kind for float→int.
            var dst = np.zeros(new Shape(3), NPTypeCode.Int32);
            Action act = () => np.concatenate(
                new[] { np.array(new double[] { 1.5, 2.5 }), np.array(new double[] { 3.5 }) },
                @out: dst);
            act.Should().Throw<InvalidCastException>().WithMessage("*same_kind*");
        }

        [TestMethod]
        public void Casting_Unsafe_AllowsFloatToInt()
        {
            var dst = np.zeros(new Shape(3), NPTypeCode.Int32);
            np.concatenate(
                new[] { np.array(new double[] { 1.5, 2.5 }), np.array(new double[] { 3.5 }) },
                @out: dst,
                casting: "unsafe");
            dst.Data<int>().Should().Equal(1, 2, 3);
        }

        [TestMethod]
        public void Casting_InvalidName_Throws()
        {
            Action act = () => np.concatenate(
                new[] { np.array(new int[] { 1 }) },
                casting: "bogus");
            act.Should().Throw<ArgumentException>().WithMessage("*casting*");
        }

        // -- Edge cases --

        [TestMethod]
        public void ZeroDimensional_Throws()
        {
            // NumPy: ValueError "zero-dimensional arrays cannot be concatenated"
            Action act = () => np.concatenate(new[] { np.array(1), np.array(2) });
            act.Should().Throw<ArgumentException>().WithMessage("*zero-dimensional*");
        }

        [TestMethod]
        public void AxisOutOfRange_Throws()
        {
            Action act = () => np.concatenate(
                new[] { np.array(new int[,] { { 1, 2 }, { 3, 4 } }), np.array(new int[,] { { 5, 6 } }) },
                axis: 5);
            act.Should().Throw<ArgumentOutOfRangeException>().WithMessage("*out of bounds*");
        }

        [TestMethod]
        public void NdimMismatch_Throws()
        {
            Action act = () => np.concatenate(
                new[] { np.array(new int[] { 1, 2 }), np.array(new int[,] { { 3, 4 } }) });
            act.Should().Throw<IncorrectShapeException>()
                .WithMessage("*same number of dimensions*");
        }

        [TestMethod]
        public void NonAxisDimMismatch_Throws()
        {
            Action act = () => np.concatenate(
                new[] { np.array(new int[,] { { 1, 2 } }), np.array(new int[,] { { 3, 4, 5 } }) });
            act.Should().Throw<IncorrectShapeException>().WithMessage("*must match exactly*");
        }

        [TestMethod]
        public void EmptyArray_PreservesNonEmptyData()
        {
            var r = np.concatenate(new[] {
                np.array(new double[] { 1.0, 2.0 }),
                np.array(new double[] { })
            });
            r.Data<double>().Should().Equal(1.0, 2.0);
        }

        [TestMethod]
        public void SingleArray_ReturnsCopy()
        {
            // NumPy: np.concatenate([a]) is NOT a.
            var a = np.array(new int[] { 1, 2, 3 });
            var r = np.concatenate(new[] { a });
            ReferenceEquals(r, a).Should().BeFalse("NumPy returns a fresh array");
            r.Data<int>().Should().Equal(1, 2, 3);
        }

        // -- Layout coverage --

        [TestMethod]
        public void FContiguous_Inputs_ProduceFContiguousOutput()
        {
            // NumPy: when all inputs are F-contig, the result is F-contig.
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } }).copy('F');
            var b = np.array(new int[,] { { 5, 6 } }).copy('F');
            var r = np.concatenate(new[] { a, b });
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void AmbiguousLayout_BothCF_Inputs_ProduceCOrder()
        {
            // NumPy 2.4.2 (PyArray_CreateMultiSortedStridePerm): inputs that are
            // BOTH C- and F-contiguous — e.g. (N,1) column views — resolve to a
            // C-ordered result. F is chosen only when all inputs are F-contig
            // AND not all are C-contig.
            var t1 = np.arange(3).reshape(1, 3).T;          // (3,1), C&F
            var t2 = (np.arange(3) + 10).reshape(1, 3).T;   // (3,1), C&F
            t1.Shape.IsFContiguous.Should().BeTrue("precondition: ambiguous layout");
            t1.Shape.IsContiguous.Should().BeTrue("precondition: ambiguous layout");

            var r = np.concatenate(new[] { t1, t2 }, 1);
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeFalse();
            r.Data<int>().Should().Equal(0, 10, 1, 11, 2, 12);
        }

        [TestMethod]
        public void MixedFOnly_And_AmbiguousInputs_ProduceFOrder()
        {
            // all F-contig, not all C-contig -> F result (probed NumPy 2.4.2)
            var fOnly = np.asfortranarray(np.arange(6).reshape(3, 2));
            var both = np.zeros((3, 1), np.int32);
            var r = np.concatenate(new[] { fOnly, both }, 1);
            r.Shape.IsFContiguous.Should().BeTrue();
            r.Shape.IsContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void StridedView_Concat_Works()
        {
            var a = np.arange(12).reshape(3, 4);
            var view = a["::2"]; // strided view, rows 0 and 2
            var b = np.arange(8).reshape(2, 4);
            var r = np.concatenate(new[] { view, b }, axis: 0);
            r.shape.Should().BeEquivalentTo(new long[] { 4, 4 });
        }

        [TestMethod]
        public void TransposedView_Concat_Works()
        {
            var a = np.arange(6).reshape(2, 3).T;     // (3,2) non-contig
            var b = np.arange(6).reshape(3, 2);
            var r = np.concatenate(new[] { a, b }, axis: 0);
            r.shape.Should().BeEquivalentTo(new long[] { 6, 2 });
        }

        // -- Dtype coverage (all 15 dtypes round-trip) --

        [DataTestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex)]
        public void AllDtypes_SameDtype_RoundTrip(NPTypeCode tc)
        {
            var a = np.array(new double[] { 1.0, 2.0 }).astype(tc);
            var b = np.array(new double[] { 3.0, 4.0 }).astype(tc);
            var r = np.concatenate(new[] { a, b });
            r.typecode.Should().Be(tc);
            r.shape[0].Should().Be(4);
        }
    }
}
