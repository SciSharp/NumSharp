using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     np.column_stack — stack 1-D arrays as columns into a 2-D array.
    ///     Expected values probed against NumPy 2.4.2
    ///     (numpy/lib/_shape_base_impl.py::column_stack).
    /// </summary>
    [TestClass]
    public class np_column_stack_test : TestClass
    {
        [TestMethod]
        public void OneD_Pair_BecomesColumns()
        {
            // np.column_stack(([1,2,3], [4,5,6])) -> [[1,4],[2,5],[3,6]]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 4, 5, 6 });
            var r = np.column_stack(a, b);
            r.shape.Should().Equal(3L, 2L);
            r.Data<int>().Should().Equal(1, 4, 2, 5, 3, 6);
        }

        [TestMethod]
        public void TwoD_StackedAsIs()
        {
            // 2-D arrays concatenate along axis 1, like hstack
            var A = np.arange(6).reshape(3, 2);
            var B = (np.arange(6) + 100).reshape(3, 2);
            var r = np.column_stack(A, B);
            r.shape.Should().Equal(3L, 4L);
            r.Data<int>().Should().Equal(0, 1, 100, 101, 2, 3, 102, 103, 4, 5, 104, 105);
        }

        [TestMethod]
        public void Mixed_1D_And_2D()
        {
            // np.column_stack((a, A)) -> [[1,0,1],[2,2,3],[3,4,5]]
            var a = np.array(new[] { 1, 2, 3 });
            var A = np.arange(6).reshape(3, 2);
            var r = np.column_stack(a, A);
            r.shape.Should().Equal(3L, 3L);
            r.Data<int>().Should().Equal(1, 0, 1, 2, 2, 3, 3, 4, 5);
        }

        [TestMethod]
        public void ZeroD_Becomes_1x1()
        {
            // np.column_stack((np.array(5), np.array(6))) -> [[5, 6]]
            var r = np.column_stack(np.asanyarray(5), np.asanyarray(6));
            r.shape.Should().Equal(1L, 2L);
            r.Data<int>().Should().Equal(5, 6);
        }

        [TestMethod]
        public void ThreeD_ConcatenatesAlongAxis1()
        {
            // ndim >= 2 passes through untouched: (2,2,2)+(2,2,2) -> (2,4,2)
            var c = np.arange(8).reshape(2, 2, 2);
            np.column_stack(c, c).shape.Should().Equal(2L, 4L, 2L);
        }

        [TestMethod]
        public void SingleInput()
        {
            np.column_stack(np.array(new[] { 1, 2, 3 })).shape.Should().Equal(3L, 1L);
            np.column_stack(np.arange(6).reshape(3, 2)).shape.Should().Equal(3L, 2L);
        }

        [TestMethod]
        public void DtypePromotion()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 4, 5, 6 });
            np.column_stack(a.astype(np.int8), b.astype(np.float32)).typecode.Should().Be(NPTypeCode.Single);
            np.column_stack(a.astype(np.uint8), b.astype(np.int8)).typecode.Should().Be(NPTypeCode.Int16);
        }

        [TestMethod]
        public void AllDtypes_Preserved()
        {
            foreach (NPTypeCode tc in new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
                NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
                NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
                NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
            })
            {
                var x = np.ones(new Shape(3), tc);
                var r = np.column_stack(x, x);
                r.typecode.Should().Be(tc, $"dtype {tc} should be preserved");
                r.shape.Should().Equal(3L, 2L);
            }
        }

        [TestMethod]
        public void EmptyInputs()
        {
            // np.column_stack((zeros(0), zeros(0))) -> (0, 2)
            np.column_stack(np.zeros(0), np.zeros(0)).shape.Should().Equal(0L, 2L);
            // np.column_stack((zeros((0,2)), zeros((0,3)))) -> (0, 5)
            np.column_stack(np.zeros((0, 2)), np.zeros((0, 3))).shape.Should().Equal(0L, 5L);
        }

        [TestMethod]
        public void AmbiguousLayoutInputs_ProduceCOrder()
        {
            // (N,1) column views are both C- and F-contiguous; NumPy's stride-perm
            // ambiguity resolution yields a C-ordered result (probed 2.4.2).
            var r = np.column_stack(np.array(new[] { 1, 2, 3 }), np.array(new[] { 4, 5, 6 }));
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void FortranOnlyInputs_ProduceFOrder()
        {
            var Af = np.asfortranarray(np.arange(6).reshape(3, 2));
            var Bf = np.asfortranarray((np.arange(6) + 100).reshape(3, 2));
            var r = np.column_stack(Af, Bf);
            r.Shape.IsFContiguous.Should().BeTrue();
            r.Shape.IsContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void StridedViews_Work()
        {
            var baseArr = np.arange(10);
            var s = baseArr["::2"]; // [0, 2, 4, 6, 8]
            var r = np.column_stack(s, s);
            r.shape.Should().Equal(5L, 2L);
            r.Data<int>().Should().Equal(0, 0, 2, 2, 4, 4, 6, 6, 8, 8);
        }

        [TestMethod]
        public void FirstDimensionMismatch_Throws()
        {
            var act = () => np.column_stack(np.array(new[] { 1, 2 }), np.array(new[] { 1, 2, 3 }));
            act.Should().Throw<IncorrectShapeException>().WithMessage(
                "*along dimension 0, the array at index 0 has size 2 and the array at index 1 has size 3*");
        }

        [TestMethod]
        public void EmptyTuple_Throws()
        {
            var act = () => np.column_stack();
            act.Should().Throw<ArgumentException>().WithMessage("*need at least one array to concatenate*");
        }

        [TestMethod]
        public void NdimMismatch_1D_vs_3D_Throws()
        {
            // 1-D becomes (3,1); a 3-D input keeps ndim 3 -> concatenate rejects
            var act = () => np.column_stack(np.array(new[] { 1, 2, 3 }), np.arange(8).reshape(2, 2, 2));
            act.Should().Throw<IncorrectShapeException>().WithMessage(
                "*must have same number of dimensions*");
        }
    }
}
