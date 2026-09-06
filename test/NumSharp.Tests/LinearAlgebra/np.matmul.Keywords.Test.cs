using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     <c>np.matmul</c>'s full keyword surface — <c>out</c>/<c>dtype</c>/<c>casting</c>/
    ///     <c>order</c>/<c>axes</c> plus the <c>axis</c>/<c>keepdims</c> rejections its 3-core-dim
    ///     signature <c>(n?,k),(k,m?)-&gt;(n?,m?)</c> requires. Every expectation probed against
    ///     NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class np_matmul_keywords_test
    {
        private static NDArray M23 => np.arange(6.0).reshape(2, 3);
        private static NDArray M32 => np.arange(6.0).reshape(3, 2);

        [TestMethod]
        public void Matmul_Bare_StillWorks()
        {
            np.matmul(M23, M32).Should().BeOfValues(10, 13, 28, 40).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Matmul_Out_ReturnsSameInstance()
        {
            var o = np.zeros(new Shape(2, 2), NPTypeCode.Double);
            var r = np.matmul(M23, M32, @out: o);
            r.Should().BeSameAs(o);
            r.Should().BeOfValues(10, 13, 28, 40).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Matmul_Out_MayBeStrided()
        {
            // out is a stride-2 column view of a (2,4) buffer — a ufunc out= accepts it.
            var backing = np.zeros(new Shape(2, 4), NPTypeCode.Double);
            var o = backing[":, ::2"];
            np.matmul(M23, M32, @out: o).Should().BeOfValues(10, 13, 28, 40).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Matmul_Dtype_SelectsTheLoop()
        {
            var r = np.matmul(M23, M32, dtype: NPTypeCode.Single);
            r.typecode.Should().Be(NPTypeCode.Single);
            r.Should().BeOfValues(10, 13, 28, 40);
        }

        [TestMethod]
        public void Matmul_Dtype_IntInputsToFloatLoop()
        {
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int64);
            var b = np.arange(6).reshape(3, 2).astype(NPTypeCode.Int64);
            np.matmul(a, b, dtype: NPTypeCode.Single).typecode.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Matmul_Dtype_UnreachableLoopUnderSameKind_Raises()
        {
            new Action(() => np.matmul(M23, M32, dtype: NPTypeCode.Boolean))
                .Should().Throw<ArgumentException>().WithMessage(
                    "Cannot cast ufunc 'matmul' input 0 from dtype('float64') to dtype('bool') with casting rule 'same_kind'");
        }

        [TestMethod]
        public void Matmul_OutCast_DefaultSameKindRejectsIntOut()
        {
            new Action(() => np.matmul(M23, M32, @out: np.zeros(new Shape(2, 2), NPTypeCode.Int32)))
                .Should().Throw<ArgumentException>().WithMessage(
                    "Cannot cast ufunc 'matmul' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'");
        }

        [TestMethod]
        public void Matmul_OutCast_UnsafeAllowsIntOut()
        {
            var o = np.zeros(new Shape(2, 2), NPTypeCode.Int32);
            var r = np.matmul(M23, M32, @out: o, casting: "unsafe");
            r.Should().BeSameAs(o);
            r.typecode.Should().Be(NPTypeCode.Int32);
            r.Should().BeOfValues(10, 13, 28, 40);
        }

        [TestMethod]
        public void Matmul_Order_C_And_F_And_K()
        {
            var af = np.asfortranarray(M23);
            var bf = np.asfortranarray(M32);
            np.matmul(af, bf, order: 'C').Shape.IsContiguous.Should().BeTrue();
            np.matmul(af, bf, order: 'F').Shape.IsFContiguous.Should().BeTrue();
            // K always yields the product's natural C-contiguous output, even for all-F inputs.
            np.matmul(af, bf, order: 'K').Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Matmul_Order_A_IsFOnlyWhenBothInputsAreF()
        {
            np.matmul(np.asfortranarray(M23), np.asfortranarray(M32), order: 'A').Shape.IsFContiguous.Should().BeTrue();
            np.matmul(M23, M32, order: 'A').Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Matmul_Order_Invalid_Raises()
        {
            new Action(() => np.matmul(M23, M32, order: 'Q'))
                .Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Q')");
        }

        [TestMethod]
        public void Matmul_Casting_Invalid_Raises()
        {
            new Action(() => np.matmul(M23, M32, casting: "bogus"))
                .Should().Throw<ValueError>().WithMessage(
                    "casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got 'bogus')");
        }

        [TestMethod]
        public void Matmul_Axis_Rejected()
        {
            new Action(() => np.matmul(M23, M32, axis: 0)).Should().Throw<TypeError>().WithMessage(
                "matmul: axis can only be used with a single shared core dimension, not with the 3 distinct ones implied by signature (n?,k),(k,m?)->(n?,m?).");
        }

        [TestMethod]
        public void Matmul_Keepdims_RejectedForAnyValue()
        {
            new Action(() => np.matmul(M23, M32, keepdims: true)).Should().Throw<TypeError>()
                .WithMessage("matmul does not support keepdims: its signature (n?,k),(k,m?)->(n?,m?) requires output 0 to have 2 core dimensions*");
            // even an explicit false rejects (NumPy's np._NoValue sentinel).
            new Action(() => np.matmul(M23, M32, keepdims: false)).Should().Throw<TypeError>()
                .WithMessage("matmul does not support keepdims*");
        }

        [TestMethod]
        public void Matmul_AxisAndAxes_TogetherRejected()
        {
            new Action(() => np.matmul(M23, M32, axes: new[] {new[] {-2, -1}, new[] {-2, -1}, new[] {-2, -1}}, axis: 0))
                .Should().Throw<TypeError>().WithMessage("cannot specify both 'axis' and 'axes'");
        }

        [TestMethod]
        public void Matmul_Axes_Batched_RelocatesCoreAxes()
        {
            var a = np.arange(24.0).reshape(3, 2, 4);
            var b = np.arange(24.0).reshape(4, 2, 3);
            np.matmul(a, b, axes: new[] {new[] {0, 2}, new[] {0, 2}, new[] {0, 2}}).Should().BeShaped(3, 2, 3);
        }

        [TestMethod]
        public void Matmul_Axes_MatrixVector()
        {
            var m = np.arange(6.0).reshape(2, 3);
            var v = np.arange(3.0);
            np.matmul(m, v, axes: new[] {new[] {-2, -1}, new[] {-1}, new[] {-1}})
                .Should().BeOfValues(5, 14).And.BeShaped(2);
        }

        [TestMethod]
        public void Matmul_Axes_OutputEntryNeverOmittable()
        {
            var m = np.arange(6.0).reshape(2, 3);
            var mt = np.arange(6.0).reshape(3, 2);
            // 2-entry axes is rejected even though the general gufunc omit rule would allow it for a
            // scalar-output product — matmul's output is treated as having core axes.
            new Action(() => np.matmul(m, mt, axes: new[] {new[] {-2, -1}, new[] {-2, -1}}))
                .Should().Throw<ValueError>().WithMessage(
                    "axes should be a list with an entry for all 3 inputs and outputs; entries for outputs can only be omitted if none of them has core axes.");
        }

        [TestMethod]
        public void Matmul_Axes_DotProductNeedsExplicitEmptyOutputEntry()
        {
            var v = np.arange(3.0);
            // the 1-D·1-D dot: the output entry must be present AND empty (0 core dims).
            np.matmul(v, v, axes: new[] {new[] {-1}, new[] {-1}, new int[] { }}).Should().BeOfValues(5).And.BeShaped();
            new Action(() => np.matmul(v, v, axes: new[] {new[] {-1}, new[] {-1}}))
                .Should().Throw<ValueError>().WithMessage("axes should be a list with an entry for all 3 inputs and outputs*");
        }

        [TestMethod]
        public void Matmul_Axes_RepeatedAxis_Raises()
        {
            new Action(() => np.matmul(M23, M32, axes: new[] {new[] {1, 1}, new[] {-2, -1}, new[] {-2, -1}}))
                .Should().Throw<ValueError>().WithMessage("axes item 0 has value 1 repeated");
        }

        [TestMethod]
        public void Matmul_Axes_WrongCoreDimCount_Raises()
        {
            new Action(() => np.matmul(M23, M32, axes: new[] {new[] {-1}, new[] {-2, -1}, new[] {-2, -1}}))
                .Should().Throw<AxisError>().WithMessage(
                    "matmul: operand 0 has 2 core dimensions, but 1 dimensions are specified by axes tuple.*");
        }

        [TestMethod]
        public void Matmul_Out_WrongCoreShape_ReportsGufuncMessage()
        {
            new Action(() => np.matmul(M23, M32, @out: np.zeros(new Shape(2, 3), NPTypeCode.Double)))
                .Should().Throw<ValueError>().WithMessage(
                    "matmul: Output operand 0 has a mismatch in its core dimension 1, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)");
        }

        [TestMethod]
        public void Matmul_Out_ExtraLeadingLoopDims_ReplicateProduct()
        {
            // out with extra leading LOOP dims is a broadcast target — NumPy replicates the product
            // into each slab (probed: out=(5,2,2) for a (2,2) product is 5 copies). The trailing CORE
            // dims still match exactly; only the leading loop dims broadcast.
            var o = np.zeros(new Shape(5, 2, 2), NPTypeCode.Double);
            var r = np.matmul(M23, M32, @out: o);
            r.Should().BeSameAs(o);
            r.Should().BeShaped(5, 2, 2).And.BeOfValues(
                10, 13, 28, 40, 10, 13, 28, 40, 10, 13, 28, 40, 10, 13, 28, 40, 10, 13, 28, 40);
        }
    }
}
