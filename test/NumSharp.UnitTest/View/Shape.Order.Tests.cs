using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.View
{
    /// <summary>
    ///     Tests for multi-order memory layout support: C/F physical, A/K logical.
    ///
    ///     PYTHON VERIFICATION (NumPy 2.4.2):
    ///     Behavior matches NumPy's flags['C_CONTIGUOUS'] / flags['F_CONTIGUOUS']
    ///     and order resolution semantics for np.empty / np.copy.
    /// </summary>
    [TestClass]
    public class ShapeOrderTests
    {
        // ================================================================
        //  Detection: IsContiguous / IsFContiguous
        // ================================================================

        [TestMethod]
        public void Scalar_IsBothCAndFContiguous()
        {
            // NumPy: np.array(42).flags -> C=True, F=True
            var scalar = new Shape();
            scalar.IsContiguous.Should().BeTrue("scalars are C-contig by definition");
            scalar.IsFContiguous.Should().BeTrue("scalars are F-contig by definition");
        }

        [TestMethod]
        public void OneDimensional_IsBothCAndFContiguous()
        {
            // NumPy: np.arange(5).flags -> C=True, F=True
            var shape = new Shape(5);
            shape.IsContiguous.Should().BeTrue();
            shape.IsFContiguous.Should().BeTrue("1-D contiguous arrays are both C and F contig");
        }

        [TestMethod]
        public void CContiguous2D_IsCOnly()
        {
            // NumPy: np.zeros((3,4)).flags -> C=True, F=False
            var shape = new Shape(3L, 4L);
            shape.IsContiguous.Should().BeTrue();
            shape.IsFContiguous.Should().BeFalse("multi-dim C-contig is not F-contig");
        }

        [TestMethod]
        public void TransposeOfCContig_IsFContiguous()
        {
            // NumPy: arr = np.arange(24).reshape(2,3,4); arr.T.flags -> C=False, F=True
            var arr = np.arange(24).reshape(2, 3, 4);
            var transposed = arr.T;

            transposed.Shape.IsContiguous.Should().BeFalse();
            transposed.Shape.IsFContiguous.Should().BeTrue(
                "transpose of C-contig produces F-contig memory layout");
        }

        [TestMethod]
        public void Shape_WithFOrder_ProducesFContigStrides()
        {
            // F-order strides for (3,4): strides[0]=1, strides[1]=3
            var shape = new Shape(new long[] { 3, 4 }, 'F');

            shape.IsFContiguous.Should().BeTrue();
            shape.IsContiguous.Should().BeFalse();
            shape.strides[0].Should().Be(1);
            shape.strides[1].Should().Be(3);
        }

        [TestMethod]
        public void Shape_WithCOrder_ProducesCContigStrides()
        {
            // C-order strides for (3,4): strides[0]=4, strides[1]=1
            var shape = new Shape(new long[] { 3, 4 }, 'C');

            shape.IsContiguous.Should().BeTrue();
            shape.IsFContiguous.Should().BeFalse();
            shape.strides[0].Should().Be(4);
            shape.strides[1].Should().Be(1);
        }

        [TestMethod]
        public void Shape_WithInvalidOrder_Throws()
        {
            // Direct Shape constructor only accepts physical orders (C/F); A/K must be resolved first.
            Action act = () => new Shape(new long[] { 3, 4 }, 'A');
            act.Should().Throw<ArgumentException>();

            Action act2 = () => new Shape(new long[] { 3, 4 }, 'X');
            act2.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Shape_3D_FOrder_HasExpectedStrides()
        {
            // F-order (2,3,4): strides = (1, 2, 6)
            var shape = new Shape(new long[] { 2, 3, 4 }, 'F');

            shape.IsFContiguous.Should().BeTrue();
            shape.strides.Should().Equal(new long[] { 1, 2, 6 });
        }

        // ================================================================
        //  OrderResolver: Logical -> Physical mapping
        // ================================================================

        [TestMethod]
        public void OrderResolver_C_ReturnsC()
        {
            OrderResolver.Resolve('C').Should().Be('C');
            OrderResolver.Resolve('c').Should().Be('C');
        }

        [TestMethod]
        public void OrderResolver_F_ReturnsF()
        {
            OrderResolver.Resolve('F').Should().Be('F');
            OrderResolver.Resolve('f').Should().Be('F');
        }

        [TestMethod]
        public void OrderResolver_A_WithoutSource_Throws()
        {
            // NumPy: np.empty((3,4), order='A') -> "only 'C' or 'F' order is permitted"
            Action act = () => OrderResolver.Resolve('A');
            act.Should().Throw<ArgumentException>()
               .WithMessage("*only 'C' or 'F'*");
        }

        [TestMethod]
        public void OrderResolver_K_WithoutSource_Throws()
        {
            Action act = () => OrderResolver.Resolve('K');
            act.Should().Throw<ArgumentException>()
               .WithMessage("*only 'C' or 'F'*");
        }

        [TestMethod]
        public void OrderResolver_A_WithCSource_ReturnsC()
        {
            var cSource = new Shape(new long[] { 3, 4 }, 'C');
            OrderResolver.Resolve('A', cSource).Should().Be('C');
        }

        [TestMethod]
        public void OrderResolver_A_WithFSource_ReturnsF()
        {
            // NumPy: np.copy(f_arr, order='A') with F-contig (not C) source -> F-contig output
            var fSource = new Shape(new long[] { 3, 4 }, 'F');
            OrderResolver.Resolve('A', fSource).Should().Be('F');
        }

        [TestMethod]
        public void OrderResolver_K_WithCSource_ReturnsC()
        {
            var cSource = new Shape(new long[] { 3, 4 }, 'C');
            OrderResolver.Resolve('K', cSource).Should().Be('C');
        }

        [TestMethod]
        public void OrderResolver_K_WithFSource_ReturnsF()
        {
            var fSource = new Shape(new long[] { 3, 4 }, 'F');
            OrderResolver.Resolve('K', fSource).Should().Be('F');
        }

        [TestMethod]
        public void OrderResolver_InvalidChar_Throws()
        {
            Action act = () => OrderResolver.Resolve('X');
            act.Should().Throw<ArgumentException>()
               .WithMessage("*'C', 'F', 'A', 'K'*");
        }

        // ================================================================
        //  np.empty integration — all 4 orders
        // ================================================================

        [TestMethod]
        public void NpEmpty_COrder_ProducesCContig()
        {
            var arr = np.empty(new Shape(3L, 4L), order: 'C');
            arr.Shape.IsContiguous.Should().BeTrue();
            arr.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void NpEmpty_FOrder_ProducesFContig()
        {
            var arr = np.empty(new Shape(3L, 4L), order: 'F');
            arr.Shape.IsContiguous.Should().BeFalse();
            arr.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void NpEmpty_AOrder_Throws()
        {
            // NumPy: np.empty(shape, order='A') -> ValueError
            Action act = () => np.empty(new Shape(3L, 4L), order: 'A');
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void NpEmpty_KOrder_Throws()
        {
            // NumPy: np.empty(shape, order='K') -> ValueError
            Action act = () => np.empty(new Shape(3L, 4L), order: 'K');
            act.Should().Throw<ArgumentException>();
        }

        // ================================================================
        //  Flags integration
        // ================================================================

        [TestMethod]
        public void Flags_FContig_ExposesFContiguousBit()
        {
            var fShape = new Shape(new long[] { 3, 4 }, 'F');
            (fShape.Flags & ArrayFlags.F_CONTIGUOUS).Should().Be(ArrayFlags.F_CONTIGUOUS);
            (fShape.Flags & ArrayFlags.C_CONTIGUOUS).Should().Be(ArrayFlags.None);
        }

        [TestMethod]
        public void Flags_CContig_ExposesCContiguousBit()
        {
            var cShape = new Shape(new long[] { 3, 4 }, 'C');
            (cShape.Flags & ArrayFlags.C_CONTIGUOUS).Should().Be(ArrayFlags.C_CONTIGUOUS);
            (cShape.Flags & ArrayFlags.F_CONTIGUOUS).Should().Be(ArrayFlags.None);
        }

        [TestMethod]
        public void Flags_1D_ExposesBothContiguousBits()
        {
            // 1-D arrays satisfy both C and F contiguity conditions
            var shape = new Shape(5L);
            (shape.Flags & ArrayFlags.C_CONTIGUOUS).Should().Be(ArrayFlags.C_CONTIGUOUS);
            (shape.Flags & ArrayFlags.F_CONTIGUOUS).Should().Be(ArrayFlags.F_CONTIGUOUS);
        }

        // ================================================================
        //  Shape.Order property — derives from actual contiguity flags
        // ================================================================

        [TestMethod]
        public void Order_CContig_ReportsC()
        {
            var shape = new Shape(new long[] { 3, 4 }, 'C');
            shape.Order.Should().Be('C');
        }

        [TestMethod]
        public void Order_FContig_ReportsF()
        {
            var shape = new Shape(new long[] { 3, 4 }, 'F');
            shape.Order.Should().Be('F');
        }

        [TestMethod]
        public void Order_Transpose_ReportsF()
        {
            // Transpose of C-contig produces F-contig memory; Order should reflect that
            var arr = np.arange(24).reshape(2, 3, 4);
            arr.Shape.Order.Should().Be('C');
            arr.T.Shape.Order.Should().Be('F');
        }

        [TestMethod]
        public void Order_1D_ReportsC()
        {
            // 1-D is both C and F contig; default to 'C'
            var shape = new Shape(5L);
            shape.Order.Should().Be('C');
        }

        [TestMethod]
        public void Order_Scalar_ReportsC()
        {
            var scalar = new Shape();
            scalar.Order.Should().Be('C');
        }

        // ================================================================
        //  Empty arrays (any dim == 0) are trivially both C and F contig
        // ================================================================

        [TestMethod]
        public void EmptyArray_IsBothCAndFContiguous()
        {
            // NumPy: np.empty((2, 0, 3)).flags -> C=True, F=True (any dim=0)
            var shape = new Shape(2L, 0L, 3L);
            shape.IsContiguous.Should().BeTrue();
            shape.IsFContiguous.Should().BeTrue();
        }
    }
}
