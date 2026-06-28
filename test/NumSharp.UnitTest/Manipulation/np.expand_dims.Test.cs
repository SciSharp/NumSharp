using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class ExpandDimsTest
    {
        [TestMethod]
        public void Simple1DArrayTo2DArray()
        {
            var input = np.array(1, 2, 3);
            var expected = np.array(new int[] {1, 2, 3});

            var result = np.expand_dims(input, 0);

            Assert.IsTrue(result.shape[0] == 1);
            Assert.IsTrue(result.shape[1] == 3);
            Assert.IsTrue(result.ndim == 2);
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<int>(), expected.Data<int>()));
        }

        [TestMethod]
        public void Simple1DArrayToTransposed2D()
        {
            var input = np.array(1, 2, 3);
            var expected = np.array(new int[] {1, 2, 3});

            var result = np.expand_dims(input, 1);

            Assert.IsTrue(result.shape[0] == 3);
            Assert.IsTrue(result.shape[1] == 1);
            Assert.IsTrue(result.ndim == 2);
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<int>(), expected.Data<int>()));
        }

        [TestMethod]
        public void Simple1DArrayToTransposed3D()
        {
            var input = np.array(1, 2, 3);
            var expected = np.array(new int[] {1, 2, 3});

            var result = np.expand_dims(np.expand_dims(input, 1), 2);

            Assert.IsTrue(result.shape[0] == 3);
            Assert.IsTrue(result.shape[1] == 1);
            Assert.IsTrue(result.shape[2] == 1);
            Assert.IsTrue(result.ndim == 3);
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<int>(), expected.Data<int>()));
        }

        // =====================================================================
        // Tuple-axis support (NumPy 2.x parity)
        // =====================================================================

        [TestMethod]
        public void TupleAxis_0_2_From2D()
        {
            // np.expand_dims(np.arange(6).reshape(2,3), (0, 2)).shape == (1, 2, 1, 3)
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, new[] { 0, 2 });

            r.shape.Should().Equal(1L, 2L, 1L, 3L);
            r.ndim.Should().Be(4);
            r.Data<int>().Should().Equal(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void TupleAxis_0_3_From2D()
        {
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, new[] { 0, 3 });

            r.shape.Should().Equal(1L, 2L, 3L, 1L);
        }

        [TestMethod]
        public void TupleAxis_NegativeAxes()
        {
            // np.expand_dims(a, (-1,-2)).shape == (2, 3, 1, 1)
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, new[] { -1, -2 });

            r.shape.Should().Equal(2L, 3L, 1L, 1L);
        }

        [TestMethod]
        public void TupleAxis_MixedSignsAndOrder()
        {
            // np.expand_dims(a, (-1, 0)).shape == (1, 2, 3, 1)
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, new[] { -1, 0 });

            r.shape.Should().Equal(1L, 2L, 3L, 1L);
        }

        [TestMethod]
        public void TupleAxis_Unsorted_2_1_0()
        {
            // np.expand_dims(a, (2,1,0)).shape == (1, 1, 1, 2, 3)
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, new[] { 2, 1, 0 });

            r.shape.Should().Equal(1L, 1L, 1L, 2L, 3L);
        }

        [TestMethod]
        public void TupleAxis_0_1_4_From2D()
        {
            // np.expand_dims(a, (0,1,4)).shape == (1, 1, 2, 3, 1)
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, new[] { 0, 1, 4 });

            r.shape.Should().Equal(1L, 1L, 2L, 3L, 1L);
        }

        [TestMethod]
        public void TupleAxis_EmptyTuple_ReturnsUnchangedShape()
        {
            // np.expand_dims(a, ()).shape == (2, 3) — no-op.
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, new int[] { });

            r.shape.Should().Equal(2L, 3L);
        }

        [TestMethod]
        public void TupleAxis_DuplicateAxis_Throws()
        {
            // NumPy: ValueError "repeated axis".
            var a = np.arange(6).reshape(2, 3);

            Action act = () => np.expand_dims(a, new[] { 0, 0 });
            act.Should().Throw<ArgumentException>().WithMessage("*repeated axis*");
        }

        [TestMethod]
        public void TupleAxis_DuplicateAfterNegativeNormalization_Throws()
        {
            // (0, -4) on ndim=2 → out_ndim=4 → -4 normalizes to 0 → duplicate.
            var a = np.arange(6).reshape(2, 3);

            Action act = () => np.expand_dims(a, new[] { 0, -4 });
            act.Should().Throw<ArgumentException>().WithMessage("*repeated axis*");
        }

        [TestMethod]
        public void TupleAxis_OutOfRange_Throws()
        {
            // NumPy: AxisError "axis 5 is out of bounds for array of dimension 3"
            // (out_ndim = ndim(2) + axis.Length(1) = 3, so valid range is [-3, 2]).
            var a = np.arange(6).reshape(2, 3);

            Action act = () => np.expand_dims(a, new[] { 5 });
            act.Should().Throw<ArgumentException>().WithMessage("*out of bounds*");
        }

        [TestMethod]
        public void TupleAxis_NegativeOutOfRange_Throws()
        {
            var a = np.arange(6).reshape(2, 3);

            Action act = () => np.expand_dims(a, new[] { -5 });
            act.Should().Throw<ArgumentException>().WithMessage("*out of bounds*");
        }

        [TestMethod]
        public void TupleAxis_IEnumerableOverload()
        {
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, new List<int> { 0, 2 });

            r.shape.Should().Equal(1L, 2L, 1L, 3L);
        }

        [TestMethod]
        public void TupleAxis_NullAxis_ReturnsUnchanged()
        {
            // Defensive: null axis is a no-op (mirrors empty-tuple semantics).
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, (int[])null);

            r.shape.Should().Equal(2L, 3L);
        }

        [TestMethod]
        public void TupleAxis_SingleElementArray_EquivalentToIntOverload()
        {
            var a = np.arange(6).reshape(2, 3);
            var r1 = np.expand_dims(a, new[] { 1 });
            var r2 = np.expand_dims(a, 1);

            r1.shape.Should().Equal(r2.shape.Select(d => (long)d).ToArray());
        }

        [TestMethod]
        public void TupleAxis_ViewSemantics_DataShared()
        {
            // Expand should produce a view of the same storage.
            var a = np.arange(6).reshape(2, 3);
            var r = np.expand_dims(a, new[] { 0, 2 });

            r.Data<int>().Should().Equal(0, 1, 2, 3, 4, 5);
        }
    }
}
