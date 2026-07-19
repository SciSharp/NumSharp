using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    ///     np.unstack (NumPy 2.1) — split an array into a sequence of arrays
    ///     along the given axis; equivalent to tuple(np.moveaxis(x, axis, 0)).
    ///     Expected values probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class np_unstack_test : TestClass
    {
        [TestMethod]
        public void Basic_Axis0()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var r = np.unstack(arr);
            r.Length.Should().Be(2);
            r[0].shape.Should().Equal(3L, 4L);
            r[0].Data<int>().Should().Equal(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
            r[1].Data<int>().Should().Equal(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
        }

        [TestMethod]
        public void Axis1()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var r = np.unstack(arr, axis: 1);
            r.Length.Should().Be(3);
            r[0].shape.Should().Equal(2L, 4L);
            // np.unstack(arr, axis=1)[0] -> [[0,1,2,3],[12,13,14,15]]
            r[0].Data<int>().Should().Equal(0, 1, 2, 3, 12, 13, 14, 15);
            r[2].Data<int>().Should().Equal(8, 9, 10, 11, 20, 21, 22, 23);
        }

        [TestMethod]
        public void NegativeAxis()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var r = np.unstack(arr, axis: -1);
            r.Length.Should().Be(4);
            r[0].shape.Should().Equal(2L, 3L);
            // np.unstack(arr, axis=-1)[0] -> [[0,4,8],[12,16,20]]
            r[0].Data<int>().Should().Equal(0, 4, 8, 12, 16, 20);
        }

        [TestMethod]
        public void ReturnsViews_WriteThrough()
        {
            // NumPy: the unstacked arrays share memory with x
            var arr = np.arange(24).reshape(2, 3, 4);
            var r = np.unstack(arr);
            r[0].SetData(999, 0, 0);
            ((int)arr.GetData(new int[] { 0, 0, 0 })).Should().Be(999);

            var r1 = np.unstack(arr, axis: 1);
            r1[2].SetData(-5, 1, 3);
            ((int)arr.GetData(new int[] { 1, 2, 3 })).Should().Be(-5);
        }

        [TestMethod]
        public void OneD_Yields_0d_Views()
        {
            var a = np.array(new[] { 10, 20, 30 });
            var r = np.unstack(a);
            r.Length.Should().Be(3);
            r[0].ndim.Should().Be(0);
            ((int)r[0]).Should().Be(10);
            ((int)r[1]).Should().Be(20);
            ((int)r[2]).Should().Be(30);
        }

        [TestMethod]
        public void RoundTrip_StackOfUnstack()
        {
            // NumPy docstring: stack(unstack(x, axis=axis), axis=axis) == x
            var arr = np.arange(24).reshape(2, 3, 4);
            var rebuilt = np.stack(np.unstack(arr, axis: 1), axis: 1);
            rebuilt.shape.Should().Equal(2L, 3L, 4L);
            ((bool)np.array_equal(rebuilt, arr)).Should().BeTrue();
        }

        [TestMethod]
        public void EmptyAlongAxis_ReturnsNoArrays()
        {
            np.unstack(np.zeros((0, 3))).Length.Should().Be(0);
        }

        [TestMethod]
        public void EmptyElsewhere_ReturnsEmptyArrays()
        {
            var r = np.unstack(np.zeros((0, 3)), axis: 1);
            r.Length.Should().Be(3);
            r[0].shape.Should().Equal(0L);
            r[0].typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void StridedInput_Views()
        {
            var baseArr = np.arange(20).reshape(10, 2);
            var s = baseArr["::2"];  // (5, 2) strided view
            var r = np.unstack(s, axis: 1);
            r.Length.Should().Be(2);
            r[0].Data<int>().Should().Equal(0, 4, 8, 12, 16);
            r[1].Data<int>().Should().Equal(1, 5, 9, 13, 17);
            // still a view chain into baseArr
            r[1].SetData(777, 0);
            ((int)baseArr.GetData(new int[] { 0, 1 })).Should().Be(777);
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
                var x = np.ones(new Shape(2, 3), tc);
                var r = np.unstack(x);
                r.Length.Should().Be(2);
                r[0].typecode.Should().Be(tc, $"dtype {tc} should be preserved");
                r[0].shape.Should().Equal(3L);
            }
        }

        [TestMethod]
        public void ZeroD_Throws()
        {
            // NumPy: ValueError("Input array must be at least 1-d.")
            var act = () => np.unstack(np.asanyarray(5.0));
            act.Should().Throw<ValueError>().WithMessage("Input array must be at least 1-d.");
        }

        [TestMethod]
        public void AxisOutOfRange_Throws()
        {
            var a = np.array(new[] { 1, 2, 3 });
            var actPos = () => np.unstack(a, axis: 2);
            actPos.Should().Throw<ArgumentOutOfRangeException>();
            var actNeg = () => np.unstack(a, axis: -2);
            actNeg.Should().Throw<ArgumentOutOfRangeException>();
        }

        [TestMethod]
        public void Null_Throws()
        {
            var act = () => np.unstack(null);
            act.Should().Throw<ArgumentNullException>();
        }
    }
}
