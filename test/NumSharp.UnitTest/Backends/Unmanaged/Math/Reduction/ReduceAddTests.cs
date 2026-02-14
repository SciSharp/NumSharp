using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Unmanaged.Math.Reduction
{
    public class ReduceAddTests
    {
        [Test]
        public void EmptyArray()
        {
            np.sum(np.array(new int[0])).Should().BeScalar(0);
        }

        [Test]
        public void Case1_Elementwise_keepdims()
        {
            var np1 = np.array(new double[] {1, 2, 3, 4, 5, 6}).reshape(3, 2);
            var mean = np.sum(np1, keepdims: true);
            mean.Shape.IsScalar.Should().BeFalse();
            mean.shape.Should().HaveCount(2).And.ContainInOrder(1, 1);
            mean.GetValue(0).Should().BeEquivalentTo(21);
        }

        [Test]
        public void Case0_Scalar()
        {
            var a = NDArray.Scalar(1);
            var ret = a.sum();
            ret.Shape.IsScalar.Should().BeTrue();
            ret.GetInt32(0).Should().Be(1);
        }

        [Test]
        public void Case0_Scalar_Axis0()
        {
            var a = NDArray.Scalar(1);
            var ret = a.sum(0);
            ret.Shape.IsScalar.Should().BeTrue();
            ret.GetInt32(0).Should().Be(1);
        }

        [Test]
        public void Case1_Elementwise()
        {
            var a = np.ones((3, 3, 3), NPTypeCode.Int32);
            var ret = a.sum();
            ret.Shape.IsScalar.Should().BeTrue();
            ret.GetInt32(0).Should().Be(3 * 3 * 3);
        }

        [Test]
        public void Case1_Axis0()
        {
            var a = np.ones((3, 3, 3), NPTypeCode.Int32);
            var ret = a.sum(0);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(9);
            FluentExtension.Should(ret.Shape).Be(new Shape(3, 3));
            ret.GetTypeCode.Should().Be(a.GetTypeCode);
            ret.Cast<int>().Should().AllBeEquivalentTo(3);
        }

        [Test]
        public void Case1_Axis1()
        {
            var a = np.ones((3, 3, 3), NPTypeCode.Int32);
            var ret = a.sum(1);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(9);
            FluentExtension.Should(ret.Shape).Be(new Shape(3, 3));
            ret.Cast<int>().Should().AllBeEquivalentTo(3);
        }

        [Test]
        public void Case1_Axis2()
        {
            var a = np.ones((3, 3, 3), NPTypeCode.Int32);
            var ret = a.sum(2);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(9);
            FluentExtension.Should(ret.Shape).Be(new Shape(3, 3));
            ret.Cast<int>().Should().AllBeEquivalentTo(3);
        }

        [Test]
        public void Case1_Axis_minus1()
        {
            var a = np.ones((3, 3, 3), NPTypeCode.Int32);
            var ret = a.sum(-1);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(9);
            FluentExtension.Should(ret.Shape).Be(new Shape(3, 3));
            ret.Cast<int>().Should().AllBeEquivalentTo(3);
        }

        [Test]
        public void Case1_Axis2_keepdims()
        {
            var a = np.ones((3, 3, 3), NPTypeCode.Int32);
            var ret = a.sum(2, keepdims: true);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(9);
            FluentExtension.Should(ret.Shape).Be(new Shape(3, 3, 1));
            ret.Cast<int>().Should().AllBeEquivalentTo(3);
        }

        [Test]
        public void Case1_Axis_minus1_keepdims()
        {
            var a = np.ones((3, 3, 3), NPTypeCode.Int32);
            var ret = a.sum(-1, keepdims: true);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(9);
            FluentExtension.Should(ret.Shape).Be(new Shape(3, 3, 1));
            ret.Cast<int>().Should().AllBeEquivalentTo(3);
        }

        [Test]
        public void Case1_Axis_1_keepdims()
        {
            var a = np.ones((3, 3, 3), NPTypeCode.Int32);
            var ret = a.sum(1, keepdims: true);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(9);
            FluentExtension.Should(ret.Shape).Be(new Shape(3, 1, 3));
            ret.Cast<int>().Should().AllBeEquivalentTo(3);
        }


        [Test]
        public void Case2_Elementwise()
        {
            var a = np.ones((2, 1, 3, 5, 1), np.int32);

            var ret = a.sum();
            ret.Shape.IsScalar.Should().BeTrue();
            ret.GetInt32(0).Should().Be(2 * 1 * 3 * 5 * 1);
        }

        [Test]
        public void Case2_Axis0()
        {
            var a = np.ones((2, 1, 3, 5, 1), np.int32);
            var ret = a.sum(0);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(15);
            FluentExtension.Should(ret.Shape).Be(new Shape(1, 3, 5, 1));
            ret.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [Test]
        public void Case2_Axis1()
        {
            var a = np.ones((2, 1, 3, 5, 1), np.int32);
            var ret = a.sum(1);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(30);
            FluentExtension.Should(ret.Shape).Be(new Shape(2, 3, 5, 1));
            ret.Cast<int>().Should().AllBeEquivalentTo(1);
        }

        [Test]
        public void Case2_Axis2()
        {
            var a = np.ones((2, 1, 3, 5, 1), np.int32);
            var ret = a.sum(2);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(10);
            FluentExtension.Should(ret.Shape).Be(new Shape(2, 1, 5, 1));
            ret.Cast<int>().Should().AllBeEquivalentTo(3);
        }

        [Test]
        public void Case2_Axis4()
        {
            var a = np.ones((2, 1, 3, 5, 1), np.int32);
            var ret = a.sum(4);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(30);
            FluentExtension.Should(ret.Shape).Be(new Shape(2, 1, 3, 5));
            ret.Cast<int>().Should().AllBeEquivalentTo(1);
        }

        [Test]
        public void Case2_Axis_minus1()
        {
            var a = np.ones((2, 1, 3, 5, 1), np.int32);
            var ret = a.sum(-1);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(30);
            FluentExtension.Should(ret.Shape).Be(new Shape(2, 1, 3, 5));
            ret.Cast<int>().Should().AllBeEquivalentTo(1);
        }

        [Test]
        public void Case2_Axis2_keepdims()
        {
            var a = np.ones((2, 1, 3, 5, 1), np.int32);
            var ret = a.sum(2, keepdims: true);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(10);
            FluentExtension.Should(ret.Shape).Be(new Shape(2, 1, 1, 5, 1));
            ret.Cast<int>().Should().AllBeEquivalentTo(3);
        }

        [Test]
        public void Case2_Axis_minus1_keepdims()
        {
            var a = np.ones((2, 1, 3, 5, 1), np.int32);
            var ret = a.sum(-1, keepdims: true);
            ret.Shape.IsScalar.Should().BeFalse();
            ret.size.Should().Be(30);
            FluentExtension.Should(ret.Shape).Be(new Shape(2, 1, 3, 5, 1));
            ret.Cast<int>().Should().AllBeEquivalentTo(1);
        }

        [Test]
        public void Case3_TurnIntoScalar()
        {
            NDArray a;
            NDArray ret;
            // >>> a = np.array(5)
            // >>> print(np.prod(a))
            // >>> print(np.prod(a).shape)
            // 5
            // ()

            a = np.array(5);
            ret = np.prod(a);
            FluentExtension.Should(ret).BeScalar(5);

            ret = np.sum(a);
            FluentExtension.Should(ret).BeScalar(5);
        }

        /// <summary>
        /// Tests that all reduction operations with keepdims=true preserve dimensions
        /// for all supported numeric dtypes. NumPy behavior: shape (M, N) with keepdims=true
        /// should return shape (1, 1), not (1) or ().
        /// </summary>
        [Test]
        [Arguments(NPTypeCode.Int32)]
        [Arguments(NPTypeCode.Int64)]
        [Arguments(NPTypeCode.Single)]
        [Arguments(NPTypeCode.Double)]
        [Arguments(NPTypeCode.Int16)]
        [Arguments(NPTypeCode.Byte)]
        public void Keepdims_AllReductions_PreservesDimensions(NPTypeCode dtype)
        {
            // Create 2D array with specific dtype
            var arr = np.arange(6).reshape(2, 3).astype(dtype);

            // Test np.sum keepdims
            var sumResult = np.sum(arr, keepdims: true);
            sumResult.ndim.Should().Be(2, $"np.sum(dtype={dtype}) should preserve 2 dimensions with keepdims=true");
            sumResult.shape[0].Should().Be(1);
            sumResult.shape[1].Should().Be(1);

            // Test np.mean keepdims (float types only for mean)
            var meanResult = np.mean(arr, keepdims: true);
            meanResult.ndim.Should().Be(2, $"np.mean(dtype={dtype}) should preserve 2 dimensions with keepdims=true");
            meanResult.shape[0].Should().Be(1);
            meanResult.shape[1].Should().Be(1);

            // Test np.amax keepdims
            var amaxResult = np.amax(arr, keepdims: true);
            amaxResult.ndim.Should().Be(2, $"np.amax(dtype={dtype}) should preserve 2 dimensions with keepdims=true");
            amaxResult.shape[0].Should().Be(1);
            amaxResult.shape[1].Should().Be(1);

            // Test np.amin keepdims
            var aminResult = np.amin(arr, keepdims: true);
            aminResult.ndim.Should().Be(2, $"np.amin(dtype={dtype}) should preserve 2 dimensions with keepdims=true");
            aminResult.shape[0].Should().Be(1);
            aminResult.shape[1].Should().Be(1);

            // Test np.prod keepdims
            var prodResult = np.prod(arr, keepdims: true);
            prodResult.ndim.Should().Be(2, $"np.prod(dtype={dtype}) should preserve 2 dimensions with keepdims=true");
            prodResult.shape[0].Should().Be(1);
            prodResult.shape[1].Should().Be(1);
        }

        /// <summary>
        /// Tests that keepdims=true works correctly for 3D arrays
        /// </summary>
        [Test]
        public void Keepdims_3D_PreservesThreeDimensions()
        {
            var arr = np.arange(24).reshape(2, 3, 4);

            var result = np.sum(arr, keepdims: true);
            result.ndim.Should().Be(3, "3D array with keepdims=true should return 3D");
            result.shape[0].Should().Be(1);
            result.shape[1].Should().Be(1);
            result.shape[2].Should().Be(1);
        }
    }
}
