using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class NDArrayGetData
    {
        [TestMethod]
        public void Case1_GetData_Nonslice()
        {
            var lhs = np.full(5, (3, 3), NPTypeCode.Int32);
            var slice = lhs.Storage.GetData(0);
            slice.Count.Should().Be(3);
            slice.Shape.IsSliced.Should().BeFalse("Slicing should occurs only when lhs is already sliced.");
            //via for
            for (int i = 0; i < 3; i++) slice.GetValue<int>(i).Should().Be(5);
            //via enumerator
            new NDIterator<int>(slice).Should().ContainInOrder(5, 5, 5);
        }

        [TestMethod]
        public void Case1_GetData_Slice()
        {
            var lhs = np.full(5, (3, 3, 3), NPTypeCode.Int32);
            lhs = lhs["1,:,:"];
            var slice = lhs.Storage.GetData(0);
            slice.Count.Should().Be(3);
            slice.Shape.IsSliced.Should().BeTrue("Slicing should occurs only when lhs is already sliced.");
            //via for
            for (int i = 0; i < 3; i++) slice.GetValue<int>(i).Should().Be(5);
            //via enumerator
            new NDIterator<int>(slice).Should().ContainInOrder(5, 5, 5);
        }

        [TestMethod]
        public void Case1_GetData_Slice2()
        {
            var lhs = np.full(5, (6, 3, 3), NPTypeCode.Int32);
            lhs = lhs["::2,:,:"];
            var slice = lhs.Storage.GetData(0, 0);
            slice.Count.Should().Be(3);
            slice.Shape.IsSliced.Should().BeTrue("Slicing should occurs only when lhs is already sliced.");
            //via for
            for (int i = 0; i < 3; i++) slice.GetValue<int>(i).Should().Be(5);
            //via enumerator
            new NDIterator<int>(slice).Should().ContainInOrder(5, 5, 5);
        }

        [TestMethod]
        public void Case2_GetData_Scalar_Nonslice()
        {
            var lhs = np.full(5, (3, 3), NPTypeCode.Int32);
            var slice = lhs.Storage.GetData(0, 1);
            slice.Count.Should().Be(1);
            slice.Shape.IsScalar.Should().BeTrue();
            slice.Shape.IsSliced.Should().BeFalse("Slicing should occurs only when lhs is already sliced.");
            //via for
            for (int i = 0; i < 1; i++) slice.GetValue<int>(i).Should().Be(5);
            //via enumerator
            new NDIterator<int>(slice).Should().ContainInOrder(5);
        }

        [TestMethod]
        public void Case2_GetData_Scalar_Slice()
        {
            var lhs = np.full(5, (3, 3, 3), NPTypeCode.Int32);
            lhs = lhs["1,:,:"];
            var slice = lhs.Storage.GetData(1, 1, 2);
            slice.Count.Should().Be(1);
            slice.Shape.IsScalar.Should().BeTrue();
            slice.Shape.IsSliced.Should().BeTrue("Slicing should occurs only when lhs is already sliced.");
            //via for
            for (int i = 0; i < 1; i++) slice.GetValue<int>(i).Should().Be(5);
            //via enumerator
            new NDIterator<int>(slice).Should().ContainInOrder(5);
        }

        [TestMethod]
        public void Case2_GetData_Scalar_Slice2()
        {
            var lhs = np.full(5, (6, 3, 3), NPTypeCode.Int32);
            lhs = lhs["::2,:,:"];
            var slice = lhs.Storage.GetData(1, 1, 2);
            slice.Count.Should().Be(1);
            slice.Shape.IsScalar.Should().BeTrue();
            slice.Shape.IsSliced.Should().BeTrue("Slicing should occurs only when lhs is already sliced.");
            //via for
            for (int i = 0; i < 1; i++) slice.GetValue<int>(i).Should().Be(5);
            //via enumerator
            new NDIterator<int>(slice).Should().ContainInOrder(5);
        }

        [TestMethod]
        public void Case3_GetData_All_Slice2()
        {
            var lhs = np.full(5, (6, 3, 3), NPTypeCode.Int32);
            lhs = lhs["::2,:,:"];
            var slice = lhs.Storage.GetData(new int[0]);
            slice.Count.Should().Be(3*3*3);
            slice.Shape.IsScalar.Should().BeFalse();
            slice.Shape.IsSliced.Should().BeTrue("Slicing should occurs only when lhs is already sliced.");

            //via enumerator
            var iter = new NDIterator<int>(slice);
            for (int i = 0; i < 3 * 3 * 3; i++, iter.HasNext()) iter.MoveNext().Should().Be(5);
        }

        [TestMethod]
        public void Case3_GetData_All()
        {
            var lhs = np.full(5, (6, 3, 3), NPTypeCode.Int32);
            var slice = lhs.Storage.GetData(new int[0]);
            slice.Count.Should().Be(6*3*3);
            slice.Shape.IsScalar.Should().BeFalse();
            slice.Shape.IsSliced.Should().BeFalse("Slicing should occurs only when lhs is already sliced.");

            //via enumerator
            var iter = new NDIterator<int>(slice);
            for (int i = 0; i < 6 * 3 * 3; i++, iter.HasNext()) iter.MoveNext().Should().Be(5);
        }

        [TestMethod]
        public void Case1_GetNDArrays_Axis0()
        {
            var a = np.full(5, (6, 3, 3), NPTypeCode.Int32);
            var ret = a.GetNDArrays(0);
            ret.Should().HaveCount(6);
            var f = ret.First();
            f.Shape.Should().Be((3, 3));
        }

        [TestMethod]
        public void Case1_GetNDArrays_Axis1()
        {
            var a = np.full(5, (6, 3, 3), NPTypeCode.Int32);
            var ret = a.GetNDArrays(1);
            ret.Should().HaveCount(6*3);
            var f = ret.First();
            f.Shape.Should().Be(Shape.Vector(3));
        }

        [TestMethod]
        public void Case1_GetNDArrays_Axis2()
        {
            var a = np.full(5, (6, 3, 3), NPTypeCode.Int32);
            var ret = a.GetNDArrays(2);
            ret.Should().HaveCount(6*3*3);
            var f = ret.First();
            f.Shape.Should().Be(Shape.Scalar);
        }

    }
}
