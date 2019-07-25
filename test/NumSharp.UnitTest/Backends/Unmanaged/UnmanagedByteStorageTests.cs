using System.Linq;
using System.Numerics;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class UnmanagedByteStorageTests
    {
        [TestMethod]
        public void Set()
        {
            var nd = new UnmanagedByteStorage<int>(new UnmanagedMemoryBlock<int>(25, 0), new Shape(5, 5));

            //fill
            for (int i = 0; i < nd.Count; i++)
            {
                nd.SetIndex(i, i);
            }

            //now get slice
            var ret = nd.Get(3);
            ret.Shape.NDim.Should().Be(1);
            ret.Shape.Size.Should().Be(5);
            ret.Shape.Dimensions[0].Should().Be(5);
            for (int i = 0; i < ret.Count; i++)
            {
                ret.SetIndex(33, i);
            }

            nd.Set(ret, 1);

            for (int i = 0; i < 5; i++)
            {
                nd.GetIndex(1 * 5 + i).Should().Be(33);
            }

            //todo so now we need to unit test this here
            //todo get my head wrapped around what kinds of sets are there and unit test them all.
        }

        [TestMethod]
        public void Set_Basic()
        {
            const int length = 30;
            var to = new UnmanagedMemoryBlock<int>(length);
            var tovec = new UnmanagedByteStorage<int>(to, new Shape(10, length / 10));
            var setvec = new UnmanagedByteStorage<int>(Enumerable.Range(100, length / 10).ToArray(), new Shape(length / 10));

            tovec.Set(setvec, 3);
            for (int i = 0; i < 3; i++)
            {
                tovec[3, i].GetIndex(0).Should().Be(100 + i);
            }
        }

        [TestMethod]
        public void ScalarCreation()
        {
            UnmanagedByteStorage<int>.Scalar(5).GetIndex(0).Should().Be(5);
            UnmanagedByteStorage<double>.Scalar(5d).GetIndex(0).Should().Be(5d);
            UnmanagedByteStorage<float>.Scalar(5f).GetIndex(0).Should().Be(5f);
            UnmanagedByteStorage<long>.Scalar(5L).GetIndex(0).Should().Be(5L);
        }

        [TestMethod]
        public void ScalarComplex()
        {
            UnmanagedByteStorage<Complex>.Scalar(new Complex(5, 5)).GetIndex(0).Should().Be(new Complex(5, 5));
        }

        [TestMethod]
        public void ScalarBoolean()
        {
            UnmanagedByteStorage<bool>.Scalar(false).GetIndex(0).Should().Be(false);
        }

        [TestMethod]
        public void SetData_Vector()
        {
            var arr = np.zeros(new Shape(10, 10), NPTypeCode.Double);
            var other = np.ones(Shape.Vector(10), NPTypeCode.Double);
            arr.SetData(other, 0);
            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    arr.GetDouble(i, j).Should().Be(1);
                }
            }
        }

        [TestMethod]
        public void SetData_Scalar()
        {
            var arr = np.zeros(new Shape(10, 10), NPTypeCode.Double);
            var other = NDArray.Scalar(1, NPTypeCode.Double);
            arr.SetData(other, 0, 3);
            arr.GetDouble(0, 3).Should().Be(1);
        }

        [TestMethod]
        public void SetData_ND()
        {
            var arr = np.zeros(new Shape(10, 10), NPTypeCode.Double);
            var other = np.ones(new Shape(10, 10), NPTypeCode.Double);
            arr.SetData(other);
            arr.Storage.InternalArray.As<ArraySlice<double>>().All(v => v == 1).Should().Be(true);
        }

        [TestMethod]
        public void GetData_Vector()
        {
            var arr = np.ones(new Shape(10, 10), NPTypeCode.Double);
            var other = arr[0];
            other.size.Should().Be(10);
            for (int i = 0; i < other.size; i++)
            {
                other.GetAtIndex<double>(i).Should().Be(1);
            }
        }

        [TestMethod]
        public void GetData_Scalar()
        {
            var arr = np.ones(new Shape(10, 10), NPTypeCode.Double);
            var other = arr[0, 0];
            other.Shape.IsScalar.Should().BeTrue();
            other.size.Should().Be(1);
            for (int i = 0; i < other.size; i++)
            {
                other.GetAtIndex<double>(i).Should().Be(1);
            }
        }

        [TestMethod]
        public void GetData_ND()
        {
            var arr = np.zeros(new Shape(1, 10, 10), NPTypeCode.Double);
            arr[0].Shape.dimensions.Should().ContainInOrder(10, 10);
        }
    }
}
