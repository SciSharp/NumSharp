using System.Linq;
using System.Numerics;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
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
        public void GetSubshape()
        {
            //initialize
            (Shape Shape, int Offset) ret;
            var nd = new UnmanagedByteStorage<int>(new UnmanagedMemoryBlock<int>(25, 0), new Shape(5, 5));
            var arr = new int[5, 5];
            var arr2 = new int[5, 1, 5];

            for (int i = 0; i < nd.Count; i++)
            {
                nd.SetIndex(i, i);
            }

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    arr[i, j] = i * 5 + j;
                }
            }

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 1; j++)
                {
                    for (int k = 0; k < 5; k++)
                    {
                        arr2[i, j, k] = i * 5 + j * 1 + k;
                    }
                }
            }


            //test case 1
            nd.Shape = new Shape(5, 5);

            ret = nd.Shape.GetSubshape(0, 0);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(0);
            arr[0, 0].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(1, 0);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5);
            arr[1, 0].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(1, 4);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5 + 4);
            arr[1, 4].Should().Be(ret.Offset);


            //test case 2
            nd.Shape = new Shape(5, 1, 5);
            ret = nd.Shape.GetSubshape(0, 0);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(0);
            arr2[0, 0, 0].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(1, 0);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(5);
            arr2[1, 0, 0].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(1, 0, 1);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5 + 1);
            arr2[1, 0, 1].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(2, 0, 1);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5 * 2 + 1);
            arr2[2, 0, 1].Should().Be(ret.Offset);

            ret = nd.Shape.GetSubshape(0, 0);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(0);

            ret = nd.Shape.GetSubshape(1, 0);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(5);

            ret = nd.Shape.GetSubshape(1, 0, 3);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(5 + 3);
            arr2[1, 0, 3].Should().Be(ret.Offset);


            //test case 3
            nd.Shape = new Shape(1, 1, 5, 5);
            ret = nd.Shape.GetSubshape(0, 0, 3, 3);
            ret.Shape.Size.Should().Be(1);
            ret.Offset.Should().Be(18);

            ret = nd.Shape.GetSubshape(0, 0, 3);
            ret.Shape.Size.Should().Be(5);
            ret.Offset.Should().Be(15);

            ret = ret.Shape.GetSubshape(2);
            ret.Shape.Size.Should().Be(1);
            ret.Shape.NDim.Should().Be(0);
            ret.Shape.IsScalar.Should().BeTrue();


            //test case 4
            nd.Shape = new Shape(1, 5, 5, 1);
            ret = nd.Shape.GetSubshape(0, 1);
            ret.Offset.Should().Be(5);
            ret.Shape.NDim.Should().Be(2);
            ret.Shape.Dimensions[0].Should().Be(5);
            ret.Shape.Dimensions[1].Should().Be(1);
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
    }
}
