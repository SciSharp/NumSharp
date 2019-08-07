using System;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class NDCoordinatesIncrementorTests : TestClass
    {
        [TestMethod]
        public void Case1()
        {
            var shape = new Shape(3, 3, 1, 2);
            var sh = new NDCoordinatesIncrementor(ref shape); //NDCoordinatesIncrementor
            sh.Index.Should().ContainInOrder(0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 0, 1);
            sh.Next().Should().ContainInOrder(0, 1, 0, 0);
            sh.Next().Should().ContainInOrder(0, 1, 0, 1);
            sh.Next().Should().ContainInOrder(0, 2, 0, 0);
            sh.Next().Should().ContainInOrder(0, 2, 0, 1);
            sh.Next().Should().ContainInOrder(1, 0, 0, 0);
            sh.Next().Should().ContainInOrder(1, 0, 0, 1);
            sh.Next().Should().ContainInOrder(1, 1, 0, 0);
            sh.Next().Should().ContainInOrder(1, 1, 0, 1);
            sh.Next().Should().ContainInOrder(1, 2, 0, 0);
            sh.Next().Should().ContainInOrder(1, 2, 0, 1);
            sh.Next().Should().ContainInOrder(2, 0, 0, 0);
            sh.Next().Should().ContainInOrder(2, 0, 0, 1);
            sh.Next().Should().ContainInOrder(2, 1, 0, 0);
            sh.Next().Should().ContainInOrder(2, 1, 0, 1);
            sh.Next().Should().ContainInOrder(2, 2, 0, 0);
            sh.Next().Should().ContainInOrder(2, 2, 0, 1);
            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case2()
        {
            var shape = new Shape(1, 1, 1, 3);
            var sh = new NDCoordinatesIncrementor(ref shape);
            sh.Index.Should().ContainInOrder(0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 0, 1);
            sh.Next().Should().ContainInOrder(0, 0, 0, 2);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case3()
        {
            var shape = new Shape(3, 1, 1, 1);
            var sh = new NDCoordinatesIncrementor(ref shape);
            sh.Index.Should().ContainInOrder(0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(1, 0, 0, 0);
            sh.Next().Should().ContainInOrder(2, 0, 0, 0);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case4()
        {
            var shape = new Shape(1, 1, 3, 1);
            var sh = new NDCoordinatesIncrementor(ref shape);
            sh.Index.Should().ContainInOrder(0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 1, 0);
            sh.Next().Should().ContainInOrder(0, 0, 2, 0);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case5()
        {
            var shape = new Shape(2, 1, 3, 1);
            var sh = new NDCoordinatesIncrementor(ref shape);
            sh.Index.Should().ContainInOrder(0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 1, 0);
            sh.Next().Should().ContainInOrder(0, 0, 2, 0);
            sh.Next().Should().ContainInOrder(1, 0, 0, 0);
            sh.Next().Should().ContainInOrder(1, 0, 1, 0);
            sh.Next().Should().ContainInOrder(1, 0, 2, 0);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case6()
        {
            var shape = new Shape(1);
            var sh = new NDCoordinatesIncrementor(ref shape);
            sh.Index.Should().ContainInOrder(0);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case7()
        {
            var shape = new Shape(2);
            var sh = new NDCoordinatesIncrementor(ref shape);
            sh.Index.Should().ContainInOrder(0);
            sh.Next().Should().ContainInOrder(1);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case8()
        {
            var shape = new Shape(100);
            var sh = new NDCoordinatesIncrementor(ref shape);
            sh.Index.Should().ContainInOrder(0);
            sh.Next().Should().ContainInOrder(1);
            sh.Next().Should().ContainInOrder(2);
        }

        [TestMethod]
        public void Case9()
        {
            var shape = new Shape(0);
            new Action(() => new NDCoordinatesIncrementor(ref shape)).Should().Throw<InvalidOperationException>();
        }

        [TestMethod]
        public void Case1_Extended()
        {
            var shape = new Shape(3, 3, 1, 2);
            var sh = new NDExtendedCoordinatesIncrementor(shape, 2);
            sh.Index[5] = 55;
            sh.Index.Should().ContainInOrder(0, 0, 0, 0, 0, 55);
            sh.Next().Should().ContainInOrder(0, 0, 0, 1, 0, 55);
            sh.Next().Should().ContainInOrder(0, 1, 0, 0, 0, 55);
            sh.Next().Should().ContainInOrder(0, 1, 0, 1, 0, 55);
            sh.Next().Should().ContainInOrder(0, 2, 0, 0, 0, 55);
            sh.Next().Should().ContainInOrder(0, 2, 0, 1, 0, 55);
            sh.Next().Should().ContainInOrder(1, 0, 0, 0, 0, 55);
            sh.Next().Should().ContainInOrder(1, 0, 0, 1, 0, 55);
            sh.Next().Should().ContainInOrder(1, 1, 0, 0, 0, 55);
            sh.Next().Should().ContainInOrder(1, 1, 0, 1, 0, 55);
            sh.Next().Should().ContainInOrder(1, 2, 0, 0, 0, 55);
            sh.Next().Should().ContainInOrder(1, 2, 0, 1, 0, 55);
            sh.Next().Should().ContainInOrder(2, 0, 0, 0, 0, 55);
            sh.Next().Should().ContainInOrder(2, 0, 0, 1, 0, 55);
            sh.Next().Should().ContainInOrder(2, 1, 0, 0, 0, 55);
            sh.Next().Should().ContainInOrder(2, 1, 0, 1, 0, 55);
            sh.Next().Should().ContainInOrder(2, 2, 0, 0, 0, 55);
            sh.Next().Should().ContainInOrder(2, 2, 0, 1, 0, 55);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case2_Extended()
        {
            var shape = new Shape(1, 1, 1, 3);
            var sh = new NDExtendedCoordinatesIncrementor(shape, 2);
            sh.Index.Should().ContainInOrder(0, 0, 0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 0, 1, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 0, 2, 0, 0);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case3_Extended()
        {
            var shape = new Shape(3, 1, 1, 1);
            var sh = new NDExtendedCoordinatesIncrementor(shape, 2);
            sh.Index.Should().ContainInOrder(0, 0, 0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(1, 0, 0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(2, 0, 0, 0, 0, 0);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case4_Extended()
        {
            var shape = new Shape(1, 1, 3, 1);
            var sh = new NDExtendedCoordinatesIncrementor(shape, 2);
            sh.Index.Should().ContainInOrder(0, 0, 0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 1, 0, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 2, 0, 0, 0);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case5_Extended()
        {
            var shape = new Shape(2, 1, 3, 1);
            var sh = new NDExtendedCoordinatesIncrementor(shape, 2);
            sh.Index.Should().ContainInOrder(0, 0, 0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 1, 0, 0, 0);
            sh.Next().Should().ContainInOrder(0, 0, 2, 0, 0, 0);
            sh.Next().Should().ContainInOrder(1, 0, 0, 0, 0, 0);
            sh.Next().Should().ContainInOrder(1, 0, 1, 0, 0, 0);
            sh.Next().Should().ContainInOrder(1, 0, 2, 0, 0, 0);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case6_Extended()
        {
            var shape = new Shape(1);
            var sh = new NDExtendedCoordinatesIncrementor(shape, 2);
            sh.Index.Should().ContainInOrder(0, 0, 0);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case7_Extended()
        {
            var shape = new Shape(2);
            var sh = new NDExtendedCoordinatesIncrementor(shape, 2);
            sh.Index.Should().ContainInOrder(0);
            sh.Next().Should().ContainInOrder(1);

            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case8_Extended()
        {
            var shape = new Shape(100);
            var sh = new NDExtendedCoordinatesIncrementor(shape, 2);
            sh.Index.Should().ContainInOrder(0, 0, 0);
            sh.Next().Should().ContainInOrder(1, 0, 0);
            sh.Index[2] = 1;
            sh.Next().Should().ContainInOrder(2, 0, 1);
        }

        [TestMethod]
        public void Case9_Extended()
        {
            var shape = new Shape(0);
            new Action(() => new NDExtendedCoordinatesIncrementor(shape, 2)).Should().Throw<InvalidOperationException>();
        }

        [TestMethod]
        public void Case10_Scalar()
        {
            var a = new UnmanagedStorage(17);
            AssertAreEqual(new int[] {17}, a.ToArray<int>());
        }

        [TestMethod]
        public void Case10_Scalar_2()
        {
            var shape = Shape.Scalar;
            var sh = new NDCoordinatesIncrementor(ref shape);
            sh.Index.Should().ContainInOrder(0);
            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case11_Scalar()
        {
            var sh = new NDCoordinatesIncrementor(new int[0]);
            sh.Index.Should().ContainInOrder(0);
            sh.Next().Should().BeNull();
        }

        [TestMethod]
        public void Case10_Scalar_Extended()
        {
            var shape = Shape.Scalar;
            var sh = new NDExtendedCoordinatesIncrementor(shape, 2);
            sh.Index.Should().ContainInOrder(0, 0, 0);
            sh.Next().Should().BeNull();
        }
    }
}
