using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class NDOffsetIncrementorTests
    {
        [TestMethod]
        public void Case1()
        {
            var shape = new Shape(3, 3, 1, 2);
            var sh = new NDOffsetIncrementor(shape);
            sh.Next().Should().Be(0);
            sh.Next().Should().Be(shape.GetOffset(0, 0, 0, 1));
            sh.Next().Should().Be(shape.GetOffset(0, 1, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(0, 1, 0, 1));
            sh.Next().Should().Be(shape.GetOffset(0, 2, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(0, 2, 0, 1));
            sh.Next().Should().Be(shape.GetOffset(1, 0, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(1, 0, 0, 1));
            sh.Next().Should().Be(shape.GetOffset(1, 1, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(1, 1, 0, 1));
            sh.Next().Should().Be(shape.GetOffset(1, 2, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(1, 2, 0, 1));
            sh.Next().Should().Be(shape.GetOffset(2, 0, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(2, 0, 0, 1));
            sh.Next().Should().Be(shape.GetOffset(2, 1, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(2, 1, 0, 1));
            sh.Next().Should().Be(shape.GetOffset(2, 2, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(2, 2, 0, 1));
            sh.Next().Should().Be(-1);
        }

        [TestMethod]
        public void Case2()
        {
            var shape = new Shape(1, 1, 1, 3);
            var sh = new NDOffsetIncrementor(shape);
            sh.Next().Should().Be(0);
            sh.Next().Should().Be(shape.GetOffset(0, 0, 0, 1));
            sh.Next().Should().Be(shape.GetOffset(0, 0, 0, 2));

            sh.Next().Should().Be(-1);
        }

        [TestMethod]
        public void Case3()
        {
            var shape = new Shape(3, 1, 1, 1);
            var sh = new NDOffsetIncrementor(shape);
            sh.Next().Should().Be(0);
            sh.Next().Should().Be(shape.GetOffset(1, 0, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(2, 0, 0, 0));

            sh.Next().Should().Be(-1);
        }

        [TestMethod]
        public void Case4()
        {
            var shape = new Shape(1, 1, 3, 1);
            var sh = new NDOffsetIncrementor(shape);
            sh.Next().Should().Be(0);
            sh.Next().Should().Be(shape.GetOffset(0, 0, 1, 0));
            sh.Next().Should().Be(shape.GetOffset(0, 0, 2, 0));

            sh.Next().Should().Be(-1);
        }

        [TestMethod]
        public void Case5()
        {
            var shape = new Shape(2, 1, 3, 1);
            var sh = new NDOffsetIncrementor(shape);
            sh.Next().Should().Be(0);
            sh.Next().Should().Be(shape.GetOffset(0, 0, 1, 0));
            sh.Next().Should().Be(shape.GetOffset(0, 0, 2, 0));
            sh.Next().Should().Be(shape.GetOffset(1, 0, 0, 0));
            sh.Next().Should().Be(shape.GetOffset(1, 0, 1, 0));
            sh.Next().Should().Be(shape.GetOffset(1, 0, 2, 0));

            sh.Next().Should().Be(-1);
        }

        [TestMethod]
        public void Case6()
        {
            var shape = new Shape(1);
            var sh = new NDOffsetIncrementor(shape);
            sh.Next().Should().Be(0);

            sh.Next().Should().Be(-1);
        }
        [TestMethod]
        public void Case7()
        {
            var shape = new Shape(2);
            var sh = new NDOffsetIncrementor(shape);
            sh.Next().Should().Be(0);
            sh.Next().Should().Be(shape.GetOffset(1));

            sh.Next().Should().Be(-1);
        }

        [TestMethod]
        public void Case8()
        {
            var shape = new Shape(100);
            var sh = new NDOffsetIncrementor(shape);
            sh.Next().Should().Be(0);
            sh.Next().Should().Be(shape.GetOffset(1));
            sh.Next().Should().Be(shape.GetOffset(2));
        }

        [TestMethod]
        public void Case9()
        {
            var shape = new Shape(0);
            var sh = new NDOffsetIncrementor(shape);
            sh.Next().Should().Be(0);

            sh.Next().Should().Be(-1);
        }
    }
}
