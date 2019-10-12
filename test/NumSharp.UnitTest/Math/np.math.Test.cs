using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class ApiMathTest
    {
        [TestMethod]
        public void AddInt32()
        {
            var x = np.arange(3);
            var y = np.arange(3);
            var z = np.add(x, y);
            z.Should().BeOfValues(0, 2, 4);

            x = np.arange(9);
            y = np.arange(9);
            z = np.add(x, y);
            z.Should().BeOfValues(0, 2, 4, 6, 8, 10, 12, 14, 16);
        }

        [TestMethod]
        public void DivideInt32()
        {
            var x = np.arange(1, 4);
            var y = np.arange(1, 4);
            var z = np.divide(x, y);
            z.Should().BeOfValues(1, 1, 1);

            x = np.arange(1, 10);
            y = np.arange(1, 10);
            z = np.divide(x, y);
            z.Should().BeOfValues(1, 1, 1, 1, 1, 1, 1, 1, 1);
        }

        [TestMethod]
        public void Sum2x2Int32()
        {
            var data = new int[,] {{0, 1}, {0, 5}};

            int s = np.sum(data);
            Assert.AreEqual(s, 6);

            var s0 = np.sum(data, axis: 0);
            s0.Should().BeOfValues(0, 6).And.BeShaped(2);

            var s1 = np.sum(data, axis: 1);
            s1.Should().BeOfValues(1, 5).And.BeShaped(2); ;

            var s2 = np.sum(data, axis: -1);
            s1.Should().BeOfValues(1, 5).And.BeShaped(2); ;
        }

        [TestMethod]
        public void Sum2x3x2Int32()
        {
            var data = np.arange(12).reshape(2, 3, 2);

            int s = (int) np.sum(data);
            Assert.AreEqual(s, 66);

            var s0 = np.sum(data, axis: 0);
            s0.Should().BeOfValues(6, 8, 10, 12, 14, 16).And.BeShaped(3,2);

            var s1 = np.sum(data, axis: 1);
            s1.Should().BeOfValues(6, 9, 24, 27).And.BeShaped(2, 2);

            var s2 = np.sum(data, axis: 2);
            s2.Should().BeOfValues(1, 5, 9, 13, 17, 21).And.BeShaped(2, 3);

            var s3 = np.sum(data, axis: -1);
            s3.Should().BeOfValues(1, 5, 9, 13, 17, 21).And.BeShaped(2,3);
        }

        [TestMethod]
        public void AddUInt8()
        {
            var x = np.arange(3).astype(np.uint8);
            var y = np.arange(3).astype(np.uint8);
            var z = np.add(x, y);
            z.Should().BeOfValues(0, 2, 4);

            x = np.arange(9).astype(np.uint8);
            y = np.arange(9).astype(np.uint8);
            z = np.add(x, y);
            z.Should().BeOfValues(0, 2, 4, 6, 8, 10, 12, 14, 16);
        }

        [TestMethod]
        public void DivideUInt8()
        {
            var x = np.arange(1, 4).astype(np.uint8);
            var y = np.arange(1, 4).astype(np.uint8);
            var z = np.divide(x, y);
            z.Should().BeOfValues(1, 1, 1);

            x = np.arange(1, 10).astype(np.uint8);
            y = np.arange(1, 10).astype(np.uint8);
            z = np.divide(x, y);
            z.Should().BeOfValues(1, 1, 1, 1, 1, 1, 1, 1, 1);
        }

        [TestMethod]
        public void AddUInt16()
        {
            var x = np.arange(3).astype(np.uint16);
            var y = np.arange(3).astype(np.uint16);
            var z = np.add(x, y);
            z.Should().BeOfValues(0, 2, 4);

            x = np.arange(9).astype(np.uint16);
            y = np.arange(9).astype(np.uint16);
            z = np.add(x, y);
            z.Should().BeOfValues(0, 2, 4, 6, 8, 10, 12, 14, 16);
        }

        [TestMethod]
        public void DivideUInt16()
        {
            var x = np.arange(1, 4).astype(np.uint16);
            var y = np.arange(1, 4).astype(np.uint16);
            var z = np.divide(x, y);
            z.Should().BeOfValues(1, 1, 1);

            x = np.arange(1, 10).astype(np.uint16);
            y = np.arange(1, 10).astype(np.uint16);
            z = np.divide(x, y);
            z.Should().BeOfValues(1, 1, 1, 1, 1, 1, 1, 1, 1);
        }

        [TestMethod]
        public void Maximum_Slice()
        {
            var boxes1 = np.array(new double[] { 12.875, 14.125, 39.75, 49 }).reshape(1, 4);
            var boxes2 = np.array(new double[]
            {
                25.875, 30.6875, 27.125, 32.3125,
                25.5, 29.625 , 27.5, 33.375,
                24.4375, 30.0625, 28.5625, 32.9375
            }).reshape(3, 4);

            var x = boxes1[Slice.Ellipsis, new Slice(":2")];
            var y = boxes2[Slice.Ellipsis, new Slice(":2")];

            var z = np.maximum(x, y);
            z.Should().BeOfValues( 25.875, 30.6875 );
        }
    }
}
