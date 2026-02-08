using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using AwesomeAssertions;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.View
{
    [TestClass]
    public class ShapeUnmanagedTests
    {
        private unsafe void TestGetOffset(Shape shape, int[] indices)
        {
            fixed (int* p = &indices[0])
            {
                var managed_result = shape.GetOffset(indices);
                shape.GetOffset(p, indices.Length).Should().Be(managed_result);
            }
        }

        [TestMethod]
        public void GetOffsetTest_unsliced()
        {
            // unsliced shape
            TestGetOffset(new Shape(1, 2, 3, 4), new int[] { 0, 0, 0, 0 });
            TestGetOffset(new Shape(1, 2, 3, 4), new int[] { 0, 1, 1, 1 });
            TestGetOffset(new Shape(1, 2, 3, 4), new int[] { 0, 1 });
            TestGetOffset(new Shape(3, 3, 3), new int[] { 2 });
            TestGetOffset(new Shape(3, 3, 3), new int[] { 2, 2 });
            TestGetOffset(new Shape(3, 3, 3), new int[] { 2, 2, 2 });

        }

        [TestMethod]
        public void GetOffsetTest_sliced()
        {
            new Shape(10, 10, 10).Slice("2:8, ::-2, 7").Should().Be(new Shape(6, 5));
            // sliced shape
            TestGetOffset(new Shape(10, 10, 10).Slice("2:8, ::-2, 7"), new int[] { 0, 0 });
            TestGetOffset(new Shape(10, 10, 10).Slice("2:8, ::-2, 7"), new int[] { 3, 2 });
            TestGetOffset(new Shape(10, 10, 10).Slice("2:8, ::-2, 7"), new int[] { 4, 4 });
            TestGetOffset(new Shape(10, 10, 10).Slice("2:8, ::-2, 7"), new int[] { 5, 4 });
        }

    }
}
