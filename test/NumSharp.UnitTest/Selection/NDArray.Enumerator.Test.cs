using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Selection
{
    public class EnumeratorTest
    {
        [TestMethod]
        public void Enumerate_1D_YieldsScalars()
        {
            // 1D arrays iterate over scalar elements
            var nd = np.arange(5);
            var values = nd.Cast<long>().ToList();
            values.Should().BeEquivalentTo(new long[] { 0, 1, 2, 3, 4 });
        }

        [TestMethod]
        public void Enumerate_2D_YieldsRows()
        {
            // 2D arrays iterate over rows (1D slices)
            var nd = np.arange(6).reshape(2, 3);
            var rows = nd.Cast<NDArray>().ToList();

            rows.Count.Should().Be(2);
            rows[0].Should().BeOfValues(0, 1, 2);
            rows[1].Should().BeOfValues(3, 4, 5);
        }

        [TestMethod]
        public void Enumerate_3D_Yields2DSlices()
        {
            // 3D arrays iterate over 2D slices
            var nd = np.arange(12).reshape(2, 3, 2);
            var slices = nd.Cast<NDArray>().ToList();

            slices.Count.Should().Be(2);
            slices[0].ndim.Should().Be(2);
            slices[0].shape.Should().BeEquivalentTo(new long[] { 3, 2 });
        }

        [TestMethod]
        public void Flat_IteratesAllElements()
        {
            // .flat always iterates over all elements as scalars
            var nd = np.arange(12).reshape(2, 3, 2);
            var elements = nd.flat.Cast<long>().ToList();

            elements.Count.Should().Be(12);
            elements.Should().BeEquivalentTo(Enumerable.Range(0, 12).Select(i => (long)i));
        }
    }
}
