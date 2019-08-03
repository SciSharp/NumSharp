using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using FluentAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NumPyArangeTest
    {
        [TestMethod]
        public void arange()
        {
            var n = np.arange(3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] {0, 1, 2}));

            n = np.arange(3, 7);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] {3, 4, 5, 6}));

            n = np.arange(3.0, 7.0, 2.0);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<double>(), new double[] {3, 5}));

            n = np.arange(0, 11, 3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] {0, 3, 6, 9}));

            // Test increments < 1
            var startd = 0.0;
            var stopd = 12.0;
            var incrementd = 0.1;
            n = np.arange(startd, stopd, incrementd);
            var r = n.Data<double>();
            var t = Enumerable.Repeat(0, (int)((stopd - startd) / incrementd)).Select((tr, ti) => tr + incrementd * ti);
            Assert.IsTrue(r.Count == 120);
            Assert.IsTrue(Enumerable.SequenceEqual(r, t));
        }

        [TestMethod]
        public void arange_negative()
        {
            np.arange(3, 0, -1).Array.Should().ContainInOrder(3,2,1);
            np.arange(3, 0, -2).Array.Should().ContainInOrder(3, 1);
            np.arange(3d, 0d, -1d).Array.Should().ContainInOrder(3, 2, 1);
            np.arange(3d, 0d, -2d).Array.Should().ContainInOrder(3, 1);
            np.arange(3f, 0f, -1f).Array.Should().ContainInOrder(3, 2, 1);
            np.arange(3f, 0f, -2f).Array.Should().ContainInOrder(3, 1);
        }

        [TestMethod]
        public void arange_case2()
        {
            np.arange(10, 1, -1).Should().ContainInOrder(10, 9, 8, 7, 6, 5, 4, 3, 2);
            np.arange(10d, 1d, -1d).Array.Should().ContainInOrder(10d, 9d, 8d, 7d, 6d, 5d, 4d, 3d, 2);
            np.arange(10f, 1f, -1f).Array.Should().ContainInOrder(10f, 9f, 8f, 7f, 6f, 5f, 4f, 3f, 2);

        }
    }
}
