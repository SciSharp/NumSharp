using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.LinearAlgebra
{
    public class TransposeTest
    {
        [Test]
        public void TransposeVector()
        {
            // NumPy: transpose of 1D array returns the array itself (view semantics)
            // Modifying x should also modify y since they share memory
            var x = np.arange(4);
            var y = np.transpose(x);
            x[0] = 3;
            Assert.IsTrue(Enumerable.SequenceEqual(x.Data<int>(), y.Data<int>()), "Transpose should share memory with original (view semantics)");
        }

        [Test]
        public void Transpose3x2()
        {
            var x = np.arange(6).reshape(3, 2).MakeGeneric<int>();

            var y = np.transpose(x).MakeGeneric<int>();

            // TODO, This should work
            // Assert.IsTrue(Enumerable.SequenceEqual(y.Data<int>(), new int[] { 0, 2, 4, 1, 3, 5 }));

            Assert.AreEqual(y[0, 0], 0);
            Assert.AreEqual(y[0, 1], 2);
            Assert.AreEqual(y[0, 2], 4);
            Assert.AreEqual(y[1, 0], 1);
            Assert.AreEqual(y[1, 1], 3);
            Assert.AreEqual(y[1, 2], 5);
        }
    }
}
