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
            // np.arange returns int64 by default (NumPy 2.x)
            var x = np.arange(4);
            var y = np.transpose(x);
            x[0] = 3;
            Assert.IsTrue(Enumerable.SequenceEqual(x.Data<long>(), y.Data<long>()), "Transpose should share memory with original (view semantics)");
        }

        [Test]
        public void Transpose3x2()
        {
            // np.arange returns int64 by default (NumPy 2.x)
            var x = np.arange(6).reshape(3, 2).MakeGeneric<long>();

            var y = np.transpose(x).MakeGeneric<long>();

            // TODO, This should work
            // Assert.IsTrue(Enumerable.SequenceEqual(y.Data<long>(), new long[] { 0, 2, 4, 1, 3, 5 }));

            Assert.AreEqual(y[0, 0], 0L);
            Assert.AreEqual(y[0, 1], 2L);
            Assert.AreEqual(y[0, 2], 4L);
            Assert.AreEqual(y[1, 0], 1L);
            Assert.AreEqual(y[1, 1], 3L);
            Assert.AreEqual(y[1, 2], 5L);
        }
    }
}
