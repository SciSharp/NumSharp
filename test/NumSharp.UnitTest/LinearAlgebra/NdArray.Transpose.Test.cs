using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using FluentAssertions;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class TransposeTest
    {
        [TestMethod]
        public void TransposeVector()
        {
            var x = np.arange(4);
            var y = np.transpose(x);
            x[0] = 3;
            Assert.IsTrue(Enumerable.SequenceEqual(x.Data<int>(), y.Data<int>()));
        }

        [TestMethod]
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

        [TestMethod, Ignore("No actual asserts inside")] //todo!
        public void Transpose10x10()
        {
            new Action(() =>
            {
                var array = np.arange(100).reshape(3, 3, 3);
                array = array.transpose();
                for (var i = 0; i < array.shape[0]; i++)
                {
                    for (var j = 0; j < array.shape[1]; j++)
                    {
                        Console.WriteLine(array[i, j].ToString());
                    }
                }
            }).Should().NotThrow("It has to run completely.");
        }
    }
}
