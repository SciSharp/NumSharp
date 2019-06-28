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
    public class NumPyZerosTest
    {
        [TestMethod]
        public void zero()
        {
            var n = np.zeros(3);

            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<double>(), new double[] {0, 0, 0}));
        }

        [TestMethod]
        public void Zeros2Dim()
        {
            var n = np.zeros(3, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<double>(), new double[] {0, 0, 0, 0, 0, 0}));
        }

        [TestMethod]
        public void Zeros1DimWithDtype()
        {
            var n = np.zeros(new Shape(3), np.int32);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] {0, 0, 0}));
        }

        [TestMethod]
        public void SimpleInt1D()
        {
            var np1 = np.zeros(new Shape(5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 5);
        }

        [TestMethod]
        public void SimpleInt2D()
        {
            var np1 = np.zeros(new Shape(5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 25);
        }

        [TestMethod]
        public void SimpleDouble3D()
        {
            var np1 = np.zeros(new Shape(5, 5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 125);
        }        
        
        [TestMethod, Description("When creating an array using zeros with type of NDArray, each value should be unique.")]
        public void DtypeNDArray()
        {
            var nd = np.zeros(new Shape(5, 5, 5), dtype: typeof(NDArray));
            var allocated = nd.GetData<NDArray>();
            for (int i = 0; i < allocated.Length; i++)
            {
                for (int j = 0; j < allocated.Length; j++)
                {
                    if (i == j)
                        continue;

                    allocated[i].Should().NotBeEquivalentTo(allocated[j]);
                }
            }
        }
    }
}
