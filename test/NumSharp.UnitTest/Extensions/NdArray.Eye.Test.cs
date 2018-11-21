using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayEyeTest
    {
        [TestMethod]
        public void SimpleIntMatrix()
        {
            var np = new NDArrayGeneric<int>().Eye(5);

            Assert.IsTrue(np[0,0] == 1);
            Assert.IsTrue(np[1,1] == 1);
            Assert.IsTrue(np[2,2] == 1);
            Assert.IsTrue(np[3,3] == 1);
            Assert.IsTrue(np[4,4] == 1);

            int[] elementsZero = np.Data.Where((x) => x == 0 ).ToArray();

            Assert.IsTrue(elementsZero.Length == 20);
        }
        [TestMethod]
        public void SimpleDoubleMatrix()
        {
            var np = new NDArrayGeneric<double>().Eye(5);

            Assert.IsTrue(np[0,0] == 1);
            Assert.IsTrue(np[1,1] == 1);
            Assert.IsTrue(np[2,2] == 1);
            Assert.IsTrue(np[3,3] == 1);
            Assert.IsTrue(np[4,4] == 1);

            double[] elementsZero = np.Data.Where((x) => x == 0.0 ).ToArray();

            Assert.IsTrue(elementsZero.Length == 20);
        }
        [TestMethod]
        public void DoubleMatrix2DiagonalLeft()
        {
            var np = new NDArrayGeneric<double>().Eye(5,-2);

            Assert.IsTrue(np[2,0] == 1);
            Assert.IsTrue(np[3,1] == 1);
            Assert.IsTrue(np[4,2] == 1);

            Assert.IsTrue(np.Data.Where(x => x ==0).ToArray().Length ==22);        
        }
        [TestMethod]
        public void DoubleMatrix2DiagonalRight()
        {
            var np = new NDArrayGeneric<double>().Eye(5,2);

            Assert.IsTrue(np[0,2] == 1);
            Assert.IsTrue(np[1,3] == 1);
            Assert.IsTrue(np[2,4] == 1);

            Assert.IsTrue(np.Data.Where(x => x ==0).ToArray().Length ==22);        
        }

    }
}
