using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NdArrayMakeGenericTester
    {
        [TestMethod]
        public void Array1DimGeneric()
        {
            var list = new double[] { 1.1, 2.2, 3.3 };
            var arrayDouble = np.array(list).MakeGeneric<double>();
            var arrayInt = np.array(list).MakeGeneric<int>();

            Assert.IsTrue(arrayDouble[0] == 1.1);
            Assert.IsTrue(arrayDouble[1] == 2.2);
            Assert.IsTrue(arrayDouble[2] == 3.3);

            Assert.IsTrue(arrayInt[0] == 1);
            Assert.IsTrue(arrayInt[1] == 2);
            Assert.IsTrue(arrayInt[2] == 3);
            
        }    
    }
}
