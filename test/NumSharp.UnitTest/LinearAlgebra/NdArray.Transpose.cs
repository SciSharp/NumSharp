using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class TransposeTest 
    {
        [TestMethod]
        public void TwoxThree()
        {
            var np1 = np.arange(6).reshape(3,2).MakeGeneric<int>();
            
            var np1Transposed = np1.transpose().MakeGeneric<int>();

            Assert.AreEqual(np1Transposed[0,0], 0);
            Assert.AreEqual(np1Transposed[0,1], 2);
            Assert.AreEqual(np1Transposed[0,2], 4);
            Assert.AreEqual(np1Transposed[1,0], 1);
            Assert.AreEqual(np1Transposed[1,1], 3);
            Assert.AreEqual(np1Transposed[1,2], 5);

        }
    }
}
