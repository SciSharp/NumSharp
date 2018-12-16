using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using System.Numerics;
using NumSharp.UnitTest.Shared;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public  class NDArrayLinSqTester
    {
        [TestMethod]
        public void DefaultTest()
        {
            double[] a = { 0, 1, 1, 1,2,1,3,1};

            var A = np.array(a,typeof(double)).reshape(4,2);

            double[] b =    {-1,0.2,0.9,2.1};

            var B = np.array(b,typeof(double)).reshape(4,1);

            var C = A.lstqr(B);

            
        }
    }
}