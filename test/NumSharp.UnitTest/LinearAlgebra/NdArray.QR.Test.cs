using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using System.Numerics;
using NumSharp.UnitTest.Shared;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    /// <summary>
    /// Test concolve with standard example from 
    /// https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html
    /// </summary>
    [TestClass]
    public class NdArrayQRTest : TestBase
    {
        [TestMethod]
        public void FullMatrix()
        {
            var nd1 = np.array(new double[]{1,1,0,1,0,1,0,1,1}).reshape(3,3);

            var T = nd1.qr();
        }
        
    }
}