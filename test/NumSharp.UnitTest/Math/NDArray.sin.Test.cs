using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class SinTest
    {
        [TestMethod]
        public void Simple1DArray()
        {
            var nd = np.array(new double[] { 0, 30, 45, 60, 90 }) * (Math.PI / 180);
            
            var nd2 = np.sin(nd);
            
            Assert.IsTrue(nd2.Data<double>(0) == 0);
            Assert.IsTrue(nd2.Data<double>(1) < 0.501);
            Assert.IsTrue(nd2.Data<double>(1) > 0.498);
            Assert.IsTrue(nd2.Data<double>(2) < 0.708);
            Assert.IsTrue(nd2.Data<double>(2) > 0.7069);
            Assert.IsTrue(nd2.Data<double>(3) < 0.867);
            Assert.IsTrue(nd2.Data<double>(3) > 0.8659);
            Assert.IsTrue(nd2.Data<double>(4) == 1);
            
        }
    }
}
