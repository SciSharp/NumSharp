using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class MatrixInvTest
    {
        [TestMethod]
        public void BasisCheck()
        {
            var np = new Matrix<double>("1 2;3 4");

            var np2 = np.Inv();

            var np3 = np.Dot(np2);
        }
    }
}
