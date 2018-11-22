using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using NumSharp.Core;

namespace NumSharp.UnitTest.Docs
{
    [TestClass]
    public class LinearAlgebraTester
    {
        NumPy np = new NumPy();

        [TestMethod]
        public void LinearRegression()
        {
            // the time array 
            var time = np.linspace<double>(0, 10, 1000);
            
            // the values over t - linear dependence with noice 
            var values = time * 3 + 5 + np.random.randn(time.Size);
            var A = np.vstack<double>(np.ones<double>(1000), time);
            var A_T = A.transpose(); 
            var A_T_A = A_T.dot<double>(A);

            //var A_PseudoInv = A.transpose().dot<double>(A).inv();

            //var param = A_PseudoInv.dot(A.transpose());

            //param = param.dot(param);

        }
    }
}
