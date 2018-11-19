using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest.Docs
{
    [TestClass]
    public class LinearAlgebraTester
    {
        [TestMethod]
        public void LinearRegression()
        {
            var np = new NumPy<double>();

            // the time array 
            var time = np.linspace(0,10,1000);
            
            // the values over t - linear dependence with noice 
            var values = time * 3 + 5 + np.random.randn(time.Data.Length);

            var A = np.vstack(np.ones(new Shape(1000)),time);
            var A_T = A.transpose(); 
            var A_T_A = A_T.dot(A);

            var A_PseudoInv = A.transpose().dot(A).inv();

            var param = A_PseudoInv.dot(A.transpose());

            //param = param.dot(param);

        }
    }
}
