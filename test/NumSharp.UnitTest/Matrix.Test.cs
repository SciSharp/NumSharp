using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class MatrixTest
    {
        [TestMethod]
        public void CheckToString()
        {
            var matrix = new Matrix<double>("1 2 3;4 5 6;7 8 9");

            //matrix.Data = new double[,] {{1,2,3},{4,5,6},{7,8,9}};

            string matrixAsString = matrix.ToString();

        }
    }
}
