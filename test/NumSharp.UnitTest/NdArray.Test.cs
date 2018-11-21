using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NDArrayTest
    {
        [TestMethod]
        public void IndexAccessorGetter()
        {
            var np = new NumPy<int>();
            var n = np.reshape(np.arange(12), 3, 4);

            Assert.IsTrue(n[1, 1] == 5);
            Assert.IsTrue(n[2, 0] == 8);

            n = np.reshape(np.arange(12), 2, 3, 2);
            var n1 = n[new Shape(1)];

            Assert.IsTrue(n1[1, 1] == 9);
            Assert.IsTrue(n1[2, 1] == 11);

            var n2 = n[new Shape(1, 2)];

            Assert.IsTrue(n2[0] == 10);
            Assert.IsTrue(n2[1] == 11);
        }

        [TestMethod]
        public void IndexAccessorSetter()
        {
            var np = new NumPy<int>();
            var n = np.reshape(np.arange(12), 3, 4);

            Assert.IsTrue(n[0, 3] == 3);
            Assert.IsTrue(n[1, 3] == 7);

            // set value
            n.Set(new Shape(0), 10);
            Assert.IsTrue(n[0, 0] == 10);
            Assert.IsTrue(n[0, 3] == 10);
            Assert.IsTrue(n[1, 3] == 7);
        }

        [TestMethod]
        public void StringCheck()
        {
            var np = new NDArray<double>().arange(9).reshape(3,3);

            var random = new Random();
            np.Data = np.Data.Select(x => x + random.NextDouble()).ToArray();
            np.Data[1] = 1;
            np.Data[2] -= 4;
            np.Data[3] -= 20;
            np.Data[8] += 23;

            var stringOfNp = np.ToString();

            Assert.IsTrue(stringOfNp.Contains("[[  0."));

            np = new NDArray<double>().arange(9).reshape(3,3);

            stringOfNp = np.ToString();        

            Assert.IsTrue(stringOfNp.Contains("([[ 0,"));
        }
        [TestMethod]
        public void CheckVectorString()
        {
            var np = new NDArray<double>().arange(9);

            var random = new Random();
            np.Data = np.Data.Select(x => x + random.NextDouble()).ToArray();
            np.Data[1] = 1;
            np.Data[2] -= 4;
            np.Data[3] -= 20;
            np.Data[8] += 23;

            var stringOfNp = np.ToString();
        }
        [TestMethod]
        public void DimOrder()
        {
            NDArray<double> np1 = new NDArray<double>().Zeros(2,2);

            np1[0,0] = 0;
            np1[1,0] = 10;
            np1[0,1] = 1;
            np1[1,1] = 11;

            // columns first than rows
            Assert.IsTrue(Enumerable.SequenceEqual(new double[] {0,1,10,11}, np1.Data ));
        }

        [TestMethod]
        public void ToDotNetArray1D()
        {
            var np1 = new NDArray<double>().arange(9);

            double[] np1_ = np1.ToDotNetArray<double[]>();

            Assert.IsTrue(Enumerable.SequenceEqual(np1_,np1.Data));
        } 

        [TestMethod]
        public void ToDotNetArray2D()
        {
            var np1 = new NDArray<double>().arange(9).reshape(3,3);

            double[][] np1_ = np1.ToDotNetArray<double[][]>();

            for (int idx = 0; idx < 3; idx ++)
            {
                for (int jdx = 0; jdx < 3; jdx ++)
                {
                    Assert.IsTrue(np1[idx,jdx] == np1_[idx][jdx]);
                }    
            }
        }

        [TestMethod]
        public void ToDotNetArray3D()
        {
            var np1 = new NDArray<double>().arange(27).reshape(3,3,3);

            double[][][] np1_ = np1.ToDotNetArray<double[][][]>();

            for (int idx = 0; idx < 3; idx ++)
            {
                for (int jdx = 0; jdx < 3; jdx ++)
                {
                    for (int kdx = 0; kdx < 3;kdx++)
                    {
                        Assert.IsTrue(np1[idx,jdx,kdx] == np1_[idx][jdx][kdx]);
                    }
                }    
            }
        }
    }
}
