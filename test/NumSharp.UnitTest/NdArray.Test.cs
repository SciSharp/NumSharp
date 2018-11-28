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
        private NumPy np = new NumPy();

        [TestMethod]
        public void StringCheck()
        {
            var nd = np.arange(9.0).reshape(3,3);

            var random = new Random();
            nd.Set(nd.Data<double>().Select(x => x + random.NextDouble()).ToArray());
            nd[1,0] = 1.0;
            nd[0,0] = 9.0;
            nd[2,2] = 7.0;
            //nd.Storage[3] -= 20;
            //nd.Storage[8] += 23;

            var stringOfNp = nd.ToString();

            /*Assert.IsTrue(stringOfNp.Contains("[[ 0."));

            nd = np.arange(9).reshape(3,3);

            stringOfNp = nd.ToString();        

            Assert.IsTrue(stringOfNp.Contains("([[ 0,"));*/
        }
        [TestMethod]
        public void CheckVectorString()
        {
            var np = new NDArrayGeneric<double>().arange(9);

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
            NDArrayGeneric<double> np1 = new NDArrayGeneric<double>().Zeros(2,2);

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
            var np1 = new NDArrayGeneric<double>().arange(9);

            double[] np1_ = np1.ToDotNetArray<double[]>();

            Assert.IsTrue(Enumerable.SequenceEqual(np1_,np1.Data));
        } 

        [TestMethod]
        public void ToDotNetArray2D()
        {
            var np1 = new NDArrayGeneric<double>().arange(9).reshape(3,3);

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
            var np1 = new NDArrayGeneric<double>().arange(27).reshape(3,3,3);

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
