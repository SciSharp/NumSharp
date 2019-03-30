using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NDArrayTest
    {
        [TestMethod]
        public void StringCheck()
        {
            
            var nd =  np.arange(9.0).reshape(3,3).MakeGeneric<double>();

            var random = new Random();
            nd.Storage.SetData(nd.Data<double>().Select(x => x + random.NextDouble()).ToArray());
            nd[1,0] = 1.0;
            nd[0,0] = 9.0;
            nd[2,2] = 7.0;
            nd[0,2] = nd[0,2] - 20.0;
            nd[2,2] += 23;

            var stringOfNp = nd.ToString();

            Assert.IsTrue(stringOfNp.Contains("[["));

            nd = np.arange(9).reshape(3,3).MakeGeneric<double>();

            stringOfNp = nd.ToString();        

            Assert.IsTrue(stringOfNp.Contains("([[ 0,"));
            
        }
        [TestMethod]
        public void CheckVectorString()
        {
            var np = new NDArray(typeof(double)).arange(9).MakeGeneric<double>();

            var random = new Random();
            np.Storage.SetData(np.Storage.GetData<double>().Select(x => x + random.NextDouble()).ToArray());
            np[1] = 1;
            np[2] -= 4;
            np[3] -= 20;
            np[8] += 23;

            var stringOfNp = np.ToString();
        }
        [TestMethod]
        public void ToDotNetArray1D()
        {
            var np1 = new NDArray(typeof(double) ).arange(9).MakeGeneric<double>();

            double[] np1_ = (double[]) np1.ToMuliDimArray<double>();

            Assert.IsTrue(Enumerable.SequenceEqual(np1_,np1.Storage.GetData<double>()));
        } 

        [TestMethod]
        public void ToDotNetArray2D()
        {
            var np1 = new NDArray(typeof(double)).arange(9).reshape(3,3).MakeGeneric<double>();
            double[][] np1_ = (double[][]) np1.ToJaggedArray<double>();

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
            var np1 = new NDArray(typeof(double)).arange(27).reshape(3,3,3);

            double[][][] np1_ = (double[][][]) np1.ToJaggedArray<double>();

            var np2 = np1.MakeGeneric<double>();

            for (int idx = 0; idx < 3; idx ++)
            {
                for (int jdx = 0; jdx < 3; jdx ++)
                {
                    for (int kdx = 0; kdx < 3;kdx++)
                    {
                        Assert.IsTrue(np2[idx,jdx,kdx] == np1_[idx][jdx][kdx]);
                    }
                }    
            }
        }
    }
}
