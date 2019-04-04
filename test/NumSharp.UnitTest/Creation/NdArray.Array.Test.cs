using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NdArrayArrayTest
    {
        [TestMethod]
        public void Array1Dim()
        {
            var list = new int[] { 1, 2, 3 };
            var n = np.array(list);

            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<int>(), new int[] { 1, 2, 3 }));
        }

        [TestMethod]
        public void Array2Dim()
        {
            var list = new int[][]
            {
                new int[] { 1, 2 },
                new int[] { 3, 4 }
            };

            var n = np.array(list);

            Assert.IsTrue(n.Storage.GetData<int>(1, 0) == 3);
        }

        [TestMethod]
        public void ArrayImage()
        {
            /*var relativePath = string.Empty;
#if NETFRAMEWORK
            relativePath = "../../../..";
#else
            relativePath = "../../..";
#endif
            var pwd = System.IO.Path.GetFullPath(relativePath);

            var imagePath = System.IO.Path.Combine(pwd,"data/image.jpg");

            if (System.IO.File.Exists(imagePath))
            {
                var image = new System.Drawing.Bitmap(imagePath);
                var imageNDArray = np.array(image);

                Assert.IsTrue(imageNDArray.Storage.GetData<byte>()[0] == 255 );
                Assert.IsTrue(imageNDArray.Storage.GetData<byte>()[1] == 253 );
                Assert.IsTrue(imageNDArray.Storage.GetData<byte>()[2] == 252 );

            }*/
        }

        [TestMethod]
        public void StringCheck()
        {
            var nd = np.arange(9.0).reshape(3, 3).MakeGeneric<double>();

            var random = new Random();
            nd.Storage.SetData(nd.Data<double>().Select(x => x + random.NextDouble()).ToArray());
            nd[1, 0] = 1.0;
            nd[0, 0] = 9.0;
            nd[2, 2] = 7.0;
            nd[0, 2] = nd[0, 2] - 20.0;
            nd[2, 2] += 23;

            var stringOfNp = nd.ToString();

            Assert.IsTrue(stringOfNp.Contains("[["));
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
            var np1 = new NDArray(typeof(double)).arange(9).MakeGeneric<double>();

            double[] np1_ = (double[])np1.ToMuliDimArray<double>();

            Assert.IsTrue(Enumerable.SequenceEqual(np1_, np1.Storage.GetData<double>()));
        }

        [TestMethod]
        public void ToDotNetArray2D()
        {
            var np1 = new NDArray(typeof(double)).arange(9).reshape(3, 3).MakeGeneric<double>();
            double[][] np1_ = (double[][])np1.ToJaggedArray<double>();

            for (int idx = 0; idx < 3; idx++)
            {
                for (int jdx = 0; jdx < 3; jdx++)
                {
                    Assert.IsTrue(np1[idx, jdx] == np1_[idx][jdx]);
                }
            }
        }

        [TestMethod]
        public void ToDotNetArray3D()
        {
            var np1 = new NDArray(typeof(double)).arange(27).reshape(3, 3, 3);

            double[][][] np1_ = (double[][][])np1.ToJaggedArray<double>();

            var np2 = np1.MakeGeneric<double>();

            for (int idx = 0; idx < 3; idx++)
            {
                for (int jdx = 0; jdx < 3; jdx++)
                {
                    for (int kdx = 0; kdx < 3; kdx++)
                    {
                        Assert.IsTrue(np2[idx, jdx, kdx] == np1_[idx][jdx][kdx]);
                    }
                }
            }
        }
    }
}
