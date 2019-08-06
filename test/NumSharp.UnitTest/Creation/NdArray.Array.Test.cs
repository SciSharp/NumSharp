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
            var list = new int[] {1, 2, 3};
            var n = np.array(list);

            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] {1, 2, 3}));
        }

        [TestMethod]
        public void Array2Dim()
        {
            var list = new int[][] {new int[] {1, 2}, new int[] {3, 4}};

            var nd = np.array(list);
            var val = nd.GetInt32(1, 0);
            Assert.IsTrue(val == 3);

            var list1 = new int[,] {{1, 2, 3}, {2, 3, 1}};

            var n1 = np.array(list1);
            Assert.IsTrue(Enumerable.SequenceEqual(n1.shape, new int[] {2, 3}));
            Assert.IsTrue(Enumerable.SequenceEqual(n1.Data<int>(), new int[] {1, 2, 3, 2, 3, 1}));
        }


        [TestMethod]
        public void Array2Dim_Accessing()
        {
            var list = new int[][] {new int[] {1, 2}, new int[] {3, 4}};
            Shape n = np.array(list).shape;


            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Console.WriteLine(n.GetOffset(i, j));
                }
            }
        }

        [TestMethod]
        public void Array3Dim()
        {
            var list = new int[,,] {{{1, 2}, {3, 4}}, {{2, 2}, {3, 3}}, {{3, 2}, {3, 1}},};

            var nd = np.array(list);
            Assert.IsTrue(Enumerable.SequenceEqual(nd.shape, new int[] {3, 2, 2}));
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<int>(), new int[] {1, 2, 3, 4, 2, 2, 3, 3, 3, 2, 3, 1}));
        }

        /*public static NDArray array(System.Drawing.Bitmap image)
        {
            var imageArray = new NDArray(typeof(Byte));

            var bmpd = image.LockBits(new System.Drawing.Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, image.PixelFormat);
            var dataSize = bmpd.Stride * bmpd.Height;

            var bytes = new byte[dataSize];
            System.Runtime.InteropServices.Marshal.Copy(bmpd.Scan0, bytes, 0, dataSize);
            image.UnlockBits(bmpd);

            imageArray.Storage.Allocate(typeof(byte),new Shape(bmpd.Height, bmpd.Width, System.Drawing.Image.GetPixelFormatSize(image.PixelFormat) / 8),1);
            imageArray.Storage.ReplaceData(bytes);

            return imageArray;
        }*/

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
        public void flatten2d()
        {
            var a = np.array(new int[,] {{1, 2}, {3, 4}});
            var b = a.flatten();
            //var c = a.flatten('F');

            //Assert.IsTrue(Enumerable.SequenceEqual(c.Data<int>(), new int[] {1, 3, 2, 4}));
            Assert.IsTrue(Enumerable.SequenceEqual(b.Data<int>(), new int[] {1, 2, 3, 4}));
            Assert.IsTrue(Enumerable.SequenceEqual(a.Data<int>(), new int[] {1, 2, 3, 4}));
        }

        [TestMethod]
        public void StringCheck()
        {
            var nd = np.arange(9d).reshape(3, 3).MakeGeneric<double>();

            var random = new Randomizer();
            nd.ReplaceData(nd.Data<double>().Select(x => x + random.NextDouble()).ToArray());
            nd[1, 0] = 1.0;
            nd[0, 0] = 9.0;
            nd[2, 2] = 7.0;
            nd[0, 2] = nd[0, 2] - 20.0;
            nd[2, 2] += 23;

            var stringOfNp = nd.ToString();

            Assert.IsTrue(stringOfNp.Contains("[["));
        }

        [TestMethod, Ignore("No assertions inside")]
        public void CheckVectorString()
        {
            var np = NumSharp.np.arange(9).MakeGeneric<double>();

            var random = new Randomizer();
            np.ReplaceData(np.Data<double>().Select(x => x + random.NextDouble()).ToArray());
            np[1] = 1;
            np[2] -= 4;
            np[3] -= 20;
            np[8] += 23;

            var stringOfNp = np.ToString();
        }

        [TestMethod]
        public void ToDotNetArray1D()
        {
            var np1 = np.arange(9d).MakeGeneric<double>();

            double[] np1_ = (double[])np1.ToMuliDimArray<double>();

            Assert.IsTrue(Enumerable.SequenceEqual(np1_, np1.Data<double>()));
        }

        [TestMethod]
        public void ToDotNetArray2D()
        {
            var np1 = np.arange(9d).reshape(3, 3).MakeGeneric<double>();
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
            var np1 = np.arange(27d).astype(np.float64).reshape(3, 3, 3);

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
