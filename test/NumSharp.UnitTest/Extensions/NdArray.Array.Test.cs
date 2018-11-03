using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayArrayTest
    {
        [TestMethod]
        public void Array1Dim()
        {
            var np = new NumPy<int>();
            var list = new int[] { 1, 2, 3 };
            var n = np.array(list);

            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 1, 2, 3 }));
        }

        [TestMethod]
        public void Array2Dim()
        {
            var np = new NumPy<int>();
            var list = new int[][]
            {
                new int[] { 1, 2 },
                new int[] { 3, 4 }
            };

            var n = np.array(list);

            Assert.IsTrue(n[1, 0] == 3);
        }

        [TestMethod]
        public void ArrayImage()
        {
            var pwd = System.IO.Path.GetFullPath("../../..");

            var imagePath = System.IO.Path.Combine(pwd,"./data/image.jpg");            

            var image = new System.Drawing.Bitmap(imagePath);

            var imageNDArray = new NDArray<byte>().Array(image);

            Assert.IsTrue(imageNDArray[0,0,0] == 255);
            Assert.IsTrue(imageNDArray[0,0,1] == 253);
            Assert.IsTrue(imageNDArray[0,0,2] == 252);

        }
    }
}
