using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using np = NumSharp.Core.NumPy;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayArrayTest
    {
        [TestMethod]
        public void Array1Dim()
        {
            var list = new int[] { 1, 2, 3 };
            var n = np.array(list);

            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] { 1, 2, 3 }));
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

            Assert.IsTrue(n.Data<int>(1, 0) == 3);
        }

        [TestMethod]
        public void ArrayImage()
        {
            var relativePath = string.Empty;
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

                Assert.IsTrue(imageNDArray.Data<byte>(0, 0, 0) == 255);
                Assert.IsTrue(imageNDArray.Data<byte>(0, 0, 1) == 253);
                Assert.IsTrue(imageNDArray.Data<byte>(0, 0, 2) == 252);
            }
        }
    }
}
