using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayArrayTest
    {
        [TestMethod]
        public void Array1Dim()
        {
            var np = new NDArray<int>();
            var list = new List<int> { 1, 2, 3 };
            // np = np.Array(list);

            // Assert.IsTrue(np.ToString().Equals("array([1, 2, 3])"));
        }

        [TestMethod]
        public void Array2Dim()
        {
            var np = new NDArray<List<int>>();
            var list = new List<List<int>>
            {
                new List<int> { 1, 2 },
                new List<int> { 3, 4 }
            };

            // np = np.Array(list);

            // Assert.IsTrue(np.ToString().Equals("array([[1, 2], [3, 4]])"));
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
