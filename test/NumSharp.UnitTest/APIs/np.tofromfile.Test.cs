using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class NumpyToFromFileTest : TestClass
    {
        [TestMethod]
        public void NumpyToFromFileTestByte1()
        {
            var testString = "Hallo World!";
            byte[] rawData = Encoding.ASCII.GetBytes(testString);


            NDArray ndArray = new NDArray(rawData);

            var testFileName = @"test." + nameof(NumpyToFromFileTestByte1);
            ndArray.tofile(testFileName);
            var loadedArray = np.fromfile(testFileName, np.uint8);

            Assert.AreEqual(np.uint8, loadedArray.dtype);
            AssertAreEqual(ndArray.shape, loadedArray.shape);
            AssertAreEqual(rawData, loadedArray.Array);
        }

        [Ignore("not support uint16")]
        [TestMethod]
        public void NumpyToFromFileTestUShort1()
        {
            var testString = "Hallo World!";
            var testData = Encoding.ASCII.GetBytes(testString);
            ushort[] testArray = Array.ConvertAll(testData, d => (ushort)d);

            NDArray ndArray = new NDArray(testArray);

            var testFileName = @"test." + nameof(NumpyToFromFileTestByte1);
            ndArray.tofile(testFileName);
            var loadedArray = np.fromfile(testFileName, np.uint16);

            Assert.AreEqual(np.uint16, loadedArray.dtype);
            AssertAreEqual(ndArray.shape, loadedArray.shape);
            AssertAreEqual(testArray, loadedArray.Array);
        }
    }
}
