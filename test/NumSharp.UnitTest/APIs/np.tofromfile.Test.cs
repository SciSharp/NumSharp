using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class NumpyToFromFileTest
    {
        [TestMethod]
        public void NumpyToFromFileTest1()
        {
            var testString = "Hallo World!";
            byte[] bytes = Encoding.ASCII.GetBytes(testString);

            NDArray byteArray = np.frombuffer(bytes, np.uint8);

            var testFileName = @"test." + nameof(NumpyToFromFileTest1);
            byteArray.tofile(testFileName);
            var loadedArray = np.fromfile(testFileName, np.uint8);
            var resultString = Encoding.ASCII.GetString(loadedArray.Data<byte>());

            Assert.IsTrue(testString == resultString);
        }    
    }
}
