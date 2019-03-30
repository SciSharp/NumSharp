using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class ApiCreationTest
    {
        [TestMethod]
        public void arange()
        {
            var np1 = new NDArray(typeof(int)).arange(4, 1);
            var np2 = new NDArray(typeof(int)).arange(5, 2);

            var np3 = np1 + np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 5, 7 }, np3.Storage.GetData<int>()));
        }
    }
}
