using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayLinSpaceTest
    {
        [TestMethod]
        public void FromNumpyDocs()
        {
            var np1 = new NDArray().linspace(2.0, 3.0,5);

            Assert.IsTrue(Enumerable.SequenceEqual(np1.Storage.GetData<double>(),new double[]{2.0,2.25,2.5,2.75,3.0}));

            var np2 = new NDArray().linspace(2.0, 3.0,5,false);

            Assert.IsTrue(Enumerable.SequenceEqual(np2.Storage.GetData<double>(),new double[]{2.0,2.20,2.4,2.6,2.8}));

        }
    }
}
