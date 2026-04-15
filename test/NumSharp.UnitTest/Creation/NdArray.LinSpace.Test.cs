using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class np_linspace_tests
    {
        [TestMethod]
        public void FromNumpyDocs()
        {
            NDArray nd;

            nd = np.linspace(2.0, 3.0, 5);
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<double>(), new double[] {2.0, 2.25, 2.5, 2.75, 3.0}));

            nd = np.linspace(2.0, 3.0, 5, false);
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<double>(), new double[] {2.0, 2.20, 2.4, 2.6, 2.8}));

            nd = np.linspace(2.0f, 3.0f, 5);
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<float>(), new float[] {2.0f, 2.25f, 2.5f, 2.75f, 3.0f}));

            nd = np.linspace(2.0f, 3.0f, 5, false);
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<float>(), new float[] {2.0f, 2.20f, 2.4f, 2.6f, 2.8f}));
        }
    }
}
