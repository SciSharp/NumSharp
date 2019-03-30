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
            //np.BackendEngine = BackendType.VectorT;
            var nd = np.arange(3);
            Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<int>(), new int[] { 0, 1, 2 }));
        }
    }
}
