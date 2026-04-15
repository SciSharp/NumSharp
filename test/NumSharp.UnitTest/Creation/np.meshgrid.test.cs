using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NumpyMeshgridTest
    {
        [TestMethod]
        public void MeshgridTest()
        {
            NDArray X = np.array(0, 1, 2);

            NDArray y = np.array(0, 1);

            Kwargs kw = new Kwargs();
            kw.indexing = "xy";
            kw.sparse = false;

            var (xx, yy) = np.meshgrid(X, y, kw);

            // Assert.IsTrue(xx, np.array(new int[,] { { 0,1,2}, { 0,1,2} }));
            // Assert.IsTrue(yy, np.array(new int[,] { { 0, 0, 0 }, { 1, 1, 1 } }));
        }
    }
}
