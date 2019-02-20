using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class RavelTest
    {
        [TestMethod]
        public void Simple2DArray()
        {
            var nd1 = np.array(new int[][]
            {
                new int[] { 3, 1, 1, 2 },
                new int[] { 3, 1, 1, 2 }
            });

            var nd2 = nd1.ravel();
            
            Assert.IsTrue(nd1.shape[0] == 2);
            Assert.IsTrue(nd1.shape[1] == 4);
            Assert.IsTrue(nd2.shape[0] == 8);
            Assert.IsFalse(Enumerable.SequenceEqual(nd1.Data<int>(), nd2.Data<int>()));
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 1, 1, 2, 3, 1, 1, 2 }, nd2.Data<int>()));
        }
    }
}
