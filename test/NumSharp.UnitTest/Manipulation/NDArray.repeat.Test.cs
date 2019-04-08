using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class RepeatTest
    {
        [TestMethod]
        public void Scalar()
        {
            var nd = np.repeat(3, 4);
            Assert.AreEqual(nd.Data<int>().Count(x => x == 3), 4);
        }

        [TestMethod]
        public void Simple2DArray()
        {
            var x = np.array(new int[][] 
            {
                new int[] { 1, 2 },
                new int[] { 3, 4 }
            });

            var nd = np.repeat(x, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 1, 1, 2, 2, 3, 3, 4, 4 }, nd.Data<int>()));
        }
    }
}
