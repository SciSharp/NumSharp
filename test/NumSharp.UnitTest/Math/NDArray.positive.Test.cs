using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using NumSharp;

namespace NumSharp.UnitTest.Maths
{
    public class NDArrayPoisitiveTest
    {
        [Test]
        public void PositiveAllNegatives()
        {
            var nd = new NDArray(np.float32, 3);
            nd.ReplaceData(new float[] {1, -2, 3.3f});
            nd = np.positive(nd);
            Assert.IsTrue(nd.Data<float>().All(v => v >= 0));
        }
    }
}
