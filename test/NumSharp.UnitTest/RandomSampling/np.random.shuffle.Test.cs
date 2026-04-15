using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling
{
    public class NpRandomShuffleTest : TestClass
    {
        [TestMethod]
        public void Base1DTest()
        {
            // NumPy-aligned: shuffle reorders elements but preserves all values
            var rnd = np.random.RandomState(42);
            var nd = np.arange(10);
            nd[8] = 5;  // [0,1,2,3,4,5,6,7,5,9]
            var originalSum = (int)np.sum(nd);

            rnd.shuffle(nd);

            // Sum should be unchanged
            Assert.AreEqual(originalSum, (int)np.sum(nd));
            // Shape unchanged
            nd.Shape.Should().BeShaped(10);
        }

        [TestMethod]
        public void Base4DTest()
        {
            // NumPy-aligned: shuffle along axis 0 (first dimension)
            // For 4D array, this shuffles the first-level "blocks"
            var rnd = np.random.RandomState(42);
            var nd = arange(2, 2, 5, 5);
            var ogshape = nd.Shape.Clone();
            var originalSum = (int)np.sum(nd);

            // Get sums of each block before shuffle
            var block0Sum = (int)np.sum(nd[0]);
            var block1Sum = (int)np.sum(nd[1]);

            rnd.shuffle(nd);

            // Total sum unchanged
            Assert.AreEqual(originalSum, (int)np.sum(nd));
            // Shape unchanged
            nd.Shape.Should().Be(ogshape);

            // Block sums still exist (just potentially swapped)
            var newBlock0Sum = (int)np.sum(nd[0]);
            var newBlock1Sum = (int)np.sum(nd[1]);
            var sums = new[] { newBlock0Sum, newBlock1Sum }.OrderBy(x => x).ToArray();
            var expectedSums = new[] { block0Sum, block1Sum }.OrderBy(x => x).ToArray();
            CollectionAssert.AreEqual(expectedSums, sums);
        }
    }
}
