using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class ApiCreationTest : TestClass
    {
        [TestMethod]
        public void arange()
        {
            var nd = np.arange(3);
            AssertAreEqual(nd.Data<int>(), new int[] {0, 1, 2});
        }

        [TestMethod]
        public void ndarray()
        {
            var x = np.ndarray((2, 3), dtype: np.int32, order: 'C');
            Console.WriteLine(x.Shape);
            x.Cast<int>().Should().AllBeEquivalentTo(0);
        }

        [TestMethod]
        public void ReshapeDoesNotSelfModify()
        {
            var x = np.arange(9);
            Assert.AreEqual(new Shape(9), new Shape(x.shape));
            x.reshape(3, 3); // <-- must not affect x!
            Assert.AreEqual(new Shape(9), new Shape(x.shape));
            var y = x.reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), new Shape(y.shape));
        }
    }
}
