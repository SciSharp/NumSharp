using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Operations {
    [TestClass]
    public class NDArrayNotTest
    {
        [TestMethod]
        public void not_1d()
        {
            var np1 = new NDArray(new[] { false, false, false, false}, new Shape(4));

            var np3 = !np1;

            Assert.IsTrue(Enumerable.SequenceEqual(new[] {true, true, true, true}, np3.Data<bool>().MemoryBlock));
        }

        [TestMethod]
        public void BoolTwo2D_NDArrayOR()
        {
            var np1 = new NDArray(new[] { false, true, false, false }, new Shape(2,2));

            var np3 = !np1;

            Assert.IsTrue(Enumerable.SequenceEqual(new[] { true, false, true, true }, np3.Data<bool>().MemoryBlock));
        }
    }
}