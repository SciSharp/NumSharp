using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class NDArrayOrTest
    {
        [Ignore("TODO: fix this test")]
        [TestMethod]
        public void BoolTwo1D_NDArrayOR()
        {
            var np1 = new NDArray(new[] {true, true, false, false}, new Shape(4));
            var np2 = new NDArray(new[] {true, false, true, false}, new Shape(4));

            var np3 = np1 | np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new[] {true, true, true, false}, np3.Data<bool>()));
        }

        [Ignore("TODO: fix this test")]
        [TestMethod]
        public void BoolTwo2D_NDArrayOR()
        {
            var np1 = new NDArray(typeof(bool), new Shape(2, 3));
            np1.ReplaceData(new bool[] {true, true, false, false, true, false});

            var np2 = new NDArray(typeof(bool), new Shape(2, 3));
            np2.ReplaceData(new bool[] {true, false, true, false, true, true});

            var np3 = np1 | np2;

            // expected
            var np4 = new bool[] {true, true, true, false, true, true};

            Assert.IsTrue(Enumerable.SequenceEqual(np3.Data<bool>(), np4));
        }
    }
}
