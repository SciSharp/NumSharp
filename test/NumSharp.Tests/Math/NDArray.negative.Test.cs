using System;
using System.Linq;

namespace NumSharp.Tests.Math
{
    [TestClass]
    public class NDArrayNegativeTest
    {
        [TestMethod]
        public void Negative_FlipsSign()
        {
            // np.negative flips the sign of each element
            // Input:  [1, -2, 3.3]
            // Output: [-1, 2, -3.3]
            NDArray nd = new[] { 1f, -2f, 3.3f };
            nd = np.negative(nd);

            var data = nd.Data<float>();
            Assert.AreEqual(-1f, data[0], 1e-6f);
            Assert.AreEqual(2f, data[1], 1e-6f);   // -2 becomes 2
            Assert.AreEqual(-3.3f, data[2], 1e-5f);
        }

        [TestMethod]
        public void Negative_AllPositiveInput()
        {
            // When all inputs are positive, all outputs are negative
            NDArray nd = new[] { 1f, 2f, 3f };
            nd = np.negative(nd);
            Assert.IsTrue(nd.Data<float>().All(v => v < 0));
        }

        [TestMethod]
        public void Negative_AllNegativeInput()
        {
            // When all inputs are negative, all outputs are positive
            NDArray nd = new[] { -1f, -2f, -3f };
            nd = np.negative(nd);
            Assert.IsTrue(nd.Data<float>().All(v => v > 0));
        }

        /// <summary>
        ///     np.negative, the instance method and the unary <c>-</c> refuse a bool array with NumPy's TypeError, an empty
        ///     one included; <c>~</c> and np.logical_not still flip it.
        /// </summary>
        [TestMethod]
        public void Negative_Bool_ThrowsLikeNumPy()
        {
            // NumPy has no negative loop for the bool dtype: np.negative(bool) and
            // unary -bool raise TypeError (even for empty arrays). NumSharp matches.
            NDArray b = new bool[] { true, false, true };
            Assert.ThrowsException<TypeError>(() => np.negative(b));
            Assert.ThrowsException<TypeError>(() => b.negative());
            Assert.ThrowsException<TypeError>(() => -b);
            NDArray empty = new bool[0];
            Assert.ThrowsException<TypeError>(() => np.negative(empty));

            // The boolean flip lives on `~` (np.invert) and np.logical_not, which
            // are unaffected by the negative guard and still return [F, T, F].
            var inv = (~b).Data<bool>();
            Assert.IsFalse(inv[0]); Assert.IsTrue(inv[1]); Assert.IsFalse(inv[2]);
            var ln = np.logical_not(b).Data<bool>();
            Assert.IsFalse(ln[0]); Assert.IsTrue(ln[1]); Assert.IsFalse(ln[2]);
        }
    }
}
