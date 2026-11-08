using System;
using System.Linq;

namespace NumSharp.Tests.Creation
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

        /// <summary>
        ///     NumPy's <c>np.ndarray((2, 3), dtype=np.int32, order='C')</c> allocates like <c>np.empty</c>: the values are
        ///     uninitialized, so only the layout is defined — int32, shape <c>(2, 3)</c>, strides <c>(12, 4)</c>,
        ///     C-contiguous. The full contract is in <c>np.ndarray.Test.cs</c> and the <c>ndarray_ctor</c> oracle tier.
        /// </summary>
        [TestMethod]
        public void ndarray()
        {
            using var x = np.ndarray((2, 3), dtype: np.int32, order: 'C');

            x.dtype.Should().Be(np.int32);
            x.shape.Should().Equal(2L, 3L);
            x.strides.Should().Equal(12L, 4L);
            x.flags.c_contiguous.Should().BeTrue();
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
