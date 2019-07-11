using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Backends.Default
{
    [TestClass]
    public class TypedArrayStorageTests
    {
        [TestMethod]
        public void ReplaceData()
        {
            var arr = np.ones(new Shape(3, 3, 3), np.int32);
            var newarr = np.zeros(new Shape(3, 3, 3), np.float32);
            arr.ReplaceData(newarr);
            arr.dtype.Should().Be(np.float32);
        }

        [TestMethod]
        public void ReplaceData_NDArray()
        {
            var arr = np.ones(new Shape(3, 3, 3), np.int32);
            var newarr = np.zeros(new Shape(3, 3, 3), np.float32);
            arr.ReplaceData(newarr);
            arr.dtype.Should().Be(np.float32);
        }
    }
}
