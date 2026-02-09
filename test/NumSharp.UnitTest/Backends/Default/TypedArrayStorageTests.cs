using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Backends.Default
{
    public class TypedArrayStorageTests
    {
        [Test]
        public void ReplaceData()
        {
            var arr = np.ones(new Shape(3, 3, 3), np.int32);
            var newarr = np.zeros(new Shape(3, 3, 3), np.float32);
            arr.ReplaceData(newarr);
            arr.dtype.Should().Be(np.float32);
        }

        [Test]
        public void ReplaceData_NDArray()
        {
            var arr = np.ones(new Shape(3, 3, 3), np.int32);
            var newarr = np.zeros(new Shape(3, 3, 3), np.float32);
            arr.ReplaceData(newarr);
            arr.dtype.Should().Be(np.float32);
        }
    }
}
