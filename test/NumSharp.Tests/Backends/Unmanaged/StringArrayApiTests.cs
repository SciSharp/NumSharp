using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends.Unmanaged
{
    [TestClass]
    public class StringArrayApiTests
    {
        private static string[] strArray = new string[] { "Hello,", " SciSharp Team!"};

        [TestMethod]
        public void StringArrayConverting()
        {
            // NumPy 2.4.2: np.array(["Hello,", " SciSharp Team!"]) -> shape (2,), dtype '<U15' (the longest text).
            // (It used to be NumSharp's "2 6 15:..." char-vector encoding.)
            var nd = np.array(strArray);
            nd.typecode.Should().Be(NPTypeCode.String);
            nd.dtype.str.Should().Be("<U15");
            nd.Should().BeShaped(2);
            nd.ToString().Should().Be("['Hello,' ' SciSharp Team!']");

            NDArray.AsStringArray(nd).Should().Equal(strArray);
        }
    }
}
