using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class StringArrayApiTests
    {
        private static string[] strArray = new string[] { "Hello,", " SciSharp Team!"};

        [TestMethod]
        public void StringArrayConverting()
        {
            var nd = np.array(strArray);
            nd.Should().BeOfType<char>()
                .And.BeShaped(28)
                .And.BeOfValues(50, 32, 54, 32, 49, 53, 58, 72, 101, 108, 108, 111, 44, 32, 83, 99, 105, 83, 104, 97, 114, 112, 32, 84, 101, 97, 109, 33);

            NDArray.AsStringArray(nd).SequenceEqual(strArray);
        }
    }
}
