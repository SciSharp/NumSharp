using System;
using System.Linq;
using System.Numerics;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class np_dtype_tests
    {
        [TestMethod]
        public void Case1()
        {
            np.dtype("?").type.Should().Be<bool>();
            np.dtype("?64").type.Should().Be<bool>();
            np.dtype("i4").type.Should().Be<Int32>();
            np.dtype("i8").type.Should().Be<Int64>();
            np.dtype("f").type.Should().Be<float>();
            np.dtype("f8").type.Should().Be<double>();
            np.dtype("d8").type.Should().Be<double>();
            np.dtype("double").type.Should().Be<double>();
            np.dtype("single16").type.Should().Be<double>();
            np.dtype("f16").type.Should().Be<double>();
        }

    }
}
