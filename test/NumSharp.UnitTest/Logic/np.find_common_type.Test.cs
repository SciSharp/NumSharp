using System;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Logic
{
    [TestClass]
    public class np_find_common_type_test
    {
        [TestMethod]
        public void Case1()
        {
            var r = np.find_common_type(new[] {np.float32}, new[] {np.int64, np.float64});
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case2()
        {
            var r = np.find_common_type(new[] {np.float32}, new[] {np.complex64});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case3()
        {
            var r = np.find_common_type(new[] {np.float32}, new[] {np.complex64});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case4()
        {
            var r = np.find_common_type(new[] {"f4", "f4", "i4",}, new[] {"c8"});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case5()
        {
            var r = np.find_common_type(new[] {"f4", "f4", "i4",}, new[] {"c8"});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case6()
        {
            var r = np.find_common_type(new[] {np.int32, np.float32});
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case7()
        {
            var r = np.find_common_type(new[] {np.int32, np.float64});
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case8()
        {
            var r = np.find_common_type(new[] {np.int32, np.float64}, new[] {np.int32, np.float64});
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case9()
        {
            var r = np.find_common_type(new[] {np.int32, np.float64}, new[] {np.int32, np.float32});
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case10()
        {
            var r = np.find_common_type(new[] {np.int32, np.float64}, new[] {np.complex64});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case11()
        {
            var r = np.find_common_type(new[] {np.uint8, np.float32}, new Type[0]);
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case12()
        {
            var r = np.find_common_type(new[] {np.@byte, np.float32}, new Type[0]);
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case13()
        {
            var r = np.find_common_type(new[] {np.float32, np.float32}, new Type[0]);
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case14()
        {
            var r = np.find_common_type(new[] {np.float32, np.@byte}, new Type[0]);
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case15()
        {
            var r = np.find_common_type(new[] {np.float64, np.float64}, new Type[0]);
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case17()
        {
            var r = np.find_common_type(new[] {np.@byte, np.@byte}, new Type[0]);
            r.Should().Be(NPTypeCode.Byte);
        }

        [TestMethod]
        public void Case18()
        {
            var r = np.find_common_type(new[] {np.complex128, np.@double}, new Type[0]);
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case19()
        {
            var r = np.find_common_type(new[] {np.complex128, np.complex128}, new Type[0]);
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case20()
        {
            var r = np.find_common_type(new[] {np.complex128, np.complex128}, new[] {np.@double});
            r.Should().Be(NPTypeCode.Complex);
        }
    }
}
