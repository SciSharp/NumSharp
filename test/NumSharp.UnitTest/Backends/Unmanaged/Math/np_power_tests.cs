using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged.Math
{
    [TestClass]
    public class np_power_tests
    {
        [TestMethod]
        public void Power_1()
        {
            var arr = np.zeros(new Shape(5, 5)) + 5d;
            var ret = np.power(arr, 2);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
            ret.GetData<double>().All(d => d==25).Should().BeTrue();
        }

        [TestMethod]
        public void Power_2()
        {
            var arr = np.ones(new Shape(5, 5));
            var ret = np.power(arr, 2);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
            ret.GetData<double>().All(d => d == 1).Should().BeTrue();
        }

        [TestMethod]
        public void PowerUpcast()
        {
            var right = np.zeros(new Shape(5, 5)).astype(NPTypeCode.Int32)+5;
            var ret = np.power(right, 2, NPTypeCode.Double);
            ret.GetTypeCode.Should().Be(NPTypeCode.Double);
            ret.GetData<double>().All(d => d == 25).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void PowerDowncast()
        {
            var right = np.zeros(new Shape(5, 5)).astype(NPTypeCode.Double) + 5;
            var ret = np.power(right, 2, NPTypeCode.Byte);
            ret.GetTypeCode.Should().Be(NPTypeCode.Byte);
            ret.GetData<byte>().All(d => d==25).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }
    }
}
