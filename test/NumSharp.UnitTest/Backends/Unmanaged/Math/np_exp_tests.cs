using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged.Math
{
    [TestClass]
    public class np_exp_tests
    {
        [TestMethod]
        public void Exp_0()
        {
            var arr = np.zeros(new Shape(5, 5));
            var ret = np.exp(arr);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
            ret.GetData<double>().All(d => System.Math.Abs(d - 1.0) < 0.0001).Should().BeTrue();
        }

        [TestMethod]
        public void Exp_1()
        {
            var arr = np.ones(new Shape(5, 5));
            var ret = np.exp(arr);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
            ret.GetData<double>().All(d => System.Math.Abs(d - System.Math.E) < 0.0001).Should().BeTrue();
        }

        [TestMethod]
        public void ExpUpcast()
        {
            var right = np.zeros(new Shape(5, 5)).astype(NPTypeCode.Int32) + 5;
            var ret = np.exp(right, NPTypeCode.Double);
            ret.GetTypeCode.Should().Be(NPTypeCode.Double);
            ret.GetData<double>().All(d => System.Math.Abs(d - 148.4131591) < 0.0001).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void ExpDowncast()
        {
            var right = np.zeros(new Shape(5, 5)).astype(NPTypeCode.Double) + 5;
            var ret = np.exp(right, NPTypeCode.Byte);
            ret.GetTypeCode.Should().Be(NPTypeCode.Byte);
            ret.GetData<byte>().All(d => d == 148).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }
    }
}
