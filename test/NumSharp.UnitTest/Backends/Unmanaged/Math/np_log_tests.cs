using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged.Math
{
    [TestClass]
    public class np_log_tests
    {
        [TestMethod]
        public void Log_1()
        {
            var arr = np.zeros(new Shape(5, 5)) + 5d;
            var ret = np.log(arr);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
            ret.GetData<double>().All(d => System.Math.Abs(d - 1.6094379124341) < 0.0001).Should().BeTrue();
        }        

        [TestMethod]
        public void Log_2()
        {
            var arr = np.ones(new Shape(5, 5));
            var ret = np.log(arr);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
            ret.GetData<double>().All(d => d == 0).Should().BeTrue();
        }

        [TestMethod]
        public void LogUpcast()
        {
            var right = np.zeros(new Shape(5, 5)).astype(NPTypeCode.Int32)+5;
            var ret = np.log(right, NPTypeCode.Double);
            ret.GetTypeCode.Should().Be(NPTypeCode.Double);
            ret.GetData<double>().All(d => System.Math.Abs(d - 1.6094379124341) < 0.0001).Should().BeTrue();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void LogDowncast()
        {
            var right = np.zeros(new Shape(5, 5)).astype(NPTypeCode.Double) + 5;
            new Action(()=> np.log(right, NPTypeCode.Byte)).Should().Throw<IncorrectTypeException>().Where(exception => exception.Message.Contains("No loop matching the specified signature and casting was found for"));
        }
    }
}
