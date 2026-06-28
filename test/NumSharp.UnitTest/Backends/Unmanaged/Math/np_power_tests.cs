using System;
using System.Linq;
using AwesomeAssertions;
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
            // NumPy 2.4.2 (probed): dtype= selects the LOOP, and a float64
            // input cannot reach a uint8 loop under the same_kind rule —
            //   np.power(f64_5x5, 2, dtype=np.uint8) → UFuncTypeError:
            //   "Cannot cast ufunc 'power' input 0 from dtype('float64') to
            //    dtype('uint8') with casting rule 'same_kind'"
            // (NumSharp previously computed the f64 loop and downcast the
            // result — misaligned; aligned by the ufunc dtype= wave.)
            var right = np.zeros(new Shape(5, 5)).astype(NPTypeCode.Double) + 5;
            new Action(() => np.power(right, 2, NPTypeCode.Byte))
                .Should().Throw<ArgumentException>()
                .WithMessage("Cannot cast ufunc 'power' input 0 from dtype('float64') to dtype('uint8') with casting rule 'same_kind'");
        }
    }
}
