using System;
using System.Numerics;

namespace NumSharp.Tests.Logic
{
    [TestClass]
    public class np_isscalar_tests
    {
        [TestMethod]
        [DataRow(typeof(double))]
        [DataRow(typeof(float))]
        [DataRow(typeof(byte))]
        [DataRow(typeof(int))]
        [DataRow(typeof(long))]
        [DataRow(typeof(char))]
        [DataRow(typeof(short))]
        [DataRow(typeof(uint))]
        [DataRow(typeof(ulong))]
        [DataRow(typeof(ushort))]
        [DataRow(typeof(decimal))]
        public void AllPrimitiveTypes(Type type)
        {
            var value = Convert.ChangeType((byte)0, type);
            Assert.IsTrue(np.isscalar(value));
        }

        [TestMethod]
        [DataRow(typeof(double))]
        [DataRow(typeof(float))]
        [DataRow(typeof(byte))]
        [DataRow(typeof(int))]
        [DataRow(typeof(long))]
        [DataRow(typeof(char))]
        [DataRow(typeof(short))]
        [DataRow(typeof(uint))]
        [DataRow(typeof(ulong))]
        [DataRow(typeof(ushort))]
        [DataRow(typeof(decimal))]
        public void AllPrimitiveArrayTypes(Type type)
        {
            var value = Convert.ChangeType((byte)0, type);
            var arr = Array.CreateInstance(type, 1);
            arr.SetValue(value, 0);
            Assert.IsFalse(np.isscalar(arr));
        }

        [TestMethod]
        public void Complex()
        {
            var value = new Complex(15, 15);
            Assert.IsTrue(np.isscalar(value));
        }

        [TestMethod]
        public void Null()
        {
            Assert.IsFalse(np.isscalar(null));
        }

        [TestMethod]
        [DataRow("")]
        [DataRow("Hi")]
        public void String(string value)
        {
            Assert.IsTrue(np.isscalar(value));
        }

        [TestMethod]
        public void NDArray()
        {
            var value = np.zeros(new Shape(3, 3));
            Assert.IsFalse(np.isscalar(value));
            NDArray nd = 1d;
            Assert.IsTrue(np.isscalar(nd));
        }

        /// <summary>
        ///     The C# spellings of NumPy and Python scalars that are no CLR primitive: NumPy 2.4.2 answers True for
        ///     <c>np.isscalar(np.float16(1.5))</c> (a <c>generic</c>), <c>np.isscalar(2**70)</c> (a <c>numbers.Number</c>),
        ///     <c>np.isscalar(np.bytes_(b'a'))</c> and <c>np.isscalar(b'a')</c>, and <c>np.isscalar(np.datetime64('2020-01-01'))</c>.
        ///     The first two answered False before the np-layer plan's NL11, as did <see cref="Char8"/> and
        ///     <see cref="DateTime64"/>.
        /// </summary>
        [TestMethod]
        public void ScalarStandIns_AreScalars()
        {
            Assert.IsTrue(np.isscalar((Half)1.5), "np.isscalar(np.float16(1.5)) is True");
            Assert.IsTrue(np.isscalar(BigInteger.Pow(2, 70)), "np.isscalar(2**70) is True");
            Assert.IsTrue(np.isscalar(new BigInteger(5)), "a small Python int is a scalar too");
            Assert.IsTrue(np.isscalar((Char8)'a'), "np.isscalar(np.bytes_(b'a')) is True");
            Assert.IsTrue(np.isscalar(new DateTime64(0)), "np.isscalar(np.datetime64('2020-01-01')) is True");
        }

        /// <summary>
        ///     A type is not a scalar: <c>np.isscalar(int)</c>, <c>np.isscalar(float)</c> and <c>np.isscalar(np.float64)</c>
        ///     are False in NumPy 2.4.2 (a class object is neither a <c>generic</c> nor a <c>numbers.Number</c>). NumSharp read
        ///     a <see cref="Type"/> as the type it names until NL11, so <c>np.isscalar(typeof(int))</c> answered True.
        /// </summary>
        [TestMethod]
        public void TypeObjects_AreNotScalars()
        {
            Assert.IsFalse(np.isscalar(typeof(int)), "np.isscalar(int) is False");
            Assert.IsFalse(np.isscalar(typeof(double)), "np.isscalar(float) is False");
            Assert.IsFalse(np.isscalar(typeof(Half)), "np.isscalar(np.float16) is False");
            Assert.IsFalse(np.isscalar(typeof(int[])), "a class object of an array type is no scalar either");
        }
    }
}
