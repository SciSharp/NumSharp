using System;
using System.Numerics;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     <c>np.fromstring</c> — parse a 1-D array from the numbers in a text string. Every expected result
    ///     was probed against NumPy 2.4.2. Text mode shares <c>np.fromfile</c>'s item reader; the binary mode
    ///     (empty separator) was removed in NumPy and raises.
    /// </summary>
    [TestClass]
    public class FromStringTests
    {
        [TestMethod]
        public void Text_BasicSeparators()
        {
            AssertD(np.fromstring("1 2 3", sep: " "), new[] { 3 }, NPTypeCode.Double, new double[] { 1, 2, 3 });
            AssertD(np.fromstring("1,2,3", NPTypeCode.Int64, sep: ","), new[] { 3 }, NPTypeCode.Int64, new double[] { 1, 2, 3 });
            AssertD(np.fromstring("1;2;3", NPTypeCode.Int32, sep: ";"), new[] { 3 }, NPTypeCode.Int32, new double[] { 1, 2, 3 });
            AssertD(np.fromstring("1.5 2.5 -3.5", sep: " "), new[] { 3 }, NPTypeCode.Double, new double[] { 1.5, 2.5, -3.5 });
            // spaces around a non-whitespace separator are a wildcard; runs of whitespace collapse.
            AssertD(np.fromstring("1, 2, 3", sep: ","), new[] { 3 }, NPTypeCode.Double, new double[] { 1, 2, 3 });
            AssertD(np.fromstring("  1   2   3  ", sep: " "), new[] { 3 }, NPTypeCode.Double, new double[] { 1, 2, 3 });
        }

        [TestMethod]
        public void Text_DefaultDtype_CountAndEmpty()
        {
            Assert.AreEqual(NPTypeCode.Double, np.fromstring("1 2", sep: " ").typecode);   // default float64
            AssertD(np.fromstring("1 2 3", NPTypeCode.Int32, 2, " "), new[] { 2 }, NPTypeCode.Int32, new double[] { 1, 2 });
            AssertD(np.fromstring("", sep: " "), new[] { 0 }, NPTypeCode.Double, Array.Empty<double>());
        }

        [TestMethod]
        public void Text_Complex()
        {
            var a = np.fromstring("1+2j 3-4j", NPTypeCode.Complex, sep: " ");
            Assert.AreEqual("2", string.Join(",", a.shape));
            Assert.AreEqual(new Complex(1, 2), (Complex)a.GetAtIndex(0));
            Assert.AreEqual(new Complex(3, -4), (Complex)a.GetAtIndex(1));
            Assert.AreEqual(new Complex(0, 5), (Complex)np.fromstring("5j", NPTypeCode.Complex, sep: " ").GetAtIndex(0));
            // imaginary unit is lowercase 'j' only — NumPy's fromstring rejects uppercase 'J'.
            Assert.ThrowsException<ValueError>(() => np.fromstring("1+2J", NPTypeCode.Complex, sep: " "));
            Assert.ThrowsException<ValueError>(() => np.fromstring("2J", NPTypeCode.Complex, sep: " "));
        }

        [TestMethod]
        public void BinaryMode_Removed_Raises()
        {
            // NumPy 2.x removed the binary mode; an empty/null separator redirects to frombuffer.
            foreach (var sep in new[] { "", (string)null })
            {
                var e = Assert.ThrowsException<ValueError>(() => np.fromstring("1 2 3", NPTypeCode.Double, -1, sep));
                Assert.AreEqual("The binary mode of fromstring is removed, use frombuffer instead", e.Message);
            }
        }

        [TestMethod]
        public void Text_UnmatchedData_Raises()
        {
            var e = Assert.ThrowsException<ValueError>(() => np.fromstring("1 x 3", sep: " "));
            Assert.AreEqual("string or file could not be read to its end due to unmatched data", e.Message);
            Assert.ThrowsException<ValueError>(() => np.fromstring("1,,3", NPTypeCode.Double, -1, ","));
        }

        [TestMethod]
        public void NullGuard()
        {
            Assert.ThrowsException<ArgumentNullException>(() => np.fromstring((string)null, NPTypeCode.Double, -1, " "));
        }

        private static void AssertD(NDArray a, int[] shape, NPTypeCode tc, double[] vals)
        {
            Assert.AreEqual(string.Join(",", shape), string.Join(",", a.shape), "shape");
            Assert.AreEqual(tc, a.typecode, "dtype");
            var r = a.ravel();
            Assert.AreEqual(vals.Length, (int)r.size, "count");
            for (int i = 0; i < vals.Length; i++)
                Assert.AreEqual(vals[i], Convert.ToDouble(r.GetAtIndex(i)), $"val[{i}]");
        }
    }
}
