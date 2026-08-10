using System;
using System.Collections.Generic;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// np.iterable parity gate. NumPy's whole body is
    /// <c>try: iter(y); return True; except TypeError: return False</c> — a pure predicate,
    /// not an iteration op. Every expectation below was probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class np_iterable_tests
    {
        // Bare scalar value types are not iterable (iter() on a NumPy scalar raises TypeError).
        [TestMethod]
        [DataRow(typeof(double))]
        [DataRow(typeof(float))]
        [DataRow(typeof(byte))]
        [DataRow(typeof(sbyte))]
        [DataRow(typeof(int))]
        [DataRow(typeof(long))]
        [DataRow(typeof(char))]
        [DataRow(typeof(short))]
        [DataRow(typeof(uint))]
        [DataRow(typeof(ulong))]
        [DataRow(typeof(ushort))]
        [DataRow(typeof(bool))]
        [DataRow(typeof(decimal))]
        public void PrimitiveScalars_AreNotIterable(Type type)
        {
            var value = Convert.ChangeType((byte)0, type);
            Assert.IsFalse(np.iterable(value));
        }

        // A C# array of any element type is iterable (the Python-list analog).
        [TestMethod]
        [DataRow(typeof(double))]
        [DataRow(typeof(float))]
        [DataRow(typeof(byte))]
        [DataRow(typeof(int))]
        [DataRow(typeof(long))]
        [DataRow(typeof(char))]
        [DataRow(typeof(short))]
        [DataRow(typeof(decimal))]
        public void PrimitiveArrays_AreIterable(Type type)
        {
            var value = Convert.ChangeType((byte)0, type);
            var arr = Array.CreateInstance(type, 1);
            arr.SetValue(value, 0);
            Assert.IsTrue(np.iterable(arr));
        }

        [TestMethod]
        public void Half_IsNotIterable()
        {
            Assert.IsFalse(np.iterable((Half)1));
        }

        [TestMethod]
        public void Complex_IsNotIterable()
        {
            Assert.IsFalse(np.iterable(new Complex(15, 15)));
        }

        [TestMethod]
        public void Null_IsNotIterable()
        {
            // NumPy's iter(None) raises TypeError -> False.
            Assert.IsFalse(np.iterable(null));
        }

        // Python strings are iterable (np.iterable("abc") == True), so C# strings match.
        [TestMethod]
        [DataRow("")]
        [DataRow("Hi")]
        public void String_IsIterable(string value)
        {
            Assert.IsTrue(np.iterable(value));
        }

        [TestMethod]
        public void Collections_AreIterable()
        {
            Assert.IsTrue(np.iterable(new List<int> { 1, 2, 3 }));
            Assert.IsTrue(np.iterable(new Dictionary<int, int> { { 1, 1 } }));
            Assert.IsTrue(np.iterable(new HashSet<int> { 1, 2 }));
            Assert.IsTrue(np.iterable(new object[] { 1, "two", 3.0 }));
            Assert.IsTrue(np.iterable(new int[2, 2])); // multidimensional array
        }

        // The one surprise NumPy documents: a 0-d array is NOT iterable, though rank>=1 is.
        [TestMethod]
        public void NDArray_ZeroD_IsNotIterable()
        {
            NDArray scalar = 1d; // implicit conversion yields a 0-d array (ndim == 0)
            Assert.AreEqual(0, scalar.ndim);
            Assert.IsFalse(np.iterable(scalar));

            Assert.IsFalse(np.iterable(NDArray.Scalar(5)));
        }

        [TestMethod]
        public void NDArray_RankOneOrMore_IsIterable()
        {
            Assert.IsTrue(np.iterable(np.arange(3)));                  // 1-d
            Assert.IsTrue(np.iterable(np.arange(4).reshape(2, 2)));    // 2-d
        }

        // Empty arrays are still iterable — iterability is a rank property, not a size property.
        [TestMethod]
        public void NDArray_Empty_IsIterable()
        {
            Assert.IsTrue(np.iterable(np.zeros(new Shape(0))));    // empty 1-d
            Assert.IsTrue(np.iterable(np.zeros(new Shape(0, 3)))); // empty 2-d
        }
    }
}
