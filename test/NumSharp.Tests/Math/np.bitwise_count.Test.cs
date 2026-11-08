using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     Pins <see cref="np.bitwise_count(NDArray,NDArray,NDArray,DType)"/> against NumPy 2.4.2. The broad
    ///     value × dtype × layout matrix is gated bit-exactly by the differential-fuzz <c>bitwise_count</c>
    ///     tier; these tests target what that tier does NOT reach cleanly — the magnitude/two's-complement
    ///     edges (signed minimum, -1), the fixed uint8 output dtype, layout/param plumbing (out=/where=/dtype=,
    ///     strided/F-order/0-d/empty), and the full error taxonomy (integer-only loop, dtype= must be uint8) —
    ///     so a regression in any of those surfaces here rather than silently.
    /// </summary>
    [TestClass]
    public class BitwiseCountTest
    {
        /// <summary>Read the whole result as a uint8 array (the only dtype bitwise_count ever produces by default).</summary>
        /// <param name="a">A bitwise_count result.</param>
        /// <returns>The elements as a <see cref="byte"/>[] in logical C-order.</returns>
        private static byte[] B(NDArray a) => a.ToArray<byte>();

        // ---------------------------------------------------------------- values / dtypes

        /// <summary>Every supported dtype maps to a uint8 result whose values are the set-bit count of the
        /// MAGNITUDE — including bool (→0/1) and NumSharp's Char (16-bit, no NumPy analog).</summary>
        [TestMethod]
        public void Values_AllDtypes()
        {
            CollectionAssert.AreEqual(new byte[] { 0, 1, 8, 1, 4 },
                B(np.bitwise_count(np.array(new byte[] { 0, 1, 255, 128, 170 }))));
            CollectionAssert.AreEqual(new byte[] { 16, 0, 1 },
                B(np.bitwise_count(np.array(new ushort[] { 65535, 0, 256 }))));
            CollectionAssert.AreEqual(new byte[] { 32, 0, 1 },
                B(np.bitwise_count(np.array(new uint[] { uint.MaxValue, 0, 1u << 31 }))));
            CollectionAssert.AreEqual(new byte[] { 64, 0, 1 },
                B(np.bitwise_count(np.array(new ulong[] { ulong.MaxValue, 0, 1UL << 63 }))));
            CollectionAssert.AreEqual(new byte[] { 1, 0, 1, 1 },
                B(np.bitwise_count(np.array(new bool[] { true, false, true, true }))));
            // Char: popcount of the 16-bit code unit ('A' == 0x41 == 2 bits).
            CollectionAssert.AreEqual(new byte[] { 16, 0, 2 },
                B(np.bitwise_count(np.array(new char[] { (char)0xFFFF, (char)0, 'A' }))));
        }

        /// <summary>Signed inputs count the MAGNITUDE (two's-complement abs), so -1 → 1 (not 8) and the
        /// signed minimum → 1 (its negation overflows to itself), matching NumPy's <c>a &lt; 0 ? -a : a</c>.</summary>
        [TestMethod]
        public void Values_SignedMagnitudeEdges()
        {
            CollectionAssert.AreEqual(new byte[] { 1, 1, 0, 1, 7 },
                B(np.bitwise_count(np.array(new sbyte[] { -128, -1, 0, 1, 127 }))));
            CollectionAssert.AreEqual(new byte[] { 1, 1, 3, 8 },
                B(np.bitwise_count(np.array(new short[] { short.MinValue, -1, 7, 255 }))));
            CollectionAssert.AreEqual(new byte[] { 1, 1, 8, 0 },
                B(np.bitwise_count(np.array(new int[] { int.MinValue, -1, 255, 0 }))));
            CollectionAssert.AreEqual(new byte[] { 1, 1, 8 },
                B(np.bitwise_count(np.array(new long[] { long.MinValue, -1, 255 }))));
        }

        /// <summary>The result dtype is uint8 for every input (probed 2.4.2), and a 0-d input yields a 0-d
        /// uint8 scalar.</summary>
        [TestMethod]
        public void OutputDtype_IsUint8_And_Scalar()
        {
            var r = np.bitwise_count(np.array(new int[] { 7 }));
            Assert.AreEqual(typeof(byte), r.dtype.type);
            var z = np.bitwise_count(np.array(7));
            Assert.AreEqual(0, z.ndim);
            Assert.AreEqual(typeof(byte), z.dtype.type);
            Assert.AreEqual((byte)3, z.GetByte(Array.Empty<long>()));
        }

        /// <summary>An empty input returns an empty uint8 array (no crash), matching NumPy.</summary>
        [TestMethod]
        public void Empty_ReturnsEmptyUint8()
        {
            var r = np.bitwise_count(np.array(new int[0]));
            Assert.AreEqual(0, r.size);
            Assert.AreEqual(typeof(byte), r.dtype.type);
        }

        // ---------------------------------------------------------------- layouts

        /// <summary>Non-contiguous inputs (strided column select, reversed) count in logical order.</summary>
        [TestMethod]
        public void Layouts_StridedAndReversed()
        {
            var m = np.array(new int[] { 1, 3, 7, 255, 256, 0 }).reshape(2, 3); // [[1,3,7],[255,256,0]]
            // columns 0,2 -> [[1,7],[255,0]] -> popcount [[1,3],[8,0]]
            CollectionAssert.AreEqual(new byte[] { 1, 3, 8, 0 }, B(np.bitwise_count(m[":, ::2"])));
            CollectionAssert.AreEqual(new byte[] { 4, 3, 2, 1 },
                B(np.bitwise_count(np.array(new int[] { 1, 3, 7, 15 })["::-1"])));
        }

        /// <summary>An F-contiguous input produces an F-contiguous result with correct logical values
        /// (NumPy preserves layout for this unary loop).</summary>
        [TestMethod]
        public void Layout_FContiguousPreserved()
        {
            var mf = np.array(new int[] { 1, 3, 7, 255 }).reshape(2, 2).copy('F'); // logical [[1,3],[7,255]]
            var r = np.bitwise_count(mf);
            Assert.IsTrue(r.Shape.IsFContiguous);
            CollectionAssert.AreEqual(new byte[] { 1, 2, 3, 8 }, B(r)); // logical [[1,2],[3,8]]
        }

        // ---------------------------------------------------------------- out= / where= / dtype=

        /// <summary>out= returns the SAME instance and, per NumPy's ufunc write-back cast, accepts a wider
        /// integer output (uint8 → int32 is a safe cast).</summary>
        [TestMethod]
        public void Out_SameInstance_And_WiderIntCast()
        {
            var o = np.zeros(new Shape(4), np.uint8);
            var r = np.bitwise_count(np.array(new int[] { 7, 7, 255, 0 }), @out: o);
            Assert.IsTrue(ReferenceEquals(r, o));
            CollectionAssert.AreEqual(new byte[] { 3, 3, 8, 0 }, B(r));

            var oi = np.zeros(new Shape(4), np.int32);
            var ri = np.bitwise_count(np.array(new int[] { 7, 7, 255, 0 }), @out: oi);
            Assert.AreEqual(typeof(int), ri.dtype.type);
            CollectionAssert.AreEqual(new int[] { 3, 3, 8, 0 }, ri.ToArray<int>());
        }

        /// <summary>where= computes only mask-true positions and leaves the prior out= contents at masked-off
        /// slots (NumPy ufunc where=).</summary>
        [TestMethod]
        public void Where_KeepsPriorContentsWhereFalse()
        {
            var baseArr = np.full(new Shape(4), (byte)99, np.uint8);
            var r = np.bitwise_count(np.array(new int[] { 7, 7, 255, 0 }),
                @out: baseArr, where: np.array(new bool[] { true, false, true, false }));
            CollectionAssert.AreEqual(new byte[] { 3, 99, 8, 99 }, B(r));
        }

        /// <summary>dtype=uint8 is the only accepted loop request (every loop outputs uint8); it is a no-op
        /// on the computation.</summary>
        [TestMethod]
        public void Dtype_Uint8_IsAccepted()
        {
            CollectionAssert.AreEqual(new byte[] { 3, 8 },
                B(np.bitwise_count(np.array(new int[] { 7, 255 }), dtype: np.uint8)));
        }

        // ---------------------------------------------------------------- error taxonomy

        /// <summary>float/complex/decimal/half inputs have no loop and raise the verbatim NumPy input-coercion
        /// TypeError.</summary>
        [TestMethod]
        public void FloatFamilyInput_Throws()
        {
            foreach (var bad in new NDArray[]
            {
                np.array(new double[] { 1.0, 2.0 }),
                np.array(new float[] { 1.0f }),
                np.array(new Complex[] { new Complex(1, 2) }),
                np.array(new decimal[] { 1m }),
                np.array(new Half[] { (Half)1.0 }),
            })
            {
                var ex = Assert.ThrowsException<TypeError>(() => np.bitwise_count(bad));
                StringAssert.Contains(ex.Message, "ufunc 'bitwise_count' not supported for the input types");
            }
        }

        /// <summary>A dtype= other than uint8 selects no loop (probed: even an integer dtype like int32 raises,
        /// because every bitwise_count loop outputs uint8).</summary>
        [TestMethod]
        public void DtypeNotUint8_Throws_NoLoop()
        {
            foreach (var dt in new DType[] { np.int32, np.int64, np.uint16, np.float64 })
            {
                var ex = Assert.ThrowsException<IncorrectTypeException>(
                    () => np.bitwise_count(np.array(new int[] { 1, 2 }), dtype: dt));
                StringAssert.Contains(ex.Message, "No loop matching the specified signature");
                StringAssert.Contains(ex.Message, "bitwise_count");
            }
        }

        /// <summary>A non-bool where= mask is rejected with the verbatim safe-cast message (validated before
        /// the loop, matching NumPy's ufunc argument order).</summary>
        [TestMethod]
        public void NonBoolWhere_Throws()
        {
            Assert.ThrowsException<ArgumentException>(() =>
                np.bitwise_count(np.array(new int[] { 1, 2 }),
                    @out: np.zeros(new Shape(2), np.uint8),
                    where: np.array(new int[] { 1, 0 })));
        }
    }
}
