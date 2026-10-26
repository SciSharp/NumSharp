using System;
using System.Linq;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     NumPy's float16 product loops (<c>HALF_dot</c> behind np.dot, np.vecdot, np.vdot, np.convolve and np.correlate;
    ///     the HALF loop of <c>matmul_inner_noblas</c> behind np.matmul, np.matvec, np.vecmat, np.inner and np.tensordot)
    ///     decide which NaN a NaN result carries by their operand order: in a product of two NaNs the SECOND factor's NaN
    ///     survives, and in the float32 sum the first NaN of the sequence does (x86's default NaN, negative, where
    ///     <c>inf * 0</c> makes one). NumSharp's kernels kept the product's NaN in the sum (RyuJIT's operand order), so
    ///     NaN results came back with another sign and payload. Every expected value is NumPy 2.4.2's raw bits
    ///     (win-amd64); NaNs are compared bit for bit, since <c>NaN == NaN</c> is false and the oracle tokenizes them.
    /// </summary>
    [TestClass]
    public class HalfProductNaNPriorityTests
    {
        private const ushort P1 = 0x7E01, N2 = 0xFE04, P3 = 0x7E05, N4 = 0xFE06, One = 0x3C00, Inf = 0x7C00, Zero = 0;

        /// <summary>A float16 array of the given raw bits.</summary>
        /// <param name="bits">The raw float16 bits.</param>
        /// <param name="shape">The shape, or none for a vector.</param>
        /// <returns>The array.</returns>
        private static NDArray H(ushort[] bits, params long[] shape)
        {
            var a = np.array(bits.Select(BitConverter.UInt16BitsToHalf).ToArray());
            return shape.Length == 0 ? a : a.reshape(new Shape(shape));
        }

        /// <summary>Asserts a float16 result's raw bits, in C order.</summary>
        /// <param name="label">The NumPy call, for the failure message.</param>
        /// <param name="r">The NumSharp result.</param>
        /// <param name="bits">NumPy's raw bits.</param>
        private static void AssertBits(string label, NDArray r, params ushort[] bits)
        {
            Assert.AreEqual(np.float16, r.dtype, label);
            var got = np.ascontiguousarray(r).ToArray<Half>().Select(BitConverter.HalfToUInt16Bits).ToArray();
            CollectionAssert.AreEqual(bits, got,
                $"{label}: numpy [{string.Join(",", bits.Select(b => b.ToString("x4")))}], numsharp [{string.Join(",", got.Select(b => b.ToString("x4")))}]");
        }

        /// <summary>The four entry points of a vector pair: matmul of the vectors and of a row and a column, vecdot, dot.</summary>
        /// <param name="label">The case, for the failure message.</param>
        /// <param name="a">The first vector's bits.</param>
        /// <param name="b">The second vector's bits.</param>
        /// <param name="expected">NumPy's bits of the one result.</param>
        private static void AssertVectorPair(string label, ushort[] a, ushort[] b, ushort expected)
        {
            AssertBits(label + " matmul", np.matmul(H(a), H(b)), expected);
            AssertBits(label + " matmul 2-D", np.matmul(H(a, 1, a.Length), H(b, b.Length, 1)), expected);
            AssertBits(label + " vecdot", np.vecdot(H(a), H(b)), expected);
            AssertBits(label + " dot", np.dot(H(a), H(b)), expected);
        }

        [TestMethod]
        public void Product_KeepsTheSecondFactorsNaN()
        {
            // np.matmul / np.vecdot / np.dot of [nan1] and [-nan2] -> -nan2 (0xfe04), and the reverse -> nan1 (0x7e01)
            AssertVectorPair("nan1 * -nan2", new[] { P1 }, new[] { N2 }, N2);
            AssertVectorPair("-nan2 * nan1", new[] { N2 }, new[] { P1 }, P1);
        }

        [TestMethod]
        public void Sum_KeepsTheFirstNaN()
        {
            // a = [nan1, -nan2], b = [1, 1]: the sum is nan1 before -nan2 arrives -> 0x7e01 (NumSharp kept 0xfe04)
            AssertVectorPair("nan1 then -nan2", new[] { P1, N2 }, new[] { One, One }, P1);
            AssertVectorPair("-nan2 then nan1", new[] { N2, P1 }, new[] { One, One }, N2);
            // a = [inf, nan1], b = [0, 1]: inf * 0 is x86's default NaN, negative, and it stays -> 0xfe00
            AssertVectorPair("inf*0 then nan1", new[] { Inf, P1 }, new[] { Zero, One }, 0xFE00);
            // a = [1, nan1], b = [-nan2, 1] -> -nan2 (0xfe04)
            AssertVectorPair("-nan2 then nan1 (second factors)", new[] { One, P1 }, new[] { N2, One }, N2);
        }

        [TestMethod]
        public void Matrices_KeepNumPysNaN()
        {
            var m = H(new[] { P1, N2, One, P3, One, N4 }, 2, 3);
            var w = H(new[] { N4, One, P3, One, N2, P1 }, 3, 2);
            // np.dot(m, w), np.inner(m, w.T), np.tensordot(m, w, 1) -> [-nan4, nan1, -nan4, nan3]
            ushort[] product = { N4, P1, N4, P3 };
            AssertBits("dot 2-D", np.dot(m, w), product);
            AssertBits("inner", np.inner(m, w.T), product);
            AssertBits("tensordot", np.tensordot(m, w, 1), product);
            // np.matmul of a stack: the second matrices reversed
            AssertBits("matmul batched", np.matmul(np.stack(new[] { m, m[":, ::-1"] }), np.stack(new[] { w, w["::-1"] })),
                N4, P1, N4, P3, N2, P1, N2, P1);
            AssertBits("matvec", np.matvec(m, H(new[] { N4, P3, One })), N4, N4);
            AssertBits("vecmat", np.vecmat(H(new[] { N2, P1 }), m), P1, N2, N2);
            AssertBits("vdot", np.vdot(H(new[] { P1, N2, One }), H(new[] { One, P3, N4 })), P1);
        }

        [TestMethod]
        public void MixedOperands_RunTheHalfLoop()
        {
            // NumPy casts int8 / bool to the float16 loop, then runs the same sum
            var a = H(new[] { P1, One, N2, One }, 2, 2);
            AssertBits("matmul f16 @ int8", np.matmul(a, np.array(new sbyte[,] { { 1, 2 }, { 3, 4 } })), P1, P1, N2, N2);
            AssertBits("matmul int8 @ f16", np.matmul(np.array(new sbyte[,] { { 1, 2 }, { 3, 4 } }), a), P1, 0x4200, P1,
                0x4700);
            AssertBits("matmul f16 @ bool", np.matmul(a, np.array(new bool[,] { { true, true }, { true, true } })), P1, P1,
                N2, N2);
            AssertBits("vecdot f16 . int8", np.vecdot(H(new[] { P1, N2, One }), np.array(new sbyte[] { 1, 2, 3 })), P1);
        }

        [TestMethod]
        public void ConvolveCorrelate_KeepNumPysNaN()
        {
            // Data [1, nan1, -nan2, 1, 1, -nan2, nan1, 1] against three ones: the window that holds both NaNs keeps the
            // first in the sum's order (NumSharp kept the later one), through the scalar kernel.
            var d = H(new[] { One, P1, N2, One, One, N2, P1, One });
            var ones3 = H(new[] { One, One, One });
            AssertBits("convolve 8 / 3 full", np.convolve(d, ones3, "full"),
                One, P1, P1, P1, N2, N2, N2, N2, P1, One);
            AssertBits("correlate 8 / 3 valid", np.correlate(d, ones3, "valid"), P1, P1, N2, N2, N2, N2);

            // NaNs in both operands (which one is the second factor), both length orders:
            // np.convolve([1, nan1, 1, 1, nan3, 1], [1, -nan4, 1]) and np.correlate([1, nan3, 1], [1, -nan2, 1, 1, -nan4, 1])
            AssertBits("convolve 6 / 3 full", np.convolve(H(new[] { One, P1, One, One, P3, One }), H(new[] { One, N4, One })),
                One, N4, N4, P1, N4, N4, P3, One);
            AssertBits("correlate 3 / 6 full",
                np.correlate(H(new[] { One, P3, One }), H(new[] { One, N2, One, One, N4, One }), "full"),
                One, N4, P3, P3, N2, P3, P3, One);

            // A long pair runs the blocked kernel: data of 300 (nan1 at 1, -nan2 at 2 and 297, nan1 at 298), kernel of
            // 40 ones -> 261 valid positions, the first two nan1, then -nan2, then the finite 40s (0x5100)
            var longData = Enumerable.Repeat(One, 300).ToArray();
            longData[1] = P1; longData[2] = N2; longData[297] = N2; longData[298] = P1;
            var valid = np.correlate(H(longData), H(Enumerable.Repeat(One, 40).ToArray()), "valid");
            var bits = np.ascontiguousarray(valid).ToArray<Half>().Select(BitConverter.HalfToUInt16Bits).ToArray();
            Assert.AreEqual(261, bits.Length);
            CollectionAssert.AreEqual(new[] { P1, P1, N2, (ushort)0x5100 }, bits.Take(4).ToArray(), "correlate 300 / 40 head");
            CollectionAssert.AreEqual(new[] { (ushort)0x5100, N2, N2, N2 }, bits.Skip(257).ToArray(), "correlate 300 / 40 tail");
            var full = np.convolve(H(longData), H(Enumerable.Repeat(One, 40).ToArray()), "full");
            var fb = np.ascontiguousarray(full).ToArray<Half>().Select(BitConverter.HalfToUInt16Bits).ToArray();
            Assert.AreEqual(339, fb.Length);
            CollectionAssert.AreEqual(new[] { One, P1, P1, P1, P1 }, fb.Take(5).ToArray(), "convolve 300 / 40 head");
            CollectionAssert.AreEqual(new[] { N2, N2, N2, P1, One }, fb.Skip(334).ToArray(), "convolve 300 / 40 tail");
        }
    }
}
