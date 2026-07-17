using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Regression tests for boolean reductions over buffers holding NON-0/1 bytes.
    ///
    ///     A boolean's numeric value is exactly 0 or 1 (NumPy: every nonzero counts as 1), but a bool
    ///     buffer can legally contain non-0/1 bytes — <see cref="np.frombuffer(byte[], NPTypeCode, long, long)"/>
    ///     returns a zero-copy VIEW (like NumPy), and framework interop wraps foreign buffers. NumSharp
    ///     used to sum the RAW storage bytes (e.g. byte 255 contributed 255) in three distinct paths:
    ///       1. flat sum/prod/mean — IL <c>EmitConvertTo</c> widened the raw byte
    ///          (DirectILKernelGenerator.cs)
    ///       2. axis sum/prod/mean — bool was aliased to byte in the widening-SIMD pair table
    ///          (DirectILKernelGenerator.Reduction.Axis.Widening.cs)
    ///       3. flat var/std       — <c>VarMomentsRealDispatch</c> read bool as byte
    ///          (Default.Reduction.Var.cs)
    ///       4. narrow casts astype(bool-&gt;i8/u8/i16/u16/f16) — the SIMD subword/widen/xToHalf cast
    ///          kernels reinterpreted the raw byte (DirectILKernelGenerator.Cast.cs; fixed by routing
    ///          every bool-source cast to the scalar normalizing ConvertValue path).
    ///     The fix normalizes nonzero-&gt;1 at each conversion point (idempotent for proper 0/1 bools).
    ///
    ///     All expected values were produced by NumPy 2.4.2:
    ///     <code>
    ///     >>> b = np.frombuffer(bytes([0,1,2,3,255,0,127,128]), dtype=bool)  # [F,T,T,T,T,F,T,T]
    ///     >>> b.sum(), b.mean(), b.var(), b.std()   # 6, 0.75, 0.1875, 0.4330127018922193
    ///     >>> b2 = np.frombuffer(bytes([1,2,0,3,4,0]), dtype=bool).reshape(2,3)  # [[T,T,F],[T,T,F]]
    ///     >>> b2.sum(), b2.sum(0).tolist(), b2.sum(1).tolist()   # 4, [2,2,0], [2,2]
    ///     >>> b2.prod(0).tolist(), [round(x,6) for x in b2.var(1)]   # [1,1,0], [0.222222, 0.222222]
    ///     </code>
    /// </summary>
    [TestClass]
    public class BoolNonBinaryReductionTests : TestClass
    {
        // bytes [0,1,2,3,255,0,127,128] -> logical [F,T,T,T,T,F,T,T] (6 True)
        private static NDArray B1D() => np.frombuffer(new byte[] { 0, 1, 2, 3, 255, 0, 127, 128 }, NPTypeCode.Boolean);

        // bytes [1,2,0,3,4,0] -> [[T,T,F],[T,T,F]]
        private static NDArray B2D() => np.frombuffer(new byte[] { 1, 2, 0, 3, 4, 0 }, NPTypeCode.Boolean).reshape(2, 3);

        // ---- flat reductions ----

        [TestMethod]
        public void Sum1D_CountsTrue()
            => np.sum(B1D()).GetInt64(0).Should().Be(6L);

        [TestMethod]
        public void Mean1D()
            => np.mean(B1D()).GetDouble(0).Should().BeApproximately(0.75, 1e-12);

        [TestMethod]
        public void Var1D()
            => np.var(B1D()).GetDouble(0).Should().BeApproximately(0.1875, 1e-12);

        [TestMethod]
        public void Std1D()
            => np.std(B1D()).GetDouble(0).Should().BeApproximately(0.4330127018922193, 1e-12);

        [TestMethod]
        public void Prod1D_AllTrue_IsOne()
            => np.prod(np.frombuffer(new byte[] { 1, 2, 3 }, NPTypeCode.Boolean)).GetInt64(0).Should().Be(1L);

        // ---- axis reductions ----

        [TestMethod]
        public void Sum2D_Total()
            => np.sum(B2D()).GetInt64(0).Should().Be(4L);

        [TestMethod]
        public void Sum2D_Axis0()
            => np.sum(B2D(), 0).ToArray<long>().Should().Equal(new long[] { 2, 2, 0 });

        [TestMethod]
        public void Sum2D_Axis1()
            => np.sum(B2D(), 1).ToArray<long>().Should().Equal(new long[] { 2, 2 });

        [TestMethod]
        public void Prod2D_Axis0()
            => np.prod(B2D(), 0).ToArray<long>().Should().Equal(new long[] { 1, 1, 0 });

        [TestMethod]
        public void Var2D_Axis1()
        {
            var v = np.var(B2D(), 1).ToArray<double>();
            v.Should().HaveCount(2);
            v[0].Should().BeApproximately(2.0 / 9.0, 1e-12);  // var([1,1,0]) = 2/9
            v[1].Should().BeApproximately(2.0 / 9.0, 1e-12);
        }

        [TestMethod]
        public void Std2D_Axis1()
        {
            var s = np.std(B2D(), 1).ToArray<double>();
            s.Should().HaveCount(2);
            s[0].Should().BeApproximately(Math.Sqrt(2.0 / 9.0), 1e-12);
            s[1].Should().BeApproximately(Math.Sqrt(2.0 / 9.0), 1e-12);
        }

        // ---- casts: astype(bool -> T) must normalize non-0/1 bytes to 0/1 for EVERY target ----
        // (narrow targets i8/u8/i16/u16/f16 used to reinterpret the raw byte). Norm is the
        // logical value of B1D(): [0,1,1,1,1,0,1,1].
        private static readonly long[] Norm = { 0, 1, 1, 1, 1, 0, 1, 1 };

        [TestMethod] public void Astype_ToInt8()
            => B1D().astype(NPTypeCode.SByte).astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(Norm);
        [TestMethod] public void Astype_ToUInt8()
            => B1D().astype(NPTypeCode.Byte).astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(Norm);
        [TestMethod] public void Astype_ToInt16()
            => B1D().astype(NPTypeCode.Int16).astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(Norm);
        [TestMethod] public void Astype_ToUInt16()
            => B1D().astype(NPTypeCode.UInt16).astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(Norm);
        [TestMethod] public void Astype_ToHalf()
            => B1D().astype(NPTypeCode.Half).astype(NPTypeCode.Double).ToArray<double>().Should().Equal(new double[] { 0, 1, 1, 1, 1, 0, 1, 1 });
        [TestMethod] public void Astype_ToInt32_Wide()
            => B1D().astype(NPTypeCode.Int32).astype(NPTypeCode.Int64).ToArray<long>().Should().Equal(Norm);
        [TestMethod] public void Astype_ToSingle_Wide()
            => B1D().astype(NPTypeCode.Single).astype(NPTypeCode.Double).ToArray<double>().Should().Equal(new double[] { 0, 1, 1, 1, 1, 0, 1, 1 });

        // ---- regression guard: canonical 0/1 buffers must keep working ----

        [TestMethod]
        public void Sum_CanonicalBytes_Unchanged()
            => np.sum(np.frombuffer(new byte[] { 0, 1, 1, 0, 1 }, NPTypeCode.Boolean)).GetInt64(0).Should().Be(3L);
    }
}
