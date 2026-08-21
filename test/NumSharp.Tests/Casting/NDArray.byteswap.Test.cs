using System;
using System.Linq;
using System.Numerics;
using System.Buffers.Binary;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     ndarray.byteswap(inplace=False) pinned against NumPy 2.4.2. byteswap reverses the bytes
    ///     WITHIN each element (toggling endian representation) without changing the dtype, so the
    ///     reinterpreted values change. Complex swaps its real/imaginary halves independently; 1-byte
    ///     dtypes are a value no-op (but inplace=False still returns a fresh copy). Every value below
    ///     was produced by running the equivalent NumPy 2.4.2 code.
    /// </summary>
    [TestClass]
    public class byteswapTests
    {
        private static string Hex(NDArray a) => Convert.ToHexString(a.tobytes()).ToLowerInvariant();

        // ---- value parity (results from NumPy 2.4.2) ---------------------------------------------

        [TestMethod]
        public void Int32_ValueSwap()
        {
            var a = np.array(new int[] { 1, 256, 8755 });
            var b = a.byteswap();
            // np.array([1,256,8755],dtype=np.int32).byteswap() -> [16777216, 65536, 857866240]
            b[0].GetInt32().Should().Be(16777216);
            b[1].GetInt32().Should().Be(65536);
            b[2].GetInt32().Should().Be(857866240);
        }

        [TestMethod]
        public void Int16_ValueSwap()
        {
            var a = np.array(new short[] { 0x0102 });          // 258
            a.byteswap()[0].GetInt16().Should().Be(0x0201);    // 513
        }

        [TestMethod]
        public void Int64_ValueSwap()
        {
            var a = np.array(new long[] { 1 });
            // 1 -> 0x0100000000000000
            a.byteswap()[0].GetInt64().Should().Be(72057594037927936L);
        }

        [TestMethod]
        public void Float64_ByteExact()
        {
            var a = np.array(new double[] { 1.0, 2.5 });
            // NumPy: 000000000000f03f0000000000000440 -> 3ff00000000000004004000000000000
            Hex(a.byteswap()).Should().Be("3ff00000000000004004000000000000");
        }

        [TestMethod]
        public void Complex128_SwapsEachHalfIndependently()
        {
            var a = np.array(new Complex[] { new Complex(1, 2) });
            // NumPy: real half and imag half reversed separately
            //   000000000000f03f 0000000000000040 -> 3ff0000000000000 4000000000000000
            Hex(a.byteswap()).Should().Be("3ff00000000000004000000000000000");
        }

        [TestMethod]
        public void Half_SwapsTwoBytes()
        {
            var a = np.array(new Half[] { (Half)1.0f });       // 0x3c00 -> bytes 003c
            Hex(a.byteswap()).Should().Be("3c00");
        }

        [TestMethod]
        public void Bool_IsValueNoOp_ButNotInplaceReturnsCopy()
        {
            var a = np.array(new bool[] { true, false, true });
            var b = a.byteswap();
            Hex(b).Should().Be(Hex(a));
            ReferenceEquals(a, b).Should().BeFalse();          // still a fresh array
        }

        [TestMethod]
        public void SByte_NoOpValues_DistinctCopy()
        {
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var b = a.byteswap();
            b[0].GetSByte().Should().Be((sbyte)1);
            b[1].GetSByte().Should().Be((sbyte)2);
            b[2].GetSByte().Should().Be((sbyte)3);
            ReferenceEquals(a.Array, b.Array).Should().BeFalse();
        }

        // ---- semantics ---------------------------------------------------------------------------

        [TestMethod]
        public void NotInplace_LeavesOriginalUntouched_ReturnsDistinctWriteableCopy()
        {
            var a = np.array(new int[] { 1, 256, 8755 });
            string before = Hex(a);
            var b = a.byteswap();
            Hex(a).Should().Be(before);                        // original untouched
            ReferenceEquals(a, b).Should().BeFalse();
            b.Shape.IsWriteable.Should().BeTrue();
            b.dtype.Should().Be(a.dtype);
            b.shape.Should().Equal(a.shape);
        }

        [TestMethod]
        public void Inplace_ReturnsSelf_AndMutates()
        {
            var a = np.array(new int[] { 1, 256, 8755 });
            var r = a.byteswap(inplace: true);
            ReferenceEquals(a, r).Should().BeTrue();
            a[0].GetInt32().Should().Be(16777216);
        }

        [TestMethod]
        public void PositionalTrue_IsInplace()
        {
            var a = np.array(new int[] { 5 });
            ReferenceEquals(a, a.byteswap(true)).Should().BeTrue();
        }

        // ---- errors ------------------------------------------------------------------------------

        [TestMethod]
        public void Inplace_OnBroadcastReadOnlyView_RaisesValueError()
        {
            var src = np.array(new int[] { 1, 2, 3, 4 }).reshape(1, 4);
            var bc = np.broadcast_to(src, new Shape(3, 4));     // read-only
            Action act = () => bc.byteswap(inplace: true);
            act.Should().Throw<ValueError>().WithMessage("array to be byte-swapped is read-only");
        }

        [TestMethod]
        public void Inplace_OnNonWriteableArray_RaisesValueError()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            a.flags.writeable = false;
            Action act = () => a.byteswap(inplace: true);
            act.Should().Throw<ValueError>();
        }

        [TestMethod]
        public void NotInplace_OnReadOnlyArray_IsAllowed()
        {
            var a = np.array(new int[] { 1, 256, 8755 });
            a.flags.writeable = false;
            var b = a.byteswap();                              // copy path — fine on read-only
            b[0].GetInt32().Should().Be(16777216);
            b.Shape.IsWriteable.Should().BeTrue();
        }

        // ---- layouts -----------------------------------------------------------------------------

        [TestMethod]
        public void ZeroDimensional()
        {
            var a = NDArray.Scalar(0x01020304);                // int32 0-d
            var b = a.byteswap();
            b.ndim.Should().Be(0);
            b.GetInt32().Should().Be(0x04030201);
            ReferenceEquals(a, b).Should().BeFalse();          // not-inplace still copies a scalar
        }

        [TestMethod]
        public void Empty_PreservesShape()
        {
            var a = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
            a.byteswap().shape.Should().Equal(a.shape);
        }

        [TestMethod]
        public void FContiguous_NotInplace_StaysFContiguous()
        {
            // NumPy: PyArray_NewCopy(ANYORDER) keeps F for an F-contiguous(-and-not-C) source.
            var f = np.arange(6).astype(NPTypeCode.Int32).reshape(2, 3).T;   // F-contiguous view
            f.Shape.IsFContiguous.Should().BeTrue();
            var b = f.byteswap();
            b.Shape.IsFContiguous.Should().BeTrue();
            // value parity: element-wise byteswap regardless of layout
            for (int i = 0; i < f.shape[0]; i++)
                for (int j = 0; j < f.shape[1]; j++)
                    b[i, j].GetInt32().Should().Be(BinaryPrimitives.ReverseEndianness(f[i, j].GetInt32()));
        }

        [TestMethod]
        public void StridedInplace_TouchesOnlyViewedElements()
        {
            var b = np.array(Enumerable.Range(0, 8).ToArray());   // int32 0..7 contiguous
            var v = b["::2"];                                     // even indices
            v.byteswap(inplace: true);
            for (int i = 0; i < 8; i++)
            {
                int expected = (i % 2 == 0) ? BinaryPrimitives.ReverseEndianness(i) : i;
                b[i].GetInt32().Should().Be(expected);
            }
        }

        [TestMethod]
        public void ReversedView_NotInplace_ValueParity()
        {
            var a = np.arange(10).astype(NPTypeCode.Int32)["::-1"];
            var b = a.byteswap();
            for (int i = 0; i < a.size; i++)
                b[i].GetInt32().Should().Be(BinaryPrimitives.ReverseEndianness(a[i].GetInt32()));
        }

        // ---- round-trip identity across every dtype ----------------------------------------------

        [TestMethod]
        public void DoubleSwap_IsIdentity_AllDtypes()
        {
            var tcs = new (NPTypeCode tc, int sz)[]
            {
                (NPTypeCode.Boolean,1),(NPTypeCode.Byte,1),(NPTypeCode.SByte,1),
                (NPTypeCode.Int16,2),(NPTypeCode.UInt16,2),(NPTypeCode.Char,2),(NPTypeCode.Half,2),
                (NPTypeCode.Int32,4),(NPTypeCode.UInt32,4),(NPTypeCode.Single,4),
                (NPTypeCode.Int64,8),(NPTypeCode.UInt64,8),(NPTypeCode.Double,8),
                (NPTypeCode.Decimal,16),(NPTypeCode.Complex,16),
            };
            var rng = new System.Random(42);
            foreach (var (tc, sz) in tcs)
            {
                var buf = new byte[sz * 13];
                rng.NextBytes(buf);
                var a = np.frombuffer(buf, tc).copy();
                string original = Hex(a);
                // not-inplace round trip
                Hex(a.byteswap().byteswap()).Should().Be(original, $"double byteswap must be identity for {tc}");
                // in-place round trip
                var c = np.frombuffer((byte[])buf.Clone(), tc).copy();
                c.byteswap(inplace: true);
                c.byteswap(inplace: true);
                Hex(c).Should().Be(original, $"double in-place byteswap must be identity for {tc}");
            }
        }

        // ---- NumSharp-only dtypes (no NumPy analog): consistent raw-itemsize reversal ------------

        [TestMethod]
        public void Char_SwapsTwoBytes()
        {
            var a = np.array(new char[] { 'A', 'ሴ' });
            var swapped = a.byteswap();
            // 'A'=0x0041 -> 0x4100 ; 'ሴ' -> 0x3412
            ((int)swapped[0].GetChar()).Should().Be(0x4100);
            ((int)swapped[1].GetChar()).Should().Be(0x3412);
        }

        [TestMethod]
        public void Decimal_ReversesSixteenBytes()
        {
            var a = np.array(new decimal[] { 1.5m, -12345.6789m });
            var swapped = a.byteswap();
            var inBytes = a.tobytes();
            var outBytes = swapped.tobytes();
            for (int e = 0; e < 2; e++)
                for (int k = 0; k < 16; k++)
                    outBytes[e * 16 + k].Should().Be(inBytes[e * 16 + 15 - k]);
        }

        // ---- large array exercises the SIMD body + non-temporal store path -----------------------

        [TestMethod]
        public void Large_Int32_SimdAndNonTemporalPath()
        {
            int n = 700_000;                                    // 2.8 MB > non-temporal threshold
            var a = np.arange(n).astype(NPTypeCode.Int32);
            var b = a.byteswap();
            b.size.Should().Be(n);
            for (long i = 0; i < n; i += 50_001)
                b[i].GetInt32().Should().Be(BinaryPrimitives.ReverseEndianness(a[i].GetInt32()));
            // round-trip identity confirms every byte moved correctly
            var round = b.byteswap();
            for (long i = 0; i < n; i += 50_001)
                round[i].GetInt32().Should().Be(a[i].GetInt32());
        }
    }
}
