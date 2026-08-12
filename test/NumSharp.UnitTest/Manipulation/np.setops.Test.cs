using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// np.union1d / intersect1d / setxor1d / setdiff1d parity with NumPy 2.4.2
    /// (arraysetops — sorted/unique set operations, flattened inputs, promoted dtype).
    /// </summary>
    [TestClass]
    public class SetOpsTests
    {
        // np.array_equal treats NaN != NaN (NumPy's equal_nan=False default), so results carrying NaN
        // need a NaN-aware element compare (NaN matches NaN by position; finite values by value).
        private static void AssertDoubleEqual(NDArray result, double[] expected)
        {
            result.typecode.Should().Be(NPTypeCode.Double);
            var got = result.ToArray<double>();
            got.Length.Should().Be(expected.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                if (double.IsNaN(expected[i])) double.IsNaN(got[i]).Should().BeTrue($"index {i}");
                else got[i].Should().Be(expected[i], $"index {i}");
            }
        }

        // ---- union1d --------------------------------------------------------------------
        [TestMethod]
        public void Union_Basic()
        {
            np.union1d(np.array(new long[] { -1, 0, 1 }), np.array(new long[] { -2, 0, 2 }))
                .array_equal(np.array(new long[] { -2, -1, 0, 1, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Union_MixedDtype_Promotes()
        {
            var r = np.union1d(np.array(new sbyte[] { 1, 2 }), np.array(new[] { 3, 400 }));
            r.typecode.Should().Be(NPTypeCode.Int32);
            r.array_equal(np.array(new[] { 1, 2, 3, 400 })).Should().BeTrue();
        }

        [TestMethod]
        public void Union_2D_Flattened()
        {
            np.union1d(np.array(new long[,] { { 1, 2 } }), np.array(new long[,] { { 3, 4 }, { 1, 5 } }))
                .array_equal(np.array(new long[] { 1, 2, 3, 4, 5 })).Should().BeTrue();
        }

        [TestMethod]
        public void Union_Empty_IsFloat64()
        {
            var r = np.union1d(np.zeros(new Shape(0), NPTypeCode.Double), np.zeros(new Shape(0), NPTypeCode.Double));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Union_Nan_KeptAtEnd()
        {
            AssertDoubleEqual(np.union1d(np.array(new[] { 1.0, double.NaN }), np.array(new[] { double.NaN, 2.0 })),
                new[] { 1.0, 2.0, double.NaN });
        }

        // ---- intersect1d ----------------------------------------------------------------
        [TestMethod]
        public void Intersect_Basic()
        {
            np.intersect1d(np.array(new long[] { 1, 3, 4, 3 }), np.array(new long[] { 3, 1, 2, 1 }))
                .array_equal(np.array(new long[] { 1, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Intersect_MixedDtype_Promotes()
        {
            var r = np.intersect1d(np.array(new[] { 1, 2, 3 }), np.array(new[] { 2.0, 3.0 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.array_equal(np.array(new[] { 2.0, 3.0 })).Should().BeTrue();
        }

        [TestMethod]
        public void Intersect_ReturnIndices()
        {
            // xy, x_ind, y_ind = intersect1d([1,1,2,3,4], [2,1,4,6], return_indices=True)
            var r = np.intersect1d(np.array(new long[] { 1, 1, 2, 3, 4 }), np.array(new long[] { 2, 1, 4, 6 }),
                assume_unique: false, return_indices: true);
            r.Length.Should().Be(3);
            r[0].array_equal(np.array(new long[] { 1, 2, 4 })).Should().BeTrue();   // values
            r[1].array_equal(np.array(new long[] { 0, 2, 4 })).Should().BeTrue();   // first idx in ar1
            r[2].array_equal(np.array(new long[] { 1, 0, 2 })).Should().BeTrue();   // first idx in ar2
        }

        [TestMethod]
        public void Intersect_ReturnIndices_AssumeUnique()
        {
            var r = np.intersect1d(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 2, 3, 4 }),
                assume_unique: true, return_indices: true);
            r[0].array_equal(np.array(new long[] { 2, 3 })).Should().BeTrue();
            r[1].array_equal(np.array(new long[] { 1, 2 })).Should().BeTrue();
            r[2].array_equal(np.array(new long[] { 0, 1 })).Should().BeTrue();
        }

        [TestMethod]
        public void Intersect_Empty()
        {
            var r = np.intersect1d(np.array(new long[] { 1, 2 }), np.array(new long[] { 3, 4 }));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void Intersect_Nan_Excluded()
        {
            // NaN != NaN, so a shared NaN is NOT in the intersection.
            np.intersect1d(np.array(new[] { 1.0, double.NaN }), np.array(new[] { double.NaN, 1.0 }))
                .array_equal(np.array(new[] { 1.0 })).Should().BeTrue();
        }

        [TestMethod]
        public void Intersect_ReturnIndices_KeywordOnly()
        {
            // NumPy: np.intersect1d(a, b, return_indices=True) — assume_unique omitted (defaults False).
            // The 3-overload split lets the keyword-only return_indices bind to the NDArray[] overload.
            var r = np.intersect1d(np.array(new long[] { 1, 1, 2, 3, 4 }), np.array(new long[] { 2, 1, 4, 6 }),
                return_indices: true);
            r.Length.Should().Be(3);
            r[0].array_equal(np.array(new long[] { 1, 2, 4 })).Should().BeTrue();   // values
            r[1].array_equal(np.array(new long[] { 0, 2, 4 })).Should().BeTrue();   // first idx in ar1
            r[2].array_equal(np.array(new long[] { 1, 0, 2 })).Should().BeTrue();   // first idx in ar2
        }

        [TestMethod]
        public void Intersect_Overloads_ReturnTypes()
        {
            var a = np.array(new long[] { 1, 3, 4, 3 });
            var b = np.array(new long[] { 3, 1, 2, 1 });
            // (ar1, ar2) and (ar1, ar2, assume_unique) are the bare-return overloads -> NDArray.
            np.intersect1d(a, b).array_equal(np.array(new long[] { 1, 3 })).Should().BeTrue();
            np.intersect1d(a, b, false).array_equal(np.array(new long[] { 1, 3 })).Should().BeTrue();
            // return_indices=false still selects the NDArray[] overload and yields exactly one slot.
            var one = np.intersect1d(a, b, assume_unique: false, return_indices: false);
            one.Length.Should().Be(1);
            one[0].array_equal(np.array(new long[] { 1, 3 })).Should().BeTrue();
        }

        // ---- setxor1d -------------------------------------------------------------------
        [TestMethod]
        public void Setxor_Basic()
        {
            np.setxor1d(np.array(new long[] { 1, 2, 3, 2, 4 }), np.array(new long[] { 2, 3, 5, 7, 5 }))
                .array_equal(np.array(new long[] { 1, 4, 5, 7 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setxor_AssumeUnique()
        {
            np.setxor1d(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 2, 3, 4 }), assume_unique: true)
                .array_equal(np.array(new long[] { 1, 4 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setxor_Empty_IsFloat64()
        {
            var r = np.setxor1d(np.zeros(new Shape(0), NPTypeCode.Double), np.zeros(new Shape(0), NPTypeCode.Double));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Setxor_Nan_BothKept()
        {
            // Each NaN is distinct (NaN != NaN), so both survive the xor.
            AssertDoubleEqual(np.setxor1d(np.array(new[] { 1.0, double.NaN }), np.array(new[] { double.NaN, 2.0 })),
                new[] { 1.0, 2.0, double.NaN, double.NaN });
        }

        // ---- setdiff1d ------------------------------------------------------------------
        [TestMethod]
        public void Setdiff_Basic()
        {
            np.setdiff1d(np.array(new long[] { 1, 2, 3, 2, 4, 1 }), np.array(new long[] { 3, 4, 5, 6 }))
                .array_equal(np.array(new long[] { 1, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setdiff_AssumeUnique_KeepsDuplicates()
        {
            // assume_unique skips the unique() on ar1, so its duplicates are preserved.
            np.setdiff1d(np.array(new long[] { 1, 1, 2, 3 }), np.array(new long[] { 3 }), assume_unique: true)
                .array_equal(np.array(new long[] { 1, 1, 2 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setdiff_ResultDtype_IsAr1Dtype()
        {
            var r = np.setdiff1d(np.array(new sbyte[] { 1, 2, 3 }), np.array(new long[] { 2 }));
            r.typecode.Should().Be(NPTypeCode.SByte);
            r.array_equal(np.array(new sbyte[] { 1, 3 })).Should().BeTrue();
        }

        [TestMethod]
        public void Setdiff_EmptyAr1_IsFloat64()
        {
            var r = np.setdiff1d(np.zeros(new Shape(0), NPTypeCode.Double), np.array(new long[] { 1, 2 }));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Setdiff_Nan_Kept()
        {
            AssertDoubleEqual(np.setdiff1d(np.array(new[] { 1.0, double.NaN, 2.0 }), np.array(new[] { 2.0 })),
                new[] { 1.0, double.NaN });
        }

        // ---- NaN canonicalization (NumPy's float sort rewrites EVERY surviving NaN to the positive
        //      canonical quiet NaN for float32/float64; float16/complex128 preserve). See np.setops.cs.
        //      NumSharp's sort yields .NET's negative NaN and unique preserves payloads, so these ops
        //      apply a NaN-canonicalization pass to stay bit-identical to NumPy 2.4.2. -----------------
        private const ulong CanonNanF64 = 0x7FF8000000000000UL;   // bits np.sort produces for any f64 NaN
        private const uint  CanonNanF32 = 0x7FC00000U;            // bits np.sort produces for any f32 NaN

        private static readonly double NegNanF64 = BitConverter.UInt64BitsToDouble(0xFFF8000000000000UL);
        private static readonly double PayloadNanF64 = BitConverter.UInt64BitsToDouble(0x7FF8000000ABCDEFUL);
        private static readonly float  NegNanF32 = BitConverter.UInt32BitsToSingle(0xFFC00000U);

        [TestMethod]
        public void Union_Float64_NegativeNan_CanonicalizedToPositive()
        {
            // .NET's double.NaN is 0xFFF8... (sign bit set); NumPy's union1d yields the positive canonical.
            var r = np.union1d(np.array(new[] { 1.0, NegNanF64 }), np.array(new[] { 2.0 }));
            var got = r.ToArray<double>();
            double.IsNaN(got[2]).Should().BeTrue();
            BitConverter.DoubleToUInt64Bits(got[2]).Should().Be(CanonNanF64);
        }

        [TestMethod]
        public void Union_Float64_PayloadNan_StrippedToCanonical()
        {
            var r = np.union1d(np.array(new[] { 1.0, PayloadNanF64 }), np.array(new[] { 2.0 }));
            BitConverter.DoubleToUInt64Bits(r.ToArray<double>()[2]).Should().Be(CanonNanF64);
        }

        [TestMethod]
        public void Setxor_Float64_NegativeNan_AllCanonicalizedToPositive()
        {
            // Two distinct NaNs survive; both must become the positive canonical.
            var r = np.setxor1d(np.array(new[] { 1.0, NegNanF64 }), np.array(new[] { PayloadNanF64, 2.0 }));
            var got = r.ToArray<double>();
            got.Length.Should().Be(4);
            BitConverter.DoubleToUInt64Bits(got[2]).Should().Be(CanonNanF64);
            BitConverter.DoubleToUInt64Bits(got[3]).Should().Be(CanonNanF64);
        }

        [TestMethod]
        public void Setdiff_Float64_NegativeNan_CanonicalizedToPositive()
        {
            var r = np.setdiff1d(np.array(new[] { 1.0, NegNanF64, 2.0 }), np.array(new[] { 2.0 }));
            BitConverter.DoubleToUInt64Bits(r.ToArray<double>()[1]).Should().Be(CanonNanF64);
        }

        [TestMethod]
        public void Union_Float32_NegativeNan_CanonicalizedToPositive()
        {
            var r = np.union1d(np.array(new[] { 1.0f, NegNanF32 }), np.array(new[] { 2.0f }));
            r.typecode.Should().Be(NPTypeCode.Single);
            BitConverter.SingleToUInt32Bits(r.ToArray<float>()[2]).Should().Be(CanonNanF32);
        }

        [TestMethod]
        public void Setxor_Float32_NegativeNan_CanonicalizedToPositive()
        {
            var r = np.setxor1d(np.array(new[] { 1.0f, NegNanF32 }), np.array(new[] { NegNanF32, 2.0f }));
            var got = r.ToArray<float>();
            BitConverter.SingleToUInt32Bits(got[2]).Should().Be(CanonNanF32);
            BitConverter.SingleToUInt32Bits(got[3]).Should().Be(CanonNanF32);
        }

        [TestMethod]
        public void Union_Float16_Nan_Preserved()
        {
            // NumPy does NOT canonicalize float16 NaN (no SIMD sort path); NumSharp must preserve it too.
            Half negNan16 = BitConverter.UInt16BitsToHalf(0xFE00);
            var r = np.union1d(np.array(new[] { (Half)1.0, negNan16 }), np.array(new[] { (Half)2.0 }));
            r.typecode.Should().Be(NPTypeCode.Half);
            var got = r.ToArray<Half>();
            BitConverter.HalfToUInt16Bits(got[2]).Should().Be((ushort)0xFE00);
        }

        [TestMethod]
        public void Union_Float64_NoNan_NegativeZeroPreserved()
        {
            // The canonicalization pass must not touch non-NaN lanes: -0.0 keeps its sign bit.
            var r = np.union1d(np.array(new[] { -0.0, 1.5 }), np.array(new[] { 2.5 }));
            var got = r.ToArray<double>();
            BitConverter.DoubleToUInt64Bits(got[0]).Should().Be(0x8000000000000000UL);   // -0.0
        }
    }
}
