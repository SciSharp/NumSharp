using System;
using System.Linq;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     ndarray.getfield(dtype, offset=0) / ndarray.setfield(val, dtype, offset=0) pinned against
    ///     NumPy 2.4.2. getfield returns a byte-reinterpreting VIEW: it reads <c>dtype</c> out of the
    ///     <c>offset</c>-th byte(s) of each element while keeping every byte-stride and dimension, so a
    ///     narrower field of a contiguous array is a strided view sharing memory. setfield is getfield
    ///     plus an assignment (writes the field, leaving the rest of each element intact). Every value
    ///     below was produced by running the equivalent NumPy 2.4.2 code.
    /// </summary>
    [TestClass]
    public class getfieldTests
    {
        private static string Hex(NDArray a) => Convert.ToHexString(a.tobytes()).ToLowerInvariant();

        // ---- getfield value parity (results from NumPy 2.4.2) ------------------------------------

        [TestMethod]
        public void Complex_RealLane_Offset0()
        {
            var x = np.array(new Complex[] { new(1, 2), new(3, 4), new(5, 6), new(7, 8) }).reshape(2, 2);
            var real = x.getfield(NPTypeCode.Double, 0);
            // np: [[1,3],[5,7]]
            real.shape.Should().Equal(2, 2);
            real.flatten().ToArray<double>().Should().Equal(1d, 3d, 5d, 7d);
        }

        [TestMethod]
        public void Complex_ImagLane_Offset8()
        {
            var x = np.array(new Complex[] { new(1, 2), new(3, 4), new(5, 6), new(7, 8) }).reshape(2, 2);
            var imag = x.getfield(NPTypeCode.Double, 8);
            imag.flatten().ToArray<double>().Should().Equal(2d, 4d, 6d, 8d);
        }

        [TestMethod]
        public void Int32_Int16_LowAndHighHalves()
        {
            var a = np.array(new int[] { 0x01020304, 0x05060708, 0x090A0B0C });
            // low half (little-endian): np -> [772, 1800, 2828]
            a.getfield(NPTypeCode.Int16, 0).flatten().ToArray<short>().Should().Equal((short)772, (short)1800, (short)2828);
            // high half: np -> [258, 1286, 2314]
            a.getfield(NPTypeCode.Int16, 2).flatten().ToArray<short>().Should().Equal((short)258, (short)1286, (short)2314);
        }

        [TestMethod]
        public void Int32_Int8_MisalignedByteOffset1()
        {
            // offset 1 is NOT a multiple of nothing (int8 size 1) — but exercise the middle byte.
            var a = np.array(new int[] { 0x01020304, 0x05060708, 0x090A0B0C });
            // np: a.getfield(int8, 1) -> [3, 7, 11]
            a.getfield(NPTypeCode.SByte, 1).flatten().ToArray<sbyte>().Should().Equal((sbyte)3, (sbyte)7, (sbyte)11);
        }

        [TestMethod]
        public void Int32_Int16_MisalignedByteOffset1_SubElementPointerShift()
        {
            // offset 1 with int16 (size 2): NumPy reads bytes [1:3] of each int32 — a mid-element start
            // that NumSharp absorbs into the wrap pointer. np -> [515, 1543, 2571].
            var a = np.array(new int[] { 0x01020304, 0x05060708, 0x090A0B0C });
            a.getfield(NPTypeCode.Int16, 1).flatten().ToArray<short>().Should().Equal((short)515, (short)1543, (short)2571);
        }

        [TestMethod]
        public void Int64_Int32_MisalignedOffset3()
        {
            var b = np.array(new long[] { 0x0102030405060708L });
            // np: b.getfield(int32, 3) -> [33752069]
            b.getfield(NPTypeCode.Int32, 3).GetInt32().Should().Be(33752069);
        }

        [TestMethod]
        public void Float64_Float32_HighHalf()
        {
            var a = np.array(new double[] { 1.5, -2.25, 3.75 });
            // np: a.getfield(float32, 4) -> [1.9375, -2.03125, 2.21875]
            a.getfield(NPTypeCode.Single, 4).flatten().ToArray<float>().Should().Equal(1.9375f, -2.03125f, 2.21875f);
            // low half is all zeros (mantissa low bits)
            a.getfield(NPTypeCode.Single, 0).flatten().ToArray<float>().Should().Equal(0f, 0f, 0f);
        }

        [TestMethod]
        public void SameSize_ReinterpretsBits()
        {
            var d = np.array(new double[] { 1.5, 2.5, -3.5 });
            // np: d.getfield(int64) -> IEEE bit patterns
            d.getfield(NPTypeCode.Int64).flatten().ToArray<long>()
                .Should().Equal(4609434218613702656L, 4612811918334230528L, -4608308318706860032L);
        }

        // ---- getfield view semantics -------------------------------------------------------------

        [TestMethod]
        public void GetField_SharesMemory_WriteThrough()
        {
            var x = np.array(new Complex[] { new(1, 2), new(3, 4) });
            var imag = x.getfield(NPTypeCode.Double, 8);   // imaginary lane
            imag.SetAtIndex(99.0, 0);
            ((Complex)x.GetAtIndex(0)).Should().Be(new Complex(1, 99));  // wrote through
        }

        [TestMethod]
        public void GetField_PreservesByteStrides_NonContiguous()
        {
            var a = np.array(new int[] { 1, 2, 3, 4 });        // C-contig, stride 4 bytes
            var g = a.getfield(NPTypeCode.Int16, 0);
            g.strides.Should().Equal(4L);                       // byte-stride kept (NOT 2)
            g.Shape.IsContiguous.Should().BeFalse();            // stride 4 != itemsize 2
        }

        [TestMethod]
        public void GetField_SameSize_StaysContiguous()
        {
            var d = np.array(new double[] { 1.0, 2.0, 3.0 });
            var g = d.getfield(NPTypeCode.Int64);
            g.strides.Should().Equal(8L);
            g.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void GetField_StridedSource_KeepsStride()
        {
            var c = np.arange(10).astype(NPTypeCode.Int32);
            var g = c["::2"].getfield(NPTypeCode.Int16, 0);     // source stride 8 bytes
            g.strides.Should().Equal(8L);
            g.flatten().ToArray<short>().Should().Equal((short)0, (short)2, (short)4, (short)6, (short)8);
        }

        [TestMethod]
        public void GetField_NegativeStrideSource()
        {
            var n = np.arange(5).astype(NPTypeCode.Int32)["::-1"];
            var g = n.getfield(NPTypeCode.Int16, 0);
            g.strides.Should().Equal(-4L);
            g.flatten().ToArray<short>().Should().Equal((short)4, (short)3, (short)2, (short)1, (short)0);
        }

        [TestMethod]
        public void GetField_TransposedSource()
        {
            var t = np.arange(6).astype(NPTypeCode.Int64).reshape(2, 3).T;   // strides (8,24) bytes
            var g = t.getfield(NPTypeCode.Int32, 0);
            g.shape.Should().Equal(3, 2);
            g.strides.Should().Equal(8L, 24L);
            g.flatten().ToArray<int>().Should().Equal(0, 3, 1, 4, 2, 5);
        }

        [TestMethod]
        public void GetField_ZeroDimensional()
        {
            var s = NDArray.Scalar(0x01020304).astype(NPTypeCode.Int32);
            var g = s.getfield(NPTypeCode.Int16, 2);            // high half
            g.ndim.Should().Be(0);
            g.GetInt16().Should().Be((short)258);
        }

        [TestMethod]
        public void GetField_Empty_PreservesShape()
        {
            var e = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
            var g = e.getfield(NPTypeCode.Int16, 0);
            g.shape.Should().Equal(0, 3);
            g.dtype.Should().Be(typeof(short));
        }

        [TestMethod]
        public void GetField_OnReadOnlyBroadcast_IsAllowed_StaysReadOnly()
        {
            var src = np.array(new int[] { 1, 2, 3 }).reshape(1, 3);
            var bc = np.broadcast_to(src, new Shape(2, 3));     // read-only
            var g = bc.getfield(NPTypeCode.Int16, 0);
            g.Shape.IsWriteable.Should().BeFalse();             // field of a read-only array stays read-only
            g.shape.Should().Equal(2, 3);
        }

        // ---- getfield errors (verbatim NumPy ValueError texts) -----------------------------------

        [TestMethod]
        public void GetField_NewTypeLarger_Raises()
        {
            var d = np.array(new int[] { 1, 2, 3 });
            Action act = () => d.getfield(NPTypeCode.Double, 0);
            act.Should().Throw<ValueError>().WithMessage("new type is larger than original type");
        }

        [TestMethod]
        public void GetField_NegativeOffset_Raises()
        {
            var d = np.array(new int[] { 1, 2, 3 });
            Action act = () => d.getfield(NPTypeCode.Int16, -1);
            act.Should().Throw<ValueError>().WithMessage("offset is negative");
        }

        [TestMethod]
        public void GetField_OffsetPlusSizeTooLarge_Raises()
        {
            var d = np.array(new int[] { 1, 2, 3 });
            Action act = () => d.getfield(NPTypeCode.Int16, 3);
            act.Should().Throw<ValueError>().WithMessage("new type plus offset is larger than original type");
        }

        [TestMethod]
        public void GetField_NullType_Raises()
        {
            var d = np.array(new int[] { 1, 2, 3 });
            Action act = () => d.getfield((Type)null);
            act.Should().Throw<ArgumentNullException>();
        }

        // ---- setfield ----------------------------------------------------------------------------

        [TestMethod]
        public void SetField_ScalarBroadcast_ImaginaryPart()
        {
            var y = np.eye(2).astype(NPTypeCode.Complex);
            y.setfield(3, NPTypeCode.Double, 8);               // imag = 3 (broadcast scalar)
            // np: [(1+3j),3j,3j,(1+3j)]
            y.flatten().ToArray<Complex>().Should().Equal(new Complex(1, 3), new Complex(0, 3), new Complex(0, 3), new Complex(1, 3));
        }

        [TestMethod]
        public void SetField_Array_LowHalfPerElement()
        {
            var a = np.array(new int[] { 0x01020304, 0x01020304, 0x01020304 });
            a.setfield(np.array(new short[] { 10, 20, 30 }), NPTypeCode.Int16, 0);
            // np: 0x0102000a, 0x01020014, 0x0102001e (high half preserved, low replaced)
            a.GetInt32(0).Should().Be(0x0102000A);
            a.GetInt32(1).Should().Be(0x01020014);
            a.GetInt32(2).Should().Be(0x0102001E);
        }

        [TestMethod]
        public void SetField_FloatIntoIntField_TruncatesTowardZero()
        {
            var c = np.array(new int[] { 0, 0, 0 });
            c.setfield(np.array(new double[] { 2.9, 3.1, -1.5 }), NPTypeCode.Int32, 0);
            c.flatten().ToArray<int>().Should().Equal(2, 3, -1);
        }

        [TestMethod]
        public void SetField_ZeroDimensional_HighHalf()
        {
            var s = NDArray.Scalar(0x01020304).astype(NPTypeCode.Int32);
            s.setfield(7, NPTypeCode.Int16, 2);
            s.GetInt32().Should().Be(0x00070304);
        }

        [TestMethod]
        public void SetField_LeavesUntouchedBytesIntact()
        {
            // set the HIGH half only; low half of each int32 must survive.
            var a = np.array(new int[] { unchecked((int)0xAABBCCDD), unchecked((int)0x11223344) });
            a.setfield(np.array(new short[] { 0x7EEE, 0x0102 }), NPTypeCode.Int16, 2);
            a.GetInt32(0).Should().Be(unchecked((int)0x7EEECCDD));
            a.GetInt32(1).Should().Be(0x01023344);
        }

        [TestMethod]
        public void SetField_ReturnsVoid_MutatesInPlace()
        {
            var m = np.arange(6).astype(NPTypeCode.Int32);
            m.setfield(0, NPTypeCode.Int16, 2);                // zero the (already-zero) high halves
            m.flatten().ToArray<int>().Should().Equal(0, 1, 2, 3, 4, 5);
        }

        // ---- setfield errors ---------------------------------------------------------------------

        [TestMethod]
        public void SetField_OnReadOnly_RaisesValueError()
        {
            var src = np.array(new int[] { 1, 2, 3 }).reshape(1, 3);
            var bc = np.broadcast_to(src, new Shape(2, 3));
            Action act = () => bc.setfield(9, NPTypeCode.Int16, 0);
            act.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
        }

        [TestMethod]
        public void SetField_ReadOnlyCheck_PrecedesDtypeValidation()
        {
            // read-only + oversized dtype: NumPy reports read-only first (PyArray_FailUnlessWriteable).
            var src = np.array(new int[] { 1, 2, 3 }).reshape(1, 3);
            var bc = np.broadcast_to(src, new Shape(2, 3));
            Action act = () => bc.setfield(9, NPTypeCode.Double, 0);   // float64 too large AND read-only
            act.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
        }

        [TestMethod]
        public void SetField_OversizedDtype_OnWriteable_RaisesTypeError()
        {
            var w = np.array(new int[] { 1, 2, 3 });
            Action act = () => w.setfield(9, NPTypeCode.Double, 0);
            act.Should().Throw<ValueError>().WithMessage("new type is larger than original type");
        }

        [TestMethod]
        public void SetField_NullValue_Raises()
        {
            var w = np.array(new int[] { 1, 2, 3 });
            Action act = () => w.setfield(null, NPTypeCode.Int16, 0);
            act.Should().Throw<ArgumentNullException>();
        }

        // ---- dtype overloads ---------------------------------------------------------------------

        [TestMethod]
        public void GetField_TypeOverload_MatchesTypeCode()
        {
            var a = np.array(new int[] { 0x01020304 });
            a.getfield(typeof(short), 2).GetInt16().Should().Be((short)258);
        }

        [TestMethod]
        public void GetField_GenericOverload_ReturnsTypedArray()
        {
            var a = np.array(new int[] { 0x01020304, 0x05060708 });
            NumSharp.Generic.NDArray<short> g = a.getfield<short>(0);
            g[0].Should().Be((short)772);
            g[1].Should().Be((short)1800);
        }

        // ---- all-dtype dispatch smoke ------------------------------------------------------------

        [TestMethod]
        public void GetField_DispatchesAll15FieldDtypes_NoThrow()
        {
            // 16-byte source so every field dtype (incl. Decimal/Complex, itemsize 16) fits at offset 0.
            var src = np.array(new Complex[] { new(1, 2), new(3, 4) });
            foreach (NPTypeCode tc in Enum.GetValues(typeof(NPTypeCode)))
            {
                if (tc == NPTypeCode.Empty || tc == NPTypeCode.String) continue;
                var g = src.getfield(tc, 0);
                g.size.Should().Be(2);
                g.typecode.Should().Be(tc);
            }
        }
    }
}
