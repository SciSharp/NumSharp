using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    /// Tests for np.packbits / np.unpackbits, verified against NumPy 2.4.2 output.
    ///
    /// NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.packbits.html
    ///                  https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html
    /// NumPy source: numpy/_core/src/multiarray/compiled_base.c (pack_bits / unpack_bits / pack_inner).
    ///
    /// Both are dtype-agnostic byte transforms over a purpose-built kernel (PackBits); the output is
    /// always uint8, keeps the input rank (except axis=None ⇒ 1-D), and is F-contiguous iff the input is.
    /// </summary>
    [TestClass]
    public class np_packbits_Test
    {
        // ==================================================================
        // np.packbits — basics & docstring example
        // ==================================================================

        [TestMethod]
        public void Packbits_Docstring_Axis_Last()
        {
            // np.packbits(a, axis=-1) where a = [[[1,0,1],[0,1,0]],[[1,1,0],[0,0,1]]]
            //   => [[[160],[64]],[[192],[32]]]  (160=1010_0000, 64=0100_0000, 192=1100_0000, 32=0010_0000)
            var a = np.array(new int[] { 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1 }).reshape(2, 2, 3);
            var r = np.packbits(a, axis: -1);
            r.Should().BeShaped(2, 2, 1).And.BeOfValues(160, 64, 192, 32);
            r.dtype.Should().Be(typeof(byte));
        }

        [TestMethod]
        public void Packbits_1D_BigOrder()
        {
            // np.packbits([1,0,1,1,0,0,0,1,1]) => [177, 128]  (10110001, 1_0000000 padded)
            var r = np.packbits(np.array(new int[] { 1, 0, 1, 1, 0, 0, 0, 1, 1 }));
            r.Should().BeShaped(2).And.BeOfValues(177, 128);
        }

        [TestMethod]
        public void Packbits_1D_LittleOrder()
        {
            // bitorder='little' reverses within each byte => [141, 1]
            var r = np.packbits(np.array(new int[] { 1, 0, 1, 1, 0, 0, 0, 1, 1 }), bitorder: "little");
            r.Should().BeShaped(2).And.BeOfValues(141, 1);
        }

        [TestMethod]
        public void Packbits_AxisNone_FlattensInCOrder()
        {
            // axis=None packs the C-order flattened array to 1-D: [[1,0,1,0,1,0,1,0],[1,1,1,1,0,0,0,0]] => [170,240]
            var a = np.array(new int[] { 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0 }).reshape(2, 8);
            var r = np.packbits(a);
            r.Should().BeShaped(2).And.BeOfValues(170, 240);
        }

        [TestMethod]
        public void Packbits_Axis0()
        {
            // np.packbits([[1,0,1],[0,1,0],[1,1,0],[0,0,1]], axis=0) => shape (1,3) [160,96,144]
            var a = np.array(new int[] { 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1 }).reshape(4, 3);
            var r = np.packbits(a, axis: 0);
            r.Should().BeShaped(1, 3).And.BeOfValues(160, 96, 144);
        }

        [TestMethod]
        public void Packbits_NonzeroSemantics_AnyNonzeroSetsBit()
        {
            // Any nonzero value (incl. high-byte-only) sets the bit: [5,0,255,0,0,0,0,0] => [160] (1010_0000)
            np.packbits(np.array(new int[] { 5, 0, 255, 0, 0, 0, 0, 0 })).Should().BeOfValues(160);
            // int16 256 = 0x0100 (low byte 0, high byte 1) is nonzero: [256,0,256,0,0,0,256,256] => [163]
            np.packbits(np.array(new short[] { 256, 0, 256, 0, 0, 0, 256, 256 })).Should().BeOfValues(163);
        }

        [TestMethod]
        public void Packbits_AllIntegerDtypes_SameResult()
        {
            // Every bool/integer dtype packs identically (nonzero-ness is dtype-agnostic).
            var bits = new int[] { 1, 0, 1, 1, 0, 0, 0, 1 };  // => 10110001 = 177
            np.packbits(np.array(bits).astype(NPTypeCode.Boolean)).Should().BeOfValues(177);
            np.packbits(np.array(bits).astype(NPTypeCode.Byte)).Should().BeOfValues(177);
            np.packbits(np.array(bits).astype(NPTypeCode.SByte)).Should().BeOfValues(177);
            np.packbits(np.array(bits).astype(NPTypeCode.Int16)).Should().BeOfValues(177);
            np.packbits(np.array(bits).astype(NPTypeCode.UInt32)).Should().BeOfValues(177);
            np.packbits(np.array(bits).astype(NPTypeCode.Int64)).Should().BeOfValues(177);
            np.packbits(np.array(bits).astype(NPTypeCode.UInt64)).Should().BeOfValues(177);
        }

        [TestMethod]
        public void Packbits_0d()
        {
            // A 0-d input is promoted to (1,): nonzero => [128] (MSB), zero => [0].
            np.packbits(np.array(7).reshape(new int[0])).Should().BeShaped(1).And.BeOfValues(128);
            np.packbits(np.array(0).reshape(new int[0])).Should().BeShaped(1).And.BeOfValues(0);
        }

        [TestMethod]
        public void Packbits_Empty()
        {
            np.packbits(np.array(new byte[0])).Should().BeShaped(0);
        }

        // ==================================================================
        // np.packbits — memory layouts (values must match logical order)
        // ==================================================================

        [TestMethod]
        public void Packbits_FContiguous_Axis1()
        {
            // F-contiguous input packs by logical order; result is F-contiguous like the input.
            var a = np.asfortranarray(np.array(new int[] { 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0 }).reshape(2, 8));
            var r = np.packbits(a, axis: 1);
            r.Should().BeShaped(2, 1).And.BeOfValues(170, 240);
        }

        [TestMethod]
        public void Packbits_Transposed_ProducesFortranOutput()
        {
            // A transposed (F-contiguous) input yields an F-contiguous output, matching NumPy's ISFORTRAN rule.
            var m = np.transpose(np.arange(24).reshape(2, 3, 4) % 2, new int[] { 2, 1, 0 }); // (4,3,2), F-contig
            var r = np.packbits(m, axis: 0);
            r.Should().BeShaped(1, 3, 2).And.BeOfValues(80, 80, 80, 80, 80, 80);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Packbits_NegativeStride_ReadsReversed()
        {
            // np.packbits(base[::-1]) reads the reversed logical order.
            var baseArr = np.array(new byte[] { 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1 });
            np.packbits(baseArr["::-1"]).Should().BeShaped(2).And.BeOfValues(245, 205);
        }

        // ==================================================================
        // np.unpackbits — basics & docstring example
        // ==================================================================

        [TestMethod]
        public void Unpackbits_Docstring_Axis1()
        {
            // np.unpackbits([[2],[7],[23]], axis=1)
            var a = np.array(new byte[] { 2, 7, 23 }).reshape(3, 1);
            var r = np.unpackbits(a, axis: 1);
            r.Should().BeShaped(3, 8).And.BeOfValues(
                0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 1, 1, 1,
                0, 0, 0, 1, 0, 1, 1, 1);
            r.dtype.Should().Be(typeof(byte));
        }

        [TestMethod]
        public void Unpackbits_1D_BigAndLittle()
        {
            // 179 = 0b10110011
            np.unpackbits(np.array(new byte[] { 179 })).Should().BeShaped(8).And.BeOfValues(1, 0, 1, 1, 0, 0, 1, 1);
            np.unpackbits(np.array(new byte[] { 179 }), bitorder: "little").Should().BeShaped(8).And.BeOfValues(1, 1, 0, 0, 1, 1, 0, 1);
        }

        // ==================================================================
        // np.unpackbits — count semantics
        // ==================================================================

        [TestMethod]
        public void Unpackbits_Count_Truncate_And_Pad()
        {
            var a = np.array(new byte[] { 179 });
            np.unpackbits(a, count: 3).Should().BeShaped(3).And.BeOfValues(1, 0, 1);              // truncate
            np.unpackbits(a, count: 0).Should().BeShaped(0);                                        // empty
            np.unpackbits(a, count: 8).Should().BeShaped(8).And.BeOfValues(1, 0, 1, 1, 0, 0, 1, 1); // exact
            np.unpackbits(a, count: 12).Should().BeShaped(12).And.BeOfValues(1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0); // pad zeros
        }

        [TestMethod]
        public void Unpackbits_Count_Negative_TrimsFromEnd()
        {
            var a = np.array(new byte[] { 179 });
            np.unpackbits(a, count: -2).Should().BeShaped(6).And.BeOfValues(1, 0, 1, 1, 0, 0);          // trim 2
            np.unpackbits(a, count: -8).Should().BeShaped(0);                                            // trim all
            np.unpackbits(a, count: -1, bitorder: "little").Should().BeShaped(7).And.BeOfValues(1, 1, 0, 0, 1, 1, 0);
        }

        [TestMethod]
        public void Unpackbits_Count_TooNegative_Throws()
        {
            new Action(() => np.unpackbits(np.array(new byte[] { 179 }), count: -9))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("-count larger than number of elements");
        }

        [TestMethod]
        public void Unpackbits_Roundtrip_WithCount()
        {
            // unpackbits(packbits(x), count=len(x)) == x
            var x = np.array(new int[] { 1, 0, 1, 1, 0, 0, 0, 1, 1 }).astype(NPTypeCode.Byte);
            var packed = np.packbits(x);
            np.unpackbits(packed, count: 9).Should().BeShaped(9).And.BeOfValues(1, 0, 1, 1, 0, 0, 0, 1, 1);
        }

        [TestMethod]
        public void Unpackbits_0d_And_Empty()
        {
            // 0-d uint8 => shape (8,)
            np.unpackbits(np.array((byte)5).reshape(new int[0])).Should().BeShaped(8).And.BeOfValues(0, 0, 0, 0, 0, 1, 0, 1);
            // empty => empty
            np.unpackbits(np.array(new byte[0])).Should().BeShaped(0);
        }

        // ==================================================================
        // Error taxonomy (NumPy-verbatim), and the validation ORDER
        // ==================================================================

        [TestMethod]
        public void Packbits_FloatInput_Throws_TypeError()
        {
            new Action(() => np.packbits(np.array(new double[] { 1.0, 0.0 })))
                .Should().ThrowExactly<TypeError>()
                .WithMessage("Expected an input array of integer or boolean data type");
        }

        [TestMethod]
        public void Unpackbits_NonUint8Input_Throws_TypeError()
        {
            new Action(() => np.unpackbits(np.array(new int[] { 1, 2 })))
                .Should().ThrowExactly<TypeError>()
                .WithMessage("Expected an input array of unsigned byte data type");
            // sbyte/int8 is also rejected (must be UNSIGNED byte).
            new Action(() => np.unpackbits(np.array(new sbyte[] { 1, 2 })))
                .Should().ThrowExactly<TypeError>();
        }

        [TestMethod]
        public void Packbits_BadBitorder_Throws_ValueError_PrefixRule()
        {
            // packbits: value must begin with "big" (3 chars) or "little" (6 chars).
            new Action(() => np.packbits(np.array(new int[] { 1, 0 }), bitorder: "bad"))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("'order' must be either 'little' or 'big'");
            // "b" alone is NOT a "big" prefix (needs the full 3 chars).
            new Action(() => np.packbits(np.array(new int[] { 1, 0 }), bitorder: "b"))
                .Should().ThrowExactly<ValueError>();
            // A "little" prefix is accepted even with a trailing suffix.
            np.packbits(np.array(new int[] { 1, 0 }), bitorder: "littleXXX").Should().BeOfValues(1);
        }

        [TestMethod]
        public void Unpackbits_BadBitorder_Throws_ValueError_FirstCharRule()
        {
            // unpackbits inspects only the FIRST character: 'l'/'b' ok, else error (different message than packbits).
            new Action(() => np.unpackbits(np.array(new byte[] { 179 }), bitorder: "x"))
                .Should().ThrowExactly<ValueError>()
                .WithMessage("'order' must begin with 'l' or 'b'");
            np.unpackbits(np.array(new byte[] { 179 }), bitorder: "laaa").Should().BeOfValues(1, 1, 0, 0, 1, 1, 0, 1); // little
            np.unpackbits(np.array(new byte[] { 179 }), bitorder: "baaa").Should().BeOfValues(1, 0, 1, 1, 0, 0, 1, 1); // big
        }

        [TestMethod]
        public void Packbits_AxisOutOfRange_Throws_AxisError()
        {
            new Action(() => np.packbits(np.zeros(new Shape(2, 2, 3), NPTypeCode.Byte), axis: 5))
                .Should().ThrowExactly<AxisError>()
                .WithMessage("axis 5 is out of bounds for array of dimension 3*");
        }

        [TestMethod]
        public void Packbits_ValidationOrder_BitorderBeforeDtype()
        {
            // NumPy validates bitorder (io_pack) BEFORE the dtype check: a float input with a bad bitorder
            // reports the bitorder ValueError, not the dtype TypeError.
            new Action(() => np.packbits(np.array(new double[] { 1.0 }), bitorder: "x"))
                .Should().ThrowExactly<ValueError>();
        }

        // ==================================================================
        // Deliberate divergence ([Misaligned]) — NumSharp is MORE correct
        // ==================================================================

        [TestMethod]
        [Misaligned]
        public void Unpackbits_EmptyInput_CountForcesOutput_ReturnsDeterministicZeros()
        {
            // NumPy's IterAllButAxis never runs over an empty input, so its count-forced output is
            // UNINITIALISED memory (garbage) — contradicting its own "add zero padding" docstring.
            // NumSharp returns the documented deterministic zeros.
            np.unpackbits(np.array(new byte[0]), count: 3).Should().BeShaped(3).And.BeOfValues(0, 0, 0);
            np.unpackbits(np.zeros(new Shape(0, 3), NPTypeCode.Byte), axis: 0, count: 3)
                .Should().BeShaped(3, 3).And.BeOfValues(0, 0, 0, 0, 0, 0, 0, 0, 0);
        }
    }
}
