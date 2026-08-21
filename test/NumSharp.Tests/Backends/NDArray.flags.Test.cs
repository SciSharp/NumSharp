using System;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     <see cref="NDArray.flags"/> (the <see cref="NDArrayFlags"/> object) and <see cref="NDArray.nbytes"/>.
    ///     Every value probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class NDArray_flags_nbytes_Test
    {
        // ---- nbytes ----------------------------------------------------------------------

        [TestMethod]
        public void Nbytes_Complex_3x5x2()
            => np.zeros(new Shape(3, 5, 2), NPTypeCode.Complex).nbytes.Should().Be(480);

        [TestMethod]
        public void Nbytes_ScalarAndEmpty()
        {
            np.array(5).nbytes.Should().Be(np.array(5).dtypesize);   // 0-d = one itemsize
            np.zeros(new Shape(0, 3), NPTypeCode.Double).nbytes.Should().Be(0);
        }

        [TestMethod]
        public void Nbytes_BroadcastUsesLogicalSize()
        {
            // A broadcast view reports its LOGICAL byte size, not its one-element backing buffer.
            var b = np.broadcast_to(np.array(new[] { 7 }).reshape(1, 1), new Shape(1000, 1000));
            b.nbytes.Should().Be(1000L * 1000L * 4);
        }

        [TestMethod]
        public void Nbytes_StridedView()
            => np.arange(24).astype(np.int32).reshape(4, 6)["...,::2"].nbytes.Should().Be(4 * 3 * 4);

        [TestMethod]
        public void Nbytes_AllDtypes_MatchNumpy()
        {
            // nbytes = size * itemsize (NumPy PyArray_NBYTES = ITEMSIZE * SIZE) on 3*5*2 = 30 elements.
            // 13 dtypes with a NumPy analog — byte-identical to NumPy 2.4.2 ndarray.nbytes.
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Boolean).nbytes.Should().Be(30);   // itemsize 1
            np.zeros(new Shape(3, 5, 2), NPTypeCode.SByte).nbytes.Should().Be(30);
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Byte).nbytes.Should().Be(30);
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Int16).nbytes.Should().Be(60);     // itemsize 2
            np.zeros(new Shape(3, 5, 2), NPTypeCode.UInt16).nbytes.Should().Be(60);
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Half).nbytes.Should().Be(60);
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Int32).nbytes.Should().Be(120);    // itemsize 4
            np.zeros(new Shape(3, 5, 2), NPTypeCode.UInt32).nbytes.Should().Be(120);
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Single).nbytes.Should().Be(120);
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Int64).nbytes.Should().Be(240);    // itemsize 8
            np.zeros(new Shape(3, 5, 2), NPTypeCode.UInt64).nbytes.Should().Be(240);
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Double).nbytes.Should().Be(240);
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Complex).nbytes.Should().Be(480);  // itemsize 16
            // 2 NumSharp-only dtypes (no NumPy analog) — 30 * their in-memory itemsize.
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Char).nbytes.Should().Be(60);      // itemsize 2
            np.zeros(new Shape(3, 5, 2), NPTypeCode.Decimal).nbytes.Should().Be(480);  // itemsize 16
        }

        [TestMethod]
        public void Nbytes_LayoutMatrix_MatchNumpy()
        {
            // nbytes tracks the LOGICAL size (independent of strides/offset/layout), so every non-copying
            // view of the 24-element int32 base still reports 24 * 4 = 96 bytes. Probed against NumPy 2.4.2.
            var b = np.arange(24).astype(np.int32).reshape(4, 6);                 // C-contiguous, 96
            np.asfortranarray(b).nbytes.Should().Be(96);                          // F-contiguous
            b.T.nbytes.Should().Be(96);                                           // transposed
            b["::-1"].nbytes.Should().Be(96);                                     // negative-stride
            np.broadcast_to(np.arange(6).astype(np.int32).reshape(1, 6), new Shape(4, 6)).nbytes.Should().Be(96); // partial broadcast
            np.zeros(new Shape(2, 1, 3, 1, 4), NPTypeCode.Int32).nbytes.Should().Be(96); // rank-5
            // Copying selections: a fresh owning array whose byte count follows its own size.
            b[new int[] { 0, 2 }].nbytes.Should().Be(48);                         // fancy-indexed (2,6)
            b[b % 2 == 0].nbytes.Should().Be(48);                                 // boolean-mask (12,)
            np.array(new[] { 9 }).nbytes.Should().Be(4);                          // one-element 1-D int32
            b[0, 0].nbytes.Should().Be(4);                                        // 0-d view
            b.astype(np.int16).nbytes.Should().Be(48);                            // astype -> itemsize 2
            b.astype(np.complex128).nbytes.Should().Be(384);                      // astype -> itemsize 16
        }

        // ---- itemsize --------------------------------------------------------------------

        [TestMethod]
        public void Itemsize_AllDtypes_MatchNumpy()
        {
            // 13 dtypes with a NumPy analog — byte-identical to NumPy 2.4.2 ndarray.itemsize.
            np.zeros(new Shape(3), NPTypeCode.Boolean).itemsize.Should().Be(1);
            np.zeros(new Shape(3), NPTypeCode.SByte).itemsize.Should().Be(1);
            np.zeros(new Shape(3), NPTypeCode.Byte).itemsize.Should().Be(1);
            np.zeros(new Shape(3), NPTypeCode.Int16).itemsize.Should().Be(2);
            np.zeros(new Shape(3), NPTypeCode.UInt16).itemsize.Should().Be(2);
            np.zeros(new Shape(3), NPTypeCode.Int32).itemsize.Should().Be(4);
            np.zeros(new Shape(3), NPTypeCode.UInt32).itemsize.Should().Be(4);
            np.zeros(new Shape(3), NPTypeCode.Int64).itemsize.Should().Be(8);
            np.zeros(new Shape(3), NPTypeCode.UInt64).itemsize.Should().Be(8);
            np.zeros(new Shape(3), NPTypeCode.Half).itemsize.Should().Be(2);
            np.zeros(new Shape(3), NPTypeCode.Single).itemsize.Should().Be(4);
            np.zeros(new Shape(3), NPTypeCode.Double).itemsize.Should().Be(8);
            np.zeros(new Shape(3), NPTypeCode.Complex).itemsize.Should().Be(16);
            // 2 NumSharp-only dtypes (no NumPy analog) — report their in-memory element size.
            np.zeros(new Shape(3), NPTypeCode.Char).itemsize.Should().Be(2);
            np.zeros(new Shape(3), NPTypeCode.Decimal).itemsize.Should().Be(16);
        }

        [TestMethod]
        public void Itemsize_IsLayoutInvariant()
        {
            // itemsize is a property of the dtype ALONE — identical across every layout (NumPy parity).
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            a.itemsize.Should().Be(8);                                             // C-contiguous
            np.array(5.0).itemsize.Should().Be(8);                                 // 0-d scalar
            np.zeros(new Shape(0, 3), NPTypeCode.Double).itemsize.Should().Be(8);  // empty
            a[":, ::2"].itemsize.Should().Be(8);                                   // strided
            a.T.itemsize.Should().Be(8);                                           // transposed
            a["::-1"].itemsize.Should().Be(8);                                     // negative-stride
            np.broadcast_to(np.array(new[] { 7 }).reshape(1, 1), new Shape(1000, 1000)).itemsize.Should().Be(4); // broadcast int32
            a.astype(np.int16).itemsize.Should().Be(2);                            // after astype
        }

        [TestMethod]
        public void Itemsize_AliasesDtypesize_AndComposesNbytes()
        {
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            a.itemsize.Should().Be(a.dtypesize);                 // alias of the legacy name
            ((long)a.size * a.itemsize).Should().Be(a.nbytes);   // size * itemsize == nbytes
        }

        // ---- flags: matrix (byte-for-byte NumPy across layouts) --------------------------

        private static (int C, int F, int own, int wr, int beh, int carr, int farr, int fnc, int forc, int num) Row(NDArray a)
        {
            var f = a.flags;
            int b(bool x) => x ? 1 : 0;
            return (b(f.c_contiguous), b(f.f_contiguous), b(f.owndata), b(f.writeable), b(f.behaved),
                    b(f.carray), b(f.farray), b(f.fnc), b(f.forc), f.num);
        }

        [TestMethod]
        public void Flags_CContiguous()
            => Row(np.arange(12).reshape(3, 4)).Should().Be((1, 0, 0, 1, 1, 1, 0, 0, 1, 1281));

        [TestMethod]
        public void Flags_FContiguous()
            => Row(np.asfortranarray(np.arange(12).reshape(3, 4))).Should().Be((0, 1, 1, 1, 1, 0, 1, 1, 1, 1286));

        [TestMethod]
        public void Flags_Strided()
            => Row(np.arange(12).reshape(3, 4)["...,::2"]).Should().Be((0, 0, 0, 1, 1, 0, 1, 0, 0, 1280));

        [TestMethod]
        public void Flags_Transposed()
            => Row(np.arange(12).reshape(3, 4).T).Should().Be((0, 1, 0, 1, 1, 0, 1, 1, 1, 1282));

        [TestMethod]
        public void Flags_Broadcast()
            => Row(np.broadcast_to(np.arange(4).reshape(1, 4), new Shape(3, 4))).Should().Be((0, 0, 0, 0, 0, 0, 1, 0, 0, 256));

        [TestMethod]
        public void Flags_ZeroD()
            => Row(np.array(5)).Should().Be((1, 1, 1, 1, 1, 1, 0, 0, 1, 1287));

        [TestMethod]
        public void Flags_Empty()
            => Row(np.zeros(new Shape(0, 3))).Should().Be((1, 1, 1, 1, 1, 1, 0, 0, 1, 1287));

        [TestMethod]
        public void Flags_NegativeStride()
            => Row(np.arange(6)["::-1"]).Should().Be((0, 0, 0, 1, 1, 0, 1, 0, 0, 1280));

        [TestMethod]
        public void Flags_FreshOwnedZeros()
            => Row(np.zeros(new Shape(3, 4))).Should().Be((1, 0, 1, 1, 1, 1, 0, 0, 1, 1285));

        // ---- flags: subscript keys -------------------------------------------------------

        [TestMethod]
        public void Flags_SubscriptKeys_ShortAndLong()
        {
            var f = np.arange(12).reshape(3, 4).flags;   // C-contiguous view
            f["C"].Should().BeTrue();
            f["C_CONTIGUOUS"].Should().BeTrue();
            f["CONTIGUOUS"].Should().BeTrue();
            f["F"].Should().BeFalse();
            f["F_CONTIGUOUS"].Should().BeFalse();
            f["FORTRAN"].Should().BeFalse();
            f["W"].Should().BeTrue();
            f["WRITEABLE"].Should().BeTrue();
            f["O"].Should().BeFalse();          // reshape view does not own
            f["OWNDATA"].Should().BeFalse();
            f["A"].Should().BeTrue();
            f["ALIGNED"].Should().BeTrue();
            f["B"].Should().BeTrue();
            f["BEHAVED"].Should().BeTrue();
            f["CA"].Should().BeTrue();
            f["CARRAY"].Should().BeTrue();
            f["FA"].Should().BeFalse();
            f["FARRAY"].Should().BeFalse();
            f["FNC"].Should().BeFalse();
            f["FORC"].Should().BeTrue();
            f["X"].Should().BeFalse();
            f["WRITEBACKIFCOPY"].Should().BeFalse();
        }

        [TestMethod]
        public void Flags_UnknownSubscript_RaisesKeyError()
        {
            var f = np.arange(3).flags;
            ((Action)(() => { var _ = f["ZZZ"]; })).Should().Throw<KeyError>().WithMessage("*Unknown flag*");
        }

        // ---- flags: repr, equality --------------------------------------------------------

        [TestMethod]
        public void Flags_ToString_MatchesNumpy()
        {
            var s = np.arange(12).reshape(3, 4).flags.ToString();
            s.Should().Be(
                "  C_CONTIGUOUS : True\n" +
                "  F_CONTIGUOUS : False\n" +
                "  OWNDATA : False\n" +
                "  WRITEABLE : True\n" +
                "  ALIGNED : True\n" +
                "  WRITEBACKIFCOPY : False\n");
        }

        [TestMethod]
        public void Flags_Equality_ByNum()
        {
            var a = np.zeros(new Shape(3, 4));       // owned, C-contig
            var b = np.zeros(new Shape(3, 4));       // owned, C-contig -> same flags
            a.flags.Should().Be(b.flags);
            (a.flags == b.flags).Should().BeTrue();

            var t = np.arange(12).reshape(3, 4).T;   // transposed -> different num
            (a.flags == t.flags).Should().BeFalse();
            (a.flags != t.flags).Should().BeTrue();
        }

        // ---- flags: writeable setter ------------------------------------------------------

        [TestMethod]
        public void Flags_Writeable_SetFalse_MakesReadOnly()
        {
            var a = np.arange(6).astype(np.int32);
            a.flags.writeable = false;
            a.flags.writeable.Should().BeFalse();
            ((Action)(() => a[0] = 9)).Should().Throw<NumSharpException>().WithMessage("*read-only*");
        }

        [TestMethod]
        public void Flags_Writeable_RoundTripOnOwnedArray()
        {
            var a = np.arange(6).astype(np.int32);
            a.flags.writeable = false;
            a.flags.writeable = true;                 // owned array can be re-enabled
            a.flags.writeable.Should().BeTrue();
            a[0] = 9;                                 // writes again
            a.GetInt32(0).Should().Be(9);
        }

        [TestMethod]
        public void Flags_Writeable_SubscriptSet()
        {
            var a = np.arange(6).astype(np.int32);
            a.flags["W"] = false;
            a.flags["WRITEABLE"].Should().BeFalse();
        }

        [TestMethod]
        public void Flags_Writebackifcopy_SetTrue_Raises()
        {
            var a = np.arange(6);
            ((Action)(() => a.flags["X"] = true)).Should().Throw<ValueError>().WithMessage("*WRITEBACKIFCOPY*");
        }
    }
}
