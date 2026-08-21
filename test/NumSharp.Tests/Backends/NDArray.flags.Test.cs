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
