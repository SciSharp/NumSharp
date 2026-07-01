using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    ///     NumPy-parity tests for the <c>order</c> parameter of <see cref="NDArray.tobytes(char)"/>.
    ///
    ///     Oracle: NumPy 2.4.2 <c>ndarray.tobytes(order)</c>. Exact byte vectors below were captured from
    ///     <c>np.arange(...).astype(...).tobytes(order).hex()</c>. The whole (15 dtype x 18 layout x 4 order)
    ///     matrix is replayed bit-exact by the fuzz/oracle harness; these are the curated, human-readable pins.
    ///
    ///     Order semantics (probed):
    ///       'C' -> row-major;  'F' -> column-major;
    ///       'A' -> 'F' iff the array is F-contiguous AND not C-contiguous, else 'C' (NumPy PyArray_ISFORTRAN);
    ///       'K' -> 'C' for every NumSharp dtype (NumPy routes numeric tobytes through CopyInto into a
    ///              C-contiguous destination, so tobytes('K') never preserves an F-contiguous source).
    /// </summary>
    [TestClass]
    public class TobytesOrderTests
    {
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
        };

        private static byte[] Hex(string h) => Convert.FromHexString(h);

        // ---- Exact, NumPy-verified bytes -------------------------------------------------

        [TestMethod]
        public void Contig2x3_Byte_C_vs_F_ExactBytes()
        {
            // [[0,1,2],[3,4,5]] uint8.  C = 000102030405 ; F (column-major) = 000301040205
            var a = np.arange(6).astype(NPTypeCode.Byte).reshape(2, 3);
            CollectionAssert.AreEqual(Hex("000102030405"), a.tobytes('C'), "C-order");
            CollectionAssert.AreEqual(Hex("000301040205"), a.tobytes('F'), "F-order");
            // 'A' on a C-contiguous array == C ; 'K' == C
            CollectionAssert.AreEqual(Hex("000102030405"), a.tobytes('A'), "A on C-contig == C");
            CollectionAssert.AreEqual(Hex("000102030405"), a.tobytes('K'), "K == C");
        }

        [TestMethod]
        public void Contig2x3_Int32_F_ExactBytes()
        {
            // [[0,1,2],[3,4,5]] int32 F-order -> elements 0,3,1,4,2,5
            var a = np.arange(6).astype(NPTypeCode.Int32).reshape(2, 3);
            CollectionAssert.AreEqual(
                Hex("000000000300000001000000040000000200000005000000"),
                a.tobytes('F'));
        }

        [TestMethod]
        public void FContiguousView_AllOrders_Int32_ExactBytes()
        {
            // arange(6).reshape(3,2).T  ==  [[0,2,4],[1,3,5]] , F-contiguous (raw buffer = 0..5).
            var t = np.arange(6).astype(NPTypeCode.Int32).reshape(3, 2).T;
            // C-order (row-major of the transposed view): 0,2,4,1,3,5
            CollectionAssert.AreEqual(Hex("000000000200000004000000010000000300000005000000"), t.tobytes('C'), "C");
            // F-order == the raw column-major buffer: 0,1,2,3,4,5
            CollectionAssert.AreEqual(Hex("000000000100000002000000030000000400000005000000"), t.tobytes('F'), "F");
            // 'A' on an F-contiguous-and-not-C view resolves to F
            CollectionAssert.AreEqual(t.tobytes('F'), t.tobytes('A'), "A == F for F-contig view");
            // 'K' must NOT preserve F — it is always C (the NumPy numeric-tobytes quirk)
            CollectionAssert.AreEqual(t.tobytes('C'), t.tobytes('K'), "K == C even for F-contig view");
            Assert.IsFalse(t.tobytes('K').AsSpan().SequenceEqual(t.tobytes('F')), "K must differ from F here");
        }

        [TestMethod]
        public void AsFortranArray_A_resolves_to_F()
        {
            // np.asfortranarray([[0,1,2],[3,4,5]]) uint8 -> F-contiguous; A == F == 000301040205 ; K == C
            var af = np.asfortranarray(np.arange(6).astype(NPTypeCode.Byte).reshape(2, 3));
            CollectionAssert.AreEqual(Hex("000301040205"), af.tobytes('A'), "A == F");
            CollectionAssert.AreEqual(Hex("000102030405"), af.tobytes('K'), "K == C");
        }

        [TestMethod]
        public void NegativeColumnStride_Byte_C_and_F_ExactBytes()
        {
            // [[0,1,2],[3,4,5]][:, ::-1] == [[2,1,0],[5,4,3]] (neither C nor F contiguous)
            var v = np.arange(6).astype(NPTypeCode.Byte).reshape(2, 3)[":, ::-1"];
            CollectionAssert.AreEqual(Hex("020100050403"), v.tobytes('C'), "C");
            CollectionAssert.AreEqual(Hex("020501040003"), v.tobytes('F'), "F");
        }

        // ---- Order resolution rules across ALL 15 dtypes (oracle-free metamorphic) --------

        [TestMethod]
        public void FOrder_Equals_ReverseTranspose_COrder_AllDtypes_2D_3D()
        {
            // Identity: a.tobytes('F') == transpose(a, reversed-axes).tobytes('C').
            // Holds for every dtype/shape and needs no external oracle.
            foreach (var tc in AllDtypes)
            {
                var a2 = np.arange(6).astype(tc).reshape(2, 3);
                CollectionAssert.AreEqual(a2.T.tobytes('C'), a2.tobytes('F'), $"2D F==T.C {tc}");

                var a3 = np.arange(24).astype(tc).reshape(2, 3, 4);
                var rev = np.transpose(a3, new int[] { 2, 1, 0 });
                CollectionAssert.AreEqual(rev.tobytes('C'), a3.tobytes('F'), $"3D F==revT.C {tc}");
            }
        }

        [TestMethod]
        public void A_and_K_Resolution_AllDtypes()
        {
            foreach (var tc in AllDtypes)
            {
                var c = np.arange(6).astype(tc).reshape(2, 3);          // C-contiguous
                var f = np.arange(6).astype(tc).reshape(3, 2).T;        // F-contiguous (not C)

                CollectionAssert.AreEqual(c.tobytes('C'), c.tobytes('A'), $"A==C on C-contig {tc}");
                CollectionAssert.AreEqual(f.tobytes('F'), f.tobytes('A'), $"A==F on F-contig {tc}");
                CollectionAssert.AreEqual(c.tobytes('C'), c.tobytes('K'), $"K==C on C-contig {tc}");
                CollectionAssert.AreEqual(f.tobytes('C'), f.tobytes('K'), $"K==C on F-contig {tc}");
            }
        }

        [TestMethod]
        public void LowercaseOrder_EqualsUppercase_AllDtypes()
        {
            foreach (var tc in AllDtypes)
            {
                var c = np.arange(6).astype(tc).reshape(2, 3);
                var f = np.arange(6).astype(tc).reshape(3, 2).T;
                CollectionAssert.AreEqual(c.tobytes('C'), c.tobytes('c'), $"c {tc}");
                CollectionAssert.AreEqual(c.tobytes('F'), c.tobytes('f'), $"f {tc}");
                CollectionAssert.AreEqual(f.tobytes('A'), f.tobytes('a'), $"a {tc}");
                CollectionAssert.AreEqual(f.tobytes('K'), f.tobytes('k'), $"k {tc}");
            }
        }

        // ---- API surface -----------------------------------------------------------------

        [TestMethod]
        public void DefaultOrder_IsC()
        {
            foreach (var tc in AllDtypes)
            {
                var a = np.arange(6).astype(tc).reshape(2, 3);
                CollectionAssert.AreEqual(a.tobytes('C'), a.tobytes(), $"tobytes()==C {tc}");
            }
        }

        [TestMethod]
        public void InvalidOrder_Throws_ArgumentException()
        {
            foreach (char bad in new[] { 'X', 'Z', '1', 'G', ' ', '\0' })
            {
                var a = np.arange(3).astype(NPTypeCode.Int32);
                Assert.ThrowsException<ArgumentException>(() => a.tobytes(bad), $"order '{bad}' must throw");
            }
        }

        // ---- Edge cases ------------------------------------------------------------------

        [TestMethod]
        public void Empty_AllOrders_AllDtypes_ReturnsEmpty()
        {
            foreach (var tc in AllDtypes)
                foreach (var o in new[] { 'C', 'F', 'A', 'K' })
                    Assert.AreEqual(0, np.zeros(new Shape(0, 3)).astype(tc).tobytes(o).Length, $"empty {tc}/{o}");
        }

        [TestMethod]
        public void Scalar0d_AllOrders_AllDtypes()
        {
            foreach (var tc in AllDtypes)
            {
                var z = np.arange(1).astype(tc).reshape(new int[0]); // 0-d
                int es = (int)z.dtypesize;
                foreach (var o in new[] { 'C', 'F', 'A', 'K' })
                    Assert.AreEqual(es, z.tobytes(o).Length, $"scalar {tc}/{o} length");
            }
        }

        [TestMethod]
        public void RoundTrip_FOrder_FromBuffer_AllDtypes()
        {
            // tobytes('F') of a view, rebuilt with frombuffer, re-emits identical bytes -> F readout survived.
            foreach (var tc in AllDtypes)
            {
                var a = np.arange(6).astype(tc).reshape(2, 3);
                byte[] fbytes = a.tobytes('F');
                var rebuilt = np.frombuffer(fbytes, tc);
                CollectionAssert.AreEqual(fbytes, rebuilt.tobytes('C'), $"roundtrip F {tc}");
                Assert.AreEqual(a.size, rebuilt.size, $"size {tc}");
            }
        }

        [TestMethod]
        public void FOrder_IsDetachedCopy()
        {
            // Even the F-contiguous fast path must hand back a fresh buffer, never an alias of storage.
            var t = np.arange(6).astype(NPTypeCode.Int32).reshape(3, 2).T; // F-contig fast path for 'F'
            byte[] b = t.tobytes('F');
            b[0] = 0xFF;
            // element [0,0] (== F-order byte slot 0) must be untouched -> ToArray materializes C-order, [0] is it.
            Assert.AreEqual(0, t.ToArray<int>()[0], "mutating tobytes('F') output must not touch source");
        }
    }
}
