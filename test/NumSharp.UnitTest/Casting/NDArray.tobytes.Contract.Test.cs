using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Casting
{
    /// <summary>
    ///     Regression + contract tests for <see cref="NDArray.tobytes"/>.
    ///
    ///     NumPy parity: <c>ndarray.tobytes()</c> returns the LOGICAL array in C (row-major) order,
    ///     honoring strides / offset / broadcasting — NOT the raw underlying buffer. Before the C6 fix,
    ///     the byte export blindly copied <c>Storage.InternalArray.BytesLength</c> from
    ///     <c>Storage.Address</c>, so a non-contiguous view leaked the whole parent buffer
    ///     (e.g. a 5-element <c>[::2]</c> view of a 10-element array returned 40 bytes of the wrong data).
    /// </summary>
    [TestClass]
    public class TobytesContractTests
    {
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
            NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
        };

        /// <summary>The independent, proven-correct C-order materialization: ToArray&lt;T&gt; → raw bytes.</summary>
        private static byte[] LogicalBytes(NDArray nd) => nd.typecode switch
        {
            NPTypeCode.Boolean => MemoryMarshal.AsBytes<bool>(nd.ToArray<bool>()).ToArray(),
            NPTypeCode.Byte    => nd.ToArray<byte>(),
            NPTypeCode.SByte   => MemoryMarshal.AsBytes<sbyte>(nd.ToArray<sbyte>()).ToArray(),
            NPTypeCode.Int16   => MemoryMarshal.AsBytes<short>(nd.ToArray<short>()).ToArray(),
            NPTypeCode.UInt16  => MemoryMarshal.AsBytes<ushort>(nd.ToArray<ushort>()).ToArray(),
            NPTypeCode.Int32   => MemoryMarshal.AsBytes<int>(nd.ToArray<int>()).ToArray(),
            NPTypeCode.UInt32  => MemoryMarshal.AsBytes<uint>(nd.ToArray<uint>()).ToArray(),
            NPTypeCode.Int64   => MemoryMarshal.AsBytes<long>(nd.ToArray<long>()).ToArray(),
            NPTypeCode.UInt64  => MemoryMarshal.AsBytes<ulong>(nd.ToArray<ulong>()).ToArray(),
            NPTypeCode.Char    => MemoryMarshal.AsBytes<char>(nd.ToArray<char>()).ToArray(),
            NPTypeCode.Half    => MemoryMarshal.AsBytes<Half>(nd.ToArray<Half>()).ToArray(),
            NPTypeCode.Single  => MemoryMarshal.AsBytes<float>(nd.ToArray<float>()).ToArray(),
            NPTypeCode.Double  => MemoryMarshal.AsBytes<double>(nd.ToArray<double>()).ToArray(),
            NPTypeCode.Decimal => MemoryMarshal.AsBytes<decimal>(nd.ToArray<decimal>()).ToArray(),
            NPTypeCode.Complex => MemoryMarshal.AsBytes<Complex>(nd.ToArray<Complex>()).ToArray(),
            _ => throw new NotSupportedException(nd.typecode.ToString())
        };

        private static void AssertLogical(string name, NDArray nd)
        {
            byte[] actual = nd.tobytes();
            Assert.AreEqual((int)(nd.size * nd.dtypesize), actual.Length, $"{name}: length must equal size*dtypesize");
            CollectionAssert.AreEqual(LogicalBytes(nd), actual, $"{name}: bytes must be the logical C-order array");
        }

        // ---- Pristine ----------------------------------------------------------------

        [TestMethod]
        public void Contiguous_AllDtypes()
        {
            foreach (var tc in AllDtypes)
                AssertLogical($"contig_{tc}", np.arange(12).astype(tc));
        }

        [TestMethod]
        public void Scalar0d_AllDtypes()
        {
            foreach (var tc in AllDtypes)
            {
                var nd = np.arange(1).astype(tc).reshape(new int[0]); // 0-d
                AssertLogical($"scalar_{tc}", nd);
            }
        }

        [TestMethod]
        public void Empty_ReturnsEmptyArray()
        {
            foreach (var tc in AllDtypes)
                Assert.AreEqual(0, np.arange(0).astype(tc).tobytes().Length, $"empty_{tc}");
        }

        // ---- Views: the C6 regressions ----------------------------------------------

        [TestMethod]
        public void PrefixSlice_AllDtypes()
        {
            foreach (var tc in AllDtypes)
                AssertLogical($"prefix_{tc}", np.arange(10).astype(tc)["0:5"]);
        }

        [TestMethod]
        public void MiddleSlice_OffsetView_AllDtypes()
        {
            foreach (var tc in AllDtypes)
                AssertLogical($"slice_{tc}", np.arange(10).astype(tc)["3:8"]);
        }

        [TestMethod]
        public void Strided_AllDtypes()
        {
            foreach (var tc in AllDtypes)
                AssertLogical($"strided_{tc}", np.arange(10).astype(tc)["::2"]);
        }

        [TestMethod]
        public void Reversed_AllDtypes()
        {
            foreach (var tc in AllDtypes)
                AssertLogical($"reversed_{tc}", np.arange(10).astype(tc)["::-1"]);
        }

        [TestMethod]
        public void Transpose_FContiguous_AllDtypes()
        {
            foreach (var tc in AllDtypes)
                AssertLogical($"transpose_{tc}", np.arange(6).astype(tc).reshape(2, 3).T);
        }

        [TestMethod]
        public void ColumnView_AllDtypes()
        {
            foreach (var tc in AllDtypes)
                AssertLogical($"column_{tc}", np.arange(6).astype(tc).reshape(2, 3)[":,1"]);
        }

        [TestMethod]
        public void Broadcast_Materializes_AllDtypes()
        {
            foreach (var tc in AllDtypes)
            {
                var b = np.broadcast_to(np.arange(3).astype(tc).reshape(1, 3), new Shape(4, 3));
                AssertLogical($"broadcast_{tc}", b);
            }
        }

        [TestMethod]
        public void ThreeD_AllDtypes()
        {
            foreach (var tc in AllDtypes)
                AssertLogical($"3d_{tc}", np.arange(8).astype(tc).reshape(2, 2, 2));
        }

        // ---- Exact, NumPy-verified bytes --------------------------------------------

        [TestMethod]
        public void Strided_Int32_ExactBytes()
        {
            // np.arange(10, dtype=np.int32)[::2].tobytes() -> [0,2,4,6,8]
            var v = np.arange(10).astype(NPTypeCode.Int32)["::2"];
            Assert.AreEqual(20, v.tobytes().Length, "5 elements * 4 bytes, NOT the 40-byte parent buffer");
            CollectionAssert.AreEqual(
                new byte[] { 0,0,0,0, 2,0,0,0, 4,0,0,0, 6,0,0,0, 8,0,0,0 },
                v.tobytes());
        }

        [TestMethod]
        public void Transpose_Int32_ExactBytes_COrder()
        {
            // np.arange(6, dtype=np.int32).reshape(2,3).T.tobytes() -> C-order [0,3,1,4,2,5]
            var t = np.arange(6).astype(NPTypeCode.Int32).reshape(2, 3).T;
            CollectionAssert.AreEqual(
                new byte[] { 0,0,0,0, 3,0,0,0, 1,0,0,0, 4,0,0,0, 2,0,0,0, 5,0,0,0 },
                t.tobytes());
        }

        [TestMethod]
        public void Reversed_Int32_ExactBytes()
        {
            // np.arange(5, dtype=np.int32)[::-1].tobytes() -> [4,3,2,1,0]
            var r = np.arange(5).astype(NPTypeCode.Int32)["::-1"];
            CollectionAssert.AreEqual(
                new byte[] { 4,0,0,0, 3,0,0,0, 2,0,0,0, 1,0,0,0, 0,0,0,0 },
                r.tobytes());
        }

        /// <summary>The original contract (a pristine contiguous array) must be unchanged.</summary>
        [TestMethod]
        public void PristineContiguous_LegacyContract_Unchanged()
        {
            var nd = np.array(new int[][] { new[] { 3, 1 }, new[] { 2, 1 } });
            CollectionAssert.AreEqual(
                new byte[] { 3,0,0,0, 1,0,0,0, 2,0,0,0, 1,0,0,0 },
                nd.tobytes());
        }

        // ---- Round-trip via frombuffer ----------------------------------------------

        [TestMethod]
        public void RoundTrip_tobytes_FromBuffer_AllDtypes_AllViews()
        {
            foreach (var tc in AllDtypes)
            {
                var full = np.arange(12).astype(tc);
                var views = new (string name, NDArray nd)[]
                {
                    ("contig",    full),
                    ("prefix",    full["0:6"]),
                    ("slice",     full["3:9"]),
                    ("strided",   full["::2"]),
                    ("reversed",  full["::-1"]),
                    ("transpose", np.arange(6).astype(tc).reshape(2, 3).T),
                };
                foreach (var (name, nd) in views)
                {
                    byte[] bytes = nd.tobytes();
                    var rebuilt = np.frombuffer(bytes, tc);
                    // A freshly built pristine array re-emits identical bytes -> the view's logical content survived.
                    CollectionAssert.AreEqual(bytes, rebuilt.tobytes(), $"{tc}/{name} round-trip");
                    Assert.AreEqual(nd.size, rebuilt.size, $"{tc}/{name} size");
                }
            }
        }

        /// <summary>Mutating a tobytes() result must never touch the source (it is a fresh copy).</summary>
        [TestMethod]
        public void Tobytes_IsDetachedCopy()
        {
            var nd = np.arange(4).astype(NPTypeCode.Int32);
            byte[] bytes = nd.tobytes();
            bytes[0] = 0xFF;
            Assert.AreEqual(0, nd.ToArray<int>()[0], "mutating the byte[] must not affect the source array");
        }
    }
}
