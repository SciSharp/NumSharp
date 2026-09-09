using System;
using System.Linq;
using System.Numerics;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    [TestClass]
    public class VBufferTests : MLNetTestBase
    {
        // ---- ToVBuffer ------------------------------------------------------------------------------

        [TestMethod]
        public void ToVBuffer_Dense_RoundTrips_RepresentativeDtypes()
        {
            AssertToVBufferRoundTrip<float>(NPTypeCode.Single);
            AssertToVBufferRoundTrip<double>(NPTypeCode.Double);
            AssertToVBufferRoundTrip<int>(NPTypeCode.Int32);
            AssertToVBufferRoundTrip<long>(NPTypeCode.Int64);
            AssertToVBufferRoundTrip<byte>(NPTypeCode.Byte);
            AssertToVBufferRoundTrip<sbyte>(NPTypeCode.SByte);
            AssertToVBufferRoundTrip<short>(NPTypeCode.Int16);
            AssertToVBufferRoundTrip<ushort>(NPTypeCode.UInt16);
            AssertToVBufferRoundTrip<uint>(NPTypeCode.UInt32);
            AssertToVBufferRoundTrip<ulong>(NPTypeCode.UInt64);
        }

        [TestMethod]
        public void ToVBuffer_Bool_RoundTrips()
        {
            using NDArray a = np.array(new[] { true, false, true });
            VBuffer<bool> vb = a.ToVBuffer<bool>();
            CollectionAssert.AreEqual(new[] { true, false, true }, vb.GetValues().ToArray());
        }

        [TestMethod]
        public void ToVBuffer_Char_CrossesAsUShort()
        {
            using NDArray a = np.array(new[] { 'x', 'y', 'z' });
            VBuffer<ushort> vb = a.ToVBuffer<ushort>();
            CollectionAssert.AreEqual(new ushort[] { 'x', 'y', 'z' }, vb.GetValues().ToArray());
        }

        [TestMethod]
        public void ToVBuffer_Half_ConvertsToSingle()
        {
            using NDArray a = np.arange(4).astype(NPTypeCode.Half);
            VBuffer<float> vb = a.ToVBuffer<float>();
            CollectionAssert.AreEqual(new float[] { 0, 1, 2, 3 }, vb.GetValues().ToArray());
        }

        [TestMethod]
        public void ToVBuffer_Decimal_ConvertsToDouble()
        {
            using NDArray a = np.array(new[] { 0m, 1.5m, 2.25m });
            VBuffer<double> vb = a.ToVBuffer<double>();
            CollectionAssert.AreEqual(new double[] { 0, 1.5, 2.25 }, vb.GetValues().ToArray());
        }

        [TestMethod]
        public void ToVBuffer_ZeroD_IsLengthOne()
        {
            using NDArray a = np.array(7f);
            VBuffer<float> vb = a.ToVBuffer<float>();
            Assert.AreEqual(1, vb.Length);
            Assert.AreEqual(7f, vb.GetValues()[0]);
        }

        [TestMethod]
        public void ToVBuffer_Empty_IsLengthZero()
        {
            using NDArray a = np.arange(0).astype(NPTypeCode.Single);
            VBuffer<float> vb = a.ToVBuffer<float>();
            Assert.AreEqual(0, vb.Length);
        }

        [TestMethod]
        public void ToVBuffer_ReadsAnyLayoutInLogicalOrder()
        {
            using NDArray a = Arange(NPTypeCode.Int32, 2, 3);
            using NDArray rev = a["::-1"];   // reversed rows: logical [3,4,5,0,1,2]
            VBuffer<int> vb = rev.ToVBuffer<int>();
            CollectionAssert.AreEqual(new[] { 3, 4, 5, 0, 1, 2 }, vb.GetValues().ToArray());
        }

        [TestMethod]
        public void ToVBuffer_WrongT_Throws()
        {
            using NDArray a = np.arange(3).astype(NPTypeCode.Int32);
            Assert.ThrowsException<ArgumentException>(() => a.ToVBuffer<long>());
        }

        [TestMethod]
        public void ToVBuffer_Complex_Throws()
        {
            using NDArray a = np.arange(3).astype(NPTypeCode.Complex);
            Assert.ThrowsException<NotSupportedException>(() => a.ToVBuffer<Complex>());
        }

        [TestMethod]
        public void ToVBuffer_Null_Throws()
        {
            NDArray a = null;
            Assert.ThrowsException<ArgumentNullException>(() => a.ToVBuffer<float>());
        }

        // ---- VBuffer -> NDArray ---------------------------------------------------------------------

        [TestMethod]
        public void VBuffer_Dense_ToNDArray()
        {
            var vb = new VBuffer<double>(4, new double[] { 1, 2, 3, 4 });
            using NDArray nd = vb.ToNDArray();
            Assert.AreEqual(NPTypeCode.Double, nd.typecode);
            AssertShape(nd, 4);
            CollectionAssert.AreEqual(new double[] { 1, 2, 3, 4 }, nd.ToArray<double>());
        }

        [TestMethod]
        public void VBuffer_Sparse_DensifiesWithZeros()
        {
            var vb = new VBuffer<float>(5, 2, new float[] { 7f, 9f }, new int[] { 1, 3 });
            Assert.IsFalse(vb.IsDense);
            using NDArray nd = vb.ToNDArray();
            CollectionAssert.AreEqual(new float[] { 0, 7, 0, 9, 0 }, nd.ToArray<float>());
        }

        [TestMethod]
        public void VBuffer_Sparse_BoundaryIndices()
        {
            var vb = new VBuffer<int>(4, 2, new int[] { 10, 40 }, new int[] { 0, 3 });
            using NDArray nd = vb.ToNDArray();
            CollectionAssert.AreEqual(new int[] { 10, 0, 0, 40 }, nd.ToArray<int>());
        }

        [TestMethod]
        public void VBuffer_Sparse_AllDefault_IsAllZeros()
        {
            var vb = new VBuffer<int>(4, 0, new int[0], new int[0]);
            Assert.IsFalse(vb.IsDense);
            using NDArray nd = vb.ToNDArray();
            CollectionAssert.AreEqual(new int[] { 0, 0, 0, 0 }, nd.ToArray<int>());
        }

        [TestMethod]
        public void VBuffer_Empty_ToNDArray()
        {
            var vb = new VBuffer<float>(0, new float[0]);
            using NDArray nd = vb.ToNDArray();
            AssertShape(nd, 0);
        }

        [TestMethod]
        public void VBuffer_LengthOne_ToNDArray()
        {
            var vb = new VBuffer<double>(1, new double[] { 42 });
            using NDArray nd = vb.ToNDArray();
            AssertShape(nd, 1);
            Assert.AreEqual(42d, (double)nd[0]);
        }

        [TestMethod]
        public void VBuffer_UShort_ComesBackAsUInt16_NotChar()
        {
            var vb = new VBuffer<ushort>(3, new ushort[] { 65, 66, 67 });
            using NDArray nd = vb.ToNDArray();
            Assert.AreEqual(NPTypeCode.UInt16, nd.typecode);
        }

        // ---- helpers -------------------------------------------------------------------------------

        private static void AssertToVBufferRoundTrip<T>(NPTypeCode code) where T : unmanaged
        {
            using NDArray a = Arange(code, 5);
            VBuffer<T> vb = a.ToVBuffer<T>();
            Assert.AreEqual(5, vb.Length);
            Assert.IsTrue(vb.IsDense);
            using NDArray back = vb.ToNDArray();
            Assert.IsTrue(np.array_equal(a, back), $"{code} VBuffer round-trip");
        }
    }
}
