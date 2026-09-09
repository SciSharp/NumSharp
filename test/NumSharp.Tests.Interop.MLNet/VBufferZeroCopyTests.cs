using System;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The zero-copy <see cref="VBuffer{T}"/> → <see cref="NDArray"/> path — <c>vbuffer.AsNDArray&lt;T&gt;()</c> —
    ///     which reaches the buffer's private backing array through a compiled-expression accessor
    ///     (<c>VBufferAccessor&lt;T&gt;</c>), pins it, and wraps it write-through (like the pythonnet / ONNX view
    ///     bridges). A sparse / empty / unknown-layout buffer falls back to a copy.
    /// </summary>
    [TestClass]
    public class VBufferZeroCopyTests : MLNetTestBase
    {
        [TestMethod]
        public void CompiledExpressionAccessor_IsAvailable_ForEveryDtype()
        {
            // If the reflected accessor bound, zero-copy actually engages (rather than silently copying).
            Assert.IsNotNull(VBufferAccessor<float>.GetValuesArray, "float _values accessor");
            Assert.IsNotNull(VBufferAccessor<double>.GetValuesArray);
            Assert.IsNotNull(VBufferAccessor<int>.GetValuesArray);
            Assert.IsNotNull(VBufferAccessor<long>.GetValuesArray);
            Assert.IsNotNull(VBufferAccessor<byte>.GetValuesArray);
            Assert.IsNotNull(VBufferAccessor<bool>.GetValuesArray);
            Assert.IsNotNull(VBufferAccessor<ushort>.GetValuesArray);
        }

        [TestMethod]
        public void AsNDArray_IsZeroCopy_WritesThroughBothWays()
        {
            var arr = new float[] { 10, 20, 30, 40 };
            var vb = new VBuffer<float>(4, arr);
            using NDArray nd = vb.AsNDArray();
            AssertShape(nd, 4);

            // NDArray write reaches the VBuffer's OWN backing array (proof it is a view, not a copy).
            nd[2] = 999f;
            Assert.AreEqual(999f, arr[2], "NDArray -> array write-through");
            Assert.AreEqual(999f, vb.GetValues()[2], "visible via VBuffer.GetValues");

            // Array write is visible in the NDArray.
            arr[0] = 111f;
            Assert.AreEqual(111f, (float)nd[0], "array -> NDArray read-through");
        }

        [TestMethod]
        public void AsNDArray_IsNonOwningView_ResizeRefuses()
        {
            var vb = new VBuffer<int>(3, new int[] { 1, 2, 3 });
            using NDArray nd = vb.AsNDArray();
            Assert.IsFalse(nd.flags.owndata, "a zero-copy view does not own its data");
            Assert.ThrowsException<IncorrectShapeException>(() => nd.resize(new Shape(9), refcheck: true));
        }

        [TestMethod]
        public void AsNDArray_PooledOversizedBuffer_ViewsOnlyLength()
        {
            var vb = new VBuffer<float>(4, new float[] { 1, 2, 3, 4 });
            VBufferEditor<float> editor = VBufferEditor.Create(ref vb, 2);   // logical length 2, backing stays 4
            VBuffer<float> shrunk = editor.Commit();
            Assert.AreEqual(2, shrunk.Length);
            using NDArray nd = shrunk.AsNDArray();
            AssertShape(nd, 2);
            Assert.AreEqual(1f, (float)nd[0]);
            Assert.AreEqual(2f, (float)nd[1]);
        }

        [TestMethod]
        public void AsNDArray_Sparse_FallsBackToOwningCopy()
        {
            var vb = new VBuffer<float>(5, 2, new float[] { 7f, 9f }, new int[] { 1, 3 });
            Assert.IsFalse(vb.IsDense);
            using NDArray nd = vb.AsNDArray();
            CollectionAssert.AreEqual(new float[] { 0, 7, 0, 9, 0 }, nd.ToArray<float>(), "sparse densified");
            Assert.IsTrue(nd.flags.owndata, "sparse fallback is an owning copy (no live pin)");
        }

        [TestMethod]
        public void AsNDArray_Empty_IsEmpty_NoLease()
        {
            int baseI = NDArrayMLNetInterop.LiveImports;
            var vb = new VBuffer<double>(0, new double[0]);
            using NDArray nd = vb.AsNDArray();
            AssertShape(nd, 0);
            Assert.AreEqual(baseI, NDArrayMLNetInterop.LiveImports, "an empty view leases nothing");
        }

        [TestMethod]
        public void AsNDArray_RoundTrips_RepresentativeDtypes()
        {
            AssertDtype(new VBuffer<double>(3, new double[] { 1, 2, 3 }), NPTypeCode.Double);
            AssertDtype(new VBuffer<int>(3, new int[] { 1, 2, 3 }), NPTypeCode.Int32);
            AssertDtype(new VBuffer<long>(3, new long[] { 1, 2, 3 }), NPTypeCode.Int64);
            AssertDtype(new VBuffer<byte>(3, new byte[] { 1, 2, 3 }), NPTypeCode.Byte);
            AssertDtype(new VBuffer<bool>(2, new[] { true, false }), NPTypeCode.Boolean);
        }

        [TestMethod]
        public void AsNDArray_UShort_ComesBackAsUInt16()
        {
            var vb = new VBuffer<ushort>(3, new ushort[] { 65, 66, 67 });
            using NDArray nd = vb.AsNDArray();
            Assert.AreEqual(NPTypeCode.UInt16, nd.typecode);
        }

        [TestMethod]
        public void AsNDArray_Lease_ReleasedOnDispose()
        {
            int baseI = NDArrayMLNetInterop.LiveImports;
            var vb = new VBuffer<float>(4, new float[] { 1, 2, 3, 4 });
            NDArray nd = vb.AsNDArray();
            Assert.AreEqual(baseI + 1, NDArrayMLNetInterop.LiveImports, "one live import lease");
            nd.Dispose();
            Assert.AreEqual(baseI, NDArrayMLNetInterop.LiveImports, "released on Dispose (no GC needed)");
        }

        [TestMethod]
        public void AsNDArray_DerivedSlice_KeepsThePinUntilBothDie()
        {
            int baseI = NDArrayMLNetInterop.LiveImports;
            RunDerivedSliceScenario(baseI);   // in a helper so the view/slice locals unroot on return
            // A NumSharp sliced view releases its ARC reference on COLLECTION (not eager Dispose), so the pin
            // settles once the slice is collectable — proving it fires only on the last reference, with no leak.
            Assert.IsTrue(WaitFor(() => NDArrayMLNetInterop.LiveImports <= baseI, 10_000),
                $"the lease outlived the last view: {baseI} -> {NDArrayMLNetInterop.LiveImports}");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void RunDerivedSliceScenario(int baseI)
        {
            var vb = new VBuffer<float>(4, new float[] { 5, 6, 7, 8 });
            NDArray view = vb.AsNDArray();
            NDArray slice = view["1:3"];        // shares the same pinned memory block
            view.Dispose();                     // parent gone, but the slice still holds the block alive
            Assert.AreEqual(baseI + 1, NDArrayMLNetInterop.LiveImports, "the lease survives while a derived slice lives");
            Assert.AreEqual(6f, (float)slice[0], "the slice still reads the pinned memory after the parent was disposed");
            slice.Dispose();
        }

        [TestMethod]
        public void AsNDArray_ForgottenView_ReleasedByFinalizer()
        {
            int baseI = NDArrayMLNetInterop.LiveImports;
            LeakAView();
            Assert.IsTrue(WaitFor(() => NDArrayMLNetInterop.LiveImports <= baseI, 10_000),
                $"finalizer did not release the pin: {baseI} -> {NDArrayMLNetInterop.LiveImports}");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void LeakAView()
        {
            var vb = new VBuffer<double>(4, new double[] { 1, 2, 3, 4 });
            NDArray nd = vb.AsNDArray();
            Assert.IsTrue(NDArrayMLNetInterop.LiveImports > 0);
            // not disposed — the finalizer must reclaim the pin
        }

        private static void AssertDtype<T>(VBuffer<T> vb, NPTypeCode expected) where T : unmanaged
        {
            using NDArray nd = vb.AsNDArray();
            Assert.AreEqual(expected, nd.typecode);
            using NDArray copy = vb.ToNDArray();
            Assert.IsTrue(np.array_equal(copy, nd), $"{expected} zero-copy view equals the copy");
        }
    }
}
