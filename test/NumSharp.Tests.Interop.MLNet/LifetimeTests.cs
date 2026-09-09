using System;
using System.Runtime.CompilerServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    [TestClass]
    public class LifetimeTests : MLNetTestBase
    {
        [TestMethod]
        public void LiveExports_Tracks_ViewLifetime()
        {
            using NDArray a = Arange(NPTypeCode.Single, 3, 3);
            int baseline = NDArrayMLNetInterop.LiveExports;
            NDArrayDataView dv = a.AsDataView("Features");
            Assert.AreEqual(baseline + 1, NDArrayMLNetInterop.LiveExports, "creating a view pins one buffer");
            dv.Dispose();
            Assert.AreEqual(baseline, NDArrayMLNetInterop.LiveExports, "disposing the view releases it");
        }

        [TestMethod]
        public void Dispose_IsIdempotent()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            NDArrayDataView dv = a.AsDataView("Features");
            dv.Dispose();
            dv.Dispose();   // must not throw or double-decrement
        }

        [TestMethod]
        public void UseAfterDispose_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            NDArrayDataView dv = a.AsDataView("Features");
            dv.Dispose();
            Assert.ThrowsException<ObjectDisposedException>(() => _ = dv.Schema);
            Assert.ThrowsException<ObjectDisposedException>(() => dv.GetRowCursor(null));
        }

        [TestMethod]
        public void DisposeSourceWhileViewAlive_IsSafe_ThePinKeepsTheBuffer()
        {
            NDArray a = Arange(NPTypeCode.Single, 3, 2);
            using NDArrayDataView dv = a.AsDataView("Features");
            a.Dispose();   // the view's ARC pin keeps the buffer alive past the source's disposal
            using NDArray back = dv.ToNDArray("Features");
            AssertShape(back, 3, 2);
            Assert.AreEqual(0f, (float)back[0, 0]);
            Assert.AreEqual(5f, (float)back[2, 1]);
        }

        [TestMethod]
        public void Cursor_HoldsOwnPin_SurvivesViewDispose()
        {
            using NDArray a = Arange(NPTypeCode.Double, 4);
            NDArrayDataView dv = a.AsDataView(new[] { "v" });
            DataViewSchema.Column col = dv.Schema["v"];
            DataViewRowCursor cur = dv.GetRowCursor(new[] { col });
            ValueGetter<double> getter = cur.GetGetter<double>(col);

            dv.Dispose();   // view gone, but the cursor's own pin keeps the buffer

            double v = 0;
            Assert.IsTrue(cur.MoveNext());
            getter(ref v);
            Assert.AreEqual(0d, v);
            Assert.IsTrue(cur.MoveNext());
            getter(ref v);
            Assert.AreEqual(1d, v);
            cur.Dispose();
        }

        [TestMethod]
        public void Finalizer_SafetyNet_ReleasesAnUndisposedView()
        {
            using NDArray a = Arange(NPTypeCode.Single, 4, 4);
            int baseline = NDArrayMLNetInterop.LiveExports;
            LeakAView(a);
            Assert.IsTrue(WaitFor(() => NDArrayMLNetInterop.LiveExports <= baseline, 10_000),
                $"finalizer did not release the pin: {baseline} -> {NDArrayMLNetInterop.LiveExports}");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void LeakAView(NDArray a)
        {
            NDArrayDataView dv = a.AsDataView("Features");
            Assert.IsTrue(NDArrayMLNetInterop.LiveExports > 0);
            // deliberately not disposed — the finalizer must reclaim the pin
        }
    }
}
