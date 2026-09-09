using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Every NDArray memory layout / flags combination — as the source of a lazy `IDataView` (the cursor reads
    ///     through the array's strides, so no layout needs materializing) — and the flags of the materialized output.
    /// </summary>
    [TestClass]
    public class LayoutFlagTests : MLNetTestBase
    {
        // ---- input layouts, VECTOR mode ------------------------------------------------------------

        [TestMethod]
        public void Vector_ReadsEveryLayout()
        {
            RoundTripVector(Arange(NPTypeCode.Double, 4, 3), "C-contiguous");
            RoundTripVector(np.asfortranarray(Arange(NPTypeCode.Double, 4, 3)), "F-contiguous");
            using (NDArray b = Arange(NPTypeCode.Double, 3, 4))
            {
                RoundTripVector(b.T, "transpose", dispose: false);
                RoundTripVector(b["::-1"], "reversed rows", dispose: false);
                RoundTripVector(b[":, ::-1"], "reversed cols", dispose: false);
                RoundTripVector(b["1:3"], "sliced offset window", dispose: false);
                RoundTripVector(b[":, ::2"], "strided cols", dispose: false);
                RoundTripVector(b["::2, ::2"], "strided both", dispose: false);
                using NDArray v = b.T;
                RoundTripVector(v["1:"], "view of a view", dispose: false);
            }
        }

        [TestMethod]
        public void Vector_BroadcastReadOnlySource()
        {
            using NDArray row = Arange(NPTypeCode.Double, 1, 3);
            using NDArray b = np.broadcast_to(row, new Shape(4, 3));   // stride-0 axis 0, read-only
            Assert.IsFalse(b.Shape.IsWriteable, "broadcast source is read-only");
            Assert.IsTrue(b.Shape.IsBroadcasted);
            using NDArrayDataView dv = b.AsDataView("F");
            using NDArray back = dv.ToNDArray("F");
            Assert.IsTrue(np.array_equal(b, back), "reads a read-only broadcast source");
            Assert.IsTrue(back.flags.writeable, "the materialized copy is writeable even though the source was not");
        }

        // ---- input layouts, SCALAR-COLUMN mode -----------------------------------------------------

        [TestMethod]
        public void ScalarColumns_ReadsEveryLayout()
        {
            using NDArray b = Arange(NPTypeCode.Int32, 4, 3);
            RoundTripScalar(b, "C-contiguous");
            RoundTripScalar(np.asfortranarray(b), "F-contiguous");
            RoundTripScalar(b["::-1"], "reversed rows");
            RoundTripScalar(b[":, ::-1"], "reversed cols");
            RoundTripScalar(b["1:4"], "sliced offset window");
            RoundTripScalar(b["::2"], "strided rows");
        }

        [TestMethod]
        public void ScalarColumns_BroadcastColumns_AllReadSameElement()
        {
            using NDArray row = np.array(new[] { 10, 20, 30 }).reshape(1, 3);
            using NDArray b = np.broadcast_to(row, new Shape(4, 3));   // 4 identical rows [10,20,30]
            using NDArrayDataView dv = b.AsDataView(new[] { "a", "b", "c" });
            using NDArray a = dv.ToNDArray("a");
            using NDArray c = dv.ToNDArray("c");
            AssertShape(a, 4);
            for (int r = 0; r < 4; r++)
            {
                Assert.AreEqual(10, (int)a[r]);
                Assert.AreEqual(30, (int)c[r]);
            }
        }

        // ---- output flags --------------------------------------------------------------------------

        [TestMethod]
        public void Output_Vector_IsOwningWriteableCContiguous()
        {
            using NDArray a = np.asfortranarray(Arange(NPTypeCode.Single, 3, 4));   // even from an F source
            using NDArrayDataView dv = a.AsDataView("F");
            using NDArray back = dv.ToNDArray("F");
            Assert.IsTrue(back.flags.c_contiguous, "C_CONTIGUOUS");
            Assert.IsTrue(back.flags.owndata, "OWNDATA");
            Assert.IsTrue(back.flags.writeable, "WRITEABLE");
            Assert.IsFalse(back.Shape.IsBroadcasted, "not broadcasted");
        }

        [TestMethod]
        public void Output_ScalarColumn_IsOwningWriteableCContiguous1D()
        {
            using NDArray a = Arange(NPTypeCode.Double, 5, 2);
            using NDArrayDataView dv = a.AsDataView(new[] { "x", "y" });
            using NDArray x = dv.ToNDArray("x");
            Assert.AreEqual(1, x.ndim);
            Assert.IsTrue(x.flags.c_contiguous);
            Assert.IsTrue(x.flags.owndata);
            Assert.IsTrue(x.flags.writeable);
        }

        [TestMethod]
        public void Output_IsIndependentCopy_MutationDoesNotTouchSource()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.AsDataView("F");
            using NDArray back = dv.ToNDArray("F");
            back[0, 0] = 777f;                       // writeable, owning
            Assert.AreEqual(0f, (float)a[0, 0], "the source is untouched by writing the materialized copy");
        }

        // ---- helpers -------------------------------------------------------------------------------

        private static void RoundTripVector(NDArray view, string label, bool dispose = true)
        {
            using NDArrayDataView dv = view.AsDataView("F");
            using NDArray back = dv.ToNDArray("F");
            Assert.IsTrue(np.array_equal(view, back), $"vector {label}");
            if (dispose)
                view.Dispose();
        }

        private static void RoundTripScalar(NDArray view, string label)
        {
            int cols = view.ndim == 1 ? 1 : (int)view.shape[1];
            var names = new string[cols];
            for (int i = 0; i < cols; i++)
                names[i] = "c" + i;
            using NDArrayDataView dv = view.AsDataView(names);
            for (int c = 0; c < cols; c++)
            {
                using NDArray col = dv.ToNDArray(names[c]);
                using NDArray expect = cols == 1 ? view : view[":, " + c];
                using NDArray expectFlat = expect.ndim == 1 ? null : expect.reshape((int)view.shape[0]);
                Assert.IsTrue(np.array_equal(expectFlat ?? expect, col), $"scalar {label} col {c}");
            }
        }
    }
}
