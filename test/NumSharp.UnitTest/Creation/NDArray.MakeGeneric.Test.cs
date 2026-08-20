using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using System.Linq;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NdArrayMakeGenericTester
    {
        [TestMethod]
        public void Array1DimGeneric()
        {
            var list = new double[] {1.1, 2.2, 3.3};
            var arrayDouble = np.array(list).MakeGeneric<double>();

            Assert.IsTrue(arrayDouble[0] == 1.1);
            Assert.IsTrue(arrayDouble[1] == 2.2);
            Assert.IsTrue(arrayDouble[2] == 3.3);
        }

        /// <summary>
        ///     MakeGeneric must return an independent view (NumPy ndarray.view() semantics):
        ///     reshaping the typed view must NOT mutate the source array's shape.
        /// </summary>
        [TestMethod]
        public void MakeGeneric_ShapeMutation_DoesNotCorruptSource()
        {
            var a = np.arange(6);                  // Int64, shape (6,)
            var g = a.MakeGeneric<long>();

            g.Shape = new Shape(2, 3);

            // NumPy: a = np.arange(6); b = a.view(); b.shape = (2,3) -> a.shape stays (6,)
            Assert.AreEqual(1, a.ndim, "source ndim must be unchanged");
            CollectionAssert.AreEqual(new long[] {6}, a.shape, "source shape must be unchanged");
            CollectionAssert.AreEqual(new long[] {2, 3}, g.shape, "typed view shape must change");
        }

        /// <summary>
        ///     MakeGeneric shares the underlying data buffer — element writes propagate both ways.
        /// </summary>
        [TestMethod]
        public void MakeGeneric_SharesData()
        {
            var a = np.arange(6);
            var g = a.MakeGeneric<long>();

            g.SetValue(999L, 0);
            Assert.AreEqual(999L, (long)a.GetValue(0), "element write through view must be visible in source");

            a.SetValue(-1L, 5);
            Assert.AreEqual(-1L, (long)g.GetValue(5), "element write through source must be visible in view");
        }

        /// <summary>
        ///     MakeGeneric over a sliced view preserves the slice geometry and the shared data.
        /// </summary>
        [TestMethod]
        public void MakeGeneric_OverSlice_PreservesViewAndShares()
        {
            var c = np.arange(10);
            var sl = c["2:6"].MakeGeneric<long>();

            CollectionAssert.AreEqual(new long[] {4}, sl.shape);
            sl.SetValue(-7L, 0);
            Assert.AreEqual(-7L, (long)c.GetValue(2), "sliced view must share data with the source");
        }

        /// <summary>
        ///     AsOrMakeGeneric's matching-dtype branch must also isolate the shape.
        /// </summary>
        [TestMethod]
        public void AsOrMakeGeneric_MatchBranch_IsolatesShape()
        {
            var e = np.arange(6);
            var g = e.AsOrMakeGeneric<long>();

            g.Shape = new Shape(3, 2);
            CollectionAssert.AreEqual(new long[] {6}, e.shape, "source shape must be unchanged");
        }

        /// <summary>
        ///     The alias holds its own reference on the shared buffer, so it stays readable/writable
        ///     after the source NDArray is disposed and finalized (no use-after-free).
        /// </summary>
        [TestMethod]
        public void MakeGeneric_AliasOutlivesDisposedSource()
        {
            var a = np.arange(100);
            var g = a.MakeGeneric<long>();

            a.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            Assert.AreEqual(77L, (long)g.GetValue(77), "alias must remain readable after source disposal");
            g.SetValue(123L, 0);
            Assert.AreEqual(123L, (long)g.GetValue(0), "alias must remain writable after source disposal");
        }

        /// <summary>
        ///     Disposing the alias must NOT free the buffer out from under the still-live source.
        /// </summary>
        [TestMethod]
        public void MakeGeneric_DisposingAlias_LeavesSourceValid()
        {
            var a = np.arange(100);
            var g = a.MakeGeneric<long>();

            g.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();

            Assert.AreEqual(33L, (long)a.GetValue(33), "source must remain valid after alias disposal");
        }

        /// <summary>
        ///     MakeGeneric over a transposed view keeps the strided geometry and shares the buffer:
        ///     writing tt[0,1] must hit t[1,0].
        /// </summary>
        [TestMethod]
        public void MakeGeneric_TransposedView_SharesStridesAndData()
        {
            var t = np.arange(6).reshape(2, 3);
            var g = t.T.MakeGeneric<long>();          // shape (3,2)

            CollectionAssert.AreEqual(new long[] {3, 2}, g.shape);
            g.SetValue(99L, 0, 1);                    // tt[0,1] == t[1,0]
            Assert.AreEqual(99L, (long)t.GetValue(1, 0), "transposed alias must share data via strides");
        }

        /// <summary>
        ///     A broadcast view is read-only (stride=0); MakeGeneric must preserve that —
        ///     reads are correct, the broadcast flag survives, and writes are rejected.
        /// </summary>
        [TestMethod]
        public void MakeGeneric_BroadcastView_ReadOnlyPreserved()
        {
            var bc = np.broadcast_to(np.arange(3).reshape(1, 3), new Shape(4, 3));
            var g = bc.MakeGeneric<long>();

            Assert.IsTrue(g.Shape.IsBroadcasted, "broadcast flag must survive the alias");
            Assert.IsFalse(g.Shape.IsWriteable, "broadcast alias must stay read-only");
            Assert.AreEqual(1L, (long)g.GetValue(2, 1), "broadcast read must be correct");
            Assert.AreEqual(12L, (long)np.sum(g).GetValue(), "engine iteration over broadcast alias must be correct");

            bool threw = false;
            try { g.SetValue(99L, 0, 0); } catch { threw = true; }
            Assert.IsTrue(threw, "writing through a broadcast view must be rejected");
        }

        /// <summary>
        ///     MakeGeneric is a NumPy view: the result reports IsView == true and chains its base
        ///     to the owning storage.
        /// </summary>
        [TestMethod]
        public void MakeGeneric_ResultIsView()
        {
            var owner = np.arange(6);
            var g = owner.MakeGeneric<long>();

            Assert.IsTrue(g.Storage.IsView, "MakeGeneric result is a view (NumPy ndarray.view semantics)");
            Assert.AreSame(owner.Storage, g.Storage.BaseStorage, "view base must chain to the owning storage");
        }

        /// <summary>
        ///     MakeGeneric round-trips every supported dtype: the typed view reports the right dtype,
        ///     reads the right value, and shares (does not reallocate) the buffer.
        /// </summary>
        [TestMethod]
        public void MakeGeneric_AllDtypes_RoundTrip()
        {
            RoundTrip(np.array(new bool[]    {true, false}),                 true,                 "bool");
            RoundTrip(np.array(new byte[]    {7, 8}),                        (byte)7,              "byte");
            RoundTrip(np.array(new sbyte[]   {-7, 8}),                       (sbyte)-7,            "sbyte");
            RoundTrip(np.array(new short[]   {-9, 8}),                       (short)-9,            "int16");
            RoundTrip(np.array(new ushort[]  {9, 8}),                        (ushort)9,            "uint16");
            RoundTrip(np.array(new int[]     {-11, 8}),                      -11,                  "int32");
            RoundTrip(np.array(new uint[]    {11, 8}),                       11u,                  "uint32");
            RoundTrip(np.array(new long[]    {-13, 8}),                      -13L,                 "int64");
            RoundTrip(np.array(new ulong[]   {13, 8}),                       13ul,                 "uint64");
            RoundTrip(np.array(new char[]    {'A', 'B'}),                    'A',                  "char");
            RoundTrip(np.array(new Half[]    {(Half)1.5, (Half)2.5}),        (Half)1.5,            "half");
            RoundTrip(np.array(new float[]   {1.25f, 2f}),                   1.25f,                "single");
            RoundTrip(np.array(new double[]  {1.1, 2.2}),                    1.1,                  "double");
            RoundTrip(np.array(new decimal[] {1.5m, 2m}),                    1.5m,                 "decimal");
            RoundTrip(np.array(new Complex[] {new Complex(1, 2), new Complex(3, 4)}), new Complex(1, 2), "complex");
        }

        private static void RoundTrip<T>(NDArray src, T expected, string name) where T : unmanaged
        {
            var g = src.MakeGeneric<T>();
            Assert.AreEqual(typeof(T), g.dtype, $"{name}: dtype");
            Assert.AreEqual(expected, g.GetValue(0), $"{name}: value");
            Assert.IsFalse(ReferenceEquals(src.Storage, g.Storage), $"{name}: must be an alias (not the same storage instance)");
        }
    }
}
