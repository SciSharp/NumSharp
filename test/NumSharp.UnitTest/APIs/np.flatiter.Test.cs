using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.APIs
{
    /// <summary>
    /// np.FlatIterator (arr.flatiter / np.flat) parity gate. Every expectation was probed against
    /// NumPy 2.4.2's flatiter. The headline capability over NumSharp's raveled <c>arr.flat</c> is
    /// write-through in logical C-order for ANY memory layout.
    /// </summary>
    [TestClass]
    public class np_flatiter_tests
    {
        private static long S(object x) => x is NDArray nd ? Convert.ToInt64(nd.GetAtIndex(0)) : Convert.ToInt64(x);
        private static string Seq(IEnumerable<object> xs) => "[" + string.Join(",", xs.Select(Convert.ToInt64)) + "]";

        [TestMethod]
        public void WriteThrough_Transposed_HitsBase()
        {
            // The defect arr.flat cannot do: a transposed array's flat write is lost via reshape-copy.
            var a = np.arange(6).reshape(2, 3);
            var t = a.T;                       // (3,2) non-contiguous
            Assert.IsFalse(t.Shape.IsContiguous);
            t.flatiter[0] = 99;                // numpy: t.flat[0]=99 -> t[0,0]=99, shares base with a
            Assert.AreEqual(99L, S(t[0, 0]));
            Assert.AreEqual(99L, S(a[0, 0]));
        }

        [TestMethod]
        public void Iterates_COrder_RegardlessOfLayout()
        {
            var a = np.arange(6).reshape(2, 3);
            Assert.AreEqual("[0,1,2,3,4,5]", Seq(a.flatiter.Cast<object>()));
            Assert.AreEqual("[0,3,1,4,2,5]", Seq(a.T.flatiter.Cast<object>()));          // transposed, logical C-order
            Assert.AreEqual("[3,4,5,0,1,2]", Seq(a["::-1"].flatiter.Cast<object>()));    // a[::-1] flips axis0
            Assert.AreEqual("[2,1,0,5,4,3]", Seq(a[":, ::-1"].flatiter.Cast<object>())); // reverse within each row
        }

        [TestMethod]
        public void Single_Get_Negative_And_Cursor()
        {
            var a = np.arange(6).reshape(2, 3);
            var f = a.flatiter;
            Assert.AreEqual(3L, S(f[3]));
            Assert.AreEqual(5L, S(f[-1]));
            Assert.AreEqual(6L, f.size);
            Assert.AreEqual(0L, f.index);
            CollectionAssert.AreEqual(new long[] { 0, 0 }, f.coords);

            Assert.AreEqual(0L, Convert.ToInt64(f.next()));
            Assert.AreEqual(1L, f.index);
            CollectionAssert.AreEqual(new long[] { 0, 1 }, f.coords);
        }

        [TestMethod]
        public void Base_Points_At_Source()
        {
            var a = np.arange(6).reshape(2, 3);
            Assert.IsTrue(ReferenceEquals(a, a.flatiter.Base));
        }

        [TestMethod]
        public void SliceSet_And_StepSet_WriteThrough()
        {
            var c = np.arange(6).reshape(2, 3);
            c.flatiter["1:4"] = np.array(new[] { 10, 20, 30 });
            CollectionAssert.AreEqual(new long[] { 0, 10, 20, 30, 4, 5 }, c.flatten().ToArray<long>());

            var d = np.arange(6).reshape(2, 3);
            d.flatiter["::2"] = np.array(new[] { -1 });   // scalar broadcast
            CollectionAssert.AreEqual(new long[] { -1, 1, -1, 3, -1, 5 }, d.flatten().ToArray<long>());
        }

        [TestMethod]
        public void FancySet_Negative_WriteThrough_And_FancyGet()
        {
            var e = np.arange(6);
            e.flatiter[new[] { -1, -2 }] = np.array(new[] { 99, 88 });
            CollectionAssert.AreEqual(new long[] { 0, 1, 2, 3, 88, 99 }, e.flatten().ToArray<long>());

            var b = np.arange(6).reshape(2, 3);
            CollectionAssert.AreEqual(new long[] { 1, 3, 5 }, b.flatiter[new[] { 1, 3, 5 }].ToArray<long>());
        }

        [TestMethod]
        public void Copy_Is_1D_COrder()
        {
            var a = np.arange(6).reshape(2, 3);
            var cp = a.flatiter.copy();
            Assert.AreEqual(1, cp.shape.Length);
            Assert.AreEqual(6, cp.shape[0]);
            CollectionAssert.AreEqual(new long[] { 0, 1, 2, 3, 4, 5 }, cp.ToArray<long>());
        }

        [TestMethod]
        public void OutOfRange_Raises_IndexError()
        {
            var a = np.arange(6).reshape(2, 3);
            var ex = Assert.ThrowsException<IndexError>(() => { var _ = a.flatiter[10]; });
            StringAssert.Contains(ex.Message, "index 10 is out of bounds for size 6");
        }

        [TestMethod]
        public void PastEnd_Coords_LeadingAxisNotWrapped()
        {
            // numpy: after exhausting a.flat, index==size and coords==(2,0) for shape (2,3) — the
            // slowest axis holds the raw remainder, it is NOT reduced mod dim[0].
            var a = np.arange(6).reshape(2, 3);
            var f = a.flatiter;
            foreach (var _ in f) { }
            Assert.AreEqual(6L, f.index);
            CollectionAssert.AreEqual(new long[] { 2, 0 }, f.coords);
        }

        [TestMethod]
        public void CrossDtype_Set_UsesNumPyCasting()
        {
            // Casting to an integer dtype must truncate toward zero and wrap on overflow (NumPy),
            // NOT round (Convert). Casting to a float dtype is exact.
            var ai = np.zeros(new Shape(3), NPTypeCode.Int32);
            ai.flatiter[0] = 3.7;    // -> 3
            ai.flatiter[1] = -3.7;   // -> -3
            Assert.AreEqual(3L, S(ai.flatiter[0]));
            Assert.AreEqual(-3L, S(ai.flatiter[1]));

            var u8 = np.zeros(new Shape(1), NPTypeCode.Byte);
            u8.flatiter[0] = 300;    // wrap -> 44
            Assert.AreEqual(44L, S(u8.flatiter[0]));

            var ad = np.zeros(new Shape(1), NPTypeCode.Double);
            ad.flatiter[0] = 5;      // int -> double exact
            Assert.AreEqual(5.0, Convert.ToDouble(ad.flatiter[0]), 1e-12);
        }

        [TestMethod]
        public void ZeroD_Flatiter()
        {
            var z = NDArray.Scalar(5L);
            Assert.AreEqual(1L, z.flatiter.size);
            Assert.AreEqual(5L, S(z.flatiter[0]));
        }

        [TestMethod]
        public void Iteration_SharesCursor_Resumes()
        {
            // numpy's iter(f) is f: a second pass resumes where the first stopped.
            var a = np.arange(4);
            var f = a.flatiter;
            f.next(); f.next();                    // consumed 0,1
            var rest = f.Cast<object>().Select(Convert.ToInt64).ToArray();
            CollectionAssert.AreEqual(new long[] { 2, 3 }, rest);
        }
    }
}
