using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.UnitTest.Backends.Iterators
{
    /// <summary>
    /// Battletest suite for NpyIter_GetGetMultiIndex factory (nditer_templ.c.src:481).
    ///
    /// NumPy generates 12 specializations over (HASINDEX × IDENTPERM × NEGPERM × BUFFER).
    /// NumSharp dispatches to 3 variants (HASINDEX and BUFFER don't affect coord logic):
    ///   1. IDENTPERM — direct copy (fast path)
    ///   2. Positive perm — apply perm[] mapping
    ///   3. NEGPERM — apply perm[] with flip decoding
    ///
    /// All expected values verified against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class NpyIterGetMultiIndexFuncTests
    {
        [TestMethod]
        public unsafe void GetMultiIndexFunc_Identity_1D()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);
            Assert.IsTrue(it.HasIdentPerm);

            var fn = it.GetMultiIndexFunc();
            Assert.IsNotNull(fn);

            Span<long> coord = stackalloc long[1];
            for (int i = 0; i < 5; i++)
            {
                it.InvokeMultiIndex(fn, coord);
                Assert.AreEqual(i, coord[0], $"at i={i}");
                it.Iternext();
            }
        }

        [TestMethod]
        public unsafe void GetMultiIndexFunc_Identity_2D()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);
            Assert.IsTrue(it.HasIdentPerm, "2D C-order should have identity perm");

            var fn = it.GetMultiIndexFunc();
            Span<long> coords = stackalloc long[2];

            var expected = new[] { (0L, 0L), (0L, 1L), (0L, 2L), (1L, 0L), (1L, 1L), (1L, 2L) };
            int i = 0;
            do
            {
                it.InvokeMultiIndex(fn, coords);
                Assert.AreEqual(expected[i].Item1, coords[0], $"coord[0] at i={i}");
                Assert.AreEqual(expected[i].Item2, coords[1], $"coord[1] at i={i}");
                i++;
            } while (it.Iternext());

            Assert.AreEqual(6, i);
        }

        [TestMethod]
        public unsafe void GetMultiIndexFunc_NegPerm_1D_Reversed()
        {
            var a = np.arange(5).astype(np.int32)["::-1"];
            using var it = NpyIterRef.New(a,
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER);

            Assert.IsTrue(it.HasNegPerm, "Reversed array under K-order should have NEGPERM");

            var fn = it.GetMultiIndexFunc();
            Span<long> coord = stackalloc long[1];

            // NumPy: iterate memory [0,1,2,3,4]; multi_index in view coords [4,3,2,1,0]
            var expected = new long[] { 4, 3, 2, 1, 0 };
            int i = 0;
            do
            {
                it.InvokeMultiIndex(fn, coord);
                Assert.AreEqual(expected[i], coord[0], $"multi_index at i={i}");
                i++;
            } while (it.Iternext());
        }

        [TestMethod]
        public unsafe void GetMultiIndexFunc_NegPerm_2D_BothReversed()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32)["::-1, ::-1"];
            using var it = NpyIterRef.New(a,
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER);

            Assert.IsTrue(it.HasNegPerm);

            var fn = it.GetMultiIndexFunc();
            Span<long> coords = stackalloc long[2];

            var expected = new[] { (1L, 2L), (1L, 1L), (1L, 0L), (0L, 2L), (0L, 1L), (0L, 0L) };
            int i = 0;
            do
            {
                it.InvokeMultiIndex(fn, coords);
                Assert.AreEqual(expected[i].Item1, coords[0], $"coord[0] at i={i}");
                Assert.AreEqual(expected[i].Item2, coords[1], $"coord[1] at i={i}");
                i++;
            } while (it.Iternext());
        }

        [TestMethod]
        public void GetMultiIndexFunc_WithoutMultiIndexFlag_ReturnsNull()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a);

            var fn = it.GetMultiIndexFunc(out string? errmsg);
            Assert.IsNull(fn);
            Assert.IsNotNull(errmsg);
            StringAssert.Contains(errmsg, "MULTI_INDEX");
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException))]
        public void GetMultiIndexFunc_WithoutMultiIndex_ThrowsOnParameterless()
        {
            var a = np.arange(5).astype(np.int32);
            using var it = NpyIterRef.New(a);
            it.GetMultiIndexFunc();
        }

        [TestMethod]
        public unsafe void GetMultiIndexFunc_AgreesWith_GetMultiIndexSpan()
        {
            var a = np.arange(12).reshape(3, 4).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            var fn = it.GetMultiIndexFunc();
            Span<long> spanCoords = stackalloc long[2];
            Span<long> fnCoords = stackalloc long[2];

            do
            {
                it.GetMultiIndex(spanCoords);
                it.InvokeMultiIndex(fn, fnCoords);
                Assert.AreEqual(spanCoords[0], fnCoords[0]);
                Assert.AreEqual(spanCoords[1], fnCoords[1]);
            } while (it.Iternext());
        }

        [TestMethod]
        public unsafe void GetMultiIndexFunc_MultiOperand()
        {
            var x = np.arange(6).reshape(2, 3).astype(np.int32);
            var y = np.zeros(new int[] { 2, 3 }, np.int32);
            using var it = NpyIterRef.MultiNew(
                nop: 2,
                op: new[] { x, y },
                flags: NpyIterGlobalFlags.MULTI_INDEX,
                order: NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_SAFE_CASTING,
                opFlags: new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });

            var fn = it.GetMultiIndexFunc();
            Span<long> coords = stackalloc long[2];

            var expectedCoords = new[] { (0L, 0L), (0L, 1L), (0L, 2L), (1L, 0L), (1L, 1L), (1L, 2L) };
            int i = 0;
            do
            {
                it.InvokeMultiIndex(fn, coords);
                Assert.AreEqual(expectedCoords[i].Item1, coords[0]);
                Assert.AreEqual(expectedCoords[i].Item2, coords[1]);
                i++;
            } while (it.Iternext());
        }

        [TestMethod]
        public unsafe void GetMultiIndexFunc_CachedDelegate_CorrectPath()
        {
            // Identity perm should dispatch to GetMultiIndex_Identity (fastest)
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);

            var fn1 = it.GetMultiIndexFunc();
            var fn2 = it.GetMultiIndexFunc();

            // The two factory calls should return delegates targeting the same method
            Assert.AreEqual(fn1.Method, fn2.Method, "Repeated factory calls should return same specialization");
        }

        [TestMethod]
        public unsafe void GetMultiIndexFunc_ArgumentValidation()
        {
            var a = np.arange(6).reshape(2, 3).astype(np.int32);
            using var it = NpyIterRef.New(a, flags: NpyIterGlobalFlags.MULTI_INDEX);
            var fn = it.GetMultiIndexFunc();

            // Span too short should throw
            Span<long> tooShort = stackalloc long[1];
            try
            {
                it.InvokeMultiIndex(fn, tooShort);
                Assert.Fail("Expected ArgumentException for too-short span");
            }
            catch (ArgumentException)
            {
                // expected
            }
        }
    }
}
