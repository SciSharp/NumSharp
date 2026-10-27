using System;
using NumSharp.Backends;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    ///     The one-pass routes of np.logical_and / or / xor and np.logical_not (<c>DefaultEngine.TryLogicalRun</c> /
    ///     <c>TryLogicalNotRun</c>, step 2 of the np-layer dtype plan's logical family): they must give the composition's
    ///     bytes, layout and result type for every call they serve, serve the calls their gates name, and decline the rest to
    ///     the composition. Held to the composition in one process through the thread-static test hooks
    ///     <c>DefaultEngine.DisableLogicalRun</c> / <c>LogicalRuns</c>.
    /// </summary>
    [TestClass]
    public class np_logical_OnePassRoute_Tests
    {
        /// <summary>Every NumSharp dtype the logical ufuncs compute (text is answered by the guard before the route).</summary>
        private static readonly DType[] Dtypes =
        {
            np.@bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64,
            np.float16, np.float32, np.float64, np.complex128, np.@decimal, NPTypeCode.Char,
        };

        /// <summary>
        ///     <paramref name="n"/> values of <paramref name="dtype"/>, zeros among them, and for the float and complex
        ///     dtypes the values whose truth is easy to get wrong: NaN (True), -0.0 (False) and a complex number with only an
        ///     imaginary part (True).
        /// </summary>
        /// <param name="dtype">The dtype.</param>
        /// <param name="n">The element count.</param>
        /// <param name="seed">Shifts the cycle of base values, so two operands differ.</param>
        /// <returns>A fresh C-contiguous 1-D array.</returns>
        private static NDArray Values(DType dtype, long n, int seed)
        {
            double[] baseValues = { 0, 1, 0, 2, 3, 0, 5, 0, 7 };
            var x = np.zeros(new Shape(n), np.float64);
            for (long i = 0; i < n; i++)
                x.SetDouble(baseValues[(i * 7 + seed) % baseValues.Length], i);
            // NaN and -0.0 before the cast, for the inexact dtypes only: an integer cast of NaN is not defined.
            bool inexact = dtype == np.float16 || dtype == np.float32 || dtype == np.float64 || dtype == np.complex128;
            if (inexact && n > 2)
            {
                x.SetDouble(double.NaN, 1);
                x.SetDouble(-0.0, 2);
            }
            var cast = x.astype(dtype);
            if (dtype == np.complex128 && n > 3)
                cast.SetAtIndex(new System.Numerics.Complex(0, 1), 3);
            return cast;
        }

        /// <summary>The raw bytes of a bool result in C order, and its type and flags, so two results can be compared exactly.</summary>
        /// <param name="r">A bool result.</param>
        /// <returns>The bytes (a bool's raw byte, not its truth) and a text of the CLR type (<c>NDArray</c> or
        ///     <c>NDArray&lt;bool&gt;</c>), dtype, shape and C / F / OWNDATA / WRITEABLE flags.</returns>
        private static (byte[] bytes, string flags) Describe(NDArray r)
        {
            var c = np.ascontiguousarray(r).view(np.uint8);
            var f = r.flags;
            return (c.ToArray<byte>(),
                $"{r.GetType().Name} {r.dtype.name} ({string.Join(",", r.shape)}) C={f.c_contiguous} F={f.f_contiguous} " +
                $"O={f.owndata} W={f.writeable}");
        }

        /// <summary>
        ///     Runs <paramref name="call"/> with the route allowed and held off, requires the same bytes and flags, and that the
        ///     route served the call exactly when <paramref name="served"/> says so.
        /// </summary>
        /// <param name="label">The case, for the failure text.</param>
        /// <param name="call">The call (it must build its operands or reuse ones it does not write).</param>
        /// <param name="served">Whether the route's gate takes this call.</param>
        private static void SameAsComposition(string label, Func<NDArray> call, bool served)
        {
            long before = DefaultEngine.LogicalRuns;
            var routed = Describe(call());
            Assert.AreEqual(served ? before + 1 : before, DefaultEngine.LogicalRuns, $"{label}: route served");
            DefaultEngine.DisableLogicalRun = true;
            try
            {
                var composed = Describe(call());
                Assert.AreEqual(composed.flags, routed.flags, $"{label}: flags");
                CollectionAssert.AreEqual(composed.bytes, routed.bytes, $"{label}: bytes");
            }
            finally
            {
                DefaultEngine.DisableLogicalRun = false;
            }
        }

        [TestMethod]
        public void EveryDtypePair_AndLayout_GivesTheCompositionsBytes()
        {
            var ops = new (string name, Func<NDArray, NDArray, NDArray> f)[]
            {
                ("and", (a, b) => np.logical_and(a, b)),
                ("or", (a, b) => np.logical_or(a, b)),
                ("xor", (a, b) => np.logical_xor(a, b)),
            };
            foreach (var da in Dtypes)
            foreach (var db in Dtypes)
            {
                // Two bool operands are already one bitwise pass: the route leaves them to the composition.
                bool served = !(da == np.@bool && db == np.@bool);
                foreach (var (name, f) in ops)
                {
                    string at = $"{name} {da.name} {db.name}";
                    // C 2-D, F 2-D, a 1-D contiguous slice at an offset, and 10,000 elements (two whole blocks and a tail).
                    var a = Values(da, 15, 0).reshape(3, 5);
                    var b = Values(db, 15, 3).reshape(3, 5);
                    SameAsComposition($"{at} C", () => f(a, b), served);
                    var fa = np.asfortranarray(a);
                    var fb = np.asfortranarray(b);
                    SameAsComposition($"{at} F", () => f(fa, fb), served);
                    var sa = Values(da, 20, 1)["4:"];
                    var sb = Values(db, 20, 5)["4:"];
                    SameAsComposition($"{at} offset", () => f(sa, sb), served);
                    var la = Values(da, 10_000, 2);
                    var lb = Values(db, 10_000, 4);
                    SameAsComposition($"{at} blocks", () => f(la, lb), served);
                    // An operand of one value, on either side, of a rank the other's covers.
                    var oneB = Values(db, 1, 3).reshape(1, 1);
                    SameAsComposition($"{at} one right", () => f(a, oneB), served);
                    var oneA = Values(da, 1, 1).reshape(Shape.Scalar);
                    SameAsComposition($"{at} one left 0-d", () => f(oneA, fb), served);
                }
            }
        }

        [TestMethod]
        public void BoolOperand_RawBytes_CombineAsTheCompositionCombinesThem()
        {
            // A bool array whose bytes are not 0 / 1 (a uint8 array viewed as bool): the composition passes a bool operand
            // through as it is, so the route must combine the same raw bytes, whole arrays and a single value alike.
            var raw = np.array(new byte[] { 0, 1, 2, 255, 0, 3, 128, 1, 0, 4 }).view(np.@bool);
            var x = Values(np.float64, 10, 1);
            SameAsComposition("raw and", () => np.logical_and(raw, x), true);
            SameAsComposition("raw or", () => np.logical_or(x, raw), true);
            SameAsComposition("raw xor", () => np.logical_xor(raw, x), true);
            SameAsComposition("raw one", () => np.logical_xor(raw["3:4"], x), true);
        }

        [TestMethod]
        public void Declined_Calls_StillGiveTheCompositionsBytes()
        {
            var a = Values(np.float64, 15, 0).reshape(3, 5);
            var b = Values(np.int32, 15, 3).reshape(3, 5);
            // Mixed memory orders, a reversed or stepped view, a broadcast operand, a row, a one-element operand of higher
            // rank, two single values, an empty result and two bool operands: each is the composition's.
            SameAsComposition("C and F", () => np.logical_and(a, np.asfortranarray(b)), false);
            SameAsComposition("reversed", () => np.logical_and(a["::-1"], b), false);
            SameAsComposition("stepped", () => np.logical_and(Values(np.float64, 30, 0).reshape(3, 10)[":, ::2"], b), false);
            SameAsComposition("broadcast", () => np.logical_and(a, np.broadcast_to(Values(np.int32, 5, 3), new Shape(3, 5))), false);
            SameAsComposition("row", () => np.logical_and(a, Values(np.int32, 5, 3)), false);
            SameAsComposition("rank-raising one",
                () => np.logical_and(Values(np.float64, 5, 0), Values(np.int32, 1, 0).reshape(1, 1)), false);
            SameAsComposition("two singles", () => np.logical_and(Values(np.float64, 1, 1), Values(np.int32, 1, 0)), false);
            SameAsComposition("empty", () => np.logical_and(np.zeros(new Shape(0, 3)), np.zeros(new Shape(0, 3), np.int32)), false);
            SameAsComposition("bool bool", () => np.logical_and(a > 2, b > 2), false);
        }

        [TestMethod]
        public void LogicalNot_GivesTheCompositionsBytes()
        {
            foreach (var dt in Dtypes)
            {
                // A bool operand takes the invert ufunc, not the route.
                bool served = dt != np.@bool;
                var a = Values(dt, 15, 0).reshape(3, 5);
                SameAsComposition($"not {dt.name} C", () => np.logical_not(a), served);
                var fa = np.asfortranarray(a);
                SameAsComposition($"not {dt.name} F", () => np.logical_not(fa), served);
                var sa = Values(dt, 20, 1)["4:"];
                SameAsComposition($"not {dt.name} offset", () => np.logical_not(sa), served);
                var la = Values(dt, 10_000, 2);
                SameAsComposition($"not {dt.name} long", () => np.logical_not(la), served);
            }
            var x = Values(np.float64, 15, 0).reshape(3, 5);
            // Declined: a reversed or broadcast operand, an empty or 0-d operand.
            SameAsComposition("not reversed", () => np.logical_not(x["::-1"]), false);
            SameAsComposition("not broadcast", () => np.logical_not(np.broadcast_to(Values(np.int32, 5, 3), new Shape(3, 5))), false);
            SameAsComposition("not empty", () => np.logical_not(np.zeros(new Shape(0, 3))), false);
            SameAsComposition("not 0-d", () => np.logical_not(Values(np.float64, 1, 1).reshape(Shape.Scalar)), false);

            // out=: a bool out of the operand's layout is served and returned itself; a C out for an F operand, an int8 out
            // and an out over the operand's own bytes are the composition's.
            var expected = Describe(np.logical_not(x));
            var o = np.zeros(new Shape(3, 5), np.@bool);
            long before = DefaultEngine.LogicalRuns;
            Assert.IsTrue(ReferenceEquals(o, np.logical_not(x, o)));
            Assert.AreEqual(before + 1, DefaultEngine.LogicalRuns, "not: C out served");
            CollectionAssert.AreEqual(expected.bytes, Describe(o).bytes);
            var bytesArray = np.array(new byte[] { 0, 1, 2, 0, 3 });
            var overlapExpected = Describe(np.logical_not(np.array(new byte[] { 0, 1, 2, 0, 3 })));
            before = DefaultEngine.LogicalRuns;
            var co = np.zeros(new Shape(3, 5), np.@bool);
            np.logical_not(np.asfortranarray(x), co);
            var io = np.zeros(new Shape(3, 5), np.int8);
            np.logical_not(x, io);
            np.logical_not(bytesArray, bytesArray.view(np.@bool));
            Assert.AreEqual(before, DefaultEngine.LogicalRuns, "not: declined outs");
            CollectionAssert.AreEqual(expected.bytes, Describe(co).bytes);
            CollectionAssert.AreEqual(expected.bytes, Describe(io.astype(np.@bool)).bytes);
            CollectionAssert.AreEqual(overlapExpected.bytes, Describe(bytesArray.view(np.@bool)).bytes);
        }

        [TestMethod]
        public void Out_IsWrittenInPlace_WhenItIsABoolOfTheRunsLayout()
        {
            var a = Values(np.float64, 15, 0).reshape(3, 5);
            var b = Values(np.int32, 15, 3).reshape(3, 5);
            var expected = Describe(np.logical_or(a, b));

            // A C bool out of the operands' shape: served, written in place, returned itself.
            var o = np.zeros(new Shape(3, 5), np.@bool);
            long before = DefaultEngine.LogicalRuns;
            Assert.IsTrue(ReferenceEquals(o, np.logical_or(a, b, o)));
            Assert.AreEqual(before + 1, DefaultEngine.LogicalRuns, "C out served");
            CollectionAssert.AreEqual(expected.bytes, Describe(o).bytes);

            // An F operand set writes an F out.
            var fo = np.asfortranarray(np.zeros(new Shape(3, 5), np.@bool));
            before = DefaultEngine.LogicalRuns;
            np.logical_or(np.asfortranarray(a), np.asfortranarray(b), fo);
            Assert.AreEqual(before + 1, DefaultEngine.LogicalRuns, "F out served");
            CollectionAssert.AreEqual(expected.bytes, Describe(fo).bytes);

            // A C out for F operands (a run order the out does not have), an int8 out (a cast) and an out that is an operand
            // (an overlap) are the composition's.
            var self = a > 1;
            var selfExpected = Describe(np.logical_or(a > 1, b));
            before = DefaultEngine.LogicalRuns;
            var co = np.zeros(new Shape(3, 5), np.@bool);
            np.logical_or(np.asfortranarray(a), np.asfortranarray(b), co);
            var io = np.zeros(new Shape(3, 5), np.int8);
            np.logical_or(a, b, io);
            np.logical_or(self, b, self);
            Assert.AreEqual(before, DefaultEngine.LogicalRuns, "declined outs");
            CollectionAssert.AreEqual(expected.bytes, Describe(co).bytes);
            CollectionAssert.AreEqual(expected.bytes, Describe(io.astype(np.@bool)).bytes);
            CollectionAssert.AreEqual(selfExpected.bytes, Describe(self).bytes);
        }
    }
}
