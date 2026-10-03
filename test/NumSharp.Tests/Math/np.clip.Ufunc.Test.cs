using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.clip as the ufunc call NumPy routes it to (<c>um.clip</c>, or <c>minimum</c> / <c>maximum</c> /
    ///     <c>positive</c> after <c>_methods._clip</c>): the array, the bounds, <c>out=</c> and <c>where=</c> broadcast
    ///     together; an out overlapping an input is computed as if from a copy; the float loop (const vs general)
    ///     follows NumPy's own per-call decision; and fromnumeric.clip's keyword rule. Every expectation below was
    ///     probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class np_clip_ufunc_tests
    {
        /// <summary>The 12-element buffer the overlap cases view twice: [0, 5, 0, 5, …].</summary>
        private static NDArray Zigzag() => np.array(new[] { 0.0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5 });

        /// <summary>The sign pattern of a float result in C order: '-' for -0.0, '+' for +0.0, '.' otherwise.</summary>
        /// <param name="r">The result.</param>
        /// <returns>One character per element.</returns>
        private static string Signs(NDArray r)
        {
            var f = r.flatten();
            var sb = new System.Text.StringBuilder();
            for (long i = 0; i < f.size; i++)
            {
                double v = Convert.ToDouble(f.GetAtIndex(i));
                sb.Append(v == 0 ? (double.IsNegative(v) ? '-' : '+') : '.');
            }
            return sb.ToString();
        }

        /// <summary>The raw bits of a float64 result in C order.</summary>
        /// <param name="r">The result.</param>
        /// <returns>The bit patterns.</returns>
        private static ulong[] Bits(NDArray r) => r.flatten().ToArray<double>().Select(BitConverter.DoubleToUInt64Bits).ToArray();

        /// <summary>A float64 array of <paramref name="dims"/> holding -0.0.</summary>
        /// <param name="dims">The shape.</param>
        /// <returns>The array.</returns>
        private static NDArray NegZeros(params long[] dims) => np.full(new Shape(dims), -0.0);

        // =====================================================================================================
        // fromnumeric.clip's keyword rule
        // =====================================================================================================

        /// <summary>min= / max= alone are the bounds (NumPy 2.x aliases).</summary>
        [TestMethod]
        public void Keywords_AloneAreTheBounds()
        {
            var a = np.arange(10);
            np.clip(a, min: 3).ToArray<long>().Should().Equal(3, 3, 3, 3, 4, 5, 6, 7, 8, 9);
            np.clip(a, min: 3, max: 5).ToArray<long>().Should().Equal(3, 3, 3, 3, 4, 5, 5, 5, 5, 5);
            np.clip(a, max: 5).ToArray<long>().Should().Equal(0, 1, 2, 3, 4, 5, 5, 5, 5, 5);
        }

        /// <summary>A keyword next to both positional bounds: NumPy's ValueError, either keyword.</summary>
        [TestMethod]
        public void Keywords_WithBothPositionals_ValueError()
        {
            var a = np.arange(10);
            const string text = "Passing `min` or `max` keyword argument when `a_min` and `a_max` are provided is forbidden.";
            Assert.ThrowsExactly<ValueError>(() => np.clip(a, 2, 5, min: 3)).Message.Should().Be(text);
            Assert.ThrowsExactly<ValueError>(() => np.clip(a, 2, 5, max: 3)).Message.Should().Be(text);
        }

        /// <summary>A keyword with only one positional: NumPy's TypeError naming the missing positional.</summary>
        [TestMethod]
        public void Keywords_WithOnePositional_TypeError()
        {
            var a = np.arange(10);
            Assert.ThrowsExactly<TypeError>(() => np.clip(a, 2, max: 3)).Message
                .Should().Be("clip() missing 1 required positional argument: 'a_max'");
            Assert.ThrowsExactly<TypeError>(() => np.clip(a, a_max: 2, min: 3)).Message
                .Should().Be("clip() missing 1 required positional argument: 'a_min'");
        }

        /// <summary>The keyword rule runs in Python before the ufunc, so it wins over a read-only out.</summary>
        [TestMethod]
        public void Keywords_RuleWinsOverReadOnlyOut()
        {
            var ro = np.broadcast_to(NDArray.Scalar(0L), new Shape(10));
            Assert.ThrowsExactly<ValueError>(() => np.clip(np.arange(10), 2, 5, ro, min: 3)).Message
                .Should().StartWith("Passing `min` or `max`");
        }

        /// <summary>
        ///     C# cannot tell an omitted argument from an explicit null, so a null bound is NumPy's None unless a keyword
        ///     is used: NumPy's <c>np.clip(a, 1)</c> raises "missing 1 required positional argument: 'a_max'", while
        ///     NumSharp clips from below (as <c>np.clip(a, 1, None)</c> does); and NumPy's
        ///     <c>np.clip(a, 2, None, min=3)</c> raises the ValueError, while NumSharp reports the missing a_max.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Keywords_NullIsNone_OmittedIsNotExpressible()
        {
            var a = np.arange(5);
            np.clip(a, 1).ToArray<long>().Should().Equal(1, 1, 2, 3, 4);
            Assert.ThrowsExactly<TypeError>(() => np.clip(a, 2, null, min: 3));
        }

        // =====================================================================================================
        // Broadcasting: array bounds, out larger than the inputs, NumPy's two iterator texts
        // =====================================================================================================

        /// <summary>The array and both bounds broadcast together, up to a shape none of them has.</summary>
        [TestMethod]
        public void Broadcast_BoundsWiderThanTheArray()
        {
            // NumPy: np.clip(arange(3.), zeros((2,1)), ones((4,1,1))) -> (4,2,3) of [0,1,1] rows.
            var r = np.clip(np.arange(3.0), np.zeros(new Shape(2, 1)), np.ones(new Shape(4, 1, 1)));
            r.Should().BeShaped(4, 2, 3);
            r.ToArray<double>().Should().Equal(Enumerable.Repeat(new[] { 0.0, 1, 1 }, 8).SelectMany(x => x));
            r.Shape.IsContiguous.Should().BeTrue();
        }

        /// <summary>A bound row against a 2-D array, and the routed one-sided forms.</summary>
        [TestMethod]
        public void Broadcast_RowAndColumnBounds()
        {
            var m = np.arange(12.0).reshape(3, 4);
            np.clip(m, np.array(new[] { 2.0, 0, 5, 0 }), 6.0).ToArray<double>()
              .Should().Equal(2, 1, 5, 3, 4, 5, 6, 6, 6, 6, 6, 6);
            np.clip(m, null, np.array(new[,] { { 1.0 }, { 5 }, { 9 } })).ToArray<double>()
              .Should().Equal(0, 1, 1, 1, 4, 5, 5, 5, 8, 9, 9, 9);
            np.clip(m, np.array(new[,] { { 1.0 }, { 5 }, { 9 } }), null).ToArray<double>()
              .Should().Equal(1, 1, 2, 3, 5, 5, 6, 7, 9, 9, 10, 11);
        }

        /// <summary>The inputs broadcast UP into an out larger than all of them (positive repeats the row).</summary>
        [TestMethod]
        public void Broadcast_IntoALargerOut()
        {
            var o = np.zeros(new Shape(2, 3));
            np.clip(np.arange(3.0), null, null, o).Should().BeSameAs(o);
            o.ToArray<double>().Should().Equal(0, 1, 2, 0, 1, 2);

            var o2 = np.zeros(new Shape(2, 3));
            np.clip(np.arange(3.0), 1, 1.5, o2);
            o2.ToArray<double>().Should().Equal(1, 1, 1.5, 1, 1, 1.5);
        }

        /// <summary>NumPy's iterator texts, every operand listed (0-d bounds as <c>()</c>) with a trailing space.</summary>
        [TestMethod]
        public void Broadcast_ErrorTexts()
        {
            var a3 = np.arange(3.0);
            Assert.ThrowsExactly<ValueError>(() => np.clip(a3, np.zeros(new Shape(4)), 1)).Message
                .Should().Be("operands could not be broadcast together with shapes (3,) (4,) () ");
            Assert.ThrowsExactly<ValueError>(() => np.clip(a3, np.zeros(new Shape(4)), 1, np.empty(new Shape(3)))).Message
                .Should().Be("operands could not be broadcast together with shapes (3,) (4,) () (3,) ");
            Assert.ThrowsExactly<ValueError>(() => np.clip(a3, np.zeros(new Shape(4)), null)).Message
                .Should().Be("operands could not be broadcast together with shapes (3,) (4,) ");
            Assert.ThrowsExactly<ValueError>(() => np.clip(a3, null, null, np.empty(new Shape(2)))).Message
                .Should().Be("operands could not be broadcast together with shapes (3,) (2,) ");
            Assert.ThrowsExactly<ValueError>(() => np.clip(a3, 0, 1, where: np.ones(new Shape(4), np.@bool))).Message
                .Should().Be("operands could not be broadcast together with shapes (3,) () () (4,) ");
            Assert.ThrowsExactly<ValueError>(() => np.clip(a3, np.zeros(new Shape(4)), 1, np.empty(new Shape(3)),
                                                           where: np.ones(new Shape(5), np.@bool))).Message
                .Should().Be("operands could not be broadcast together with shapes (3,) (4,) () (3,) (5,) ");
            Assert.ThrowsExactly<ValueError>(() => np.clip(np.zeros(new Shape(2, 3)), 0, 1, np.empty(new Shape(3)))).Message
                .Should().Be("non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (2,3)");
            Assert.ThrowsExactly<ValueError>(() => np.clip(a3, np.zeros(new Shape(2, 1)), 1, np.empty(new Shape(3)))).Message
                .Should().Be("non-broadcastable output operand with shape (3,) doesn't match the broadcast shape (2,3)");
        }

        // =====================================================================================================
        // where=
        // =====================================================================================================

        /// <summary>Only the mask's True positions are written; the others keep out's prior contents.</summary>
        [TestMethod]
        public void Where_KeepsPriorContents()
        {
            var a3 = np.arange(3.0);

            var o1 = np.full(new Shape(3), 9.0);
            np.clip(a3, 1, 1, o1, where: false);
            o1.ToArray<double>().Should().Equal(9, 9, 9);

            var o2 = np.full(new Shape(3), 9.0);
            np.clip(a3, 1, 1, o2, where: NDArray.Scalar(true));
            o2.ToArray<double>().Should().Equal(1, 1, 1);

            var o3 = np.full(new Shape(2, 3), 9.0);
            np.clip(a3, 1, 1, o3, where: np.array(new[,] { { true }, { false } }));
            o3.ToArray<double>().Should().Equal(1, 1, 1, 9, 9, 9);

            // The maximum route and a dtype= loop under a mask (NumPy: [9, 1, 2] and [0.5, 9, 1.5]).
            var o4 = np.full(new Shape(3), 9.0);
            np.clip(a3, 1, null, o4, where: np.array(new[] { false, true, true }));
            o4.ToArray<double>().Should().Equal(9, 1, 2);

            var o5 = np.full(new Shape(3), 9.0);
            np.clip(a3, 0.5, 1.5, o5, np.float32, where: np.array(new[] { true, false, true }));
            o5.ToArray<double>().Should().Equal(0.5, 9, 1.5);

            // An int8 clip cast into a float64 out under a mask: [1, 9, 0].
            var o6 = np.full(new Shape(3), 9.0);
            np.clip(np.array(new sbyte[] { 1, 100, -100 }), 0, 5, o6, where: np.array(new[] { true, false, true }));
            o6.ToArray<double>().Should().Equal(1, 9, 0);
        }

        /// <summary>A mask without out: the fresh result takes the operands' shape (the mask votes).</summary>
        [TestMethod]
        public void Where_WithoutOut_ShapeFromAllOperands()
        {
            np.clip(np.arange(3.0), 0, 1, where: np.ones(new Shape(2, 3), np.@bool)).Should().BeShaped(2, 3);
            np.clip(np.arange(3.0), null, null, where: np.ones(new Shape(2, 3), np.@bool)).Should().BeShaped(2, 3);
        }

        /// <summary>
        ///     A non-bool mask is refused with NumPy's 'safe'-cast text (the house ArgumentException), AFTER a read-only
        ///     out and BEFORE the Python-int OverflowError, the dtype= casts and the broadcast check.
        /// </summary>
        [TestMethod]
        public void Where_NonBoolMask_ErrorOrder()
        {
            const string text = "Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'";
            var i8 = np.array(new sbyte[] { 1, 100, -100 });
            var bad = np.array(new long[] { 1, 0, 1 });
            var ro = np.broadcast_to(NDArray.Scalar((sbyte)0), new Shape(3));

            Assert.ThrowsExactly<ValueError>(() => np.clip(i8, 0, 5, ro, where: bad)).Message.Should().Be("output array is read-only");
            Assert.ThrowsExactly<ArgumentException>(() => np.clip(i8, 1000, 2000, where: bad)).Message.Should().Be(text);
            Assert.ThrowsExactly<ArgumentException>(() => np.clip(np.arange(3.0), 0, 1, dtype: np.int8, where: bad)).Message.Should().Be(text);
            Assert.ThrowsExactly<ArgumentException>(() => np.clip(np.arange(3.0), np.zeros(new Shape(4)), 1, where: bad)).Message.Should().Be(text);
        }

        /// <summary>No bound left on a bool array is positive, which has no bool loop — with or without a mask.</summary>
        [TestMethod]
        public void Where_PositiveRoute_BoolHasNoLoop()
        {
            var b = np.array(new[] { true, false });
            Assert.ThrowsExactly<TypeError>(() => np.clip(b, null, null, where: np.array(new[] { true, false })));
            Assert.ThrowsExactly<TypeError>(() => np.clip(b, null, null, np.zeros(new Shape(2), np.@bool), where: np.array(new[] { true, false })));
        }

        // =====================================================================================================
        // out overlapping an input: COPY_IF_OVERLAP
        // =====================================================================================================

        /// <summary>Every overlap pattern computes as if from a copy of the input (NumPy 2.4.2's values).</summary>
        [TestMethod]
        public void Overlap_AsIfCopied()
        {
            NDArray Run(Action<NDArray> f)
            {
                var x = Zigzag();
                f(x);
                return x;
            }

            Run(x => np.clip(x[":-1"], 1, 4, x["1:"])).ToArray<double>().Should().Equal(0, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1);
            Run(x => np.clip(x["1:"], 1, 4, x[":-1"])).ToArray<double>().Should().Equal(4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 5);
            Run(x => np.clip(x["::-1"], 1, 4, x)).ToArray<double>().Should().Equal(4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1);
            Run(x => np.clip(x, 1, 4, x)).ToArray<double>().Should().Equal(1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4);
            Run(x => np.clip(x[":-1"], null, null, x["1:"])).ToArray<double>().Should().Equal(0, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0);
            Run(x => np.clip(x[":-1"], 1, null, x["1:"])).ToArray<double>().Should().Equal(0, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1);
            Run(x => np.clip(x["1:"], x[":-1"], 10, x["1:"])).ToArray<double>().Should().Equal(0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5);
            Run(x => np.clip(x["1:"], -10, x[":-1"], x["1:"])).ToArray<double>().Should().Equal(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            Run(x => np.clip(x.reshape(3, 4).T, 1, 4, x.reshape(4, 3))).ToArray<double>().Should().Equal(1, 1, 1, 4, 4, 4, 1, 1, 1, 4, 4, 4);
            Run(x => np.clip(x[":3"], 1, 4, x.reshape(4, 3))).ToArray<double>().Should().Equal(1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 4, 1);
        }

        /// <summary>
        ///     Overlap under a mask: NumPy flags a masked caller out READWRITE, so the copy it computes into starts from
        ///     out's values and the masked-off slots come back unchanged. They came back as heap bytes before (6.95e-310).
        /// </summary>
        [TestMethod]
        public void Overlap_WithMask_KeepsMaskedOffValues()
        {
            var x = Zigzag();
            np.clip(x[":-1"], 1, 4, x["1:"], where: np.arange(11) % 3 != 0);
            x.ToArray<double>().Should().Equal(0, 5, 4, 1, 0, 1, 4, 5, 4, 1, 0, 1);

            // An int64 input cast into the float64 out it was copied from.
            var y = np.array(new[] { 0.0, 5, 0, 5, 0, 5 });
            np.clip(y[":-1"].astype(np.int64), 1, 4, y["1:"]);
            y.ToArray<double>().Should().Equal(0, 1, 4, 1, 4, 1);
        }

        /// <summary>
        ///     The same READWRITE rule in the shared ufunc routes np.clip routes to (maximum / minimum / positive) and in
        ///     their siblings: the masked-off slots of an out overlapping its input kept heap bytes there too.
        /// </summary>
        [TestMethod]
        public void Overlap_WithMask_RoutedUfuncs()
        {
            NDArray Run(Action<NDArray> f)
            {
                var x = Zigzag();
                f(x);
                return x;
            }

            var m = np.arange(11) % 3 != 0;
            Run(x => np.add(x[":-1"], 1, x["1:"], m)).ToArray<double>().Should().Equal(0, 5, 6, 1, 0, 1, 6, 5, 6, 1, 0, 1);
            Run(x => np.add(x[":-1"], 1, @out: x["1:"], where: m, dtype: np.float32)).ToArray<double>().Should().Equal(0, 5, 6, 1, 0, 1, 6, 5, 6, 1, 0, 1);
            Run(x => np.maximum(x[":-1"], 1, x["1:"], m)).ToArray<double>().Should().Equal(0, 5, 5, 1, 0, 1, 5, 5, 5, 1, 0, 1);
            Run(x => np.minimum(x[":-1"], 4, x["1:"], m)).ToArray<double>().Should().Equal(0, 5, 4, 0, 0, 0, 4, 5, 4, 0, 0, 0);
            Run(x => np.positive(x[":-1"], x["1:"], m)).ToArray<double>().Should().Equal(0, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0);
            Run(x => np.sqrt(x[":-1"], x["1:"], m)).ToArray<double>()
                .Should().Equal(0, 5, System.Math.Sqrt(5), 0, 0, 0, System.Math.Sqrt(5), 5, System.Math.Sqrt(5), 0, 0, 0);
            Run(x => np.evaluate((NDExpr)x[":-1"] + 1, x["1:"], m)).ToArray<double>().Should().Equal(0, 5, 6, 1, 0, 1, 6, 5, 6, 1, 0, 1);
        }

        /// <summary>
        ///     NumPy computes a masked, overlapping out of ANOTHER dtype through a temporary of the loop dtype, so its
        ///     masked-off values round-trip through that dtype: np.less (a bool loop) into a float64 out turns a kept 5.0
        ///     into 1.0 ("best-effort", ufunc_object.c). NumSharp's temporary has out's own dtype and keeps 5.0.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Overlap_WithMaskAndCast_MaskedOffValuesKeptExactly()
        {
            var x = Zigzag();
            np.less(x[":-1"], 3, x["1:"], np.arange(11) % 3 != 0);
            // NumPy 2.4.2: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            x.ToArray<double>().Should().Equal(0, 5, 0, 1, 0, 1, 0, 5, 0, 1, 0, 1);
        }

        // =====================================================================================================
        // The float loop NumPy runs: const (a signed-zero tie keeps the element) vs general (takes the bound)
        // =====================================================================================================

        /// <summary>
        ///     -0.0 clipped by +0.0 bounds: the sign of each result tells which loop ran. NumPy's choice is its own per-call
        ///     decision (trivial loop, NpyIter axis order, buffering), reproduced case by case.
        /// </summary>
        [TestMethod]
        public void FloatLoop_ConstOrGeneral_FollowsNumPy()
        {
            Signs(np.clip(NegZeros(4), np.zeros(new Shape(1), np.float32), 1.0)).Should().Be("----");
            Signs(np.clip(NegZeros(2, 3), np.zeros(new Shape(1, 1)), 1.0)).Should().Be("------");
            Signs(np.clip(NegZeros(6)["::2"], np.zeros(new Shape(1)), 1.0)).Should().Be("---");
            Signs(np.clip(NegZeros(1, 1), np.zeros(new Shape(1)), 1.0)).Should().Be("-");
            Signs(np.clip(NegZeros(1), np.zeros(new Shape(1)), 1.0)).Should().Be("+");
            Signs(np.clip(NegZeros(4), np.zeros(new Shape(1)), np.ones(new Shape(1)))).Should().Be("----");
            Signs(np.clip(NegZeros(4), np.zeros(new Shape(1)), np.ones(new Shape(4)))).Should().Be("++++");
            Signs(np.clip(NegZeros(10000), np.zeros(new Shape(1)), 1.0)[":4"]).Should().Be("----");
            Signs(np.clip(NegZeros(2, 10000), np.zeros(new Shape(2, 1)), 1.0)[":, :2"]).Should().Be("----");
            Signs(np.clip(np.asfortranarray(NegZeros(2, 3)), np.zeros(new Shape(1, 1)), 1.0)).Should().Be("------");
            Signs(np.clip(NegZeros(4, 3)["::2"], np.zeros(new Shape(1, 1)), 1.0)).Should().Be("------");
            Signs(np.clip(NegZeros(4), np.zeros(new Shape(1)), 1.0, np.empty(new Shape(4)))).Should().Be("----");
            Signs(np.clip(np.full(new Shape(4), -0.0f), np.zeros(new Shape(1), np.float32), 1.0)).Should().Be("----");
            Signs(np.clip(NegZeros(4), np.zeros(new Shape(1), np.float32), np.ones(new Shape(1), np.float32))).Should().Be("----");
            Signs(np.clip(NegZeros(2, 3), np.zeros(new Shape(2, 1)), np.ones(new Shape(2, 1)))).Should().Be("------");
            Signs(np.clip(NegZeros(3), np.zeros(new Shape(2, 1)), 1.0)).Should().Be("------");
            Signs(np.clip(NDArray.Scalar(-0.0), np.zeros(new Shape(1)), 1.0)).Should().Be("+");
        }

        /// <summary>
        ///     One-element calls NumPy keeps OFF its trivial loop take the const loop: NpyIter gives every extent-1 axis
        ///     stride 0. In place (a one-element input is never "ahead" of its output) and a 2-D bound needing a cast.
        /// </summary>
        [TestMethod]
        public void FloatLoop_OneElement_OffTheTrivialLoop_IsConst()
        {
            var a = NegZeros(1);
            np.clip(a, 0.0, np.ones(new Shape(2))["::-1"][":1"], a);
            Signs(a).Should().Be("-");

            var b = NegZeros(1);
            Signs(np.clip(b, 0.0, np.ones(new Shape(2))["::-1"][":1"])).Should().Be("+");

            Signs(np.clip(np.full(new Shape(1, 1), -0.0f), np.zeros(new Shape(1, 1), np.float16), 1.0)).Should().Be("-");
            Signs(np.clip(NegZeros(1, 1), np.zeros(new Shape(1, 1), np.float32), np.ones(new Shape(1, 1), np.float32))).Should().Be("-");
        }

        /// <summary>
        ///     The const loop fills a call with a NaN bound's OWN bits (the lower one wins), even over a NaN element; the
        ///     general loop and the routed maximum / minimum keep a NaN element's bits.
        /// </summary>
        [TestMethod]
        public void FloatLoop_NaNBits()
        {
            double negPayload = BitConverter.UInt64BitsToDouble(0xFFF8000000000123UL);
            double posPayload = BitConverter.UInt64BitsToDouble(0x7FF8000000000456UL);
            double elemNaN = BitConverter.UInt64BitsToDouble(0xFFF8000000000AAAUL);
            double boundNaN = BitConverter.UInt64BitsToDouble(0x7FF8000000000BBBUL);
            var two = np.array(new[] { 1.0, 2.0 });

            Bits(np.clip(two, negPayload, 3.0)).Should().Equal(0xFFF8000000000123UL, 0xFFF8000000000123UL);
            Bits(np.clip(two, 0.0, posPayload)).Should().Equal(0x7FF8000000000456UL, 0x7FF8000000000456UL);
            Bits(np.clip(two, negPayload, posPayload)).Should().Equal(0xFFF8000000000123UL, 0xFFF8000000000123UL);
            Bits(np.clip(np.array(new[] { elemNaN, 1.0 }), boundNaN, 3.0)).Should().Equal(0x7FF8000000000BBBUL, 0x7FF8000000000BBBUL);
            Bits(np.clip(np.array(new[] { elemNaN, 2.0 }), 0.0, 3.0)).Should().Equal(0xFFF8000000000AAAUL, 0x4000000000000000UL);
            Bits(np.clip(np.array(new[] { elemNaN, 1.0 }), np.array(new[] { boundNaN, boundNaN }), 3.0))
                .Should().Equal(0xFFF8000000000AAAUL, 0x7FF8000000000BBBUL);
            Bits(np.clip(np.array(new[] { elemNaN, 1.0, 2.0 }), boundNaN, null))
                .Should().Equal(0xFFF8000000000AAAUL, 0x7FF8000000000BBBUL, 0x7FF8000000000BBBUL);
            Bits(np.clip(np.array(new[,] { { elemNaN, 1.0 }, { 2.0, 3.0 } }), np.array(new[] { boundNaN, 0.0 }), 3.0))
                .Should().Equal(0xFFF8000000000AAAUL, 0x3FF0000000000000UL, 0x7FF8000000000BBBUL, 0x4008000000000000UL);
            var f32 = np.clip(np.array(new[] { 1.0f, 2.0f }), NDArray.Scalar(BitConverter.UInt32BitsToSingle(0x7FC00000u)), 3.0);
            f32.ToArray<float>().Select(BitConverter.SingleToUInt32Bits).Should().Equal(0x7FC00000u, 0x7FC00000u);
        }

        // =====================================================================================================
        // Result layout: NumPy's ufunc output layout (the operands' memory order; C wins ties; the mask votes)
        // =====================================================================================================

        /// <summary>A fresh result takes the operands' memory order, as NumPy's iterator allocates it.</summary>
        [TestMethod]
        public void Layout_FollowsTheOperands()
        {
            var f23 = np.asfortranarray(np.zeros(new Shape(2, 3)));
            var a3 = np.arange(3.0);

            np.clip(a3, f23, 1).Shape.IsFContiguous.Should().BeTrue();
            np.clip(f23, 0, 1, where: np.ones(new Shape(2, 3), np.@bool)).Shape.IsContiguous.Should().BeTrue();
            np.clip(np.zeros(new Shape(2, 3)), 0, 1, where: np.asfortranarray(np.ones(new Shape(2, 3), np.@bool))).Shape.IsContiguous.Should().BeTrue();
            np.clip(a3, 0, 1, where: np.asfortranarray(np.ones(new Shape(2, 3), np.@bool))).Shape.IsFContiguous.Should().BeTrue();
            np.clip(f23, 0, null).Shape.IsFContiguous.Should().BeTrue();
            np.clip(f23, np.zeros(new Shape(2, 3)), 1).Shape.IsContiguous.Should().BeTrue();
            np.clip(np.zeros(new Shape(2, 3)), f23, 1).Shape.IsContiguous.Should().BeTrue();
            np.clip(f23, null, null).Shape.IsFContiguous.Should().BeTrue();
            np.clip(a3, null, null, where: np.asfortranarray(np.ones(new Shape(2, 3), np.@bool))).Shape.IsFContiguous.Should().BeTrue();

            var t3 = np.arange(24.0).reshape(2, 3, 4).transpose(new[] { 1, 0, 2 });
            var p = np.clip(t3, np.zeros(new Shape(4)), 1);
            p.strides.Should().Equal(4L * 8, 12L * 8, 1L * 8);
            np.clip(t3, 0, 1).strides.Should().Equal(4L * 8, 12L * 8, 1L * 8);
        }
    }
}
