using System;
using System.Linq;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    ///     The result layout of <c>np.any</c> / <c>np.all</c> / <c>np.count_nonzero</c> along an axis (plan api-parameter-parity
    ///     X10). NumPy allocates a reduction's result with <c>npyiter_new_temp_array</c> over the reduce iterator: the input's
    ///     axes in NpyIter's order with the reduced ones dropped — so an F input gives an F result and a permuted input keeps
    ///     its memory order, where NumSharp used to allocate C. <c>count_nonzero</c> is NumPy's
    ///     <c>a.astype(bool).sum(axis, dtype=intp)</c> and inherits the sum's layout. Every expected stride (in elements) and
    ///     flag was probed against NumPy 2.4.2; the values are checked against the same reduction of a C-ordered copy, so a
    ///     result written through the wrong strides fails too.
    /// </summary>
    [TestClass]
    public class NpAllAnyResultLayoutTest
    {
        /// <summary>A (3, 5, 4) operand of the given kind with a mix of zero and nonzero elements.</summary>
        /// <param name="kind">"bool", "int32" or "U2" (text: '' or 'ab').</param>
        /// <returns>The C-ordered operand.</returns>
        private static NDArray Base(string kind)
        {
            var m = np.arange(60).reshape(3, 5, 4) % 3;
            return kind switch
            {
                "bool" => m.astype(np.@bool),
                "int32" => m.astype(np.int32),
                _ => np.where(m == 0, np.array(""), np.array("ab")).astype("U2"),
            };
        }

        /// <summary>Runs one of the three reductions.</summary>
        /// <param name="fn">"any", "all" or "count_nonzero".</param>
        /// <param name="a">The operand.</param>
        /// <param name="axis">The axis.</param>
        /// <param name="keepdims">Keep the reduced axis with extent 1.</param>
        /// <returns>The result.</returns>
        private static NDArray Reduce(string fn, NDArray a, int axis, bool keepdims) => fn switch
        {
            "any" => np.any(a, axis, keepdims),
            "all" => np.all(a, axis, keepdims),
            _ => np.count_nonzero(a, axis, keepdims),
        };

        /// <summary>
        ///     Asserts a result's dims, its element strides on the axes longer than 1 (NumPy assigns an extent-1 axis an
        ///     arbitrary stride), its contiguity flags, ownership, and its values against the reduction of a C copy.
        /// </summary>
        /// <param name="r">The result under test.</param>
        /// <param name="reference">The same reduction over a C-ordered copy (the values must be equal in logical order).</param>
        /// <param name="dims">NumPy's result dims.</param>
        /// <param name="strides">NumPy's element strides (only the axes longer than 1 are compared).</param>
        /// <param name="c">NumPy's C_CONTIGUOUS flag.</param>
        /// <param name="f">NumPy's F_CONTIGUOUS flag.</param>
        /// <param name="because">The case label shown on a failure.</param>
        private static void AssertLayout(NDArray r, NDArray reference, long[] dims, long[] strides, bool c, bool f, string because)
        {
            r.shape.Should().Equal(dims, because);
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] > 1)
                    r.Shape.strides[i].Should().Be(strides[i], $"{because}: stride of axis {i}");
            r.flags.c_contiguous.Should().Be(c, because);
            r.flags.f_contiguous.Should().Be(f, because);
            r.flags.owndata.Should().BeTrue(because);
            r.flags.writeable.Should().BeTrue(because);
            if (r.typecode == NPTypeCode.Boolean)
                r.ToArray<bool>().Should().Equal(reference.ToArray<bool>(), because);
            else
                r.ToArray<long>().Should().Equal(reference.ToArray<long>(), because);
        }

        /// <summary>An F operand reduced along each axis gives an F result (probed: axis 0 → (5, 4) strides (1, 5)).</summary>
        [TestMethod]
        public void FInput_TheResultKeepsFOrder()
        {
            var cases = new (int axis, bool kd, long[] dims, long[] strides)[]
            {
                (0, false, new long[] { 5, 4 }, new long[] { 1, 5 }),
                (1, false, new long[] { 3, 4 }, new long[] { 1, 3 }),
                (2, false, new long[] { 3, 5 }, new long[] { 1, 3 }),
                (0, true, new long[] { 1, 5, 4 }, new long[] { 1, 1, 5 }),
                (1, true, new long[] { 3, 1, 4 }, new long[] { 1, 3, 3 }),
                (2, true, new long[] { 3, 5, 1 }, new long[] { 1, 3, 15 }),
            };
            foreach (var kind in new[] { "bool", "int32", "U2" })
            foreach (var fn in new[] { "any", "all", "count_nonzero" })
            foreach (var (axis, kd, dims, strides) in cases)
            {
                var c = Base(kind);
                var f = np.asfortranarray(c);
                AssertLayout(Reduce(fn, f, axis, kd), Reduce(fn, c, axis, kd), dims, strides, c: false, f: true,
                    $"{fn}({kind} F, axis={axis}, keepdims={kd})");
            }
        }

        /// <summary>A permuted operand keeps its memory order: (1, 0, 2) and (2, 0, 1) transposes of a C (3, 5, 4) array.</summary>
        [TestMethod]
        public void PermutedInput_TheResultFollowsTheMemoryOrder()
        {
            var cases = new (int[] perm, int axis, long[] dims, long[] strides, bool c, bool f)[]
            {
                (new[] { 1, 0, 2 }, 0, new long[] { 3, 4 }, new long[] { 4, 1 }, true, false),
                (new[] { 1, 0, 2 }, 1, new long[] { 5, 4 }, new long[] { 4, 1 }, true, false),
                (new[] { 1, 0, 2 }, 2, new long[] { 5, 3 }, new long[] { 1, 5 }, false, true),
                (new[] { 2, 0, 1 }, 0, new long[] { 3, 5 }, new long[] { 5, 1 }, true, false),
                (new[] { 2, 0, 1 }, 1, new long[] { 4, 5 }, new long[] { 1, 4 }, false, true),
                (new[] { 2, 0, 1 }, 2, new long[] { 4, 3 }, new long[] { 1, 4 }, false, true),
            };
            foreach (var kind in new[] { "bool", "int32", "U2" })
            foreach (var fn in new[] { "any", "all", "count_nonzero" })
            foreach (var (perm, axis, dims, strides, c, f) in cases)
            {
                var t = Base(kind).transpose(perm);
                AssertLayout(Reduce(fn, t, axis, false), Reduce(fn, np.ascontiguousarray(t), axis, false), dims, strides, c, f,
                    $"{fn}({kind} transpose({string.Join(",", perm)}), axis={axis})");
            }
        }

        /// <summary>A stepped slice whose axes are already in C memory order keeps a C result (NpyIter's sort changes nothing).</summary>
        [TestMethod]
        public void SteppedInputInCOrder_TheResultIsC()
        {
            var s = (np.arange(120).reshape(3, 10, 4) % 3 == 0)[":, ::2, :"];
            foreach (var fn in new[] { "any", "all", "count_nonzero" })
            for (int axis = 0; axis < 3; axis++)
            {
                var r = Reduce(fn, s, axis, false);
                r.flags.c_contiguous.Should().BeTrue($"{fn} axis {axis}");
                AsLongs(r).Should().Equal(AsLongs(Reduce(fn, np.ascontiguousarray(s), axis, false)), $"{fn} axis {axis}");
            }
        }

        /// <summary>A bool or int64 result's values in logical C order, as longs (True = 1), for one comparison form.</summary>
        /// <param name="r">The result.</param>
        /// <returns>The values.</returns>
        private static long[] AsLongs(NDArray r) =>
            r.typecode == NPTypeCode.Boolean ? r.ToArray<bool>().Select(x => x ? 1L : 0L).ToArray() : r.ToArray<long>();

        /// <summary>
        ///     A tuple axis over an F 4-D operand keeps the remaining axes in F order, with and without keepdims (probed:
        ///     axis (0, 2) of (2, 3, 4, 5) → (3, 5) strides (1, 3); keepdims (1, 3, 1, 5) strides (1, 1, 3, 3)).
        /// </summary>
        [TestMethod]
        public void TupleAxis_FInput_KeepsTheRemainingAxesInFOrder()
        {
            var c = (np.arange(120).reshape(2, 3, 4, 5) % 7) != 0;
            var f = np.asfortranarray(c);
            var cases = new (int[] axes, bool kd, long[] dims, long[] strides)[]
            {
                (new[] { 0, 2 }, false, new long[] { 3, 5 }, new long[] { 1, 3 }),
                (new[] { 0, 2 }, true, new long[] { 1, 3, 1, 5 }, new long[] { 1, 1, 3, 3 }),
                (new[] { 1, 3 }, false, new long[] { 2, 4 }, new long[] { 1, 2 }),
                (new[] { 1, 3 }, true, new long[] { 2, 1, 4, 1 }, new long[] { 1, 2, 2, 8 }),
                (new[] { 0, 1 }, false, new long[] { 4, 5 }, new long[] { 1, 4 }),
            };
            foreach (var fn in new[] { "any", "all" })
            foreach (var (axes, kd, dims, strides) in cases)
            {
                var r = fn == "any" ? np.any(f, axes, kd) : np.all(f, axes, kd);
                var reference = fn == "any" ? np.any(c, axes, kd) : np.all(c, axes, kd);
                AssertLayout(r, reference, dims, strides, c: false, f: true, $"{fn}(F, axis=({string.Join(",", axes)}), keepdims={kd})");
            }
        }

        /// <summary>
        ///     keepdims results own their data, as NumPy's do: <c>np.count_nonzero</c> and the bool <c>np.sum</c> count route
        ///     used to return np.evaluate's reshaped view (OWNDATA false).
        /// </summary>
        [TestMethod]
        public void Keepdims_TheResultOwnsItsData()
        {
            var b = np.array(new[] { true, false, true });
            var r1 = np.count_nonzero(b, 0, true);
            r1.shape.Should().Equal(new long[] { 1 });
            r1.flags.owndata.Should().BeTrue();
            r1.flags.writeable.Should().BeTrue();
            r1.ToArray<long>().Should().Equal(2L);

            var i = np.asfortranarray(np.array(new[,] { { 1, 0, 2 }, { 0, 0, 3 } }));
            var r2 = np.count_nonzero(i, 0, true);
            r2.shape.Should().Equal(new long[] { 1, 3 });
            r2.flags.owndata.Should().BeTrue();
            r2.ToArray<long>().Should().Equal(1L, 0L, 2L);

            var s = np.sum(np.array(new[,] { { true, false }, { true, true } }), 1, true, np.int64);
            s.shape.Should().Equal(new long[] { 2, 1 });
            s.flags.owndata.Should().BeTrue();
            s.ToArray<long>().Should().Equal(1L, 2L);
        }

        /// <summary>
        ///     <c>np.count_nonzero</c> along an axis reads every dtype's truth as NumPy's bool cast does: NaN and inf count,
        ///     <c>-0.0</c> does not, a complex value counts when either part is nonzero.
        /// </summary>
        [TestMethod]
        public void CountNonzero_Axis_UsesTheBoolCast()
        {
            var d = np.array(new[,] { { double.NaN, -0.0, 0.0 }, { double.PositiveInfinity, 1e-300, -0.0 } });
            np.count_nonzero(d, 0).ToArray<long>().Should().Equal(2L, 1L, 0L);
            np.count_nonzero(d, 1).ToArray<long>().Should().Equal(1L, 2L);
            var z = np.array(new[] { new System.Numerics.Complex(0, 0), new System.Numerics.Complex(0, -1), new System.Numerics.Complex(2, 0) });
            np.count_nonzero(z, 0).GetInt64().Should().Be(2L);
            var e = np.zeros(new Shape(0L, 3L));
            np.count_nonzero(e, 0).ToArray<long>().Should().Equal(0L, 0L, 0L);
            np.count_nonzero(e, 1).size.Should().Be(0);
        }
    }
}
