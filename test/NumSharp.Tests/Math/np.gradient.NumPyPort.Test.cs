using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.gradient held to NumPy 2.4.2's own statements (<c>numpy/lib/_function_base_impl.py::gradient</c>), bit for bit:
    ///     every expected value is NumPy's, probed and written as raw bits, so a change in a dtype rule or an operation order
    ///     fails here even when the values stay close.
    /// </summary>
    [TestClass]
    public class NpGradientNumPyPortTest
    {
        /// <summary>The float32 elements of a result as raw bits.</summary>
        /// <param name="a">A float32 array.</param>
        /// <returns>Each element's IEEE bits, in C order.</returns>
        private static uint[] Bits32(NDArray a) => a.ToArray<float>().Select(BitConverter.SingleToUInt32Bits).ToArray();

        /// <summary>The float64 elements of a result as raw bits.</summary>
        /// <param name="a">A float64 array.</param>
        /// <returns>Each element's IEEE bits, in C order.</returns>
        private static ulong[] Bits64(NDArray a) => a.ToArray<double>().Select(BitConverter.DoubleToUInt64Bits).ToArray();

        /// <summary>
        ///     A spacing of another dtype than float64 is read through its own dtype. np.gradient read every spacing value
        ///     with <c>GetDouble</c>, which indexes the float64 lane of the storage whatever the dtype: float32 / float16 /
        ///     bool coordinate diffs were read as other bytes and past their buffer (the results changed from run to run), a
        ///     0-d int64 spacing became the denormal its bits spell, and an empty axis' coordinates read an empty buffer.
        /// </summary>
        [TestMethod]
        public void SpacingOfAnotherDtype_IsReadInItsOwnDtype()
        {
            var f32 = np.array(new[] { 1f, 4f, 9f, 16f, 25f });
            var f64 = np.array(new[] { 1.0, 4.0, 9.0, 16.0, 25.0 });

            // np.gradient(f32, np.array([0, 0.5, 1.5, 2, 3], np.float32)): float32 coefficients, a float32 result.
            NDArray r = np.gradient(f32, np.array(new[] { 0f, 0.5f, 1.5f, 2f, 3f }));
            r.dtype.Should().Be(np.float32);
            Bits32(r).Should().Equal(0x40c00000u, 0x40b55555u, 0x41300001u, 0x41455556u, 0x41100000u);
            // The same call again: nothing outside the arrays is read, so the bits cannot change.
            Bits32((NDArray)np.gradient(f32, np.array(new[] { 0f, 0.5f, 1.5f, 2f, 3f })))
                .Should().Equal(0x40c00000u, 0x40b55555u, 0x41300001u, 0x41455556u, 0x41100000u);

            // np.gradient(f64, np.array([0, 0.5, 1.5, 2, 3], np.float16)): the float16 diffs make float16 coefficients.
            r = np.gradient(f64, np.array(new[] { 0.0, 0.5, 1.5, 2.0, 3.0 }).astype(np.float16));
            r.dtype.Should().Be(np.float64);
            Bits64(r).Should().Equal(0x4018000000000000ul, 0x4016aa4000000000ul, 0x4025fd8000000000ul,
                0x4028ab2000000000ul, 0x4022000000000000ul);

            // np.gradient(f64, np.array(2)): the 0-d int64 spacing is the step 2, [1.5, 2, 3, 4, 4.5].
            r = np.gradient(f64, np.array(2L));
            Bits64(r).Should().Equal(0x3ff8000000000000ul, 0x4000000000000000ul, 0x4008000000000000ul,
                0x4010000000000000ul, 0x4012000000000000ul);

            // np.gradient(f64, np.array([False, True, True, False, True])): NumPy keeps bool diffs, and the interior's
            // `-(dx2)` is its boolean-negative TypeError.
            Action boolCoords = () => np.gradient(f64, np.array(new[] { false, true, true, false, true }));
            boolCoords.Should().Throw<TypeError>().WithMessage(
                "The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not " +
                "function instead.");

            // np.gradient(np.zeros((0, 3)), np.zeros(0), 1.0): the empty axis' empty diff has no diffx[0].
            Action emptyAxis = () => np.gradient(np.zeros(new Shape(0, 3)), np.zeros(new Shape(0)), 1.0);
            emptyAxis.Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0");
        }

        /// <summary>
        ///     A scalar spacing keeps its NEP 50 kind through NumPy's statements (<c>2. * ax_dx</c>, the quotients, the edge
        ///     coefficients), where every scalar was converted to a C# double, a weak Python float. Probed 2.4.2:
        ///     <list type="bullet">
        ///         <item>a Python float step keeps a float32 input in float32; the same value as a 0-d float64 array, or as a
        ///         collapsed float64 coordinate step (<c>diffx[0]</c>, a NumPy scalar), divides in float64, rounded once;</item>
        ///         <item>a float16 step (a NumPy scalar) doubles and divides the edge coefficients in float16 scalarmath;</item>
        ///         <item>a char step is NumPy's uint16 (it was refused by the double conversion, as was a float16 one);</item>
        ///         <item>float32 coordinates make float32 edge coefficients (the edges were widened to float64), and complex
        ///         coordinates complex ones whose real part the store keeps;</item>
        ///         <item>a str or None step is refused by Python's own <c>2. * ax_dx</c> (a str was parsed as a number), and a
        ///         zero step raises ZeroDivisionError for a Python float in <c>-1.5 / ax_dx</c>, inf / nan for a NumPy
        ///         scalar.</item>
        ///     </list>
        /// </summary>
        [TestMethod]
        public void SpacingKind_IsNumPysOperandKind()
        {
            var f32 = np.array(new[] { 1.25f, 5f, 11.25f, 20f, 31.25f, 45f, 61.25f });
            var f64 = np.array(new[] { 1.0, 4.0, 9.0, 16.0, 25.0 });
            // s = 1 + 2**-24: twice it is 2 + 2**-23, which rounds to 2.0 in float32 (a tie, to even), so a float32 quotient
            // by the Python float is exact and a float64 quotient by the NumPy scalar lies just below it.
            double s = 1.0 + System.Math.Pow(2, -24);

            NDArray r = np.gradient(f32, s);
            r.dtype.Should().Be(np.float32);
            Bits32(r).Should().Equal(0x40700000u, 0x40a00000u, 0x40f00000u, 0x41200000u, 0x41480000u, 0x41700000u,
                0x41820000u);
            r = np.gradient(f32, np.array(s));
            r.dtype.Should().Be(np.float32);
            var strong = new[] { 0x406fffffu, 0x409fffffu, 0x40efffffu, 0x411fffffu, 0x4147ffffu, 0x416fffffu, 0x4181ffffu };
            Bits32(r).Should().Equal(strong);
            // np.gradient(f32, np.arange(7) * s): exact coordinates, every difference s, so diffx[0] is np.float64(s).
            var coords = np.array(Enumerable.Range(0, 7).Select(k => k * s).ToArray());
            Bits32((NDArray)np.gradient(f32, coords)).Should().Equal(strong);
            // edge_order=2: the weak coefficients are float32 (-1.5 / s rounds s to 1.0), the strong ones float64.
            Bits32((NDArray)np.gradient(f32, s, edge_order: 2)).Should().Equal(0x401ffffeu, 0x40a00000u, 0x40f00000u,
                0x41200000u, 0x41480000u, 0x41700000u, 0x418c0000u);
            var strong2 = new[] { 0x401fffffu, 0x409fffffu, 0x40efffffu, 0x411fffffu, 0x4147ffffu, 0x416fffffu, 0x418bffffu };
            Bits32((NDArray)np.gradient(f32, np.array(s), edge_order: 2)).Should().Equal(strong2);
            Bits32((NDArray)np.gradient(f32, coords, edge_order: 2)).Should().Equal(strong2);

            // np.gradient(f64, np.float16(0.1)): `2. * np.float16(0.1)` is np.float16(0.19995), the interior's divisor; with
            // edge_order=2 the coefficients -1.5 / step, 2. / step, -0.5 / step round to the float16 -15, 20, -5.
            r = np.gradient(f64, (Half)0.1);
            r.dtype.Should().Be(np.float64);
            Bits64(r).Should().Equal(0x403e01e01e01e01eul, 0x4044014014014014ul, 0x404e01e01e01e01eul, 0x4054014014014014ul,
                0x4056816816816817ul);
            Bits64((NDArray)np.gradient(f64, (Half)0.1, edge_order: 2)).Should().Equal(0x4034000000000000ul,
                0x4044014014014014ul, 0x404e01e01e01e01eul, 0x4054014014014014ul, 0x4059000000000000ul);

            // np.gradient(f32, np.uint16(3)): `2. * np.uint16(3)` is np.float64(6.0), so the float32 input divides in float64.
            r = np.gradient(f32, (char)3);
            r.dtype.Should().Be(np.float32);
            Bits32(r).Should().Equal(0x3fa00000u, 0x3fd55555u, 0x40200000u, 0x40555555u, 0x40855555u, 0x40a00000u,
                0x40ad5555u);
            Bits32((NDArray)np.gradient(f32, (char)3, edge_order: 2)).Should().Equal(0x3f555555u, 0x3fd55555u, 0x40200000u,
                0x40555555u, 0x40855555u, 0x40a00000u, 0x40baaaabu);

            // np.gradient(f32, np.array([0, 0.5, 1.5, 2, 3, 3.5, 4.5], np.float32), edge_order=2): float32 coefficients,
            // the edges included (float64 coordinates give 0x40fd5555 at the front edge).
            Bits32((NDArray)np.gradient(f32, np.array(new[] { 0f, 0.5f, 1.5f, 2f, 3f, 3.5f, 4.5f }), edge_order: 2))
                .Should().Equal(0x40fd5554u, 0x40e2aaaau, 0x415c0001u, 0x4176aaabu, 0x41b0aaaau, 0x41be0000u, 0x410c0000u);

            // np.gradient(f64, [0, 0.5+0.5j, 1.5+1j, 2, 3+0.5j]): complex coefficients, the float64 component keeps the real
            // part of each complex value (NumPy's setitem, with its ComplexWarning).
            var complexCoords = np.array(new[]
            {
                new System.Numerics.Complex(0, 0), new System.Numerics.Complex(0.5, 0.5),
                new System.Numerics.Complex(1.5, 1), new System.Numerics.Complex(2, 0), new System.Numerics.Complex(3, 0.5),
            });
            r = np.gradient(f64, complexCoords);
            r.dtype.Should().Be(np.float64);
            Bits64(r).Should().Equal(0x4008000000000000ul, 0x400a762762762763ul, 0xbfd9999999999990ul, 0x3fd9999999999990ul,
                0x401ccccccccccccdul);
            Bits64((NDArray)np.gradient(f64, complexCoords, edge_order: 2)).Should().Equal(0x400589d89d89d89bul,
                0x400a762762762763ul, 0xbfd9999999999990ul, 0x3fd9999999999990ul, 0x402c000000000000ul);

            // A str or None step: Python's `2. * ax_dx` refuses it.
            Action text = () => np.gradient(f32, "2");
            text.Should().Throw<TypeError>().WithMessage("can't multiply sequence by non-int of type 'float'");
            Action none = () => np.gradient(np.ones(new Shape(3, 4)), null, 1.0);
            none.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for *: 'float' and 'NoneType'");

            // A zero step with edge_order=2: CPython's `-1.5 / 0.0` raises; np.float16(0) gives -inf, inf, -inf
            // coefficients (scalarmath), so the edges are inf - inf, a NaN (its sign is the CPU's default NaN).
            Action pyZero = () => np.gradient(f32, 0.0, edge_order: 2);
            pyZero.Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
            float[] halfZero = ((NDArray)np.gradient(f32, (Half)0.0, edge_order: 2)).ToArray<float>();
            float.IsNaN(halfZero[0]).Should().BeTrue();
            float.IsNaN(halfZero[6]).Should().BeTrue();
            halfZero.Skip(1).Take(5).Should().AllSatisfy(v => float.IsPositiveInfinity(v).Should().BeTrue());
        }

        /// <summary>
        ///     Asserts every component of a gradient: its byte strides and its C / F / OWNDATA flags.
        /// </summary>
        /// <param name="r">The gradient.</param>
        /// <param name="strides">NumPy's byte strides of every component (NumPy gives each component the same layout).</param>
        /// <param name="c">NumPy's C_CONTIGUOUS flag.</param>
        /// <param name="f">NumPy's F_CONTIGUOUS flag.</param>
        private static void AssertLayout(np.GradientResult r, long[] strides, bool c, bool f)
        {
            foreach (NDArray component in r.ToArray())
            {
                component.strides.Should().Equal(strides);
                component.flags.c_contiguous.Should().Be(c);
                component.flags.f_contiguous.Should().Be(f);
                component.flags.owndata.Should().BeTrue();
            }
        }

        /// <summary>
        ///     Each component is NumPy's <c>np.empty_like(f, dtype=otype)</c>: it keeps the input's memory order (order 'K'),
        ///     where a C-ordered array was allocated. Probed 2.4.2: an F-ordered, a broadcast or a transposed input gives
        ///     F-ordered components, a permuted one its permutation's order, a reversed or stepped one C order; an integer input
        ///     converts to float64 with its order kept first.
        /// </summary>
        [TestMethod]
        public void ComponentLayout_IsNumPysEmptyLike()
        {
            var squares = np.power(np.arange(12).astype(np.float64), 2).reshape(3, 4);
            var squares3 = np.power(np.arange(24).astype(np.float64), 2).reshape(2, 3, 4);

            var fOrdered = np.gradient(np.asfortranarray(squares));
            AssertLayout(fOrdered, new long[] { 8, 24 }, c: false, f: true);
            // np.gradient(np.asfortranarray(x))[1] is the C-order values [[1, 2, 4, 5], [9, 10, 12, 13], [17, 18, 20, 21]].
            fOrdered[1].ToArray<double>().Should().Equal(1, 2, 4, 5, 9, 10, 12, 13, 17, 18, 20, 21);

            AssertLayout(np.gradient(np.broadcast_to(np.array(new[,] { { 1.0, 4.0, 9.0, 16.0 } }), new Shape(3, 4))),
                new long[] { 8, 24 }, c: false, f: true);
            AssertLayout(np.gradient(np.transpose(squares3, new[] { 1, 0, 2 })), new long[] { 32, 96, 8 }, c: false, f: false);
            AssertLayout(np.gradient(squares3.T, axis: 0), new long[] { 8, 32, 96 }, c: false, f: true);
            AssertLayout(np.gradient(squares["::-1, ::-1"]), new long[] { 32, 8 }, c: true, f: false);
            AssertLayout(np.gradient(np.power(np.arange(24).astype(np.float64), 2).reshape(3, 8)[":, ::2"]),
                new long[] { 32, 8 }, c: true, f: false);
            AssertLayout(np.gradient(np.asfortranarray(np.arange(12).astype(np.int8).reshape(3, 4))),
                new long[] { 8, 24 }, c: false, f: true);
        }

        /// <summary>
        ///     A Python list spacing (a <c>List&lt;T&gt;</c> or an <c>object[]</c>, the house map) is coordinates, as NumPy's
        ///     <c>np.ndim</c> reads it, converted by np.array's rules. It was read as a scalar step: a <c>List&lt;double&gt;</c>
        ///     reached the double conversion and its InvalidCastException, and a ragged list reached a KeyNotFoundException
        ///     in dtype discovery. Probed 2.4.2.
        /// </summary>
        [TestMethod]
        public void SpacingList_IsCoordinates()
        {
            var f64 = np.array(new[] { 1.0, 4.0, 9.0, 16.0, 25.0 });

            // np.gradient(f64, [0.0, 0.5, 1.5, 2.0, 3.0]): the same bits as the coordinates as an ndarray.
            var uneven = new ulong[] { 0x4018000000000000ul, 0x4016aaaaaaaaaaabul, 0x4025fffffffffffful, 0x4028aaaaaaaaaaaaul,
                0x4022000000000000ul };
            Bits64((NDArray)np.gradient(f64, new List<double> { 0, 0.5, 1.5, 2, 3 })).Should().Equal(uneven);
            // An object[] is ONE spacing only when it is passed as an object: as an object[] it binds the spacing-list
            // overload (NumPy's *varargs), five scalar steps here. Its C# ints are Python ints (int64 coordinates).
            Bits64((NDArray)np.gradient(f64, (object)new object[] { 0, 1, 4, 5, 8 })).Should().Equal(0x4008000000000000ul,
                0x4005555555555555ul, 0x4016aaaaaaaaaaabul, 0x4017fffffffffffful, 0x4008000000000000ul);
            // Evenly spaced coordinates collapse to their step, np.float64(0.5).
            Bits64((NDArray)np.gradient(f64, new List<double> { 0, 0.5, 1, 1.5, 2 })).Should().Equal(0x4018000000000000ul,
                0x4020000000000000ul, 0x4028000000000000ul, 0x4030000000000000ul, 0x4032000000000000ul);

            // np.gradient(np.arange(12.0).reshape(3, 4) ** 2, [0.0, 0.5, 1.5], [0, 1, 4, 5]): a list per axis.
            var (g0, g1) = np.gradient(np.power(np.arange(12).astype(np.float64), 2).reshape(3, 4),
                new List<double> { 0, 0.5, 1.5 }, new List<int> { 0, 1, 4, 5 });
            Bits64(g0).Should().Equal(0x4040000000000000ul, 0x4048000000000000ul, 0x4050000000000000ul, 0x4054000000000000ul,
                0x4042aaaaaaaaaaaaul, 0x4049555555555556ul, 0x4050000000000000ul, 0x4053555555555555ul, 0x4048000000000000ul,
                0x404c000000000000ul, 0x4050000000000000ul, 0x4052000000000000ul);
            Bits64(g1).Should().Equal(0x3ff0000000000000ul, 0x3ff0000000000000ul, 0x4010000000000000ul, 0x4014000000000000ul,
                0x4022000000000000ul, 0x401eaaaaaaaaaaa8ul, 0x4025555555555556ul, 0x402a000000000000ul, 0x4031000000000000ul,
                0x402caaaaaaaaaaaaul, 0x4031555555555558ul, 0x4035000000000000ul);

            // np.ndim's refusal of a ragged list, then np.gradient's own checks of a list's rank and length.
            Action ragged = () => np.gradient(f64, (object)new object[] { 0.0, new object[] { 0.5 } });
            ragged.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence. The requested array " +
                "has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");
            Action column = () => np.gradient(np.ones(new Shape(3, 4)),
                new List<List<double>> { new() { 0 }, new() { 1 }, new() { 2 } }, 1.0);
            column.Should().Throw<ValueError>().WithMessage("distances must be either scalars or 1d");
            Action shortList = () => np.gradient(f64, new List<double> { 0, 1 });
            shortList.Should().Throw<ValueError>()
                .WithMessage("when 1d, distances must match the length of the corresponding dimension");
        }
    }
}
