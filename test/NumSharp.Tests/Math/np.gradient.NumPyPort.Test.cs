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
