using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.array</c> and <c>np.asanyarray</c> of a C# object list - an <c>object[]</c>, a jagged <c>object[][]</c>, an
    ///     <c>object[,]</c> grid of rows, a <see cref="List{T}"/> of objects, a tuple - which the boundary map reads as NumPy's
    ///     Python list (a C# integer a Python int, a float or double a Python float, null None, a typed C# array or an
    ///     <see cref="NDArray"/> an ndarray item). Every expectation is NumPy 2.4.2's answer for the Python spelling
    ///     (probed: <c>np.array([1, 2.5])</c>, <c>np.array([300], dtype=np.int8)</c>, ...).
    /// </summary>
    /// <remarks>
    ///     <c>np.array</c> of an object array threw for any list without text (<c>ArgumentOutOfRangeException</c>, or
    ///     <c>Cannot interpret 'System.Object' as a data type</c>), and <c>np.asanyarray</c> converted it with C#'s types (a C#
    ///     int list was int32, a dtype request a cast that wrapped 300 into int8); both are NumPy's coercion of a sequence
    ///     now (<c>UnmanagedStorage.FromSequence</c>).
    /// </remarks>
    [TestClass]
    public class np_array_PythonList_Tests
    {
        /// <summary>Python ints are int64, a Python float among them makes float64, a bool among ints int64.</summary>
        [TestMethod]
        public void Discovery_PythonScalars()
        {
            foreach (var f in new Func<object[], NDArray>[] { a => np.array(a), a => np.asanyarray(a) })
            {
                var ints = f(new object[] { 1, 2 });
                ints.dtype.Should().Be(np.int64);
                ints.ToArray<long>().Should().Equal(1L, 2L);

                var mixed = f(new object[] { 1, 2.5 });
                mixed.dtype.Should().Be(np.float64);
                mixed.ToArray<double>().Should().Equal(1.0, 2.5);

                var boolInt = f(new object[] { true, 2 });
                boolInt.dtype.Should().Be(np.int64);
                boolInt.ToArray<long>().Should().Equal(1L, 2L);

                // A C# float is a Python float (float64); a NumPy float16 scalar promotes with it to float64 too.
                f(new object[] { 1.5f, 2.5f }).dtype.Should().Be(np.float64);
                f(new object[] { (Half)1, 2.5 }).dtype.Should().Be(np.float64);

                // 2**64 - 1 is uint64, 1 is int64: the promotion is float64.
                var u64 = f(new object[] { ulong.MaxValue, 1 });
                u64.dtype.Should().Be(np.float64);
                u64.ToArray<double>().Should().Equal(18446744073709551615.0, 1.0);

                var complexInt = f(new object[] { new Complex(1, 2), 1 });
                complexInt.dtype.Should().Be(np.complex128);
            }
        }

        /// <summary>Nested lists and grids give their shape; empty lists end the dimensions (<c>[[], []]</c> is (2, 0)).</summary>
        [TestMethod]
        public void Discovery_Shapes()
        {
            var nested = np.array(new object[] { new object[] { 1, 2 }, new object[] { 3, 4 } });
            nested.dtype.Should().Be(np.int64);
            nested.shape.Should().Equal(2L, 2L);
            nested.ToArray<long>().Should().Equal(1L, 2L, 3L, 4L);

            var grid = np.array(new object[,] { { 1, 2.5 }, { 3, 4 } });
            grid.dtype.Should().Be(np.float64);
            grid.shape.Should().Equal(2L, 2L);
            grid.ToArray<double>().Should().Equal(1.0, 2.5, 3.0, 4.0);

            var empty = np.array(new object[0]);
            empty.dtype.Should().Be(np.float64);
            empty.shape.Should().Equal(0L);

            var emptyRows = np.asanyarray(new object[] { new object[0], new object[0] });
            emptyRows.dtype.Should().Be(np.float64);
            emptyRows.shape.Should().Equal(2L, 0L);

            // ndarray items keep their dtype: [int32 array, int32 array] is int32 (2, 2), and a typed C# array is one.
            var rows = np.array(new object[] { np.array(new[] { 1, 2 }), new[] { 3, 4 } });
            rows.dtype.Should().Be(np.int32);
            rows.shape.Should().Equal(2L, 2L);
            rows.ToArray<int>().Should().Equal(1, 2, 3, 4);
        }

        /// <summary>
        ///     A ragged list is NumPy's ValueError, and the requested order is checked before any item: an invalid order
        ///     raises even when an item would be refused.
        /// </summary>
        [TestMethod]
        public void Errors_RaggedAndOrder()
        {
            Action ragged = () => np.array(new object[] { new object[] { 1 }, new object[] { 2, 3 } });
            ragged.Should().Throw<ValueError>().WithMessage(
                "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");

            Action order = () => np.array(new object[] { 300 }, np.int8, order: 'Z');
            order.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Z')");
            Action anyOrder = () => np.asanyarray(new object[] { 300 }, np.int8, 'Z');
            anyOrder.Should().Throw<ValueError>().WithMessage("order must be one of 'C', 'F', 'A', or 'K' (got 'Z')");
        }

        /// <summary>
        ///     With a dtype every item is stored by the dtype's setitem (NumPy's <c>PyArray_Pack</c>): a float truncates into
        ///     an integer, an int out of range raises, None is NaN for a float and <c>int()</c>'s TypeError for an integer, a
        ///     complex into a real dtype raises.
        /// </summary>
        [TestMethod]
        public void Request_PacksEveryItem()
        {
            foreach (var f in new Func<object[], DType, NDArray>[] { (a, d) => np.array(a, d), (a, d) => np.asanyarray(a, d) })
            {
                var truncated = f(new object[] { 1, 2.5 }, np.int8);
                truncated.dtype.Should().Be(np.int8);
                truncated.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)2);

                Action overflow = () => f(new object[] { 300 }, np.int8);
                overflow.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");

                var nan = f(new object[] { 1, null }, np.float64);
                double.IsNaN(nan.ToArray<double>()[1]).Should().BeTrue();

                Action noneInt = () => f(new object[] { 1, null }, np.int8);
                noneInt.Should().Throw<TypeError>().WithMessage(
                    "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");

                Action complexReal = () => f(new object[] { new Complex(1, 2), 1 }, np.float64);
                complexReal.Should().Throw<TypeError>().WithMessage("float() argument must be a string or a real number, not 'complex'");

                var floats = f(new object[] { 1, 2 }, np.float32);
                floats.dtype.Should().Be(np.float32);
                floats.ToArray<float>().Should().Equal(1f, 2f);

                // An ndarray item is cast (unsafe), not packed: NumPy's PyArray_AssignArray of the coercion cache.
                var cast = f(new object[] { np.array(new[] { 300, 1 }) }, np.int8);
                cast.ToArray<sbyte>().Should().Equal((sbyte)44, (sbyte)1);

                // A text dtype over ndarray items casts each item into the class adapted to it (int32 is <U11).
                var text = f(new object[] { np.array(new[] { 1, 2 }), np.array(new[] { 3, 4 }) }, np.dtype("U"));
                text.dtype.ToString().Should().Be("<U11");
                text.shape.Should().Equal(2L, 2L);
            }
        }

        /// <summary>
        ///     A tuple, a <see cref="List{T}"/> of objects and an <c>object[,]</c> grid are Python sequences too
        ///     (<c>np.asanyarray((1, 2))</c> is int64); an empty grid keeps its own dimensions (NumPy's nested lists cannot
        ///     state a grid of no rows: <c>[]</c> is (0,)).
        /// </summary>
        [TestMethod]
        public void OtherSequences()
        {
            np.asanyarray((1, 2)).dtype.Should().Be(np.int64);
            np.asanyarray(new List<object> { 1, 2.5 }).dtype.Should().Be(np.float64);
            np.asanyarray(new object[,] { { (byte)1, (byte)2 } }).dtype.Should().Be(np.int64);

            var noRows = np.asanyarray(new object[0, 3]);
            noRows.dtype.Should().Be(np.float64);
            noRows.shape.Should().Equal(0L, 3L);
            np.array(new object[0, 3]).shape.Should().Equal(0L, 3L);
        }

        /// <summary>
        ///     NumPy's object arrays are refused: None or a Python int past 64 bits without a dtype, and an ndarray item of a
        ///     text dtype (NumPy builds a text array from it, which the coercion of a sequence does not build).
        /// </summary>
        [TestMethod]
        public void Refusals()
        {
            Action none = () => np.array(new object[] { 1, null });
            none.Should().Throw<NotSupportedException>();
            Action big = () => np.asanyarray(new object[] { 1, BigInteger.Pow(2, 70) });
            big.Should().Throw<NotSupportedException>();
            Action textRows = () => np.asanyarray(new object[] { np.array(new[] { "a", "b" }), np.array(new[] { "c", "d" }) });
            textRows.Should().Throw<NotSupportedException>().WithMessage("An array item of dtype <U1 makes NumPy build a text array*");
        }
    }
}
