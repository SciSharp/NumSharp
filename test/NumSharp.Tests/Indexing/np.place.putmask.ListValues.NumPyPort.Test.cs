using System;
using System.Collections.Generic;
using AwesomeAssertions;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     <c>np.place</c> / <c>np.putmask</c> with their values given as a C# array, which both document as NumPy's Python
    ///     LIST: NumPy converts the list with the target's descriptor (<c>PyArray_FromAny(values0, PyArray_DESCR(self))</c>
    ///     in <c>arr_place</c> / <c>PyArray_PutMask</c>), every item by its setitem and after the mask's size check. Every
    ///     expected value, type and text is NumPy 2.4.2's for the same call with a Python list (a C# <c>object[]</c>; a typed
    ///     array's items are the Python numbers they spell: a C# int a Python int, a double a Python float).
    /// </summary>
    /// <remarks>
    ///     Found by P3's batch 10 of the np-layer dtype plan, reading what R1 still listed in the two facades: the list was
    ///     discovered on its own (<c>np.array</c>) and then cast UNSAFELY into the target - <c>[1000, 2]</c> into int8 stored
    ///     <c>-24</c> where NumPy raises OverflowError, <c>[1, 2.5]</c> into <c>&lt;U3</c> stored <c>'1.0'</c> (the
    ///     discovery's float64) where NumPy stores <c>'1'</c>, a NaN into an integer array stored a value where NumPy raises,
    ///     and a ragged list raised before a wrong-size mask did. (np.place into a StringDType array crashes NumPy 2.4.2
    ///     itself, so its StringDType case is not here.)
    /// </remarks>
    [TestClass]
    public class np_place_putmask_list_values_numpy_port_tests
    {
        /// <summary>The mask of every case: the first and the last of three elements.</summary>
        /// <returns>A fresh bool array <c>[True, False, True]</c>.</returns>
        private static NDArray Mask() => np.array(new[] { true, false, true });

        /// <summary>A Python int outside the target's integer range raises NumPy's OverflowError (it was wrapped).</summary>
        [TestMethod]
        public void OutOfRangePythonInt_RaisesOverflowError()
        {
            foreach (var list in new Array[] { new object[] { 1000, 2 }, new[] { 1000, 2 } })
            {
                Action place = () => np.place(np.zeros(new Shape(3), np.int8), Mask(), list);
                place.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
                Action putmask = () => np.putmask(np.zeros(new Shape(3), np.int8), Mask(), list);
                putmask.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
            }
            foreach (var list in new Array[] { new object[] { -1, 2 }, new long[] { -1, 2 } })
            {
                Action place = () => np.place(np.zeros(new Shape(3), np.uint8), Mask(), list);
                place.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
                Action putmask = () => np.putmask(np.zeros(new Shape(3), np.uint8), Mask(), list);
                putmask.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
            }
        }

        /// <summary>
        ///     A list of numbers into a text array is each number's own <c>str()</c>: NumPy's <c>[1, 2.5]</c> into
        ///     <c>&lt;U3</c> is <c>['1', '2.5']</c> (np.place) and <c>['1', '1']</c> (np.putmask, by position); the list's
        ///     float64 discovery gave <c>'1.0'</c>.
        /// </summary>
        [TestMethod]
        public void NumbersIntoText_AreEachItemsStr()
        {
            var placed = np.zeros(new Shape(3), np.dtype("U3"));
            np.place(placed, Mask(), new object[] { 1, 2.5 });
            placed.ToString(true).Should().Be("array(['1', '', '2.5'], dtype='<U3')");

            var putmasked = np.zeros(new Shape(3), np.dtype("U3"));
            np.putmask(putmasked, Mask(), new object[] { 1, 2.5 });
            putmasked.ToString(true).Should().Be("array(['1', '', '1'], dtype='<U3')");

            var text = np.array(new[] { "x", "y", "z" }, np.dtypes.StringDType());
            np.putmask(text, Mask(), new object[] { 1, 2.5 });
            text.ToString(true).Should().Be("array(['1', 'y', '1'], dtype=StringDType())");
        }

        /// <summary>
        ///     A typed array into fixed-width text is each item's <c>str()</c> too (text targets take the per-item stores): a
        ///     C# <c>float</c> is the Python float it holds, so <c>1.1f</c> is <c>'1.100000023841858'</c> where the float32
        ///     discovery wrote <c>'1.1'</c>; a C# <c>Half</c> is a NumPy float16 scalar (<c>'1.1'</c>). NumPy 2.4.2's
        ///     answers for the same lists.
        /// </summary>
        [TestMethod]
        public void TypedArraysIntoText_AreEachItemsStr()
        {
            NDArray Placed(string dt, Array values)
            {
                var a = np.zeros(new Shape(3), np.dtype(dt));
                np.place(a, Mask(), values);
                return a;
            }

            Placed("U32", new[] { 1.1f, 2f }).ToString(true).Should().Be("array(['1.100000023841858', '', '2.0'], dtype='<U32')");
            Placed("S32", new[] { 1.1f, 2f }).ToString(true).Should().Be("array([b'1.100000023841858', b'', b'2.0'], dtype='|S32')");
            Placed("U5", new[] { 1, 2 }).ToString(true).Should().Be("array(['1', '', '2'], dtype='<U5')");
            Placed("U5", new[] { true, false }).ToString(true).Should().Be("array(['True', '', 'False'], dtype='<U5')");
            Placed("U8", new[] { 0.1, 1e16 }).ToString(true).Should().Be("array(['0.1', '', '1e+16'], dtype='<U8')");
            Placed("U32", new[] { (Half)1.1f, (Half)2f }).ToString(true).Should().Be("array(['1.1', '', '2.0'], dtype='<U32')");

            var putmasked = np.zeros(new Shape(3), np.dtype("U32"));
            np.putmask(putmasked, Mask(), new[] { 1.1f, 2f });
            putmasked.ToString(true).Should().Be("array(['1.100000023841858', '', '1.100000023841858'], dtype='<U32')");
        }

        /// <summary>A NaN into an integer array is <c>int()</c>'s ValueError, as NumPy's setitem raises it.</summary>
        [TestMethod]
        public void NaNIntoIntegers_RaisesIntsValueError()
        {
            foreach (var list in new Array[] { new object[] { double.NaN, 2 }, new[] { double.NaN, 2 } })
            {
                Action place = () => np.place(np.zeros(new Shape(3), np.int8), Mask(), list);
                place.Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
                Action putmask = () => np.putmask(np.zeros(new Shape(3), np.int8), Mask(), list);
                putmask.Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
            }
        }

        /// <summary>
        ///     The list is converted where NumPy converts it, after the mask's size check: a wrong-size mask is reported
        ///     before a ragged list, and with a right-size mask the ragged list raises NumPy's inhomogeneous-shape text.
        /// </summary>
        [TestMethod]
        public void ListIsConvertedAfterTheMasksSizeCheck()
        {
            var ragged = new object[] { new object[] { 1 }, 2 };
            Action placeSize = () => np.place(np.zeros(new Shape(3)), np.array(new[] { true, false }), ragged);
            placeSize.Should().Throw<ValueError>().WithMessage("place: mask and data must be the same size");
            Action putmaskSize = () => np.putmask(np.zeros(new Shape(3)), np.array(new[] { true, false }), ragged);
            putmaskSize.Should().Throw<ValueError>().WithMessage("putmask: mask and data must be the same size");

            const string inhomogeneous = "setting an array element with a sequence. The requested array has an " +
                                         "inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.";
            Action placeRagged = () => np.place(np.zeros(new Shape(3)), Mask(), ragged);
            placeRagged.Should().Throw<ValueError>().Where(e => e.Message == inhomogeneous);
            Action putmaskRagged = () => np.putmask(np.zeros(new Shape(3)), Mask(), ragged);
            putmaskRagged.Should().Throw<ValueError>().Where(e => e.Message == inhomogeneous);
        }

        /// <summary>
        ///     A typed array takes the cast route where the cast stores what each item's setitem stores (Python floats into a
        ///     narrower float class, any numbers into bool): NumPy's values for the list (float16: <c>[1.0, inf, 0.0, nan,
        ///     0.1]</c>, bits <c>15360 31744 0 32256 11878</c>; float32: <c>[0.1, inf, -0.0]</c>; bool: <c>[False, False,
        ///     True, True]</c>), byte for byte the per-item route's for the same values given as an <c>object[]</c>.
        /// </summary>
        [TestMethod]
        public void TypedArrayCastRoute_StoresWhatEachItemStores()
        {
            static byte[] Bytes(NDArray a) => a.tobytes();

            var all5 = np.array(new[] { true, true, true, true, true });
            // Python's float('nan') is the POSITIVE quiet NaN; .NET's double.NaN is the negative one (0xfff8...), whose
            // sign both routes keep, as NumPy does.
            double pythonNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            var halves = new[] { 1.0, 65520.0, 1e-8, pythonNaN, 0.1 };
            var viaCast = np.zeros(new Shape(5), np.float16);
            np.putmask(viaCast, all5, halves);
            viaCast.view(np.uint16).ToArray<ushort>().Should().Equal(15360, 31744, 0, 32256, 11878);
            var viaItems = np.zeros(new Shape(5), np.float16);
            np.putmask(viaItems, all5, Boxed(halves));
            Bytes(viaCast).Should().Equal(Bytes(viaItems));

            var all3 = np.array(new[] { true, true, true });
            foreach (var typed in new Array[] { new[] { 0.1, 1e39, -1e-46 }, new[] { 0.1f, float.PositiveInfinity, -0f } })
            {
                var cast = np.zeros(new Shape(3), np.float32);
                np.place(cast, all3, typed);
                cast.view(np.uint32).ToArray<uint>().Should().Equal(1036831949u, 2139095040u, 2147483648u);
                var items = np.zeros(new Shape(3), np.float32);
                np.place(items, all3, Boxed(typed));
                Bytes(cast).Should().Equal(Bytes(items), typed.GetType().Name);
            }

            var all4 = np.array(new[] { true, true, true, true });
            foreach (var typed in new Array[] { new[] { 0.0, -0.0, double.NaN, 2.5 }, new[] { 0, 0, 1, -7 } })
            {
                var cast = np.zeros(new Shape(4), np.@bool);
                np.putmask(cast, all4, typed);
                cast.ToArray<bool>().Should().Equal(false, false, true, true);
                var items = np.zeros(new Shape(4), np.@bool);
                np.putmask(items, all4, Boxed(typed));
                Bytes(cast).Should().Equal(Bytes(items), typed.GetType().Name);
            }
        }

        /// <summary>
        ///     Python ints and floats into an integer class: the list is cast when one fused min / max pass shows every item
        ///     in range (a float truncated toward zero, as <c>int()</c> truncates) and stored item by item otherwise, which
        ///     raises at the first item out of range. NumPy: int8 &lt;- <c>[1.9, -1.9, 127.99, -128.5]</c> is <c>[1, -1, 127,
        ///     -128]</c>, &lt;- <c>[-128, 127, 5]</c> stores them, &lt;- <c>[128.0]</c> raises "Python integer 128 out of
        ///     bounds for int8"; uint8 &lt;- <c>[0.5, -0.5, 255.9]</c> is <c>[0, 0, 255]</c>; uint64 &lt;- <c>[-1]</c> raises.
        ///     Each list is stored as given (the pass's scalar loop) and tiled to 35 items (its vector lanes and a scalar
        ///     tail), each stored result byte for byte the per-item route's for the same values given as an
        ///     <c>object[]</c>; each error is raised for a one-item list and for a 35-item list with the bad item at each of
        ///     its positions (every vector lane and the tail).
        /// </summary>
        [TestMethod]
        public void IntegerTargets_CastWhenEveryItemFits()
        {
            foreach (var (dt, given, expectedGiven) in new (string, Array, long[])[]
                     {
                         ("int8", new[] { 1.9, -1.9, 127.99, -128.5 }, new long[] { 1, -1, 127, -128 }),
                         ("int8", new[] { 1.9f, -1.9f, 127.99f, -128.5f }, new long[] { 1, -1, 127, -128 }),
                         ("int8", new[] { -128, 127, 5 }, new long[] { -128, 127, 5 }),
                         ("uint8", new[] { 0.5, -0.5, 255.9 }, new long[] { 0, 0, 255 }),
                         ("int32", new uint[] { 0, 2147483647 }, new long[] { 0, 2147483647 }),
                         ("uint16", new long[] { 0, 65535 }, new long[] { 0, 65535 }),
                     })
            {
                foreach (var typed in new[] { given, Tile(given, 35) })
                {
                    string label = $"{dt} <- {typed.GetType().Name} of {typed.Length}";
                    var expected = new long[typed.Length];
                    for (int i = 0; i < expected.Length; i++)
                        expected[i] = expectedGiven[i % expectedGiven.Length];
                    var cast = np.zeros(new Shape(typed.Length), np.dtype(dt));
                    StoreAll(cast, typed);
                    cast.astype(np.int64).ToArray<long>().Should().Equal(expected, label);
                    var items = np.zeros(new Shape(typed.Length), np.dtype(dt));
                    StoreAll(items, Boxed(typed));
                    cast.tobytes().Should().Equal(items.tobytes(), label);
                }
            }

            // The bad item at every position of a 35-item list: every lane of every vector of the pass, and its tail.
            var positions = new List<(int count, int at)> { (1, 0) };
            for (int at = 0; at < 35; at++)
                positions.Add((35, at));
            foreach (var (count, at) in positions)
            {
                string where = $"{count} items, the bad one at {at}";
                Action past = () => StoreAll(np.zeros(new Shape(count), np.int8), Tile(new[] { 1.0 }, count, 128.0, at));
                past.Should().Throw<OverflowException>(where).WithMessage("Python integer 128 out of bounds for int8");
                Action negative = () => StoreAll(np.zeros(new Shape(count), np.uint64), Tile(new long[] { 1 }, count, -1L, at));
                negative.Should().Throw<OverflowException>(where).WithMessage("Python integer -1 out of bounds for uint64");
                // NumPy's text past 2**31 follows the platform's C long (Windows: "Python int too large to convert to C
                // long"); the exception type is what both platforms share.
                Action wide = () => StoreAll(np.zeros(new Shape(count), np.int32), Tile(new uint[] { 1 }, count, 4294967295u, at));
                wide.Should().Throw<OverflowException>(where);
                Action infinite = () => StoreAll(np.zeros(new Shape(count), np.int8),
                    Tile(new[] { 1.0 }, count, double.PositiveInfinity, at));
                infinite.Should().Throw<OverflowException>(where).WithMessage("cannot convert float infinity to integer");
                Action nan = () => StoreAll(np.zeros(new Shape(count), np.int8), Tile(new[] { 1.0 }, count, double.NaN, at));
                nan.Should().Throw<ValueError>(where).WithMessage("cannot convert float NaN to integer");
            }
        }

        /// <summary>
        ///     The range pass's bounds are NumPy's exactly at the 64-bit classes, where a double cannot hold every integer:
        ///     NumPy 2.4.2 stores <c>2**63 - 1</c> (a Python int) and the doubles <c>-2**63</c>, <c>2**63 - 1024</c> and
        ///     <c>2**64 - 2048</c> (each the last double inside its class), <c>-0.5</c> into uint64 (truncated to 0),
        ///     <c>2147483647.9</c> and <c>-2147483648.9</c> into int32; it raises OverflowError for <c>2**63</c> into int64,
        ///     the doubles <c>-2**63 - 2048</c> and <c>2**63</c> into int64, <c>2**64</c> into uint64 and <c>-2147483649.0</c>
        ///     into int32 (texts of the platform's C long; the type is pinned), and <c>-1.0</c> into uint64 ("Python
        ///     integer -1 out of bounds for uint64"). Each as one item and as 35 equal items.
        /// </summary>
        /// <remarks>
        ///     The first range pass built its bounds as <c>(decimal)Math.Pow(2, 63)</c>, which .NET rounds to 15
        ///     significant digits (9223372036854780000): 2^63 into int64 and <c>-2**63 - 2048</c> passed and wrapped to
        ///     -2^63 (the one-item lists, stored item by item, were right).
        /// </remarks>
        [TestMethod]
        public void IntegerTargets_RangePassBoundsAreExact()
        {
            foreach (int count in new[] { 1, 35 })
            {
                var i64 = np.zeros(new Shape(count), np.int64);
                StoreAll(i64, Tile(new[] { 9223372036854775807UL }, count));
                i64.ToArray<long>().Should().OnlyContain(v => v == long.MaxValue);
                StoreAll(i64, Tile(new[] { -9223372036854775808.0 }, count));
                i64.ToArray<long>().Should().OnlyContain(v => v == long.MinValue);
                StoreAll(i64, Tile(new[] { 9223372036854774784.0 }, count));
                i64.ToArray<long>().Should().OnlyContain(v => v == 9223372036854774784L);

                var u64 = np.zeros(new Shape(count), np.uint64);
                StoreAll(u64, Tile(new[] { 18446744073709549568.0 }, count));
                u64.ToArray<ulong>().Should().OnlyContain(v => v == 18446744073709549568UL);
                StoreAll(u64, Tile(new[] { -0.5 }, count));
                u64.ToArray<ulong>().Should().OnlyContain(v => v == 0UL);

                var i32 = np.zeros(new Shape(count), np.int32);
                StoreAll(i32, Tile(new[] { 2147483647.9 }, count));
                i32.ToArray<int>().Should().OnlyContain(v => v == int.MaxValue);
                StoreAll(i32, Tile(new[] { -2147483648.9 }, count));
                i32.ToArray<int>().Should().OnlyContain(v => v == int.MinValue);

                foreach (var (dt, list) in new (string, Array)[]
                         {
                             ("int64", new[] { 9223372036854775808UL }),
                             ("int64", new[] { -9223372036854777856.0 }),
                             ("int64", new[] { 9223372036854775808.0 }),
                             ("uint64", new[] { 18446744073709551616.0 }),
                             ("int32", new[] { -2147483649.0 }),
                         })
                {
                    Action overflow = () => StoreAll(np.zeros(new Shape(count), np.dtype(dt)), Tile(list, count));
                    overflow.Should().Throw<OverflowException>($"{dt} <- {list.GetValue(0)} x {count}");
                }
                Action negative = () => StoreAll(np.zeros(new Shape(count), np.uint64), Tile(new[] { -1.0 }, count));
                negative.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint64");
            }
        }

        /// <summary>np.putmask over an all-True mask of <paramref name="a"/>'s size: every value stored, by position.</summary>
        /// <param name="a">The target.</param>
        /// <param name="values">The values (a C# array: a Python list).</param>
        private static void StoreAll(NDArray a, Array values) => np.putmask(a, np.ones(new Shape(a.size), np.@bool), values);

        /// <summary>The items of a typed C# array boxed into an <c>object[]</c>: the same Python list, stored item by item.</summary>
        /// <param name="typed">A one-dimensional typed array.</param>
        /// <returns>A new <c>object[]</c> of the boxed items.</returns>
        private static object[] Boxed(Array typed)
        {
            var items = new object[typed.Length];
            for (int i = 0; i < typed.Length; i++)
                items[i] = typed.GetValue(i);
            return items;
        }

        /// <summary>
        ///     <paramref name="typed"/>'s items cycled to <paramref name="count"/> items of the same C# element type, with
        ///     <paramref name="bad"/> (when given) at position <paramref name="at"/> (the last when negative).
        /// </summary>
        /// <param name="typed">A one-dimensional typed array.</param>
        /// <param name="count">The length of the result.</param>
        /// <param name="bad">An item that replaces the one at <paramref name="at"/>, or null.</param>
        /// <param name="at">The position of <paramref name="bad"/>; negative for the last.</param>
        /// <returns>A new array of <paramref name="typed"/>'s element type.</returns>
        private static Array Tile(Array typed, int count, object bad = null, int at = -1)
        {
            var tiled = Array.CreateInstance(typed.GetType().GetElementType()!, count);
            for (int i = 0; i < count; i++)
                tiled.SetValue(typed.GetValue(i % typed.Length), i);
            if (bad != null)
                tiled.SetValue(bad, at < 0 ? count - 1 : at);
            return tiled;
        }

        /// <summary>
        ///     What did not change, with NumPy's answers: in-range items, floats truncated into integers, str items parsed
        ///     (the error quoting the Python str), a widening typed array (the discover-and-cast route) and float32 targets.
        /// </summary>
        [TestMethod]
        public void InRangeLists_KeepNumPysValues()
        {
            var a = np.zeros(new Shape(3), np.int64);
            np.place(a, Mask(), new[] { 7, 8 });
            a.ToString(true).Should().Be("array([7, 0, 8])");

            var f = np.zeros(new Shape(3), np.float32);
            np.putmask(f, Mask(), new object[] { 1, 2.5 });
            f.ToString(true).Should().Be("array([1., 0., 1.], dtype=float32)");

            var i8 = np.zeros(new Shape(3), np.int8);
            np.place(i8, Mask(), new object[] { 1, 2.5 });
            i8.ToString(true).Should().Be("array([1, 0, 2], dtype=int8)");

            Action parse = () => np.place(np.zeros(new Shape(3), np.int32), Mask(), new object[] { "7", "8x" });
            parse.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: '8x'");

            var d = np.zeros(new Shape(3));
            np.place(d, Mask(), new[] { 1.5, 2.5 });
            d.ToString(true).Should().Be("array([1.5, 0. , 2.5])");
        }
    }
}
