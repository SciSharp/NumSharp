using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.Tests.Backends.Unmanaged
{
    /// <summary>
    ///     Regression gates for the public element members of <see cref="NDArray"/> / <see cref="UnmanagedStorage"/> —
    ///     <c>GetValue</c>, <c>GetValue&lt;T&gt;</c>, <c>GetAtIndex</c>, the typed <c>GetDouble</c> … getters, <c>SetValue</c>,
    ///     <c>SetValue&lt;T&gt;</c>, <c>SetAtIndex</c>, the typed <c>SetDouble</c> … setters, <c>itemset</c> and
    ///     <c>item(i, j, …)</c> — after the parity QA fixes QA-12 and QA-33.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     QA-12: a <c>T</c> other than the dtype used to read or write <c>sizeof(T)</c> raw bytes (a <c>Debug.Assert</c> was
    ///     the only type test), so <c>a.SetValue(2.5, 0)</c> on a float32 array gave <c>[0, 2.0625, 3, 4]</c> and the last
    ///     element wrote past the buffer in Release. The members now cast with NumPy's cast (the cast <c>astype</c> runs);
    ///     the conversion matrix below holds every member to <c>astype</c>'s answer for every dtype pair.
    ///     </para>
    ///     <para>
    ///     QA-33: the coordinates were not checked, so <c>m.GetDouble(0, 5)</c> read the next row and <c>a.SetDouble(9.0,
    ///     10)</c> wrote outside the array. Every public member now wraps a negative coordinate and raises NumPy's
    ///     <c>IndexError</c> texts before touching memory.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ElementAccessParityTests
    {
        /// <summary>The 15 numeric dtypes every conversion test covers (the text lane is gated separately).</summary>
        private static readonly NPTypeCode[] Numeric =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
            NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
            NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        // ----- QA-12: conversion --------------------------------------------------------------------------------------

        /// <summary>
        ///     The parity QA pin: <c>SetValue&lt;double&gt;</c> into float32 stores the cast value and leaves the neighbours,
        ///     and <c>itemset&lt;double&gt;</c> (the same body) does too — in Debug and Release alike now that no
        ///     <c>Debug.Assert</c> stands in for the type test.
        /// </summary>
        [TestMethod]
        public void SetValueGeneric_OtherElementType_CastsInsteadOfWritingRawBytes()
        {
            using var a = np.array(new[] { 1f, 2f, 3f, 4f });
            using var b = np.array(new[] { 1f, 2f, 3f, 4f });

            a.SetValue(2.5, 0);
            a.SetValue(2.5, 3);          // the last element: used to write four bytes past the buffer
            b.itemset(new[] { 1 }, 2.5);

            a.ToArray<float>().Should().Equal(2.5f, 2f, 3f, 2.5f);
            b.ToArray<float>().Should().Equal(1f, 2.5f, 3f, 4f);
        }

        /// <summary>
        ///     The parity QA pin for the reads: <c>GetValue&lt;double&gt;(1)</c>, <c>GetAtIndex&lt;double&gt;(1)</c> and
        ///     <c>GetDouble(1)</c> of float32 <c>[1, 2, 3, 4]</c> are 2.0 (they read elements 1 and 2 as one double).
        /// </summary>
        [TestMethod]
        public void GetValueGeneric_OtherElementType_CastsInsteadOfReadingRawBytes()
        {
            using var a = np.array(new[] { 1f, 2f, 3f, 4f });

            a.GetValue<double>(1).Should().Be(2.0);
            a.GetAtIndex<double>(1).Should().Be(2.0);
            a.GetDouble(1).Should().Be(2.0);
            a.GetValue<double>(3).Should().Be(4.0);   // the last element: used to read past the buffer
        }

        /// <summary>
        ///     The int literal into an int64 array (QA-12's <c>[38654705665, 2, 3, 4]</c>: the four bytes landed in element
        ///     0's high half) and a double into uint8 / int16 (NumPy's C cast: truncate, then wrap).
        /// </summary>
        [TestMethod]
        public void SetValueGeneric_NarrowerAndWiderTypes_StoreNumPysCast()
        {
            using var i64 = np.array(new long[] { 1, 2, 3, 4 });
            using var u8 = np.zeros(new Shape(3), np.uint8);
            using var i16 = np.zeros(new Shape(2), np.int16);

            i64.SetValue(9, 1);
            u8.SetValue(258.9, 1);       // 258 truncated, wrapped to 2
            i16.SetValue(-2.75, 0);      // truncation toward zero

            i64.ToArray<long>().Should().Equal(1L, 9L, 3L, 4L);
            u8.ToArray<byte>().Should().Equal((byte)0, (byte)2, (byte)0);
            i16.ToArray<short>().Should().Equal((short)-2, (short)0);
        }

        /// <summary>
        ///     Every generic and typed member converts exactly as <c>astype</c> does, for every pair of the 15 numeric dtypes:
        ///     the value written by <c>SetValue&lt;S&gt;</c> into a <c>D</c> array equals <c>np.array(new S[]{v}).astype(D)</c>,
        ///     and the neighbours stay zero.
        /// </summary>
        [TestMethod]
        public void SetValueGeneric_EveryDtypePair_MatchesAstype()
        {
            CheckSetAgainstAstype(true);
            CheckSetAgainstAstype((byte)200);
            CheckSetAgainstAstype((sbyte)-100);
            CheckSetAgainstAstype((short)-3000);
            CheckSetAgainstAstype((ushort)60000);
            CheckSetAgainstAstype(-70000);
            CheckSetAgainstAstype(4000000000u);
            CheckSetAgainstAstype(-5000000000L);
            CheckSetAgainstAstype(10000000000UL);
            CheckSetAgainstAstype('A');
            CheckSetAgainstAstype((Half)1.5);
            CheckSetAgainstAstype(-2.75f);
            CheckSetAgainstAstype(300.625);
            CheckSetAgainstAstype(2.5m);
            CheckSetAgainstAstype(new Complex(2.5, -1.0));
        }

        /// <summary>
        ///     Every generic read converts exactly as <c>astype</c> does: <c>GetValue&lt;T&gt;</c> /
        ///     <c>GetAtIndex&lt;T&gt;</c> of an <c>S</c> element equals the element of <c>astype(T)</c>, for every pair.
        /// </summary>
        [TestMethod]
        public void GetValueGeneric_EveryDtypePair_MatchesAstype()
        {
            foreach (var source in Numeric)
            {
                using var a = np.array(new[] { 0.0, 300.625, -2.75 }).astype(source);
                CheckGetAgainstAstype<bool>(a);
                CheckGetAgainstAstype<byte>(a);
                CheckGetAgainstAstype<sbyte>(a);
                CheckGetAgainstAstype<short>(a);
                CheckGetAgainstAstype<ushort>(a);
                CheckGetAgainstAstype<int>(a);
                CheckGetAgainstAstype<uint>(a);
                CheckGetAgainstAstype<long>(a);
                CheckGetAgainstAstype<ulong>(a);
                CheckGetAgainstAstype<char>(a);
                CheckGetAgainstAstype<Half>(a);
                CheckGetAgainstAstype<float>(a);
                CheckGetAgainstAstype<double>(a);
                CheckGetAgainstAstype<decimal>(a);
                CheckGetAgainstAstype<Complex>(a);
            }
        }

        /// <summary>
        ///     The typed getters / setters (<c>GetDouble</c>, <c>GetInt32</c>, <c>SetByte</c>, …) on an array of another
        ///     dtype convert like the generic members (GetDouble's old doc promised a NullReferenceException; the bytes were
        ///     reinterpreted instead).
        /// </summary>
        [TestMethod]
        public void TypedAccessors_OtherDtype_Convert()
        {
            using var i32 = np.array(new[] { 7, -9 });
            using var f64 = np.array(new[] { -2.7, 0.5 });
            using var i16 = np.array(new short[] { 300, 2 });

            i32.GetDouble(1).Should().Be(-9.0);
            f64.GetInt32(0).Should().Be(-2);
            f64.GetBoolean(1).Should().BeTrue();
            i16.GetByte(0).Should().Be((byte)44);

            f64.SetInt32(5, 1);
            i32.SetDouble(3.9, 0);
            f64.ToArray<double>().Should().Equal(-2.7, 5.0);
            i32.ToArray<int>().Should().Equal(3, -9);
        }

        // ----- QA-33: coordinate checks -------------------------------------------------------------------------------

        /// <summary>
        ///     The parity QA pin: <c>m.GetDouble(0, 5)</c> raises NumPy's IndexError (it read the next row) and
        ///     <c>m.GetDouble(1, -1)</c> wraps to the last column.
        /// </summary>
        [TestMethod]
        public void TypedCoordinateGetters_CheckAndWrapLikeItem()
        {
            using var m = np.arange(6.0).reshape(2, 3);

            Action pastTheAxis = () => m.GetDouble(0, 5);

            pastTheAxis.Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 1 with size 3");
            m.GetDouble(1, -1).Should().Be(5.0);
        }

        /// <summary>
        ///     The parity QA pin for the NumPy-named form: <c>item(1, -2)</c> is 4.0 and <c>item(0, 4)</c> raises (they read
        ///     1.0 and 4.0 through the unchecked offset).
        /// </summary>
        [TestMethod]
        public void ItemWithSeveralCoordinates_ChecksAndWrapsEachAxis()
        {
            using var m = np.arange(6.0).reshape(2, 3);
            using var c = np.arange(24.0).reshape(2, 3, 4);

            Action pastTheAxis = () => m.item(0, 4);

            ((double)m.item(1, -2)).Should().Be(4.0);
            ((double)c.item(1, -1, -1)).Should().Be(23.0);
            m.item<int>(-1, -1).Should().Be(5);
            pastTheAxis.Should().Throw<IndexError>().WithMessage("index 4 is out of bounds for axis 1 with size 3");
        }

        /// <summary>
        ///     Every public setter refuses an out-of-range coordinate with NumPy's text before writing: the array is
        ///     unchanged afterwards (the writes used to land outside it).
        /// </summary>
        [TestMethod]
        public void Setters_OutOfRangeCoordinate_RaiseAndWriteNothing()
        {
            using var a = np.array(new[] { 1.0, 2.0, 3.0 });

            Action setDouble = () => a.SetDouble(9.0, 10);
            Action setValueGeneric = () => a.SetValue(9.0, 3);
            Action setValueBoxed = () => a.SetValue((object)9.0, -4);
            Action setAtIndex = () => a.SetAtIndex(9.0, 3);
            Action setAtIndexBoxed = () => a.SetAtIndex((object)9.0, -4);
            Action itemset = () => a.itemset(new[] { 7 }, (object)9.0);

            setDouble.Should().Throw<IndexError>().WithMessage("index 10 is out of bounds for axis 0 with size 3");
            setValueGeneric.Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
            setValueBoxed.Should().Throw<IndexError>().WithMessage("index -4 is out of bounds for axis 0 with size 3");
            setAtIndex.Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
            setAtIndexBoxed.Should().Throw<IndexError>().WithMessage("index -4 is out of bounds for axis 0 with size 3");
            itemset.Should().Throw<IndexError>().WithMessage("index 7 is out of bounds for axis 0 with size 3");
            a.ToArray<double>().Should().Equal(1.0, 2.0, 3.0);
        }

        /// <summary>
        ///     Every public getter refuses an out-of-range coordinate or flat index with NumPy's text (the reads used to
        ///     return memory before or after the array), and wraps a negative one.
        /// </summary>
        [TestMethod]
        public void Getters_OutOfRangeCoordinate_RaiseAndNegativeWraps()
        {
            using var a = np.array(new[] { 1.0, 2.0, 3.0 });
            using var m = np.arange(6.0).reshape(2, 3);

            Action getDouble = () => a.GetDouble(10);
            Action getValue = () => a.GetValue(-4);
            Action getValueGeneric = () => a.GetValue<double>(3);
            Action getAtIndex = () => a.GetAtIndex(3);
            Action getAtIndexFlat2d = () => m.GetAtIndex<double>(10);

            getDouble.Should().Throw<IndexError>().WithMessage("index 10 is out of bounds for axis 0 with size 3");
            getValue.Should().Throw<IndexError>().WithMessage("index -4 is out of bounds for axis 0 with size 3");
            getValueGeneric.Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
            getAtIndex.Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
            getAtIndexFlat2d.Should().Throw<IndexError>().WithMessage("index 10 is out of bounds for size 6");

            a.GetDouble(-1).Should().Be(3.0);
            ((double)a.GetValue(-3)).Should().Be(1.0);
            a.GetAtIndex<double>(-1).Should().Be(3.0);
            m.GetAtIndex<double>(-2).Should().Be(4.0);
        }

        /// <summary>
        ///     More coordinates than dimensions is NumPy's "too many indices" IndexError (the extra coordinate was silently
        ///     added to the offset); fewer keep NumSharp's documented "the rest are filled with 0".
        /// </summary>
        [TestMethod]
        public void Coordinates_TooManyRaise_FewerFillWithZero()
        {
            using var m = np.arange(6.0).reshape(2, 3);

            Action tooMany = () => m.GetDouble(0, 1, 2);

            tooMany.Should().Throw<IndexError>()
                .WithMessage("too many indices for array: array is 2-dimensional, but 3 were indexed");
            m.GetDouble(1).Should().Be(3.0);
        }

        /// <summary>
        ///     A 0-d array's only element takes no coordinate or the flat index 0 / -1; anything else raises NumPy's text.
        /// </summary>
        [TestMethod]
        public void ZeroDim_TakesNoCoordinateOrItsFlatIndex()
        {
            using var s = np.array(5.0);

            Action two = () => s.GetValue(0, 0);
            Action one = () => s.GetDouble(1);

            ((double)s.GetValue()).Should().Be(5.0);
            s.GetDouble(0).Should().Be(5.0);
            s.GetValue<int>(-1).Should().Be(5);
            two.Should().Throw<IndexError>().WithMessage("too many indices for array: array is 0-dimensional, but 2 were indexed");
            one.Should().Throw<IndexError>().WithMessage("index 1 is out of bounds for size 1");
        }

        /// <summary>
        ///     Coordinates address views through their strides and offset: a reversed view's <c>-1</c> is the base's first
        ///     element, a transposed view's <c>(2, 0)</c> the base's <c>(0, 2)</c>, and a write through an offset view lands in
        ///     the addressed element of the base.
        /// </summary>
        [TestMethod]
        public void Coordinates_OnViews_AddressTheViewedElement()
        {
            using var a = np.arange(6.0).reshape(2, 3);
            using var reversed = a["::-1, ::-1"];
            using var transposed = a.T;
            using var row = a["1"];

            reversed.GetDouble(-1, -1).Should().Be(0.0);
            transposed.GetDouble(2, 0).Should().Be(2.0);
            row.SetDouble(40.0, -1);
            row.SetValue(30, 0);

            a.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 30.0, 4.0, 40.0);
        }

        /// <summary>
        ///     <c>SetData(object scalar, coordinates)</c> on a view stores at the addressed element: its scalar branch used
        ///     to hand the storage OFFSET to <c>SetAtIndex</c>, which reads a logical flat index and transformed it again
        ///     (one element too far for a view at offset 1).
        /// </summary>
        [TestMethod]
        public void SetDataScalar_OnOffsetView_StoresAtTheAddressedElement()
        {
            using var a = np.arange(5L);
            using var tail = a["1:"];

            tail.SetData((object)70L, new[] { 0 });
            tail.Storage.SetData((object)90L, new long[] { -1 });   // the long[] overload lives on the storage

            a.ToArray<long>().Should().Equal(0L, 70L, 2L, 3L, 90L);
        }

        /// <summary>
        ///     The read-only check still comes first: a typed setter on a read-only array raises NumPy's ValueError.
        /// </summary>
        [TestMethod]
        public void TypedSetter_ReadOnly_RaisesValueError()
        {
            using var a = np.array(new[] { 1.0, 2.0 });
            a.setflags(write: false);

            Action set = () => a.SetDouble(5.0, 0);

            set.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
        }

        // ----- helpers ------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Writes <paramref name="value"/> with <c>SetValue&lt;S&gt;</c> into the middle of a zeroed array of every numeric
        ///     dtype and compares the stored element with <c>astype</c>'s cast of the same value.
        /// </summary>
        /// <typeparam name="S">The value's C# type.</typeparam>
        /// <param name="value">The value to write.</param>
        /// <exception cref="AssertFailedException">A stored element differs from astype's, or a neighbour changed.</exception>
        private static void CheckSetAgainstAstype<S>(S value) where S : unmanaged
        {
            foreach (var target in Numeric)
            {
                using var src = np.array(new[] { value });
                NDArray expected;
                try
                {
                    expected = src.astype(target);
                }
                catch (Exception)
                {
                    continue; // astype refuses the pair, so there is no NumPy answer to hold SetValue to
                }

                using (expected)
                using (var dst = np.zeros(new Shape(3), target))
                using (var zero = np.zeros(new Shape(1), target))
                {
                    dst.SetValue(value, 1);
                    dst.GetAtIndex(1).Should().Be(expected.GetAtIndex(0), "SetValue<{0}>({1}) into {2}", typeof(S).Name, value, target);
                    dst.GetAtIndex(0).Should().Be(zero.GetAtIndex(0), "the left neighbour of a {0} write into {1}", typeof(S).Name, target);
                    dst.GetAtIndex(2).Should().Be(zero.GetAtIndex(0), "the right neighbour of a {0} write into {1}", typeof(S).Name, target);
                }
            }
        }

        /// <summary>
        ///     Reads every element of <paramref name="a"/> with <c>GetValue&lt;T&gt;</c> and <c>GetAtIndex&lt;T&gt;</c> and
        ///     compares them with the elements of <c>a.astype(T)</c>.
        /// </summary>
        /// <typeparam name="T">The type to read the elements as.</typeparam>
        /// <param name="a">A 1-D array of any numeric dtype.</param>
        /// <exception cref="AssertFailedException">A read differs from astype's element.</exception>
        private static void CheckGetAgainstAstype<T>(NDArray a) where T : unmanaged
        {
            NDArray expected;
            try
            {
                expected = a.astype(InfoOf<T>.NPTypeCode);
            }
            catch (Exception)
            {
                return; // astype refuses the pair
            }

            using (expected)
            {
                for (long i = 0; i < a.size; i++)
                {
                    a.GetValue<T>(i).Should().Be(expected.GetAtIndex<T>(i), "GetValue<{0}> of {1} element {2}", typeof(T).Name, a.typecode, i);
                    a.GetAtIndex<T>(i).Should().Be(expected.GetAtIndex<T>(i), "GetAtIndex<{0}> of {1} element {2}", typeof(T).Name, a.typecode, i);
                }
            }
        }
    }
}
