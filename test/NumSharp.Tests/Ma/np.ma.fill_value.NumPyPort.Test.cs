using System;
using System.Numerics;
using AwesomeAssertions;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    ///     numpy.ma's fill value as NumPy STORES it (NumPy 2.4.2's <c>_check_fill_value</c>, the <c>fill_value</c> getter
    ///     and setter, <c>_update_from</c> / <c>__array_finalize__</c>), for numeric dtypes — the text path's model, which
    ///     the numeric dtypes now share. Every expected value is NumPy's, probed.
    /// </summary>
    /// <remarks>
    ///     The rundown's bug 4 (2026-10-08): a numeric fill was the caller's raw C# value, never converted
    ///     (<c>astype(int64)</c> kept 7.5, <c>fill_value=300</c> on int8 was accepted) and never materialized (each read
    ///     computed the default again). The tier <c>ma_fill.jsonl</c> gates the whole matrix; these tests pin one case of
    ///     each rule with its reason.
    ///     <para>
    ///         Every boxed fill goes through <see cref="Exactly"/> and every error through <see cref="Raises{T}"/>, because
    ///         AwesomeAssertions' object <c>Be</c> compares numbers ACROSS types (<c>999999L</c> equals <c>999999UL</c>) and
    ///         <c>WithMessage</c> is a case-insensitive wildcard pattern: a planted bug that stored the unsigned default as
    ///         int64, or formatted <c>1e+20</c> as C#'s <c>1E+20</c>, passed the first version of these tests and was caught
    ///         only by the oracle tier.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_ma_fill_value_numpy_port_tests
    {
        private static NDMaskedArray Ma(NDArray data, object fill = null)
            => np.ma.array(data, mask: np.array(new[] { false, true, false, false }), fill_value: fill);

        private static NDArray I8() => np.array(new sbyte[] { 1, 2, 3, 4 });
        private static NDArray F64() => np.array(new[] { 1.5, 2.5, 3.5, 4.5 });

        /// <summary>
        ///     Asserts the getter's boxed scalar is NumPy's scalar TYPE as well as its value (<c>np.uint64(999999)</c> is a
        ///     <see cref="ulong"/>, never a <see cref="long"/> that happens to compare equal).
        /// </summary>
        /// <param name="actual">The <c>fill_value</c> getter's result.</param>
        /// <param name="expected">NumPy's scalar, as its C# type.</param>
        /// <param name="because">Why the value is expected, shown on a failure.</param>
        private static void Exactly(object actual, object expected, string because = "")
        {
            actual.Should().NotBeNull(because);
            actual.GetType().Should().Be(expected.GetType(), because);
            actual.Should().Be(expected, because);
        }

        /// <summary>
        ///     Asserts the call raises <typeparamref name="T"/> with exactly NumPy's text — compared whole and
        ///     case-sensitively, which <c>WithMessage</c> (a case-insensitive wildcard pattern) does not.
        /// </summary>
        /// <typeparam name="T">The expected exception type; a derived type also satisfies it.</typeparam>
        /// <param name="act">The call.</param>
        /// <param name="text">NumPy's message.</param>
        private static void Raises<T>(Action act, string text) where T : Exception
            => act.Should().Throw<T>().Which.Message.Should().Be(text);

        /// <summary>The getter materializes the default at the DEFAULT's own dtype: int64 for signed integer data, uint64
        /// for unsigned, float64 for every float, complex128, bool.</summary>
        [TestMethod]
        public void Default_IsStoredAtItsOwnDtype()
        {
            var x = Ma(I8());
            x._fill_value.Should().BeNull("nothing is stored until something reads the fill");
            Exactly(x.fill_value, 999999L);
            x._fill_value.dtype.Should().Be(np.int64);
            Exactly(Ma(np.array(new byte[] { 1, 2, 3, 4 })).fill_value, 999999UL, "default_fill_value is np.uint for kind 'u'");
            Exactly(Ma(np.array(new float[] { 1, 2, 3, 4 })).fill_value, 1e20);
            Exactly(Ma(np.array(new[] { Half.One, Half.One, Half.One, Half.One })).fill_value, 1e20);
            Exactly(Ma(np.array(new[] { true, false, true, false })).fill_value, true);
            Exactly(Ma(np.array(new[] { Complex.One, Complex.One, Complex.One, Complex.One })).fill_value, new Complex(1e20, 0));
        }

        /// <summary>The constructor converts a given fill to the data dtype: a Python scalar (a C# number) is packed with
        /// PyArray_Pack's range checks, a NumPy scalar (Half) or an array is cast.</summary>
        [TestMethod]
        public void Constructor_ConvertsLikeNumPysAsarray()
        {
            Exactly(Ma(I8(), 5.7).fill_value, (sbyte)5);
            Exactly(Ma(I8(), (Half)300).fill_value, (sbyte)44, "np.float16(300) is cast into int8");
            Exactly(Ma(I8(), NDArray.Scalar(1000L)).fill_value, (sbyte)-24, "a 0-d int64 array is cast");
            Exactly(Ma(I8(), true).fill_value, (sbyte)1);
            Exactly(Ma(np.array(new[] { Half.One, Half.One, Half.One, Half.One }), 70000).fill_value, Half.PositiveInfinity);
            Exactly(Ma(np.array(new[] { true, false, true, false }), double.NaN).fill_value, true);
            Exactly(Ma(F64(), 5).fill_value, 5.0, "the fill takes the data's dtype");
        }

        /// <summary>A Python scalar that does not fit raises NumPy's TypeError (OverflowError / ValueError caught and
        /// rewrapped); a complex into a real dtype raises Python's own int() / float() TypeError; a str for a numeric array
        /// is refused by its type.</summary>
        [TestMethod]
        public void Constructor_RefusesWithNumPysTexts()
        {
            Raises<TypeError>(() => Ma(I8(), 300), "Cannot convert fill_value 300 to dtype int8");
            Raises<TypeError>(() => Ma(I8(), 1e20), "Cannot convert fill_value 1e+20 to dtype int8");
            Raises<TypeError>(() => Ma(np.array(new long[] { 1, 2, 3, 4 }), double.NaN), "Cannot convert fill_value nan to dtype int64");
            Raises<TypeError>(() => Ma(np.array(new byte[] { 1, 2, 3, 4 }), -1), "Cannot convert fill_value -1 to dtype uint8");
            Raises<TypeError>(() => Ma(I8(), new Complex(1, 2)),
                              "int() argument must be a string, a bytes-like object or a real number, not 'complex'");
            Raises<TypeError>(() => Ma(F64(), new Complex(1, 2)),
                              "float() argument must be a string or a real number, not 'complex'");
            Raises<TypeError>(() => Ma(I8(), "x"), "Cannot set fill value of string with array of dtype int8");
        }

        /// <summary>A set converts to the data dtype and is stored when nothing is; once a default was read, the set is
        /// written INTO the stored int64 array, so the same assignment reads back another type.</summary>
        [TestMethod]
        public void Setter_DependsOnWhatIsStored()
        {
            var unread = Ma(I8());
            unread.fill_value = 5;
            Exactly(unread.fill_value, (sbyte)5);

            var read = Ma(I8());
            _ = read.fill_value;
            read.fill_value = 5;
            Exactly(read.fill_value, 5L, "written into the materialized int64 default");

            // `fill_value = None` writes the int64 default 999999 into the stored int8 5: 63.
            var reset = Ma(I8(), 5);
            reset.fill_value = null;
            Exactly(reset.fill_value, (sbyte)63);

            Raises<TypeError>(() => Ma(I8()).fill_value = 300, "Cannot convert fill_value 300 to dtype int8");
            Raises<ValueError>(() => Ma(I8(), 5).fill_value = np.array(new long[] { 1, 2 }), "setting an array element with a sequence.");
        }

        /// <summary>astype re-checks a STORED fill against the new dtype (an array cast: never raising); an unread array
        /// stores nothing, so the result reads its own default.</summary>
        [TestMethod]
        public void Astype_ConvertsTheStoredFill_DependingOnReads()
        {
            Exactly(Ma(F64(), 7.5).astype(np.int64).fill_value, 7L);
            Exactly(Ma(F64()).astype(np.int64).fill_value, 999999L, "nothing stored: the result's own default");
            var read = Ma(F64());
            _ = read.fill_value;
            Exactly(read.astype(np.int64).fill_value, long.MinValue, "1e20 cast into int64 (MSVC's out-of-range cast)");
            Exactly(read.astype(np.int8).fill_value, (sbyte)0);
        }

        /// <summary>A view of the _arraymethod family SHARES the stored object (a set reaches both); a copy-based
        /// derivation (astype, view, byteswap) stores its own converted copy.</summary>
        [TestMethod]
        public void Inheritance_SharesOrCopies_AsNumPy()
        {
            var x = Ma(np.array(new long[] { 1, 4, 9, 16 }), 5);
            var slice = (NDMaskedArray)x["1:"];
            var copy = x.copy();
            var cast = x.astype(np.float64);
            x.fill_value = 7;
            Exactly(slice.fill_value, 7L);
            Exactly(copy.fill_value, 7L, "x.copy() is an _arraymethod: _update_from shares the fill");
            Exactly(cast.fill_value, 5.0, "astype stored a converted copy");

            slice.fill_value = 8;
            Exactly(x.fill_value, 8L, "a set on the slice reaches the parent too");

            var unread = Ma(np.array(new long[] { 1, 4, 9, 16 }));
            var child = (NDMaskedArray)unread["1:"];
            unread.fill_value = 7;
            Exactly(child.fill_value, 999999L, "nothing was stored to share: the child reads its own default");
        }

        /// <summary>filled() casts the stored fill into the data under 'same_kind' (the int64 default 999999 into int8 is
        /// 63); a fill of another kind shared with a derived array is NumPy's object-array fallback, which NumSharp
        /// refuses.</summary>
        [TestMethod]
        public void Filled_CastsSameKind_AndRefusesTheObjectFallback()
        {
            Ma(I8()).filled().ToArray<sbyte>().Should().Equal(1, 63, 3, 4);
            Ma(I8()).filled(2.5).ToArray<sbyte>().Should().Equal(1, 2, 3, 4);
            Raises<TypeError>(() => Ma(I8()).filled(300), "Cannot convert fill_value 300 to dtype int8");

            // ma.absolute of a complex array shares its complex fill with the float result: NumPy returns
            // array([2.23606797749979, np.complex128(1+1j)], dtype=object).
            var c = np.ma.array(np.array(new[] { new Complex(1, 2), new Complex(3, 0) }), mask: np.array(new[] { false, true }),
                                fill_value: new Complex(1, 1));
            Action objectFallback = () => np.ma.absolute(c).filled();
            objectFallback.Should().Throw<NotSupportedException>().WithInnerException<InvalidCastException>();
            Raises<NotSupportedException>(objectFallback,
                "filled(): the fill value of dtype complex128 cannot be cast to the data dtype float64 under the 'same_kind' " +
                "rule, so NumPy returns an object array (np.choose over the data as objects); NumSharp has no object dtype. " +
                "Pass a fill_value of the data's kind.");
        }

        /// <summary>The repr materializes the default and prints it by the STORED dtype: the plain str when it is the
        /// data's, NumPy's scalar repr otherwise.</summary>
        [TestMethod]
        public void Repr_PrintsTheStoredFillsDtype()
        {
            var x = Ma(I8());
            x.ToString(true).Should().Contain("fill_value=np.int64(999999)");
            x._fill_value.Should().NotBeNull("the repr read the getter");
            Ma(I8(), 1).ToString(true).Should().Contain("fill_value=1,");
            Ma(np.array(new float[] { 1, 2, 3, 4 })).ToString(true).Should().Contain("fill_value=np.float64(1e+20)");
            Ma(np.array(new long[] { 1, 2, 3, 4 })).ToString(true).Should().Contain("fill_value=999999)");
            Ma(np.array(new[] { true, false, true, false }), true).ToString(true).Should().Contain("fill_value=True)");
            // ma.sqrt of an int8 array keeps the int8 fill object: np.int8(7) beside float16 data, which prints as the
            // Python floats astype(object) makes of it.
            np.ma.sqrt(Ma(I8(), 7)).ToString(true).Should().Be(
                "masked_array(data=[1.0, --, 1.732421875, 2.0],\n             mask=[False,  True, False, False],\n" +
                "       fill_value=np.int8(7),\n            dtype=float16)");
        }

        /// <summary>The masked singleton stores nothing and refuses the getter's write, as NumPy's MaskedConstant does.</summary>
        [TestMethod]
        public void MaskedSingleton_RefusesItsFill()
        {
            Raises<AttributeError>(() => _ = np.ma.masked.fill_value, "attributes of masked are not writeable");
            Raises<AttributeError>(() => np.ma.masked.fill_value = 5, "attributes of masked are not writeable");
            Raises<AttributeError>(() => np.ma.masked.filled(), "attributes of masked are not writeable");
            np.ma.masked.filled(3).GetDouble().Should().Be(3.0);
            np.ma.masked._fill_value.Should().BeNull();
        }

        /// <summary>
        ///     <c>ma.default_fill_value(obj)</c> takes an array or a scalar as well as a dtype (NumPy's <c>_get_dtype_of</c>:
        ///     the descriptor, the array's dtype, else <c>np.asanyarray(obj).dtype</c>); only the kind decides the value.
        ///     None is a 0-d object array, whose default is <c>'?'</c>. Every expected value is NumPy 2.4.2's.
        /// </summary>
        [TestMethod]
        public void DefaultFillValue_OfAnArrayOrAScalar()
        {
            Exactly(np.ma.default_fill_value(np.array(new byte[] { 1 })), 999999L);
            Exactly(np.ma.default_fill_value(np.ma.array(np.array(new[] { 1.5f }))), 1e20);
            Exactly(np.ma.default_fill_value(np.array(new sbyte[0])), 999999L, "an empty int8 array keeps its dtype");
            Exactly(np.ma.default_fill_value((sbyte)3), 999999L, "np.int8(3)");
            Exactly(np.ma.default_fill_value(3), 999999L);
            Exactly(np.ma.default_fill_value(1.5), 1e20);
            Exactly(np.ma.default_fill_value(true), true);
            Exactly(np.ma.default_fill_value((Half)1), 1e20, "np.float16(1.0)");
            Exactly(np.ma.default_fill_value(new Complex(1, 2)), new Complex(1e20, 0));
            Exactly(np.ma.default_fill_value(new[] { 1, 2 }), 999999L, "[1, 2]");
            Exactly(np.ma.default_fill_value(new System.Collections.Generic.List<int> { 1, 2 }), 999999L, "a List is a Python list");
            Exactly(np.ma.default_fill_value(new object[] { 1.5, 2 }), 1e20, "[1.5, 2] is float64");
            Exactly(np.ma.default_fill_value(0), 999999L, "a literal 0 binds the object overload, not NPTypeCode.Empty");
            Exactly(np.ma.default_fill_value(new double[0]), 1e20, "[] is float64");
            Exactly(np.ma.default_fill_value(np.ma.masked), 1e20, "the masked singleton is float64");
            Exactly(np.ma.default_fill_value((object)"x"), "N/A", "a str is text");
            Exactly(np.ma.default_fill_value(np.array(new[] { "ab" })), "N/A");
            ((byte[])np.ma.default_fill_value(new byte[] { (byte)'x' })).Should().Equal((byte)'N', (byte)'/', (byte)'A');
            ((byte[])np.ma.default_fill_value(NDArray.Scalar(new byte[] { 97, 98 }))).Should().Equal((byte)'N', (byte)'/', (byte)'A');
            Exactly(np.ma.default_fill_value((DType)null), "?", "None: default_filler['O']");
            Exactly(np.ma.default_fill_value((object)null), "?");
            Exactly(np.ma.default_fill_value(np.complex128), new Complex(1e20, 0));
        }

        /// <summary>Neighbours of the inheritance rules: byteswap / argpartition / cumsum give the result its own mask
        /// (a mask write through it never reaches the source); argpartition converts the fill; a C# int literal added to
        /// a narrow masked array is numpy.ma's strong int64.</summary>
        [TestMethod]
        public void Neighbours_MaskCopies_AndTheIntLiteral()
        {
            foreach (var derive in new Func<NDMaskedArray, NDMaskedArray>[] { m => m.byteswap(), m => m.argpartition(1), m => np.ma.cumsum(m) })
            {
                var x = Ma(np.array(new long[] { 3, 1, 2, 5 }), 7);
                var r = derive(x);
                r._mask.SetBoolean(true, 0);
                x._mask.GetBoolean(0).Should().BeFalse();
            }
            var f = Ma(F64(), 7.5);
            Exactly(f.argpartition(1).fill_value, 7L);
            var h = Ma(I8(), 7);
            h.harden_mask();
            h.byteswap().hardmask.Should().BeTrue();
            (Ma(I8()) + 1).dtype.Should().Be(np.int64);
            (1 + Ma(I8())).dtype.Should().Be(np.int64);
            (Ma(I8()) * 2).dtype.Should().Be(np.int64);
            (Ma(np.array(new[] { true, false, true, false })) + 1).dtype.Should().Be(np.int64);
        }
    }
}
