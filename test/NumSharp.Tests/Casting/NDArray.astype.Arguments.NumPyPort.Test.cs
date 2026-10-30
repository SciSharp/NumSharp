using System;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     <c>ndarray.astype</c>'s argument parsing, NumPy 2.4.2's <c>array_astype</c> (<c>methods.c</c>): its
    ///     <c>npy_parse_arguments</c> runs the converters in the parameter order - the dtype, the order
    ///     (<c>PyArray_OrderConverter</c>), then astype's casting converter (<c>PyArray_CastingConverterSameValue</c>, which also
    ///     takes NumPy 2.4's <c>'same_value'</c>) - before the dtype is adapted, the <c>copy=False</c> shortcut is taken or the
    ///     rule is checked. Every expected outcome below was probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     Found by P3 batch 9b's replay of astype's snapshot: the numeric path validated the casting name only for a cast
    ///     between two storage types and with the five-name text of <c>np.can_cast</c>, the text path read the casting name
    ///     before the order, and the chararray's numeric path took its <c>copy=False</c> shortcut before either.
    /// </remarks>
    [TestClass]
    public class AstypeArgumentsNumPyPortTests
    {
        private const string OrderError = "order must be one of 'C', 'F', 'A', or 'K' (got 'Z')";

        private const string CastingError =
            "casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe', 'same_value' (got 'bogus')";

        /// <summary>
        ///     A bad order AND a bad casting name report the order: NumPy's order converter runs first
        ///     (<c>np.arange(3.0).astype('f8', order='Z', casting='bogus')</c> is the order's ValueError).
        /// </summary>
        [TestMethod]
        public void OrderIsCheckedBeforeCasting_Numeric()
        {
            var a = np.arange(3.0);
            new Action(() => a.astype(np.float64, true, 'Z', "bogus"))
                .Should().Throw<ValueError>().WithMessage(OrderError);
        }

        /// <summary>
        ///     The same order on the string path (<c>np.array(['a']).astype('U3', order='Z', casting='bogus')</c>); it read the
        ///     casting name first.
        /// </summary>
        [TestMethod]
        public void OrderIsCheckedBeforeCasting_Text()
        {
            var a = np.array(new[] { "a" });
            new Action(() => a.astype(np.dtype("U3"), true, 'Z', "bogus"))
                .Should().Throw<ValueError>().WithMessage(OrderError);
        }

        /// <summary>
        ///     The casting name is parsed before the <c>copy=False</c> shortcut, so a cast that needs nothing still refuses an
        ///     unknown name (<c>np.arange(3.0).astype('f8', casting='bogus', copy=False)</c>); NumSharp returned the array
        ///     itself. The text lists <c>'same_value'</c>, astype's own converter's sixth name.
        /// </summary>
        [TestMethod]
        public void CastingNameIsCheckedBeforeTheCopyFalseShortcut()
        {
            var a = np.arange(3.0);
            new Action(() => a.astype(np.float64, false, 'K', "bogus"))
                .Should().Throw<ValueError>().WithMessage(CastingError);
            new Action(() => a.astype(np.float64, true, 'K', "bogus"))
                .Should().Throw<ValueError>().WithMessage(CastingError);
        }

        /// <summary>
        ///     A cast between two storage types names <c>'same_value'</c> in the list too
        ///     (<c>np.arange(3).astype(np.float64, casting='bogus')</c>); NumSharp used <c>np.can_cast</c>'s five-name text.
        /// </summary>
        [TestMethod]
        public void CastingNameErrorListsSameValue()
        {
            var a = np.arange(3L);
            new Action(() => a.astype(np.float64, true, 'K', "bogus"))
                .Should().Throw<ValueError>().WithMessage(CastingError);
        }

        /// <summary>
        ///     A chararray over int8 data parses its arguments before its own <c>copy=False</c> shortcut too
        ///     (<c>np.char.array(['1', '-5']).astype(np.int8)</c>, then <c>.astype(np.int8, copy=False, casting='bogus')</c> and
        ///     <c>order='Z'</c>).
        /// </summary>
        [TestMethod]
        public void CharArrayParsesItsArgumentsBeforeItsShortcut()
        {
            NDCharArray digits = np.@char.array(new[] { "1", "-5" }).astype(np.int8);
            new Action(() => digits.astype(np.int8, false, 'K', "bogus"))
                .Should().Throw<ValueError>().WithMessage(CastingError);
            new Action(() => digits.astype(np.int8, false, 'Z', "bogus"))
                .Should().Throw<ValueError>().WithMessage(OrderError);
            digits.astype(np.int8, false, 'K', "no").Should().BeSameAs(digits);
        }

        /// <summary>
        ///     <c>'same_value'</c> within one storage type copies the values (<c>np.arange(3.0).astype(np.float64,
        ///     casting='same_value')</c> is <c>[0., 1., 2.]</c>): no value can change, so NumSharp needs no value check there.
        /// </summary>
        [TestMethod]
        public void SameValueWithinOneStorageTypeRuns()
        {
            var a = np.arange(3.0);
            var copy = a.astype(np.float64, true, 'K', "same_value");
            copy.Should().NotBeSameAs(a);
            copy.dtype.Should().Be(np.float64);
            copy.ToArray<double>().Should().Equal(0.0, 1.0, 2.0);
        }

        /// <summary>
        ///     <c>'same_value'</c> between two numeric storage types needs NumPy 2.4's value-checking cast loops, which
        ///     NumSharp's numeric casts do not have: a <see cref="NotSupportedException"/> naming both dtypes, where the name used
        ///     to be refused as unknown. NumPy's own answers (<c>[2.0, 3.0]</c> to int32 works, <c>[1.5]</c> raises
        ///     <c>could not cast 'same_value' double to long</c>) are pinned in <see cref="OpenBugs"/>.
        /// </summary>
        [TestMethod]
        public void SameValueAcrossNumericStorageTypesIsNotSupported()
        {
            var a = np.array(new[] { 2.0, 3.0 });
            new Action(() => a.astype(np.int32, true, 'K', "same_value"))
                .Should().Throw<NotSupportedException>().WithMessage("*same_value*float64*int32*");
        }
    }
}
