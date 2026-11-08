using System;
using AwesomeAssertions;
using NumSharp.Backends;

namespace NumSharp.Tests.Ma
{
    /// <summary>
    ///     numpy.ma over NEP 55 <c>StringDType</c> (strings phase 1, P1.7d): the default fill value <c>'N/A'</c>, the
    ///     extremum members' <c>Unsuitable type</c> refusal, the set routines' refusals (reached without computing the
    ///     uniques before them), the legacy products' descriptor and object-product answers, and the stores. Every
    ///     expected value and message is NumPy 2.4.2's output for the same call; the <c>tma</c> section of the string
    ///     oracle covers the whole surface, these pin the decisions by example.
    /// </summary>
    [TestClass]
    public class np_ma_stringdtype_tests
    {
        /// <summary>NumPy's <c>StringDType()</c>.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>NumPy's <c>StringDType(na_object=None)</c>.</summary>
        private static DType TN => np.dtypes.StringDType(na_object: null);

        /// <summary>NumPy's <c>StringDType(na_object=np.nan)</c>.</summary>
        private static DType TNan => np.dtypes.StringDType(na_object: double.NaN);

        /// <summary>NumPy's <c>StringDType(coerce=False)</c>.</summary>
        private static DType TC => np.dtypes.StringDType(coerce: false);

        /// <summary>NumPy's <c>ma.array(np.array(['ab', 'c', 'de'], dt), mask=[1, 0, 0])</c>.</summary>
        /// <param name="dt">The descriptor (default <see cref="T"/>).</param>
        /// <param name="hard">The <c>hard_mask=</c> argument.</param>
        /// <returns>The masked array, its first element masked.</returns>
        private static NDMaskedArray MT(DType dt = null, bool hard = false)
            => np.ma.array(np.array(new[] { "ab", "c", "de" }, dt ?? T), mask: new[] { true, false, false }, hard_mask: hard);

        /// <summary>An empty <c>StringDType</c> masked array with an explicit (all-False, empty) mask, NumPy's
        ///     <c>ma.array(np.zeros(0, T), mask=np.zeros(0, bool))</c>.</summary>
        /// <returns>The empty masked array.</returns>
        private static NDMaskedArray MTe() => np.ma.array(np.array(new string[0], T), mask: new bool[0]);

        /// <summary>
        ///     The default fill value of every <c>StringDType</c> is NumPy's <c>'N/A'</c> (<c>default_filler['T']</c>), and
        ///     <c>filled()</c> writes it into the masked slots; an NA element that is not masked stays the NA object.
        /// </summary>
        [TestMethod]
        public void DefaultFill_IsNA_AndFilledWritesIt()
        {
            np.ma.default_fill_value(T).Should().Be("N/A");
            np.ma.default_fill_value(TN).Should().Be("N/A");
            MT().fill_value.Should().Be("N/A");
            using var filled = MT().filled();
            filled.GetAtIndex(0).Should().Be("N/A");
            filled.GetAtIndex(1).Should().Be("c");

            var na = np.ma.array(np.array(new object[] { "ab", null, "de" }, TN), mask: new[] { true, false, false });
            using var naFilled = na.filled();
            naFilled.GetAtIndex(0).Should().Be("N/A");
            naFilled.GetAtIndex(1).Should().BeNull();
        }

        /// <summary>
        ///     numpy.ma's extremum fills are keyed by the dtype's scalar type, and a <c>StringDType</c>'s is Python's
        ///     <c>str</c>, which no entry names: every member that asks for one refuses with <c>Unsuitable type</c> — before
        ///     it reads the mask, so an all-False mask refuses too (the median), and the NA descriptor is in the text.
        /// </summary>
        [TestMethod]
        public void ExtremumMembers_RaiseUnsuitableType()
        {
            Action min = () => np.ma.min(MT());
            min.Should().Throw<TypeError>().WithMessage("Unsuitable type StringDType() for calculating minimum.");
            Action max = () => MT(TN).max();
            max.Should().Throw<TypeError>().WithMessage("Unsuitable type StringDType(na_object=None) for calculating maximum.");
            Action fill = () => np.ma.minimum_fill_value(T);
            fill.Should().Throw<TypeError>().WithMessage("Unsuitable type StringDType() for calculating minimum.");
            Action argsort = () => MT().argsort(-1);
            argsort.Should().Throw<TypeError>().WithMessage("Unsuitable type StringDType() for calculating minimum.");
            Action sort = () => np.ma.sort(MT());
            sort.Should().Throw<TypeError>().WithMessage("Unsuitable type StringDType() for calculating minimum.");
            Action median = () => np.ma.median(np.ma.array(np.array(new[] { "ab", "c", "de" }, T), mask: new[] { false, false, false }));
            median.Should().Throw<TypeError>().WithMessage("Unsuitable type StringDType() for calculating minimum.");
            // The axis is checked first (_ureduce's normalize_axis_tuple).
            Action medianAxis = () => np.ma.median(MT(), 5);
            medianAxis.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
        }

        /// <summary>
        ///     <c>ma.unique</c>: a masked slot makes <c>_unique1d</c> sort the masked array (the refusal); without one NumPy
        ///     hashes the text and sorts the distinct values (strings phase 2, P2.2 over <c>StringDType</c>): NumPy's
        ///     <c>masked_array(data=['ab', 'c', 'de'], mask=False)</c>, its mask <c>nomask</c>.
        /// </summary>
        [TestMethod]
        public void Unique_MaskedRefuses_UnmaskedIsHashedOrShapeDecided()
        {
            Action masked = () => np.ma.unique(MT());
            masked.Should().Throw<TypeError>().WithMessage("Unsuitable type StringDType() for calculating minimum.");
            var three = np.ma.unique(np.ma.array(np.array(new[] { "de", "c", "ab", "c" }, T)));
            three.shape.Should().Equal(3L);
            three._data.GetAtIndex(0).Should().Be("ab");
            three._data.GetAtIndex(1).Should().Be("c");
            three._data.GetAtIndex(2).Should().Be("de");
            three.dtype.Should().Be(T);
            ReferenceEquals(three.mask, np.ma.nomask).Should().BeTrue();

            var empty = np.ma.unique(MTe());
            empty.size.Should().Be(0);
            empty.dtype.Should().Be(T);
            ReferenceEquals(empty.mask, np.ma.nomask).Should().BeTrue();
            var one = np.ma.unique(np.ma.array(np.array("ab", T)));
            one.shape.Should().Equal(1L);
            one._data.GetAtIndex(0).Should().Be("ab");
        }

        /// <summary>
        ///     The set routines end in a refusal whenever a <c>StringDType</c> operand is involved, and NumSharp reaches it
        ///     WITHOUT the computations NumPy makes before it: <c>isin(MU, MT)</c> sorts the <c>U</c> text of <c>MU</c> first
        ///     (which NumSharp's phase 1 cannot do) and then refuses at <c>unique(MT)</c>.
        /// </summary>
        [TestMethod]
        public void SetRoutines_RefuseWithoutTheUniquesBeforeIt()
        {
            const string unsuitable = "Unsuitable type StringDType() for calculating minimum.";
            Action isin = () => np.ma.isin(MT(), np.array(new[] { 1.5 }));
            isin.Should().Throw<TypeError>().WithMessage(unsuitable);
            Action isinU = () => np.ma.isin(np.ma.array(np.array(new[] { "ab", "c" }), mask: new[] { true, false }), MT());
            isinU.Should().Throw<TypeError>().WithMessage(unsuitable);
            Action in1d = () => np.ma.in1d(np.array(new[] { 1.5, 2.0, 3.0 }), MT());
            in1d.Should().Throw<TypeError>().WithMessage(unsuitable);
            // An empty StringDType operand is not shortcut: in1d's argsort of the join refuses it.
            Action setdiff = () => np.ma.setdiff1d(MTe(), MTe());
            setdiff.Should().Throw<TypeError>().WithMessage(unsuitable);
            Action intersect = () => np.ma.intersect1d(np.ma.array(np.array(new[] { "ab", "c" }), mask: new[] { true, false }),
                np.ma.array(np.array(new[] { "x", "y" }, T)));
            intersect.Should().Throw<TypeError>().WithMessage(unsuitable);

            const string promotion = "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class " +
                "'numpy.dtypes.Float64DType'>. This means that no common DType exists for the given inputs. For example they " +
                "cannot be stored in a single array unless the dtype is `object`. The full list of DTypes is: (<class " +
                "'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.Float64DType'>)";
            Action setdiffF = () => np.ma.setdiff1d(MTe(), 1.5);
            setdiffF.Should().Throw<DTypePromotionError>().WithMessage(promotion);
            Action setxorF = () => np.ma.setxor1d(MTe(), 1.5);
            setxorF.Should().Throw<DTypePromotionError>().WithMessage(promotion);

            // Nothing to compare: the empty intersection is NumPy's result.
            var empty = np.ma.intersect1d(MTe(), MTe());
            empty.size.Should().Be(0);
            empty.dtype.Should().Be(T);
        }

        /// <summary>
        ///     <c>np.unique</c> over <c>StringDType</c>: at most one element is the shape's answer (index outputs included);
        ///     an axis over two or more dimensions refuses with NumPy's two texts (the structured view of the rows, or —
        ///     for zero-width rows — the view of the field-less result back as <c>StringDType</c>).
        /// </summary>
        [TestMethod]
        public void NpUnique_ShapeDecidedAndAxisRefusals()
        {
            var r = np.unique(np.array(new[] { "a" }, T), return_index: true, return_inverse: true, return_counts: true);
            r.values.GetAtIndex(0).Should().Be("a");
            r.indices.GetInt64(0).Should().Be(0);
            r.inverse_indices.GetInt64(0).Should().Be(0);
            r.counts.GetInt64(0).Should().Be(1);

            Action axis = () => np.unique(np.array(new[,] { { "a", "b" } }, T), axis: 0);
            axis.Should().Throw<TypeError>().WithMessage("The axis argument to unique is not supported for dtype StringDType()");
            Action zeroWidth = () => np.unique(np.zeros(new Shape(3, 0), T), axis: 0);
            zeroWidth.Should().Throw<TypeError>().WithMessage("Cannot change data-type for array of references.");
            Action outOfRange = () => np.unique(np.array(new[] { "a" }, T), axis: 5);
            outOfRange.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 1*");
        }

        /// <summary>
        ///     The legacy products (<c>dot</c>, <c>inner</c>, <c>convolve</c>, <c>correlate</c>) look a legacy descriptor up
        ///     first: a <c>StringDType</c> first operand (or a <c>U</c> one beside it) is refused there; any other class has no
        ///     common DType with <c>StringDType</c>, so NumPy multiplies Python objects and its first product decides
        ///     (<c>1.5 * 'x'</c>, <c>b'a' * 'x'</c>, <c>1.5 * None</c>); a product that does not raise builds an object array.
        /// </summary>
        [TestMethod]
        public void Products_LegacyDescriptorOrFirstObjectProduct()
        {
            const string legacy = "This function currently only supports native NumPy dtypes and old-style user dtypes, but " +
                "the dtype was StringDType().\n(The function may need to be updated to support arbitraryuser dtypes.)";
            var f3 = np.array(new[] { 1.5, 2.0, 3.0 });
            Action dotTF = () => np.ma.dot(MT(), f3);
            dotTF.Should().Throw<TypeError>().Which.Message.Should().Be(legacy);
            Action dotFT = () => np.ma.dot(f3, MT());
            dotFT.Should().Throw<TypeError>().WithMessage("can't multiply sequence by non-int of type 'float'");
            Action dotST = () => np.dot(np.array(new[] { "a", "b" }, np.dtype("|S1")), np.array(new[] { "x", "y" }, T));
            dotST.Should().Throw<TypeError>().WithMessage("can't multiply sequence by non-int of type 'str'");
            Action dotNA = () => np.dot(np.array(new[] { 1.5, 2.0 }), np.array(new object[] { null, "y" }, TN));
            dotNA.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for *: 'float' and 'NoneType'");
            // An int repeats the text: NumPy returns the object result 'xxyyy', which NumSharp has no dtype for.
            Action dotIT = () => np.dot(np.array(new long[] { 2, 3 }), np.array(new[] { "x", "y" }, T));
            dotIT.Should().Throw<NotSupportedException>().Which.Should().NotBeOfType<StringComputationNotSupportedException>();
            // The alignment check follows the object conversion.
            Action misaligned = () => np.dot(f3, np.array(new[] { "x", "y" }, T));
            misaligned.Should().Throw<IncorrectShapeException>().WithMessage("shapes (3,) and (2,) not aligned: 3 (dim 0) != 2 (dim 0)");

            // np.convolve swaps by length in Python first: the StringDType operand then comes first.
            Action convolveSwap = () => np.convolve(np.array(new[] { 1.5, 2.0 }), np.array(new[] { "x", "y", "z" }, T));
            convolveSwap.Should().Throw<TypeError>().Which.Message.Should().Be(legacy);
            Action convolve = () => np.convolve(f3, np.array(new[] { "x", "y", "z" }, T));
            convolve.Should().Throw<TypeError>().WithMessage("can't multiply sequence by non-int of type 'float'");
            // np.correlate's C swap makes the StringDType operand the left factor: 'x' * 1.5 names the float too.
            Action correlateSwap = () => np.correlate(np.array(new[] { 1.5, 2.0 }), np.array(new object[] { "x", "y", "z" }, TN));
            correlateSwap.Should().Throw<TypeError>().WithMessage("can't multiply sequence by non-int of type 'float'");
            Action correlateEmpty = () => np.correlate(np.array(new[] { 1.5, 2.0 }), np.array(new string[0], T));
            correlateEmpty.Should().Throw<ValueError>().WithMessage("second array argument cannot be empty");
            // numpy.ma slides the masks first: a scalar kernel fails np.correlate's depth check there.
            Action maCorrelate = () => np.ma.correlate(MT(), "ab");
            maCorrelate.Should().Throw<ValueError>().WithMessage("object of too small depth for desired array");
        }

        /// <summary>
        ///     <c>np.vander</c> (and numpy.ma's) promotes the input with <c>int</c> before it allocates, and
        ///     <c>StringDType</c> has no common DType with it; a non-1-D input is refused first.
        /// </summary>
        [TestMethod]
        public void Vander_RefusesByPromotion()
        {
            const string text = "The DTypes <class 'numpy.dtypes.StringDType'> and <class 'numpy.dtypes.Int64DType'> do not " +
                "have a common DType. For example they cannot be stored in a single array unless the dtype is `object`.";
            Action vander = () => np.vander(np.array(new[] { "a", "b" }, T));
            vander.Should().Throw<DTypePromotionError>().WithMessage(text);
            Action maVander = () => np.ma.vander(MT());
            maVander.Should().Throw<DTypePromotionError>().WithMessage(text);
            Action twoD = () => np.vander(np.array(new[,] { { "a" } }, T));
            twoD.Should().Throw<ValueError>().WithMessage("x must be a one-dimensional array or sequence.");
        }

        /// <summary>
        ///     The shifts have no <c>StringDType</c> loop, and NumPy's refusal names a bare Python number by NEP 50's abstract
        ///     class.
        /// </summary>
        [TestMethod]
        public void Shift_RefusalNamesThePythonNumberClass()
        {
            Action left = () => np.ma.left_shift(MT(), 1.5);
            left.Should().Throw<TypeError>().WithMessage("ufunc 'left_shift' did not contain a loop with signature matching " +
                "types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes._PyFloatDType'>) -> None");
            Action right = () => np.ma.right_shift(MT(), 1);
            right.Should().Throw<TypeError>().WithMessage("ufunc 'right_shift' did not contain a loop with signature matching " +
                "types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes._PyLongDType'>) -> None");
        }

        /// <summary>
        ///     Stores: under a HARD mask NumPy asks one element for its <c>size</c>, and a <c>StringDType</c> element is a
        ///     Python object (a str, or the NA object); a slice reads a sub-array and stores. A C# number is the Python number
        ///     NumPy stores: <c>coerce=False</c> refuses it, a coercing descriptor writes its str().
        /// </summary>
        [TestMethod]
        public void Stores_HardMaskElementAndCoercion()
        {
            Action hard = () => MT(hard: true)[1] = "zz";
            hard.Should().Throw<AttributeError>().WithMessage("'str' object has no attribute 'size'");
            Action hardNone = () => np.ma.array(np.array(new object[] { "ab", null, "de" }, TN), mask: new[] { true, false, false },
                hard_mask: true)[1] = "zz";
            hardNone.Should().Throw<AttributeError>().WithMessage("'NoneType' object has no attribute 'size'");
            Action hardNan = () => np.ma.array(np.array(new object[] { "ab", double.NaN, "de" }, TNan), mask: new[] { true, false, false },
                hard_mask: true)[1] = "zz";
            hardNan.Should().Throw<AttributeError>().WithMessage("'float' object has no attribute 'size'");

            var sliced = MT(hard: true);
            sliced["1:2"] = "zz";
            sliced._data.GetAtIndex(1).Should().Be("zz");
            sliced._data.GetAtIndex(0).Should().Be("ab");

            Action coerceFalse = () => MT(TC)[1] = 5;
            coerceFalse.Should().Throw<ValueError>().WithMessage("StringDType only allows string data when string coercion is disabled.");
            var coerced = MT();
            coerced[1] = 5;
            coerced._data.GetAtIndex(1).Should().Be("5");
        }

        /// <summary>
        ///     <c>masked_values</c> fills the masked slots with the value first (<c>filled(x, value)</c>), whose check
        ///     refuses a number for <c>StringDType(coerce=False)</c> before the comparison's own refusal.
        /// </summary>
        [TestMethod]
        public void MaskedValues_ChecksTheFillBeforeComparing()
        {
            Action coerceFalse = () => np.ma.masked_values(MT(TC), 1.5);
            coerceFalse.Should().Throw<TypeError>().WithMessage("Cannot convert fill_value 1.5 to dtype StringDType(coerce=False)");
            Action compare = () => np.ma.masked_values(MT(), 1.5);
            compare.Should().Throw<TypeError>().WithMessage("ufunc 'equal' did not contain a loop with signature matching types " +
                "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes._PyFloatDType'>) -> None");
        }

        /// <summary>
        ///     <c>ma.array(list, dtype=StringDType)</c> converts the list INTO the descriptor (<c>np.array(data,
        ///     dtype=dtype)</c>): a None is an NA element, a number its str(), refused by <c>coerce=False</c>.
        /// </summary>
        [TestMethod]
        public void ArrayFromList_ConvertsIntoTheDescriptor()
        {
            var na = np.ma.array(new object[] { "ab", null }, dtype: TN, mask: new[] { false, true });
            na.dtype.Should().Be(TN);
            na._data.GetAtIndex(0).Should().Be("ab");
            na._data.GetAtIndex(1).Should().BeNull();

            var numbers = np.ma.array(new object[] { 1, 2.5 }, dtype: T);
            numbers._data.GetAtIndex(0).Should().Be("1");
            numbers._data.GetAtIndex(1).Should().Be("2.5");

            Action coerceFalse = () => np.ma.array(new object[] { 1, 2.5 }, dtype: TC);
            coerceFalse.Should().Throw<ValueError>().WithMessage("StringDType only allows string data when string coercion is disabled.");
        }
    }
}
