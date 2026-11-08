using System;
using System.Collections;
using System.Linq;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     <c>numpy.char</c> phase 3 (docs/plans/np-char.md): <see cref="NDCharArray"/> BEHAVIOR — the operators with
    ///     Python's dispatch, the sort and search members, NumPy's finalize error for the index functions, the string
    ///     partition behind <c>np.partition</c> and the statistics, and the <c>np.*</c> functions that keep the class or drop
    ///     it. Every literal expectation was probed against NumPy 2.4.2; the string oracle's sections <c>strcharops</c>
    ///     (3,266 cases) and <c>strcharfn</c> (2,633 cases) replay the full matrices over every layout. The
    ///     <see cref="MisalignedAttribute"/> tests pin the documented divergences.
    /// </summary>
    [TestClass]
    public class CharArrayBehaviorTests
    {
        /// <summary>NumPy's text of <c>chararray.__array_finalize__</c> for data that is not string data.</summary>
        private const string Finalize = "Can only create a chararray from string data.";

        /// <summary>NumPy's text when <c>ndarray.partition(kth, axis=...)</c> meets <c>chararray.partition(sep)</c>.</summary>
        private const string StringPartition = "chararray.partition() got an unexpected keyword argument 'axis'";

        /// <summary><c>np.char.array(["ab  ", "b", "ab", "a "])</c>: trailing whitespace on two elements.</summary>
        /// <returns>A new 1-d chararray.</returns>
        private static NDCharArray Words() => np.@char.array(new[] { "ab  ", "b", "ab", "a " });

        /// <summary>The elements of an array as NumPy's <c>tolist()</c> gives them (unstripped).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The elements.</returns>
        private static object[] List(NDArray a) => ((IList)a.tolist()).Cast<object>().ToArray();

        // ------------------------------------------------------------------ operators

        /// <summary>
        ///     The comparisons strip trailing whitespace with a chararray on EITHER side — <c>u == c</c> runs the chararray's
        ///     reflected method, as Python dispatches to a subclass first — and the static type decides: the same chararray
        ///     behind an <see cref="NDArray"/> reference compares unstripped (NumPy's <c>np.asarray(c) == "ab"</c>).
        /// </summary>
        [TestMethod]
        public void Comparisons_Strip_PlainArrayOnTheLeftToo_NotThroughAnNDArrayReference()
        {
            var c = Words();
            NDArray u = np.array(new[] { "ab", "b", "ab  ", "a" });

            (c == "ab").ToArray<bool>().Should().Equal(true, false, true, false);
            (u == c).ToArray<bool>().Should().Equal(true, true, true, true);
            NDArray plain = c;
            (plain == "ab").ToArray<bool>().Should().Equal(false, false, true, false);
        }

        /// <summary>
        ///     <c>+</c> is <c>np.add</c> in the written order and returns a chararray VIEW of the computed array (the stored
        ///     text keeps its whitespace); <c>*</c> repeats and copies a <c>U</c> result; a count that is not an integer and
        ///     a <c>None</c> operand raise NumPy's errors (a short chararray's trivial-loop copy into object is the finalize
        ///     error, a long one reaches CPython's concatenation error).
        /// </summary>
        [TestMethod]
        public void AddAndMultiply_ReturnChararrays_WithNumPysErrors()
        {
            var c = Words();

            NDCharArray sum = c + "x";
            List(sum).Should().Equal("ab  x", "bx", "abx", "a x");
            sum.dtype.str.Should().Be("<U5");
            sum.flags.owndata.Should().BeFalse();
            List("x" + c).Should().Equal("xab  ", "xb", "xab", "xa ");

            NDCharArray twice = c * 2;
            List(twice).Should().Equal("ab  ab  ", "bb", "abab", "a a ");
            twice.dtype.str.Should().Be("<U8");
            twice.flags.owndata.Should().BeTrue();

            new Action(() => _ = c * 2.5).Should().Throw<ValueError>().WithMessage("Can only multiply by integers");
            new Action(() => _ = c + (object)null).Should().Throw<ValueError>().WithMessage(Finalize);
            var big = np.@char.array(Enumerable.Repeat("a", 9000).ToArray());
            new Action(() => _ = big + (object)null).Should().Throw<TypeError>()
                .WithMessage("can only concatenate str (not \"NoneType\") to str");
        }

        // ------------------------------------------------------------------ members

        /// <summary><c>sort</c> sorts in place (unstripped order); <c>argsort</c> and <c>searchsorted</c> return plain
        ///     arrays.</summary>
        [TestMethod]
        public void SortInPlace_ArgsortAndSearchsortedArePlain()
        {
            var c = Words();
            NDArray order = c.argsort();
            order.GetType().Should().Be(typeof(NDArray));
            order.ToArray<long>().Should().Equal(3L, 2L, 0L, 1L);

            c.sort();
            List(c).Should().Equal("a ", "ab", "ab  ", "b");
            NDArray at = c.searchsorted(np.array(new[] { "ab", "b" }), "right");
            at.GetType().Should().Be(typeof(NDArray));
            at.ToArray<long>().Should().Equal(2L, 4L);
        }

        // ------------------------------------------------------------------ the finalize error and the string partition

        /// <summary>
        ///     <c>np.argmax</c> / <c>np.argmin</c> / <c>np.argpartition</c> create their int64 result as the subclass: NumPy's
        ///     finalize error, unless an <c>out=</c> receives the indices.
        /// </summary>
        [TestMethod]
        public void IndexFunctions_RaiseTheFinalizeError_UnlessOutReceivesTheIndices()
        {
            var c = Words();
            new Action(() => np.argmax(c)).Should().Throw<ValueError>().WithMessage(Finalize);
            new Action(() => np.argmin(c, 0)).Should().Throw<ValueError>().WithMessage(Finalize);
            new Action(() => np.argpartition(c, 0)).Should().Throw<ValueError>().WithMessage(Finalize);
            new Action(() => c.argmax()).Should().Throw<ValueError>().WithMessage(Finalize);

            var @out = np.zeros(Shape.Scalar, np.int64);
            var r = np.argmax(c, null, @out);
            ReferenceEquals(r, @out).Should().BeTrue();
            r.GetAtIndex<long>(0).Should().Be(1L);
        }

        /// <summary>
        ///     <c>np.partition</c>, <c>np.median</c>, <c>np.quantile</c> (one axis and a tuple axis) call
        ///     <c>a.partition(kth, axis=...)</c>, which for a chararray is the STRING partition: NumPy's TypeError, after the
        ///     checks that come first (an unknown method's ValueError).
        /// </summary>
        [TestMethod]
        public void PartitionAndTheStatistics_MeetTheStringPartition()
        {
            var c = Words();
            new Action(() => np.partition(c, 0)).Should().Throw<TypeError>().WithMessage(StringPartition);
            new Action(() => np.median(c)).Should().Throw<TypeError>().WithMessage(StringPartition);
            new Action(() => np.quantile(c, 0.5, method: "lower")).Should().Throw<TypeError>().WithMessage(StringPartition);
            new Action(() => np.quantile(c, 0.5, new[] { 0 }, method: "lower")).Should().Throw<TypeError>()
                .WithMessage(StringPartition);
            new Action(() => np.percentile(c, 50.0, method: "bad")).Should().Throw<ArgumentException>()
                .WithMessage("'bad' is not a valid method*");
        }

        // ------------------------------------------------------------------ the np.* functions

        /// <summary>
        ///     The functions NumPy answers through a method or an <c>np.asanyarray</c> view keep the class (owning exactly
        ///     where the plain result owns): <c>np.sort</c>, <c>np.reshape</c>, <c>np.unique</c>, <c>np.take</c> (a 0-d take is
        ///     the unstripped NumPy scalar), <c>np.flip</c> of a 0-d chararray (its element, stripped: <c>m[()]</c>), and
        ///     <c>np.resize</c> of an empty chararray (<c>np.zeros_like</c>).
        /// </summary>
        [TestMethod]
        public void FunctionsThatKeepTheClass()
        {
            var c = Words();

            var sorted = np.sort(c);
            sorted.Should().BeOfType<NDCharArray>();
            List(sorted).Should().Equal("a ", "ab", "ab  ", "b");
            sorted.flags.owndata.Should().BeTrue();

            var grid = np.reshape(c, new Shape(2, 2));
            grid.Should().BeOfType<NDCharArray>();
            grid.flags.owndata.Should().BeFalse();

            NDArray values = np.unique(c);
            values.Should().BeOfType<NDCharArray>();
            List(values).Should().Equal("a ", "ab", "ab  ", "b");

            var element = np.take(c, 0);
            element.Should().BeOfType<NDCharArray>();
            ((NDCharArray)element).IsScalarStandIn.Should().BeTrue();
            element.item().Should().Be("ab  ");
            np.take(c, new[] { 0, 1 }).Should().BeOfType<NDCharArray>();

            var c0 = (NDCharArray)np.array("ab  ").view(typeof(NDCharArray));
            var flipped = np.flip(c0);
            ((NDCharArray)flipped).IsScalarStandIn.Should().BeTrue();
            flipped.item().Should().Be("ab");

            var empty = (NDCharArray)np.zeros(new Shape(0), "U2").view(typeof(NDCharArray));
            var resized = np.resize(empty, new Shape(3));
            resized.Should().BeOfType<NDCharArray>();
            resized.flags.owndata.Should().BeTrue();
            List(resized).Should().Equal("", "", "");
        }

        /// <summary>
        ///     The functions that convert with <c>np.asarray</c> / <c>np.concatenate</c> / <c>np.where</c> return plain arrays:
        ///     <c>np.concatenate</c>, <c>np.stack</c>, <c>np.where</c>, and <c>np.resize</c> of a non-empty chararray.
        /// </summary>
        [TestMethod]
        public void FunctionsThatDropTheClass()
        {
            var c = Words();
            np.concatenate(new NDArray[] { c, c }).GetType().Should().Be(typeof(NDArray));
            np.stack(new NDArray[] { c, c }).GetType().Should().Be(typeof(NDArray));
            var picked = np.where(c == "ab", c, "z");
            picked.GetType().Should().Be(typeof(NDArray));
            List(picked).Should().Equal("ab  ", "z", "ab", "z");
            var resized = np.resize(c, new Shape(3));
            resized.GetType().Should().Be(typeof(NDArray));
            List(resized).Should().Equal("ab  ", "b", "ab");
        }

        /// <summary>
        ///     <c>np.unique</c> with a return flag sorts the chararray and finds its runs with the chararray's STRIPPING
        ///     <c>!=</c>, so strings that differ only in trailing whitespace are one value (the run's first element); without
        ///     a flag NumPy compares the plain data.
        /// </summary>
        [TestMethod]
        public void Unique_WithAFlag_MergesStrippedEqualNeighbours()
        {
            var c = np.@char.array(new[] { "b ", "a", "b" });
            var (values, counts) = np.unique(c, return_counts: true);
            values.Should().BeOfType<NDCharArray>();
            List(values).Should().Equal("a", "b");
            counts.ToArray<long>().Should().Equal(1L, 2L);

            List(np.unique(c)).Should().Equal("a", "b", "b ");
        }

        /// <summary>
        ///     The logical ufuncs over a SHORT chararray: NumPy's trivial loop copies it into bool as a chararray — the
        ///     finalize error — unless a mask selects the masked loop (<c>where=False</c>, an array <c>np.array(True)</c>);
        ///     the C# literal <c>true</c> is NumPy's <c>where=True</c>, no mask. A chararray longer than
        ///     <c>np.getbufsize()</c> is not copied and computes plain.
        /// </summary>
        [TestMethod]
        public void LogicalUfuncs_CopyAShortChararrayIntoBool()
        {
            var c = Words();
            new Action(() => np.logical_and(c, c)).Should().Throw<ValueError>().WithMessage(Finalize);
            new Action(() => np.logical_and(c, c, null, true)).Should().Throw<ValueError>().WithMessage(Finalize);
            np.logical_and(c, c, null, false).shape.Should().Equal(4L);
            np.logical_and(c, c, null, np.array(true)).ToArray<bool>().Should().Equal(true, true, true, true);

            var big = np.@char.array(Enumerable.Repeat("a", 9000).ToArray());
            var truth = np.logical_and(big, big);
            truth.GetType().Should().Be(typeof(NDArray));
            truth.dtype.Should().Be(np.bool_);
        }

        /// <summary><c>np.zeros_like</c> allocates as the subclass: a text result is a chararray, a numeric dtype is the
        ///     finalize error before anything is filled.</summary>
        [TestMethod]
        public void ZerosLike_KeepsTheClass_NumericDtypeRaises()
        {
            var c = Words();
            var zeros = np.zeros_like(c);
            zeros.Should().BeOfType<NDCharArray>();
            List(zeros).Should().Equal("", "", "", "");
            new Action(() => np.zeros_like(c, np.float64)).Should().Throw<ValueError>().WithMessage(Finalize);
        }

        /// <summary><c>np.unstack</c> of a 1-d chararray iterates it: each element is <c>__getitem__</c>'s stripped
        ///     value.</summary>
        [TestMethod]
        public void Unstack_OneDimensional_GivesStrippedElements()
        {
            var items = np.unstack(Words());
            items.Should().HaveCount(4);
            items.Select(a => a.item()).Should().Equal("ab", "b", "ab", "a");
            items.All(a => a is NDCharArray { IsScalarStandIn: true }).Should().BeTrue();
        }

        // ------------------------------------------------------------------ documented divergences

        /// <summary>
        ///     The static rule (the owner's decision, <see cref="NDMatrix"/>'s rule): a chararray behind an
        ///     <see cref="NDArray"/> reference binds <see cref="NDArray"/>'s operators and members, which compute plain —
        ///     <c>==</c> does not strip, <c>+</c> returns a plain array, <c>argmax()</c> returns the index and an element keeps
        ///     its whitespace. NumPy dispatches on the object, so each would be the chararray's answer there. The <c>np.*</c>
        ///     functions still find the chararray at run time: <c>np.reshape</c> of the same reference is a chararray.
        /// </summary>
        [TestMethod, Misaligned]
        public void ThroughAnNDArrayReference_OperatorsAndMembersArePlain()
        {
            NDArray plain = Words();
            (plain == "ab").ToArray<bool>().Should().Equal(false, false, true, false);
            var sum = plain + "x";
            sum.GetType().Should().Be(typeof(NDArray));
            List(sum).Should().Equal("ab  x", "bx", "abx", "a x");
            plain.argmax().Should().Be(1L);
            plain[0].item().Should().Be("ab  ");
            np.reshape(plain, new Shape(2, 2)).Should().BeOfType<NDCharArray>();
        }

        /// <summary>
        ///     The C# <c>null</c> literal beside a chararray keeps <see cref="NDArray"/>'s reference test for <c>==</c> and
        ///     <c>!=</c> (a 0-d <c>False</c> / <c>True</c>), because C# code tests a reference with <c>x == null</c>. NumPy's
        ///     <c>c == None</c> and <c>c != None</c> raise <c>TypeError: comparison of non-string arrays</c>.
        /// </summary>
        [TestMethod, Misaligned]
        public void EqualsNull_IsAReferenceTest()
        {
            var c = Words();
            var equal = c == (object)null;
            equal.ndim.Should().Be(0);
            equal.GetAtIndex<bool>(0).Should().BeFalse();
            (c != (object)null).GetAtIndex<bool>(0).Should().BeTrue();
        }

        /// <summary>
        ///     <c>np.squeeze</c> of a chararray with no length-1 axis returns a NEW chararray view. NumPy's
        ///     <c>PyArray_Squeeze</c> returns the operand itself (<c>np.squeeze(c) is c</c>); the values and the class
        ///     agree.
        /// </summary>
        [TestMethod, Misaligned]
        public void Squeeze_WithoutALengthOneAxis_ReturnsANewView()
        {
            var c = Words();
            var squeezed = np.squeeze(c);
            squeezed.Should().BeOfType<NDCharArray>();
            ReferenceEquals(squeezed, c).Should().BeFalse();
            squeezed.flags.owndata.Should().BeFalse();
            List(squeezed).Should().Equal(List(c));
        }

        /// <summary>
        ///     A function of an element stand-in (<c>c[0]</c>, NumPy's Python <c>str</c>) computes as for a plain 0-d array,
        ///     so a VIEW of the read-only stand-in is read-only. NumPy's <c>np.atleast_2d(c[0])</c> converts the <c>str</c>
        ///     into a fresh array first, so its result is writeable.
        /// </summary>
        [TestMethod, Misaligned]
        public void AViewOfAnElementStandIn_IsReadOnly()
        {
            var c = Words();
            var grid = np.atleast_2d(c[0]);
            grid.GetType().Should().Be(typeof(NDArray));
            grid.shape.Should().Equal(1L, 1L);
            grid.flags.writeable.Should().BeFalse();
        }

        /// <summary>
        ///     <c>chararray + StringDType</c> where NumPy's trivial loop does not copy the chararray (2-d here): NumPy returns a
        ///     PLAIN StringDType array, which the <see cref="NDCharArray"/> result type of the <c>+</c> operator cannot hold,
        ///     so NumSharp refuses with <see cref="NotSupportedException"/>. <c>np.add</c> returns NumPy's answer.
        /// </summary>
        [TestMethod, Misaligned]
        public void AddStringDType_TheOperatorRefuses_NpAddComputes()
        {
            var c = np.@char.array(new[] { "ab", "c", "d", "e" }).reshape(2, 2);
            var t = np.array(new[] { "q", "r", "s", "t" }, np.dtypes.StringDType()).reshape(2, 2);
            new Action(() => _ = c + t).Should().Throw<NotSupportedException>().WithMessage("*call np.add*");
            var sum = np.add(c, t);
            sum.GetType().Should().Be(typeof(NDArray));
            sum.dtype.ToString().Should().Be("StringDType()");
            List(sum.ravel()).Should().Equal("abq", "cr", "ds", "et");
        }
    }
}
