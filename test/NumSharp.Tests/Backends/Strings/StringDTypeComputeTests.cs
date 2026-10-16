using System;
using System.Linq;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Computing on NEP 55 <c>StringDType</c> text (strings phase 2, P2.T): the six comparisons and their NA rules, the
    ///     truth readers, <c>add</c> / <c>multiply</c> / <c>maximum</c> / <c>minimum</c> with NumPy's descriptor resolution
    ///     (the compatible NA object, the first operand's rules, a <c>StringDType</c> out= as the loop's own output, any
    ///     other out= receiving the default <c>StringDType()</c> through the output cast), the reductions and the
    ///     accumulation on both of <c>PyUFunc_Accumulate</c>'s paths, the statistics that refuse after computing, and
    ///     <c>np.char</c>'s wrappers. Every expected value and message was probed against <c>numpy==2.4.2</c>; the string
    ///     oracle's sections <c>tcomp</c> / <c>tarith</c> / <c>treduce</c> hold the same contract over every layout.
    /// </summary>
    [TestClass]
    public class StringDTypeComputeTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=nan)</c> descriptor (a NaN-like NA: it propagates and sorts last).</summary>
        private static DType TNaN => np.dtypes.StringDType(na_object: double.NaN);

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor (a None-like NA: ordering and arithmetic refuse it).</summary>
        private static DType TNone => np.dtypes.StringDType(na_object: null);

        /// <summary>A fresh <c>StringDType(na_object='NA')</c> descriptor (a string NA: it reads as its text).</summary>
        private static DType TStr => np.dtypes.StringDType(na_object: "NA");

        /// <summary>A 1-D array of the given elements in the given descriptor (texts and NA objects).</summary>
        /// <param name="d">The descriptor.</param>
        /// <param name="values">The elements.</param>
        /// <returns>The new array.</returns>
        private static NDArray A(DType d, params object[] values) => np.array(values, d);

        /// <summary>The elements of an array in C order (texts and NA objects).</summary>
        /// <param name="a">A <c>StringDType</c> or <c>U</c> array.</param>
        /// <returns>The values.</returns>
        private static object[] V(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => a.Storage.GetAtIndex(i)).ToArray();

        /// <summary>The flags of a bool array in C order.</summary>
        /// <param name="a">A bool array.</param>
        /// <returns>The flags.</returns>
        private static bool[] B(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => (bool)a.Storage.GetAtIndex(i)).ToArray();

        // ------------------------------------------------------------------------------------------------------------
        // comparisons
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>_compare</c> under a NaN-like NA: every ordering with an NA is False and <c>==</c> / <c>!=</c> with an
        ///     NA are both False (NaN's comparison semantics), whichever side the NA is on.
        /// </summary>
        [TestMethod]
        public void Compare_NaNLikeNA_IsFalseForEveryOperator()
        {
            using var x = A(TNaN, "b", "a", double.NaN);
            using var lt = A(TNaN, "a", double.NaN, double.NaN);
            using var eq = A(TNaN, "b", double.NaN, double.NaN);
            B(x < lt).Should().Equal(false, false, false);
            B(x >= lt).Should().Equal(true, false, false);
            B(x == eq).Should().Equal(true, false, false);
            B(x != eq).Should().Equal(false, false, false);
        }

        /// <summary>
        ///     A None-like NA: <c>==</c> / <c>!=</c> treat two NAs as equal and an NA as unequal to text; an ordering raises
        ///     NumPy's ValueError naming the ufunc.
        /// </summary>
        [TestMethod]
        public void Compare_NoneLikeNA_EqualsButNeverOrders()
        {
            using var x = A(TNone, "b", null);
            using var y = A(TNone, "a", null);
            B(x == y).Should().Equal(false, true);
            B(x != y).Should().Equal(true, false);
            using var z = A(TNone, "a", "b");
            Action lt = () => _ = x < z;
            lt.Should().Throw<ValueError>().WithMessage("'less' not supported for null values that are not nan-like or strings.");
        }

        /// <summary>
        ///     A string NA compares as its text; a <c>U</c> array or a C# string is cast to the default descriptor and
        ///     compared by code point (UTF-8 order, so <c>'😀' &lt; 'z'</c> is False); two different NA objects have no loop.
        /// </summary>
        [TestMethod]
        public void Compare_StringNA_UText_AndIncompatibleNA()
        {
            using var s = A(TStr, "b", "NA");
            using var s2 = A(TStr, "NA", "NB");
            B(s < s2).Should().Equal(false, true);
            using var t = A(T, "a", "é", "\U0001F600");
            using var u = np.array(new[] { "b", "e", "z" });
            B(t < u).Should().Equal(true, false, false);
            using var t2 = A(T, "a", "bc");
            B(t2 == "bc").Should().Equal(false, true);
            using var n1 = A(TNaN, "a");
            using var n2 = A(TNone, "a");
            Action eq = () => np.equal(n1, n2);
            eq.Should().Throw<TypeError>().WithMessage("Cannot find a compatible null string value for null strings 'nan' and 'None'");
        }

        // ------------------------------------------------------------------------------------------------------------
        // truth values
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The text → bool cast: an empty text is False; a NaN-like NA is True (Python's <c>bool(nan)</c>), a None-like NA
        ///     False, and a string NA is NumPy's <c>default_string.size == 0</c> — False for <c>na_object='NA'</c> but True for
        ///     <c>na_object=''</c>, the inverse of the text's own truth (NumPy's <c>string_to_bool</c>, reproduced), while
        ///     <c>np.nonzero</c> reads that NA as the empty text it stands for.
        /// </summary>
        [TestMethod]
        public void Truth_FollowsTheNARule()
        {
            using var n = A(TNaN, "", "a", double.NaN);
            B(n.astype(np.@bool)).Should().Equal(false, true, true);
            using var o = A(TNone, "", "a", null);
            B(o.astype(np.@bool)).Should().Equal(false, true, false);
            using var s = A(TStr, "", "a", "NA");
            B(s.astype(np.@bool)).Should().Equal(false, true, false);
            using var e = A(np.dtypes.StringDType(na_object: ""), "", "x");
            B(e.astype(np.@bool)).Should().Equal(true, true);
            np.nonzero(e)[0].ToArray<long>().Should().Equal(1);
            using var nz = A(TNaN, "", "a", double.NaN, "b");
            np.nonzero(nz)[0].ToArray<long>().Should().Equal(1, 2, 3);
            using var other = A(T, "x", "y", "z");
            B(np.logical_and(n, other)).Should().Equal(false, true, true);
        }

        /// <summary><c>np.isnan</c> is True exactly for the NA cells of a NaN-like descriptor.</summary>
        [TestMethod]
        public void IsNaN_OnlyForNaNLikeNA()
        {
            using var n = A(TNaN, "", "a", double.NaN);
            B(np.isnan(n)).Should().Equal(false, false, true);
            using var o = A(TNone, "", null);
            B(np.isnan(o)).Should().Equal(false, false);
        }

        // ------------------------------------------------------------------------------------------------------------
        // add / multiply / maximum / minimum
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>add</c>: a NaN-like NA propagates, a string NA concatenates as its text, a None-like NA raises; the result
        ///     takes the compatible NA object and the FIRST operand's coerce; two different NA objects have no loop.
        /// </summary>
        [TestMethod]
        public void Add_NARules_AndTheResultDescriptor()
        {
            using var r1 = np.add(A(TNaN, "a", double.NaN), A(TNaN, "b", "c"));
            V(r1).Should().Equal("ab", double.NaN);
            r1.dtype.ToString(true).Should().Be("StringDType(na_object=nan)");
            using var r2 = np.add(A(TStr, "a", "NA"), A(TStr, "b", "NA"));
            V(r2).Should().Equal("ab", "NANA");
            Action none = () => np.add(A(TNone, "a", null), A(TNone, "b", "c"));
            none.Should().Throw<ValueError>().WithMessage("Cannot add null that is not a nan-like value");
            using var r3 = np.add(A(np.dtypes.StringDType(coerce: false), "a", "b"), np.array(new[] { "c", "d" }));
            V(r3).Should().Equal("ac", "bd");
            r3.dtype.ToString(true).Should().Be("StringDType(coerce=False)");
            using var r4 = np.add(A(T, "a"), A(TNaN, "b"));
            r4.dtype.ToString(true).Should().Be("StringDType(na_object=nan)");
            Action conflict = () => np.add(A(TNone, "a"), A(TNaN, "b"));
            conflict.Should().Throw<TypeError>().WithMessage("Cannot find a compatible null string value for null strings 'None' and 'nan'");
        }

        /// <summary>
        ///     <c>multiply</c>: a negative count overflows the size (NumPy converts it to <c>size_t</c>), so does a product past
        ///     the size limit; a string NA repeats its text; a None-like NA is a TypeError; the count may come first.
        /// </summary>
        [TestMethod]
        public void Multiply_Counts_AndNARules()
        {
            Action negative = () => np.multiply(A(TNaN, "ab", double.NaN, "c"), np.array(new long[] { 2, 3, -1 }));
            negative.Should().Throw<OverflowException>().WithMessage("Overflow encountered in string multiply");
            using var r = np.multiply(np.array(3L), A(TStr, "ab", "NA"));
            V(r).Should().Equal("ababab", "NANANA");
            Action none = () => np.multiply(A(TNone, "ab", null), np.array(2L));
            none.Should().Throw<TypeError>().WithMessage("Cannot multiply null that is not a nan-like value");
            Action huge = () => np.multiply(A(T, "ab"), np.array(1L << 62));
            huge.Should().Throw<OverflowException>().WithMessage("Overflow encountered in string multiply");
        }

        /// <summary>
        ///     <c>maximum</c> / <c>minimum</c>: a NaN-like NA wins <c>maximum</c> and loses <c>minimum</c> (it orders last), a
        ///     None-like NA raises.
        /// </summary>
        [TestMethod]
        public void MaximumMinimum_NaNLikeNAOrdersLast()
        {
            using var mx = np.maximum(A(TNaN, "b", "a", double.NaN), A(TNaN, "a", double.NaN, double.NaN));
            V(mx).Should().Equal("b", double.NaN, double.NaN);
            using var mn = np.minimum(A(TNaN, "b", "a", double.NaN), A(TNaN, "a", double.NaN, "z"));
            V(mn).Should().Equal("a", "a", "z");
            Action none = () => np.maximum(A(TNone, "b", null), A(TNone, "a", "c"));
            none.Should().Throw<ValueError>().WithMessage("Cannot compare null that is not a nan-like value");
        }

        /// <summary>
        ///     The out= descriptor of a text loop: a <c>StringDType</c> out is the loop's own output (an NA it packs reads as
        ///     THAT descriptor's NA object); a <c>U</c> out receives the default <c>StringDType()</c>, whose NA cells cast to
        ///     <c>''</c>; a numeric out is refused naming <c>StringDType()</c>; a where= mask hides a refused NA.
        /// </summary>
        [TestMethod]
        public void Out_DescriptorResolution()
        {
            using var x = A(TNaN, "a", double.NaN);
            using var y = A(TNaN, "b", "c");
            using (var u = np.full(new Shape(2), "k", np.dtype("U3")))
            {
                np.add(x, y, u);
                V(u).Should().Equal("ab", "");
            }
            using (var t = np.full(new Shape(2), "k", TNone))
            {
                np.add(x, y, t);
                V(t).Should().Equal(new object[] { "ab", null });
            }
            using (var f = np.zeros(new Shape(2)))
            {
                Action cast = () => np.add(x, y, f);
                cast.Should().Throw<ArgumentException>()
                    .WithMessage("Cannot cast ufunc 'add' output from StringDType() to dtype('float64') with casting rule 'same_kind'");
            }
            using (var u = np.full(new Shape(2), "q", np.dtype("U3")))
            {
                np.add(A(TNone, "a", null), A(TNone, "b", "c"), u, np.array(new[] { true, false }));
                V(u).Should().Equal("ab", "q");
            }
            using (var u = np.full(new Shape(2), "k", np.dtype("U3")))
            {
                np.maximum(x, y, u);
                V(u).Should().Equal("b", "");
            }
        }

        /// <summary>
        ///     COPY_IF_OVERLAP: an out that overlaps an input other than element for element computes from a copy
        ///     (<c>np.add(a[:-1], a[1:], out=a[1:])</c> reads the original texts).
        /// </summary>
        [TestMethod]
        public void Out_OverlapComputesFromACopy()
        {
            using var a = A(T, "a", "b", "c");
            np.add(a[":-1"], a["1:"], a["1:"]);
            V(a).Should().Equal("a", "ab", "bc");
        }

        // ------------------------------------------------------------------------------------------------------------
        // reductions
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.sum</c> is the concatenation in order; a NaN-like NA makes the sum the NA, a string NA adds its text, a
        ///     None-like NA raises; the loop is not reorderable (one axis at most) and has no identity (an empty sum raises).
        /// </summary>
        [TestMethod]
        public void Sum_ConcatenatesInOrder()
        {
            using (var r = np.sum(A(T, "a", "b", "c")))
                V(r).Should().Equal("abc");
            using (var m = np.array(new[,] { { "a", "b" }, { "c", "d" } }, T))
            {
                using var r = np.sum(m, 0);
                V(r).Should().Equal("ac", "bd");
                Action both = () => np.sum(m);
                both.Should().Throw<ValueError>().WithMessage("reduction operation 'add' is not reorderable, so at most one axis may be specified");
            }
            using (var r = np.sum(A(TNaN, "a", double.NaN, "c")))
                V(r).Should().Equal(double.NaN);
            using (var r = np.sum(A(TStr, "a", "NA", "c")))
                V(r).Should().Equal("aNAc");
            Action none = () => np.sum(A(TNone, "a", null, "c"));
            none.Should().Throw<ValueError>().WithMessage("Cannot add null that is not a nan-like value");
            Action empty = () => np.sum(np.zeros(new Shape(0), T));
            empty.Should().Throw<ValueError>().WithMessage("zero-size array to reduction operation add which has no identity");
        }

        /// <summary>
        ///     <c>np.max</c> / <c>np.min</c> / <c>np.argmax</c> / <c>np.argmin</c>: a NaN-like NA is the maximum (it orders
        ///     last); <c>initial=</c> takes part; a where= mask needs <c>initial=</c>; the first extreme wins a tie.
        /// </summary>
        [TestMethod]
        public void MaxMinArg_NaNOrdersLast_AndKeywords()
        {
            using (var r = np.max(A(TNaN, "b", double.NaN, "c")))
                V(r).Should().Equal(double.NaN);
            using (var r = np.min(A(TNaN, "b", double.NaN, "c")))
                V(r).Should().Equal("b");
            using (var r = np.max(A(T, "b", "a", "c"), initial: "z"))
                V(r).Should().Equal("z");
            using var m = np.array(new[,] { { "b", "a" }, { "c", "d" } }, T);
            using var mask = np.array(new[,] { { false, true }, { true, true } });
            using (var r = np.max(m, 1, null, false, "", mask))
                V(r).Should().Equal("a", "d");
            Action noInitial = () => np.max(m, 1, null, false, null, mask);
            noInitial.Should().Throw<ValueError>()
                .WithMessage("reduction operation 'maximum' does not have an identity, so to use a where mask one has to specify 'initial'");
            np.argmax(A(TNaN, "b", double.NaN, "c")).Should().Be(1);
            np.argmin(A(T, "b", "a", "c", "a")).Should().Be(1);
        }

        /// <summary>
        ///     <c>np.cumsum</c>: each output is the running concatenation; a NaN-like NA makes every later output the NA, a
        ///     string NA appends its text, a None-like NA raises; along an axis each lane accumulates on its own.
        /// </summary>
        [TestMethod]
        public void CumSum_RunningConcatenation()
        {
            using (var r = np.cumsum(A(TNaN, "a", "b", double.NaN, "c")))
                V(r).Should().Equal("a", "ab", double.NaN, double.NaN);
            using (var r = np.cumsum(A(TStr, "a", "NA", "c")))
                V(r).Should().Equal("a", "aNA", "aNAc");
            Action none = () => np.cumsum(A(TNone, "a", null, "c"));
            none.Should().Throw<ValueError>().WithMessage("Cannot add null that is not a nan-like value");
            using var m = np.array(new[,] { { "a", "b" }, { "c", "d" } }, T);
            using (var r = np.cumsum(m, 0))
                V(r).Should().Equal("a", "b", "ac", "bd");
        }

        /// <summary>
        ///     The accumulation's out= on <c>PyUFunc_Accumulate</c>'s two paths: the direct path's wrong-size error, the
        ///     iterator path's <c>op_axes</c> and non-broadcastable texts, an out with an extra trailing axis (written at index
        ///     0), an input of extent 1 along the axis (only the first cell written), the <c>U</c> out's first-element cast
        ///     (an NA first element is the input's <c>na_name</c>), and <c>enforce_uniform_args</c>.
        /// </summary>
        [TestMethod]
        public void CumSum_OutOnBothPaths()
        {
            using var m = np.array(new[,] { { "a", "b" }, { "c", "d" } }, T);
            using (var o = np.empty(new Shape(2), T))
            {
                Action r = () => np.cumsum(m, 0, null, o);
                r.Should().Throw<IncorrectShapeException>()
                    .WithMessage("Iterator input op_axes[0][0] (==1) is not a valid axis of op[0], which has 1 dimensions*");
            }
            using (var o = np.empty(new Shape(1, 2), T))
            {
                Action r = () => np.cumsum(m, 0, null, o);
                r.Should().Throw<IncorrectShapeException>()
                    .WithMessage("non-broadcastable output operand with shape (1,2) [remapped to (2,1)] doesn't match the broadcast shape (2,2)");
            }
            using (var o = np.full(new Shape(2, 2, 2), "k", T))
            {
                np.cumsum(m, 0, null, o);
                V(o).Should().Equal("a", "k", "b", "k", "ac", "k", "bd", "k");
            }
            using (var v = A(T, "a", "b", "c"))
            using (var o = np.empty(new Shape(2), T))
            {
                Action r = () => np.cumsum(v, null, null, o);
                r.Should().Throw<IncorrectShapeException>().WithMessage("provided out is the wrong size for the accumulation.");
            }
            using (var row = np.array(new[,] { { "x", "yy", "z" } }, T))
            using (var o = np.full(new Shape(2, 3), "k", T))
            {
                np.cumsum(row, 0, null, o);
                V(o).Should().Equal("x", "yy", "z", "k", "k", "k");
            }
            using (var u = np.empty(new Shape(3), np.dtype("U5")))
            {
                np.cumsum(A(TNaN, double.NaN, "1", "2"), null, null, u);
                V(u).Should().Equal("nan", "nan1", "nan12");
            }
            using (var v = A(T, "a", "b", "c"))
            using (var o = np.empty(new Shape(3), TNaN))
            {
                Action r = () => np.cumsum(v, null, null, o);
                r.Should().Throw<TypeError>().WithMessage(
                    "the resolved dtypes are not compatible with add.accumulate. Resolved (StringDType(na_object=nan), StringDType(), StringDType(na_object=nan))");
            }
        }

        /// <summary>
        ///     A 1-D accumulation into a 0-d out crashes NumPy 2.4.2 for every dtype, an empty input included (its direct
        ///     path reads the out's stride along the axis first); NumSharp raises the direct path's wrong-size error for text
        ///     as for numbers.
        /// </summary>
        [TestMethod]
        public void CumSum_ZeroDimOut_IsTheWrongSizeError()
        {
            const string Message = "provided out is the wrong size for the accumulation.";
            foreach (var input in new[] { A(T, "a", "b"), A(T, "a"), np.array(new string[0], T) })
            {
                using (input)
                using (var zeroD = np.array("", T))
                {
                    Action text = () => np.cumsum(input, null, null, zeroD);
                    text.Should().Throw<IncorrectShapeException>().WithMessage(Message, $"{input.size} elements");
                }
            }
            using var numbers = np.arange(0L);
            using var zeroDNumber = np.array(0L);
            Action number = () => np.cumsum(numbers, null, null, zeroDNumber);
            number.Should().Throw<IncorrectShapeException>().WithMessage(Message);
        }

        /// <summary>
        ///     NumPy's exact alias <c>np.cumsum(a, out=a)</c>: no copy, and the first-element cast onto itself empties a short
        ///     first string, which the running sum then follows.
        /// </summary>
        [TestMethod]
        public void CumSum_ExactAlias_IsNumPys()
        {
            using var a = A(T, "abc", "de", "f");
            np.cumsum(a, null, null, a);
            V(a).Should().Equal("", "de", "def");
        }

        /// <summary><c>np.trace</c>'s add.reduce into a <c>U</c> out casts the first element like the accumulation does.</summary>
        [TestMethod]
        public void Trace_IntoU_FirstNACast()
        {
            using var m = np.array(new object[] { double.NaN, "x", "y", "b" }, TNaN).reshape(2, 2);
            using var o = np.array("", np.dtype("U9"));
            np.trace(m, 0, 0, 1, null, o);
            V(o).Should().Equal("nanb");
        }

        /// <summary>
        ///     <c>np.trace</c> reaches its add.reduce after the diagonal's axis checks, so a bad axis wins over everything,
        ///     then the <c>dtype=</c> instance refusal, then the read-only out (a direct <c>ufunc.reduce</c> parses its out
        ///     first; NumPy's order, probed).
        /// </summary>
        [TestMethod]
        public void Trace_ErrorOrder_AxisThenDTypeThenReadOnlyOut()
        {
            using var m = A(T, "a", "b", "c", "d").reshape(2, 2);
            using var ro = np.array("", T);
            ro.setflags(write: false);
            const string Instance = "Cannot pass a new user DType instance to the `dtype` or `signature` arguments of ufuncs. Pass the DType class instead.";
            Action axis = () => np.trace(m, 0, 5, 1, np.dtypes.StringDType(), ro);
            axis.Should().Throw<AxisError>().WithMessage("axis1: axis 5 is out of bounds for array of dimension 2*");
            Action both = () => np.trace(m, 0, 0, 1, np.dtypes.StringDType(), ro);
            both.Should().Throw<TypeError>().WithMessage(Instance);
            Action readOnly = () => np.trace(m, 0, 0, 1, null, ro);
            readOnly.Should().Throw<ValueError>().WithMessage("output array is read-only");
        }

        // ------------------------------------------------------------------------------------------------------------
        // cell layouts and the arena reservation (the mutation pass found these unpinned by the value contracts)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The 16 bytes of cell <paramref name="i"/> of an owning <c>StringDType</c> array, as NumPy's
        ///     <c>bytes.hex()</c> prints them (lower case), so a literal can be pasted from a NumPy probe.</summary>
        /// <param name="a">An owning <c>StringDType</c> array (cell 0 at the start of its buffer).</param>
        /// <param name="i">The cell index.</param>
        /// <returns>32 hex digits.</returns>
        private static unsafe string Cell(NDArray a, int i)
            => Convert.ToHexString(new ReadOnlySpan<byte>(a.Storage.VStringSlice.Address + 16 * i, 16)).ToLowerInvariant();

        /// <summary>
        ///     A short text stored over an arena cell keeps the old offset's high bytes past its size (NumPy's
        ///     <c>NpyString_newemptysize</c> writes only byte 15 over the freed cell), and every comparison reads the text alone
        ///     — the ordering's inline first-word test against an arena text included, where the stale byte would otherwise
        ///     order <c>'q'</c> after <c>'q\0\x05…'</c>. Cells and answers probed from NumPy (<c>t[1] = 'q'</c> over offset
        ///     0x135 leaves the stale 0x01).
        /// </summary>
        [TestMethod]
        public void Compare_ShortCellWithStaleBytes_ReadsOnlyItsText()
        {
            using var t = A(T, new string('L', 300), new string('m', 40), "z");
            Cell(t, 1).Should().Be("35010000000000002800000000000040");
            t[1] = "q";
            Cell(t, 1).Should().Be("71010000000000000000000000000061", "the 0x01 past the size is the freed offset's");
            using var q = t["1:2"];
            using var b = A(T, "q\0\u0005" + new string('z', 15));
            B(np.less(q, b)).Should().Equal(true);
            B(np.less_equal(q, b)).Should().Equal(true);
            B(np.greater(q, b)).Should().Equal(false);
            B(np.less(b, q)).Should().Equal(false);
            B(np.greater(b, q)).Should().Equal(true);
            B(np.equal(q, b)).Should().Equal(false);
            B(np.not_equal(q, b)).Should().Equal(true);
        }

        /// <summary>
        ///     An emptied short cell (byte 15 <c>INITIALIZED | OUTSIDE_ARENA</c>, size 0: what <c>NpyString_free</c> leaves
        ///     after <c>e[0] = ''</c>) wins <c>maximum</c> / <c>minimum</c> into a fresh output as NumPy's <c>pack_empty</c>
        ///     over a never-written cell: the output cell stays all zero, so a later longer store takes the ARENA, where the
        ///     emptied input itself sends it to the heap. Cells probed from NumPy.
        /// </summary>
        [TestMethod]
        public void MaximumMinimum_EmptiedShortWinner_IsTheNeverWrittenCell()
        {
            using var e = A(T, "abc", "x");
            e[0] = "";
            Cell(e, 0).Should().Be("00000000000000000000000000000060");
            foreach (bool max in new[] { true, false })
            {
                using var r = max ? np.maximum(e, e) : np.minimum(e, e);
                Cell(r, 0).Should().Be("00000000000000000000000000000000");
                Cell(r, 1).Should().Be("78000000000000000000000000000061");
                r[0] = new string('w', 20);
                Cell(r, 0).Should().Be("01000000000000001400000000000040", "a never-written cell takes its first long text into the arena");
                r.Storage.VStringSlice.Allocator.HeapStrings.Should().Be(0);
            }
            e[0] = new string('w', 20);
            Cell(e, 0).Substring(30).Should().Be("70", "an initialized cell's longer text goes to the heap");
            e.Storage.VStringSlice.Allocator.HeapStrings.Should().Be(1);
        }

        /// <summary>
        ///     A fresh output's arena is reserved once, at exactly the bytes its texts take (a medium text — up to 255 bytes —
        ///     its size plus a 1-byte prefix, a long one an 8-byte prefix), and filled with no heap string; the cells
        ///     (offsets, sizes, flags) are NumPy's, probed. <c>maximum</c> / <c>minimum</c> reserve the larger candidate of each
        ///     pair without comparing (an upper bound: <c>minimum</c> of a 300- and a 299-byte text reserves 308 and uses 307),
        ///     and over inputs whose winners are all cells they reserve nothing.
        /// </summary>
        [TestMethod]
        public void Writers_ReserveTheArenaExactly()
        {
            static void Exact(NDArray r, long reserved, long used, params string[] cells)
            {
                var allocator = r.Storage.VStringSlice.Allocator;
                allocator.ArenaCapacity.Should().Be(reserved, "the size pass reserved its total once");
                allocator.ArenaUsed.Should().Be(used);
                allocator.HeapStrings.Should().Be(0);
                for (int i = 0; i < cells.Length; i++)
                    Cell(r, i).Should().Be(cells[i], $"cell {i}");
            }
            const string Medium255 = "0100000000000000ff00000000000040", Long300 = "08010000000000002c01000000000050";
            using (var x = A(T, new string('a', 14), new string('b', 254), new string('c', 299)))
            using (var d = A(T, "d"))
            using (var r = np.add(x, d))
                Exact(r, 564, 564, "6161616161616161616161616161646f", Medium255, Long300);
            using (var x = A(T, "abc", new string('q', 85), new string('r', 100)))
            using (var three = np.array(3L))
            using (var r = np.multiply(x, three))
                Exact(r, 564, 564, "61626361626361626300000000000069", Medium255, Long300);
            using (var m1 = A(T, new string('a', 255), new string('b', 300), "zz"))
            using (var m2 = A(T, new string('a', 254) + "b", new string('b', 299), "zy"))
            {
                using (var r = np.maximum(m1, m2))
                    Exact(r, 564, 564, Medium255, Long300, "7a7a0000000000000000000000000062");
                using (var r = np.minimum(m1, m2))
                    Exact(r, 564, 563, Medium255, "08010000000000002b01000000000050", "7a790000000000000000000000000062");
            }
            using (var s1 = A(T, "ab", "cd"))
            using (var s2 = A(T, "ac", "ca"))
            using (var r = np.maximum(s1, s2))
                Exact(r, 0, 0, "61630000000000000000000000000062", "63640000000000000000000000000062");
        }

        // ------------------------------------------------------------------------------------------------------------
        // statistics and np.char
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The statistics compute their sum / extremes, then meet NumPy's next statement: an ndarray mean has no divide
        ///     loop, a NaN-like NA sum is the Python float NaN divided by the count, <c>np.ptp</c> subtracts two Python
        ///     strings, and <c>np.average</c>'s weights find no common DType.
        /// </summary>
        [TestMethod]
        public void Statistics_RefuseAfterComputing()
        {
            Action var = () => np.var(A(T, "a", "b"));
            var.Should().Throw<TypeError>().WithMessage(
                "ufunc 'divide' did not contain a loop with signature matching types (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.Int64DType'>) -> None");
            using (var r = np.mean(A(TNaN, "a", double.NaN)))
                double.IsNaN(r.GetDouble()).Should().BeTrue();
            Action ptp = () => np.ptp(A(T, "a", "b"));
            ptp.Should().Throw<TypeError>().WithMessage(
                "ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U1'), dtype('<U1')) -> None");
            Action avg = () => np.average(A(T, "a", "b"), weights: np.array(new long[] { 1, 2 }));
            avg.Should().Throw<DTypePromotionError>().WithMessage(
                "The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes.Int64DType'>.*The full list of DTypes is: (<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.Int64DType'>)");
        }

        /// <summary>
        ///     <c>np.char.add</c> / <c>np.char.multiply</c> over <c>StringDType</c> are the ufuncs; <c>np.char.multiply</c>
        ///     turns the loop's TypeError (a None-like NA) into its own <c>Can only multiply by integers</c>.
        /// </summary>
        [TestMethod]
        public void Char_AddMultiply()
        {
            using var t = A(TNaN, "ab", "c", double.NaN);
            using (var r = np.@char.add(t, t))
                V(r).Should().Equal("abab", "cc", double.NaN);
            using (var r = np.@char.multiply(t, np.array(new long[] { -1, 2, 3 })))
                V(r).Should().Equal("", "cc", double.NaN);
            Action none = () => np.@char.multiply(A(TNone, "ab", null), np.array(2L));
            none.Should().Throw<ValueError>().WithMessage("Can only multiply by integers");
        }

        /// <summary>
        ///     NumPy 2.4.2 HANGS on an empty <c>StringDType</c> text times a negative count: its multiply converts the count
        ///     to <c>size_t</c>, the overflow check multiplies it by the size 0 and passes, and the repeat loop then runs
        ///     2^64 - 1 times copying nothing. NumSharp returns the answer that loop would end with, the empty text (a
        ///     non-empty text still raises NumPy's overflow error).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Multiply_EmptyTextTimesANegativeCount_IsTheEmptyText()
        {
            using var r = np.multiply(A(T, "", "ab"), np.array(new long[] { -1, 2 }));
            V(r).Should().Equal("", "abab");
            using var first = np.multiply(np.array(long.MinValue), A(T, ""));
            V(first).Should().Equal("");
            Action nonEmpty = () => np.multiply(A(T, "a"), np.array(-1L));
            nonEmpty.Should().Throw<OverflowException>().WithMessage("Overflow encountered in string multiply");
        }

        /// <summary>
        ///     A comparison with <c>where=</c> into an <c>out=</c> of another dtype (a <c>U</c> or an integer array) compares
        ///     only the selected pairs: NumPy's masked inner loop never meets a masked-off None-like NA, which an ordering
        ///     would refuse, and the masked-off slots of <c>out=</c> keep their contents.
        /// </summary>
        [TestMethod]
        public void Comparison_WhereIntoACastOut_ComparesOnlyTheSelectedPairs()
        {
            using var a = A(TNone, "a", null, "b", "");
            using var b = A(TNone, "x", "y", null, "z");
            using var mask = np.array(new[] { true, false, false, true });
            using var textOut = np.array(new[] { "-", "-", "-", "-" }, np.dtype("U5"));
            np.less(a, b, @out: textOut, where: mask);
            V(textOut).Should().Equal("True", "-", "-", "True");
            using var intOut = np.full(new Shape(4), 7, np.int32);
            np.less(a, b, @out: intOut, where: mask);
            V(intOut).Should().Equal(1, 7, 7, 1);
        }

        /// <summary>
        ///     A masked <c>StringDType</c> reduction lays its fresh result out as NumPy's reduction iterator does: the input
        ///     AND the <c>where=</c> mask vote on the axis order, and where they disagree C order wins — a permuted input with
        ///     a C-ordered mask gives a C result, the same input without a mask the input's own order.
        /// </summary>
        [TestMethod]
        public void MaskedReduction_ResultLayoutFollowsTheInputAndTheMask()
        {
            using var flat = np.array(Enumerable.Range(0, 12).Select(i => (object)((char)('a' + i)).ToString()).ToArray(), T);
            using var cube = flat.reshape(3, 2, 2);
            using var x = cube.transpose(new[] { 2, 1, 0 });   // (2, 2, 3), element strides (1, 2, 4)
            using var mask = np.ones(new Shape(2, 2, 3), np.@bool);
            mask[0, 0, 0] = false;
            using var masked = np.max(x, axis: 1, keepdims: true, initial: A(T, "a")["0"], where: mask);
            masked.shape.Should().Equal(2L, 1L, 3L);
            masked.strides.Should().Equal(48L, 48L, 16L);
            masked.flags.c_contiguous.Should().BeTrue();
            V(masked.copy()).Should().Equal("c", "g", "k", "d", "h", "l");
            using var plain = np.max(x, axis: 1, keepdims: true);
            plain.strides.Should().Equal(16L, 32L, 32L);
            plain.flags.c_contiguous.Should().BeFalse();
        }
    }
}
