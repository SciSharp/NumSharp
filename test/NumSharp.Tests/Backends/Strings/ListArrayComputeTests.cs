using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Text;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The ndarray members of numpy.char's object result (<see cref="NDListArray"/>) that compare, sort, reduce, compute and
    ///     assign lists — NumPy's object loops over Python lists. Every literal was probed against NumPy 2.4.2 over
    ///     <c>np.char.split(np.array(['a b', 'c', '', 'd e f']))</c> (<c>r</c>), its bytes twin (<c>s</c>) and
    ///     <c>np.char.split(np.array([['a b', 'c', 'x y'], ['', 'd e f', 'g']]))</c> (<c>a</c>); the string oracle's section
    ///     <c>strcharlist</c> replays the full matrix over every class and core layout.
    /// </summary>
    [TestClass]
    public class ListArrayComputeTests
    {
        /// <summary>The (4,) split result <c>r</c>.</summary>
        /// <returns>A new list array (the caller disposes it).</returns>
        private static NDListArray R() => np.@char.split(np.array(new[] { "a b", "c", "", "d e f" }));

        /// <summary>The (4,) bytes split result <c>s</c>.</summary>
        /// <returns>A new list array.</returns>
        private static NDListArray S()
        {
            using var text = np.array(new[] { "a b", "c", "", "d e f" });
            using var bytes = text.astype(np.dtype("S"));
            return np.@char.split(bytes);
        }

        /// <summary>The (2, 3) split result <c>a</c>.</summary>
        /// <returns>A new list array.</returns>
        private static NDListArray A() => np.@char.split(np.array(new[,] { { "a b", "c", "x y" }, { "", "d e f", "g" } }));

        /// <summary>The empty (0,) split result.</summary>
        /// <returns>A new list array.</returns>
        private static NDListArray E() => np.@char.split(np.array(new string[0]));

        /// <summary>The (24,) split result of repeated words: ties for the sort family.</summary>
        /// <returns>A new list array.</returns>
        private static NDListArray Ties()
            => np.@char.split(np.array(Enumerable.Repeat(new[] { "c", "a", "c", "b", "a", "c" }, 4).SelectMany(x => x).ToArray()));

        /// <summary>Python's repr of a nested .NET list of pieces (str or bytes), as NumPy's <c>tolist()</c> prints.</summary>
        /// <param name="o">A list, a string, bytes, a bool, a number, or null.</param>
        /// <returns>The repr.</returns>
        private static string Py(object o) => o switch
        {
            null => "None",
            string s => "'" + s + "'",
            byte[] b => "b'" + Encoding.ASCII.GetString(b) + "'",
            bool v => v ? "True" : "False",
            IList l => "[" + string.Join(", ", l.Cast<object>().Select(Py)) + "]",
            _ => o.ToString(),
        };

        /// <summary>The values of a list array, or of a list stand-in (its one list).</summary>
        /// <param name="l">The list array.</param>
        /// <returns>Python's repr of its <c>tolist()</c>.</returns>
        private static string Lists(NDListArray l) => Py(l.tolist());

        /// <summary>The values of a numeric / bool / text array in Python's repr.</summary>
        /// <param name="a">The array.</param>
        /// <returns>Python's repr of its <c>tolist()</c> (a scalar's own repr for 0-d).</returns>
        private static string Values(NDArray a) => a.ndim == 0 ? Py(a.GetAtIndex(0)) : Py(a.tolist());

        // =============================================================================================================
        // Comparisons
        // =============================================================================================================

        /// <summary>
        ///     The comparisons are Python's list_richcompare per broadcast pair: the first unequal items decide (a shorter
        ///     prefix is smaller), str against bytes orders as CPython's TypeError, any other Python value is never equal and
        ///     never ordered; the reflected operator names the swapped symbol.
        /// </summary>
        [TestMethod]
        public void Comparisons_AreListRichCompare()
        {
            using var r = R();
            using var rev = r["::-1"];
            using (var lt = r < rev) Values(lt).Should().Be("[True, False, True, False]");
            using (var gt = r > rev) Values(gt).Should().Be("[False, True, False, True]");
            using (var ne = r != rev) Values(ne).Should().Be("[True, True, True, True]");
            using (var col = r.reshape(4, 1))
            using (var eq = r == col)
                Values(eq).Should().Be("[[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]");
            using var s = S();
            using (var eq = r == s) Values(eq).Should().Be("[False, False, True, False]", "an empty list equals the empty bytes list");
            using (var eq = r == 1) Values(eq).Should().Be("[False, False, False, False]");

            Action strBytes = () => _ = r < s;
            strBytes.Should().Throw<TypeError>().WithMessage("'<' not supported between instances of 'str' and 'bytes'");
            Action intLeft = () => _ = 1 < r;
            intLeft.Should().Throw<TypeError>().WithMessage("'>' not supported between instances of 'list' and 'int'");
            Action str = () => _ = r < "x";
            str.Should().Throw<TypeError>().WithMessage("'<' not supported between instances of 'list' and 'str'");
            using var head = r[":3"];
            Action broadcast = () => _ = r < head;
            broadcast.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (4,) (3,) ");
            using var e = E();
            using (var empty = e < 1) empty.size.Should().Be(0, "an empty broadcast compares nothing");
        }

        /// <summary>
        ///     An element (<c>r["0"]</c>) is NumPy's Python LIST: against an ARRAY the ufunc converts it with np.asarray into its
        ///     items (str), while two lists compare as lists.
        /// </summary>
        [TestMethod]
        public void ListElement_IsItsItemsAgainstAnArray()
        {
            using var r = R();
            using var first = r["0"];
            using var head = r[":2"];
            using (var eq = head == first) Values(eq).Should().Be("[False, False]", "['a', 'b'] is compared item by item: a list is never a str");
            Action ordered = () => _ = head < first;
            ordered.Should().Throw<TypeError>().WithMessage("'<' not supported between instances of 'list' and 'str'");
            Action reflected = () => _ = first < head;
            reflected.Should().Throw<TypeError>().WithMessage("'>' not supported between instances of 'list' and 'str'");
            Action shapes = () => _ = r == first;
            shapes.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (4,) (2,) ");
            using var again = r["0"];
            using (var same = first == again) Values(same).Should().Be("True", "two lists compare as Python lists");
            using var empty = r["2"];
            Action emptyAdd = () => _ = empty + r;
            emptyAdd.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (0,) (4,) ");
        }

        /// <summary>
        ///     A list against an ndarray: list's comparison is NotImplemented, so the ARRAY's operator runs with the list's
        ///     items as its operand (<c>lst &lt; a</c> is <c>np.greater(a, items)</c>, <c>a &lt; lst</c> is
        ///     <c>np.less(a, items)</c>); <c>==</c> of a pair with no loop is the operator's all-False constant, an ordering's
        ///     no-loop error names the ufunc the reflection picks, and an empty list's float64 items do not broadcast (probed
        ///     2.4.2; the mutation pass found this path untested).
        /// </summary>
        [TestMethod]
        public void ListElement_AgainstAnNDArray_RunsTheArraysOperator()
        {
            using var u = np.array(new[] { "a b", "c d e", "" });
            using var r = np.@char.split(u);
            using var lst = r["0"];
            using var a = np.array(new[] { "a", "c" });
            using (var lt = lst < a) Values(lt).Should().Be("[False, True]");
            using (var reflected = a < lst) Values(reflected).Should().Be("[False, False]");
            using (var ge = lst >= a) Values(ge).Should().Be("[True, False]");
            using (var eq = lst == a) Values(eq).Should().Be("[True, False]");
            using var bytes = np.array(new object[] { new byte[] { 97 }, new byte[] { 99 } });
            Action noLoop = () => _ = lst < bytes;
            noLoop.Should().Throw<TypeError>().WithMessage(
                "ufunc 'greater' did not contain a loop with signature matching types (<class 'numpy.dtypes.BytesDType'>, " +
                "<class 'numpy.dtypes.StrDType'>) -> None");
            using (var never = lst == bytes) Values(never).Should().Be("[False, False]");
            using var empty = r["2"];
            Action shapes = () => _ = empty == a;
            shapes.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (2,) (0,) ");
        }

        // =============================================================================================================
        // Truth and conversions
        // =============================================================================================================

        /// <summary>A list's truth is its non-emptiness: all / any / nonzero / astype(bool); a 0-d nonzero is NumPy 2's
        ///     refusal.</summary>
        [TestMethod]
        public void Truth_IsNonEmptiness()
        {
            using var a = A();
            using (var all = a.all()) Values(all).Should().Be("False");
            using (var any = a.any()) Values(any).Should().Be("True");
            using (var all0 = a.all(0)) Values(all0).Should().Be("[False, True, True]");
            using (var b = a.astype(np.dtype("bool"))) Values(b).Should().Be("[[True, True, True], [False, True, True]]");
            using var r = R();
            var nz = r.nonzero();
            Values(nz[0]).Should().Be("[0, 1, 3]");
            foreach (var x in nz)
                x.Dispose();
            using var z = np.@char.split(np.array("a b"));
            Action zeroD = () => z.nonzero();
            zeroD.Should().Throw<ValueError>()
                .WithMessage("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.");
        }

        /// <summary>astype converts each list as Python does: StringDType holds its repr, numbers refuse, and only the
        ///     'unsafe' rule casts from object (a 0-d array is a scalar in the text).</summary>
        [TestMethod]
        public void Astype_ConvertsLikePython()
        {
            using var r = R();
            using (var t = r.astype(np.dtypes.StringDType()))
                Values(t).Should().Be("['['a', 'b']', '['c']', '[]', '['d', 'e', 'f']']");
            Action toInt = () => r.astype(np.dtype("int64"));
            toInt.Should().Throw<TypeError>().WithMessage("int() argument must be a string, a bytes-like object or a real number, not 'list'");
            Action toFloat = () => r.astype(np.dtype("float64"));
            toFloat.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence.");
            Action safe = () => r.astype(np.dtype("bool"), casting: "safe");
            safe.Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('O') to dtype('bool') according to the rule 'safe'");
            using var z = np.@char.split(np.array("a b"));
            Action scalar = () => z.astype(np.dtype("bool"), casting: "safe");
            scalar.Should().Throw<TypeError>().WithMessage("Cannot cast scalar from dtype('O') to dtype('bool') according to the rule 'safe'");
        }

        // =============================================================================================================
        // Sort family
        // =============================================================================================================

        /// <summary>
        ///     The default sort is NumPy's introsort over the lists (UNSTABLE: ties keep its order, not the input's); 'stable'
        ///     is timsort's order; the errors are NumPy's.
        /// </summary>
        [TestMethod]
        public void Sort_IsNumPysAlgorithms()
        {
            using var r = R();
            r.sort();
            Lists(r).Should().Be("[[], ['a', 'b'], ['c'], ['d', 'e', 'f']]");
            using var t = Ties();
            using (var quick = t.argsort())
                Values(quick).Should().Be("[1, 4, 19, 7, 10, 22, 13, 16, 21, 3, 9, 15, 0, 20, 18, 17, 11, 12, 8, 6, 5, 2, 14, 23]");
            using (var stable = t.argsort(kind: "stable"))
                Values(stable).Should().Be("[1, 4, 7, 10, 13, 16, 19, 22, 3, 9, 15, 21, 0, 2, 5, 6, 8, 11, 12, 14, 17, 18, 20, 23]");
            using var a = A();
            using (var flat = a.argsort(null)) Values(flat).Should().Be("[3, 0, 1, 4, 5, 2]");
            Action kind = () => a.sort(kind: "x");
            kind.Should().Throw<ValueError>().WithMessage("sort kind must be one of 'quick', 'heap', or 'stable' (got 'x')");
            using var z = np.@char.split(np.array("a b"));
            using (var one = z.argsort()) Values(one).Should().Be("[0]", "a 0-d array is raveled to one element first");
            using var ro = R();
            ro.setflags(write: false);
            Action readOnly = () => ro.sort();
            readOnly.Should().Throw<ValueError>().WithMessage("sort array is read-only");
        }

        /// <summary>
        ///     searchsorted is npy_binsearch, whose bounds carry from key to key: over an UNSORTED haystack the answers are
        ///     NumPy's, not a fresh binary search's; the sorter and side variants too.
        /// </summary>
        [TestMethod]
        public void Searchsorted_CarriesTheBounds()
        {
            using var r = np.@char.split(np.array(new[] { "c", "a", "", "b a", "b", "a" }));
            using (var left = r.searchsorted(r)) Values(left).Should().Be("[6, 0, 0, 3, 3, 3]");
            using (var rev = r["::-1"])
            using (var right = r.searchsorted(rev, "right")) Values(right).Should().Be("[3, 6, 6, 0, 3, 6]");
            using (var range = np.arange(6L))
            using (var order = range["::-1"])
            using (var sorted = r.searchsorted(r, sorter: order)) Values(sorted).Should().Be("[5, 4, 0, 5, 5, 4]");
            using var grid = r.reshape(2, 3);
            Action deep = () => grid.searchsorted(r);
            deep.Should().Throw<ValueError>().WithMessage("object too deep for desired array");
        }

        /// <summary>argmax / argmin return the FIRST extreme list; an empty array is NumPy's ValueError.</summary>
        [TestMethod]
        public void Argmax_IsTheFirstExtreme()
        {
            using var r = R();
            using (var i = r.argmax()) Values(i).Should().Be("3");
            using (var i = r.argmin()) Values(i).Should().Be("2");
            using var t = Ties();
            using (var i = t.argmax()) Values(i).Should().Be("0", "the first of the tied ['c'] lists");
            using var e = E();
            Action empty = () => e.argmax();
            empty.Should().Throw<ValueError>().WithMessage("attempt to get argmax of an empty sequence");
        }

        // =============================================================================================================
        // Reductions
        // =============================================================================================================

        /// <summary>
        ///     max / min fold in NumPy's visiting order with npy_ObjectMax (a tie keeps the earlier list); a reduction to one
        ///     element is the LIST itself; initial / where / out are NumPy's.
        /// </summary>
        [TestMethod]
        public void MaxMin_FoldTheLists()
        {
            using var r = R();
            using (var m = r.max()) Lists(m).Should().Be("['d', 'e', 'f']");
            using (var m = r.min()) Lists(m).Should().Be("[]");
            using var a = A();
            using (var m = a.max(0)) Lists(m).Should().Be("[['a', 'b'], ['d', 'e', 'f'], ['x', 'y']]");
            using (var m = a.max(0, initial: new[] { "m" })) Lists(m).Should().Be("[['m'], ['m'], ['x', 'y']]");
            using (var mask = np.array(new[] { true, false, true }))
            {
                Action masked = () => a.max(where: mask);
                masked.Should().Throw<ValueError>()
                    .WithMessage("reduction operation 'maximum' does not have an identity, so to use a where mask one has to specify 'initial'");
            }
            using var e = E();
            Action empty = () => e.max();
            empty.Should().Throw<ValueError>().WithMessage("zero-size array to reduction operation maximum which has no identity");
            using var source = R();
            using var head = source[":3"];
            using var o = head.copy();
            a.max(0, o).Should().BeSameAs(o);
            Lists(o).Should().Be("[['a', 'b'], ['d', 'e', 'f'], ['x', 'y']]");
        }

        /// <summary>sum concatenates in NumPy's visiting order (a transposed array in MEMORY order), starting from initial;
        ///     prod is list * list's TypeError past one element.</summary>
        [TestMethod]
        public void SumProd_ConcatenateInVisitingOrder()
        {
            using var a = A();
            using (var t = a.T)
            using (var s = t.sum()) Lists(s).Should().Be("['a', 'b', 'c', 'x', 'y', 'd', 'e', 'f', 'g']");
            using (var s = a.sum(1)) Lists(s).Should().Be("[['a', 'b', 'c', 'x', 'y'], ['d', 'e', 'f', 'g']]");
            using (var s = a.sum(initial: new[] { "z" })) Lists(s).Should().Be("['z', 'a', 'b', 'c', 'x', 'y', 'd', 'e', 'f', 'g']");
            Action initial0 = () => a.sum(initial: 0);
            initial0.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for +: 'int' and 'list'");
            using var r = R();
            Action prod = () => r.prod();
            prod.Should().Throw<TypeError>().WithMessage("can't multiply sequence by non-int of type 'list'");
            using (var one = r["1:2"])
            using (var p = one.prod(initial: 2)) Lists(p).Should().Be("['c', 'c']", "2 * ['c']");
            using (var b = r.sum(null, np.dtype("bool"))) Values(b).Should().Be("True", "the OR of the lists' truths");
            // where= without initial (get_initial_from_ufunc): a non-empty object reduction uses no identity, so the mask is
            // NumPy's refusal; an EMPTY one takes the identity, the int 0, which a list array cannot hold — no refusal (the
            // mutation pass found this split untested).
            using (var mask = np.array(new[] { true, false, true, true }))
            {
                Action masked = () => r.sum(where: mask);
                masked.Should().Throw<ValueError>().WithMessage(
                    "reduction operation 'add' does not have an identity, so to use a where mask one has to specify 'initial'");
            }
            using (var none = r["0:0"])
            using (var noMask = np.zeros(new Shape(0L), np.@bool))
            {
                Action emptySum = () => none.sum(where: noMask);
                emptySum.Should().Throw<NotSupportedException>("NumPy's empty masked sum is the identity 0, not the refusal");
            }
        }

        /// <summary>cumsum / trace / clip / mean / var / std are NumPy's object statements.</summary>
        [TestMethod]
        public void Accumulations_AndStatistics()
        {
            using var a = A();
            using (var c = a.cumsum(1))
                Lists(c).Should().Be("[[['a', 'b'], ['a', 'b', 'c'], ['a', 'b', 'c', 'x', 'y']], [[], ['d', 'e', 'f'], ['d', 'e', 'f', 'g']]]");
            using (var tr = a.trace(1)) Lists(tr).Should().Be("['c', 'g']");
            using var r = R();
            using (var lo = r["1:2"])
            using (var c = r.clip(lo)) Lists(c).Should().Be("[['c'], ['c'], ['c'], ['d', 'e', 'f']]");
            Action unary = () => r.clip();
            unary.Should().Throw<TypeError>().WithMessage("bad operand type for unary +: 'list'");
            Action mean = () => r.mean();
            mean.Should().Throw<TypeError>().WithMessage(
                "ufunc 'divide' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");
            Action var = () => r.var();
            var.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for /: 'list' and 'int'");
            using var e = E();
            using (var m = e.mean()) double.IsNaN(m.GetDouble()).Should().BeTrue();
            Action emptyVar = () => e.var();
            emptyVar.Should().Throw<DivideByZeroException>().WithMessage("division by zero");
            using var ze = np.@char.split(np.array(""));
            using (var v = ze.var()) v.GetDouble().Should().Be(0.0, "a 0-d empty list's deviations sum to 0");
        }

        // =============================================================================================================
        // Arithmetic
        // =============================================================================================================

        /// <summary>+ concatenates and * repeats per broadcast pair into new lists; the other Python values are CPython's
        ///     TypeErrors; the size limits are its MemoryError (no text) and OverflowError.</summary>
        [TestMethod]
        public void AddMultiply_AreListConcatAndRepeat()
        {
            using var r = R();
            using (var rev = r["::-1"])
            using (var c = r + rev) Lists(c).Should().Be("[['a', 'b', 'd', 'e', 'f'], ['c'], ['c'], ['d', 'e', 'f', 'a', 'b']]");
            using (var m = r * 2) Lists(m).Should().Be("[['a', 'b', 'a', 'b'], ['c', 'c'], [], ['d', 'e', 'f', 'd', 'e', 'f']]");
            using (var n = np.array(new long[] { 0, 1, 2, 3 }))
            using (var m = r * n) Lists(m).Should().Be("[[], ['c'], [], ['d', 'e', 'f', 'd', 'e', 'f', 'd', 'e', 'f']]");
            Action addInt = () => _ = r + 1;
            addInt.Should().Throw<TypeError>().WithMessage("can only concatenate list (not \"int\") to list");
            Action strLeft = () => _ = "x" + r;
            strLeft.Should().Throw<TypeError>().WithMessage("can only concatenate str (not \"list\") to str");
            Action mulFloat = () => _ = r * 1.5;
            mulFloat.Should().Throw<TypeError>().WithMessage("can't multiply sequence by non-int of type 'float'");
            Action mulStrLeft = () => _ = "x" * r;
            mulStrLeft.Should().Throw<TypeError>().WithMessage("can't multiply sequence by non-int of type 'list'");
            using var one = r["1:2"];
            using (var big = NDArray.Scalar(1L << 62))
            {
                Action memory = () => _ = one * big;
                memory.Should().Throw<OutOfMemoryException>().Where(x => x.Message.Length == 0);
            }
            using (var huge = np.array(new ulong[] { 1UL << 63 }))
            {
                Action overflow = () => _ = one * huge;
                overflow.Should().Throw<OverflowException>().WithMessage("cannot fit 'int' into an index-sized integer");
            }
            using var c1 = np.@char.array(new[] { "x", "y", "z", "w" });
            Action chararray = () => _ = r + c1;
            chararray.Should().Throw<ValueError>().WithMessage("Can only create a chararray from string data.");
        }

        /// <summary>dot is OBJECT_dot (products summed in k order), a 0-d operand a multiply; the alignment and out= texts
        ///     are NumPy's.</summary>
        [TestMethod]
        public void Dot_SumsTheProducts()
        {
            using var r = R();
            using (var w = np.array(new long[] { 2, 0, 1, 1 }))
            using (var d = r.dot(w)) Lists(d).Should().Be("['a', 'b', 'a', 'b', 'd', 'e', 'f']");
            using var a = A();
            using (var w = np.array(new long[] { 1, 2, 0 }))
            using (var d = a.dot(w)) Lists(d).Should().Be("[['a', 'b', 'c', 'c'], ['d', 'e', 'f', 'd', 'e', 'f']]");
            using (var w = np.array(new long[] { 1, 2 }))
            {
                Action align = () => a.dot(w);
                align.Should().Throw<IncorrectShapeException>().WithMessage("shapes (2,3) and (2,) not aligned: 3 (dim 1) != 2 (dim 0)");
            }
            using (var w = np.array(new long[] { 1, 2, 0 }))
            using (var strided = r["::2"])
            {
                Action badOut = () => a.dot(w, strided);
                badOut.Should().Throw<ValueError>()
                    .WithMessage("output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)");
            }
        }

        // =============================================================================================================
        // Assignment
        // =============================================================================================================

        /// <summary>
        ///     Assignment copies POINTERS: an element takes a list (r["3"]); a sub-array takes an array's lists by broadcasting
        ///     (overlap copied first), while a list assigned to a sub-array is the sequence of its items (NumPy's shape error);
        ///     the read-only check runs before the index.
        /// </summary>
        [TestMethod]
        public void SetItem_CopiesPointers()
        {
            using var source = R();
            using var r = R();
            using (var v = source["3"]) r["1"] = v;
            Lists(r).Should().Be("[['a', 'b'], ['d', 'e', 'f'], [], ['d', 'e', 'f']]");
            using var o = R();
            using (var head = o[":3"]) o["1:"] = head;
            Lists(o).Should().Be("[['a', 'b'], ['a', 'b'], ['c'], []]", "the overlapping source is read before it is written");
            using var a = A();
            using (var row = source["1:4"]) a[":, :"] = row;
            Lists(a).Should().Be("[[['c'], [], ['d', 'e', 'f']], [['c'], [], ['d', 'e', 'f']]]");
            using var q = R();
            using (var list = source["3"])
            {
                Action sequence = () => q["1:3"] = list;
                sequence.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (3,) into shape (2,)");
            }
            using var ro = R();
            ro.setflags(write: false);
            using (var list = source["2"])
            {
                Action readOnly = () => ro["9"] = list;
                readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
            }
        }

        /// <summary>fill stores ONE list in every element; put cycles the values' lists through NumPy's modes. Appending to a
        ///     split result's own store must keep its lists exact (the store's capacity is not its piece count).</summary>
        [TestMethod]
        public void FillPut_StoreLists()
        {
            using var source = R();
            using var r = R();
            r.fill(new[] { "z", "w" });
            Lists(r).Should().Be("[['z', 'w'], ['z', 'w'], ['z', 'w'], ['z', 'w']]");
            using var p = R();
            using (var values = source["1:3"])
            using (var indices = np.array(new long[] { 0, 1, 2 }))
                p.put(indices, values);
            Lists(p).Should().Be("[['c'], [], ['c'], ['d', 'e', 'f']]");
            using var c = R();
            using (var values = source["1:2"])
            using (var indices = np.array(new long[] { 10 }))
                c.put(indices, values, "clip");
            Lists(c).Should().Be("[['a', 'b'], ['c'], [], ['c']]");
            using var bad = R();
            using (var values = source["1:2"])
            using (var indices = np.array(new long[] { 0 }))
            {
                Action mode = () => bad.put(indices, values, "x");
                mode.Should().Throw<ValueError>().WithMessage("clipmode must be one of 'clip', 'raise', or 'wrap' (got 'x')");
            }
        }

        /// <summary>take / compress write into out= with take's checks (shape, then writeability, then the indices); resize
        ///     shrinks in place; byteswap / getfield / view(dtype) / choose are NumPy's object-array answers.</summary>
        [TestMethod]
        public void OutResizeAndObjectRefusals()
        {
            using var r = R();
            using var head = r[":2"];
            using var o = head.copy();
            using (var indices = np.array(new long[] { 3, 0 }))
                r.take(indices, null, o).Should().BeSameAs(o);
            Lists(o).Should().Be("[['d', 'e', 'f'], ['a', 'b']]");
            using (var three = r[":3"])
            using (var bad = three.copy())
            using (var indices = np.array(new long[] { 3, 0 }))
            {
                Action shape = () => r.take(indices, null, bad);
                shape.Should().Throw<ValueError>().WithMessage("output array does not match result of ndarray.take");
            }
            using var a = A();
            a.resize(new Shape(3, 1), false);
            Lists(a).Should().Be("[[['a', 'b']], [['c']], [['x', 'y']]]");
            using var g = R();
            Action grow = () => g.resize(6);
            grow.Should().Throw<NotSupportedException>("NumPy fills the new elements with the int 0");
            using (var swapped = r.byteswap()) Lists(swapped).Should().Be(Lists(r));
            // byteswap's copy keeps an F-contiguous array F (order 'A'); a transposed C array is one (the mutation pass
            // found this untested, as it found take into a READ-ONLY out of another heap: take's own refusal comes first).
            using (var fresh = A())
            using (var t = fresh.T)
            using (var swappedT = t.byteswap())
            {
                swappedT.flags.f_contiguous.Should().BeTrue();
                swappedT.flags.c_contiguous.Should().BeFalse();
                Lists(swappedT).Should().Be(Lists(t));
            }
            using (var readOnly = np.@char.split(np.array(new[] { "x", "y" })))
            using (var indices = np.array(new long[] { 3, 0 }))
            {
                readOnly.setflags(write: false);
                Action intoReadOnly = () => r.take(indices, null, readOnly);
                intoReadOnly.Should().Throw<ValueError>().WithMessage("WRITEBACKIFCOPY base is read-only");
            }
            Action field = () => r.getfield(np.dtype("int64"));
            field.Should().Throw<TypeError>().WithMessage("Cannot get/set field of an object array");
            Action view = () => r.view(np.dtype("int64"));
            view.Should().Throw<TypeError>().WithMessage("Cannot change data-type for array of references.");
            Action choose = () => r.choose(new[] { 0 });
            choose.Should().Throw<TypeError>().WithMessage("Cannot cast array data from dtype('O') to dtype('int64') according to the rule 'safe'");
        }

        /// <summary>tofile writes each list's repr (or <c>format % (list,)</c>) in text mode; binary mode is NumPy's OSError.</summary>
        [TestMethod]
        public void Tofile_WritesTheReprs()
        {
            using var r = R();
            using var text = new MemoryStream();
            r.tofile(text, "|");
            Encoding.ASCII.GetString(text.ToArray()).Should().Be("['a', 'b']|['c']|[]|['d', 'e', 'f']");
            using var padded = new MemoryStream();
            r.tofile(padded, "|", "%-12s");
            Encoding.ASCII.GetString(padded.ToArray()).Should().Be("['a', 'b']  |['c']       |[]          |['d', 'e', 'f']");
            using var binary = new MemoryStream();
            Action bin = () => r.tofile(binary);
            bin.Should().Throw<OSError>().WithMessage("cannot write object arrays to a file in binary mode");
            using var number = new MemoryStream();
            Action d = () => r.tofile(number, "|", "%d");
            d.Should().Throw<TypeError>().WithMessage("%d format: a real number is required, not list");
        }
    }
}
