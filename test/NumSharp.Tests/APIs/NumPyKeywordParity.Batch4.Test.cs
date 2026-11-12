using System;
using System.Linq;
using System.Reflection;
using NumSharp.Generic;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 4 (2026-11-12, docs/plans/API_PARITY_DIFF.md): the <see cref="NDAxis"/> type (the owner's OD1
    ///     decision, with public constructors beside the implicit conversions) and ONE axis signature for np.flip,
    ///     np.expand_dims, np.moveaxis, np.squeeze / ndarray.squeeze, np.all / np.any / ndarray.all / .any, np.count_nonzero,
    ///     np.ptp, np.median / np.nanmedian, np.percentile / np.nanpercentile and np.quantile / np.nanquantile (q as one
    ///     array), np.average / np.average_returned, np.trim_zeros, np.size and np.gradient. Every expected value or text is
    ///     NumPy 2.4.2's answer for the same call (probed 2026-11-12: batch4_numpy_evidence.py, batch4_matrix_evidence.py,
    ///     batch4_edges.py).
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch4Tests
    {
        /// <summary>The public static overloads of <c>np.<paramref name="name"/></c>.</summary>
        /// <param name="name">The function name.</param>
        /// <returns>Every public static method of that name on <see cref="np"/>.</returns>
        private static MethodInfo[] NpForms(string name)
            // Without the kept int / int[] / int? forwards of an NDAxis member (the owner's batch-6 decision).
            => typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == name).WithoutKeptIntegerForms();

        /// <summary>The overloads of <c>np.<paramref name="name"/></c> that take an <c>axis</c> parameter.</summary>
        /// <param name="name">The function name.</param>
        /// <returns>The axis-taking overloads.</returns>
        private static MethodInfo[] AxisForms(string name)
            => NpForms(name).Where(m => m.GetParameters().Any(p => p.Name == "axis")).ToArray();

        /// <summary><c>np.arange(24.0).reshape(2, 3, 4)</c>, the evidence scripts' <c>a</c>.</summary>
        private static NDArray A => np.arange(24.0).reshape(2, 3, 4);

        /// <summary>The 2 x 3 matrix <c>[[1, 5, 2], [4, 0, 6]]</c>, batch4_matrix_evidence.py's <c>m</c>.</summary>
        private static NDArray M => np.matrix(np.array(new double[,] { { 1, 5, 2 }, { 4, 0, 6 } }));

        /// <summary>The bool matrix <c>[[True, False, True], [True, True, True]]</c>, the evidence's <c>mb</c>.</summary>
        private static NDArray MB => np.matrix(np.array(new bool[,] { { true, false, true }, { true, true, true } }));

        // ------------------------------------------------------------------ fix 1: NDAxis

        /// <summary>Each spelling is NumPy's axis argument: an int one axis, a C# tuple or an int[] a tuple, the empty array
        ///     the empty tuple; the constructors build the same values; the repr is Python's (<c>(1,)</c> keeps its comma).</summary>
        [TestMethod]
        public void NDAxis_SpellingsAndConstructors_ArePythonsArgument()
        {
            NDAxis one = 1, pair = (0, 2), triple = (0, 1, 2), array = new[] { 0, 2 }, empty = new int[0];
            one.IsTuple.Should().BeFalse();
            one.ToString().Should().Be("1");
            pair.ToString().Should().Be("(0, 2)");
            triple.ToString().Should().Be("(0, 1, 2)");
            array.Should().Be(pair);
            empty.ToString().Should().Be("()");
            new NDAxis(new[] { 1 }).ToString().Should().Be("(1,)");
            new NDAxis(1).Should().Be(one);
            new NDAxis(0, 2).Should().Be(pair);
            new NDAxis((0, 2)).Should().Be(pair);
            new NDAxis((0, 1, 2)).Should().Be(triple);
            // 1 != (1,) in Python: a single axis never equals a one-element tuple.
            (one == new NDAxis(new[] { 1 })).Should().BeFalse();
            // default(NDAxis) and new NDAxis() are the EMPTY tuple, as the params constructor with no argument.
            default(NDAxis).Should().Be(empty);
            new NDAxis().IsTuple.Should().BeTrue();
            new NDAxis().Count.Should().Be(0);
            pair.Count.Should().Be(2);
            pair[1].Should().Be(2);
            pair.ToArray().Should().Equal(0, 2);
        }

        /// <summary>The array a tuple was built from is copied: changing it afterwards does not change the value.</summary>
        [TestMethod]
        public void NDAxis_CopiesItsArray()
        {
            var axes = new[] { 0, 2 };
            NDAxis axis = axes;
            axes[0] = 1;
            axis.ToArray().Should().Equal(0, 2);
            axis.ToArray()[0] = 9;   // ToArray hands out a copy too
            axis[0].Should().Be(0);
        }

        /// <summary>NumPy's <c>normalize_axis_tuple</c>: <c>((0, -1), 3)</c> is <c>(0, 2)</c>, <c>((0, 3), 3)</c> is
        ///     <c>AxisError: axis 3 is out of bounds for array of dimension 3</c>, <c>((0, 0), 3)</c> is <c>ValueError: repeated
        ///     axis</c>, an int is the one-element tuple and <c>()</c> stays empty.</summary>
        [TestMethod]
        public void NDAxis_Normalize_IsNormalizeAxisTuple()
        {
            ((NDAxis)(0, -1)).Normalize(3).Should().Equal(0, 2);
            ((NDAxis)1).Normalize(3).Should().Equal(1);
            ((NDAxis)new int[0]).Normalize(3).Should().BeEmpty();
            Action range = () => ((NDAxis)(0, 3)).Normalize(3);
            range.Should().Throw<AxisError>().WithMessage("axis 3 is out of bounds for array of dimension 3*");
            Action repeat = () => ((NDAxis)(0, 0)).Normalize(3);
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis");
            Action named = () => ((NDAxis)(0, 0)).Normalize(3, "source");
            named.Should().Throw<ValueError>().WithMessage("repeated axis in `source` argument");
        }

        /// <summary>A null <c>int[]</c> is None for an <c>NDAxis?</c> parameter (what a null int[] axis meant before) and
        ///     an <see cref="ArgumentNullException"/> for a non-nullable <see cref="NDAxis"/>; a null <c>int?</c> is None
        ///     through the lifted conversion.</summary>
        [TestMethod]
        public void NDAxis_NullArrayAndNullInt_AreNone()
        {
            int[] none = null;
            int? noAxis = null;
            NDAxis? fromArray = none;
            NDAxis? fromInt = noAxis;
            fromArray.HasValue.Should().BeFalse();
            fromInt.HasValue.Should().BeFalse();
            np.flip(A, none).ToArray<double>().First().Should().Be(23.0);   // flip(a, None) reverses every axis
            Action nonNullable = () => np.expand_dims(A, none);
            nonNullable.Should().Throw<ArgumentNullException>();
        }

        // ------------------------------------------------------------------ fix 2: flip

        /// <summary>One signature; NumPy: <c>flip(a, None)[0, 0]</c> is <c>[23, 22, 21, 20]</c>, <c>flip(a, (0, 2))[0, 0]</c>
        ///     <c>[15, 14, 13, 12]</c>, <c>(0, 0)</c> repeated, <c>(0, 0, 5)</c> the AxisError of 5 (every range first),
        ///     <c>flip(np.array(5.), ())</c> 5.0 and <c>flip(np.array(5.), 0)</c> an AxisError.</summary>
        [TestMethod]
        public void Flip_OneSignature_NoneIntTuple_AsNumPy()
        {
            AxisForms("flip").Should().HaveCount(1);
            np.flip(A, null)[0, 0].ToArray<double>().Should().Equal(23, 22, 21, 20);
            np.flip(A, (0, 2))[0, 0].ToArray<double>().Should().Equal(15, 14, 13, 12);
            np.flip(A, new[] { 0, 2 })[0, 0].ToArray<double>().Should().Equal(15, 14, 13, 12);
            Action repeat = () => np.flip(A, (0, 0));
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis");
            Action range = () => np.flip(A, (0, 0, 5));
            range.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 3*");
            ((double)np.flip(np.array(5.0), new int[0])).Should().Be(5.0);
            Action zeroD = () => np.flip(np.array(5.0), 0);
            zeroD.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");
        }

        // ------------------------------------------------------------------ fix 3: expand_dims

        /// <summary>One signature; NumPy: <c>expand_dims(f, (0, 3))</c> and <c>(0, -1)</c> are <c>(1, 2, 3, 1)</c>,
        ///     <c>(0, 4)</c> is <c>axis 4 is out of bounds for array of dimension 4</c>, <c>(1, 1)</c> repeated; a matrix
        ///     operand gives a plain array.</summary>
        [TestMethod]
        public void ExpandDims_OneSignature_Tuple_AsNumPy()
        {
            NpForms("expand_dims").Should().HaveCount(1);
            var f = np.arange(6.0).reshape(2, 3);
            np.expand_dims(f, (0, 3)).shape.Should().Equal(1L, 2L, 3L, 1L);
            np.expand_dims(f, (0, -1)).shape.Should().Equal(1L, 2L, 3L, 1L);
            Action range = () => np.expand_dims(f, (0, 4));
            range.Should().Throw<AxisError>().WithMessage("axis 4 is out of bounds for array of dimension 4*");
            Action repeat = () => np.expand_dims(f, (1, 1));
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis");
            var fromMatrix = np.expand_dims(M, (0, 1));
            (fromMatrix is NDMatrix).Should().BeFalse("NumPy: expand_dims(m, (0, 1)) is an ndarray (1, 1, 2, 3)");
            fromMatrix.shape.Should().Equal(1L, 1L, 2L, 3L);
        }

        // ------------------------------------------------------------------ fix 4: moveaxis

        /// <summary>One signature; NumPy: <c>moveaxis(a, (0, 1), (2, 0))</c> is <c>(3, 4, 2)</c>, a length mismatch and a
        ///     repeated destination are its ValueErrors, an out-of-range source names the argument.</summary>
        [TestMethod]
        public void MoveAxis_OneSignature_Tuples_AsNumPy()
        {
            NpForms("moveaxis").Should().HaveCount(1);
            np.moveaxis(A, (0, 1), (2, 0)).shape.Should().Equal(3L, 4L, 2L);
            np.moveaxis(A, 0, -1).shape.Should().Equal(3L, 4L, 2L);
            Action mismatch = () => np.moveaxis(A, (0, 1), new[] { 2 });
            mismatch.Should().Throw<ValueError>()
                .WithMessage("`source` and `destination` arguments must have the same number of elements");
            Action repeat = () => np.moveaxis(A, (0, 1), (1, 1));
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis in `destination` argument");
            Action range = () => np.moveaxis(A, (0, 5), (1, 2));
            range.Should().Throw<AxisError>().WithMessage("source: axis 5 is out of bounds for array of dimension 3*");
        }

        // ------------------------------------------------------------------ fix 5: squeeze

        /// <summary>NumPy: <c>squeeze(z (1, 3, 1), (0, 2))</c> is <c>(3,)</c>, <c>(0, 1)</c> the length error, <c>(0, 0)</c>
        ///     <c>duplicate value in 'axis'</c>; each item's range and repeat are checked in order (<c>(0, 0, 5)</c> is the
        ///     repeat, <c>(5, 0, 0)</c> the range); <c>()</c> removes nothing.</summary>
        [TestMethod]
        public void Squeeze_Tuple_AsNumPy()
        {
            var z = np.zeros(new Shape(1, 3, 1));
            np.squeeze(z, (0, 2)).shape.Should().Equal(3L);
            z.squeeze((0, 2)).shape.Should().Equal(3L);
            np.squeeze(z, (2, 0)).shape.Should().Equal(3L);
            np.squeeze(z, new int[0]).shape.Should().Equal(1L, 3L, 1L);
            Action size = () => np.squeeze(z, (0, 1));
            size.Should().Throw<ValueError>().WithMessage("cannot select an axis to squeeze out which has size not equal to one");
            Action duplicate = () => np.squeeze(z, (0, 0));
            duplicate.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            Action repeatFirst = () => np.squeeze(z, (0, 0, 5));
            repeatFirst.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            Action rangeFirst = () => np.squeeze(z, (5, 0, 0));
            rangeFirst.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 3*");
        }

        /// <summary>NumPy's integer form keeps its 0-d case (<c>squeeze(np.array(5.), 0)</c> is the array), the tuple form
        ///     has none (<c>(0,)</c> is an AxisError, <c>()</c> the array); a matrix stays a 1 x 3 matrix, a chararray a
        ///     chararray.</summary>
        [TestMethod]
        public void Squeeze_ZeroDAndSubclasses_AsNumPy()
        {
            var s = np.array(5.0);
            ((double)np.squeeze(s, 0)).Should().Be(5.0);
            ((double)np.squeeze(s, new int[0])).Should().Be(5.0);
            Action tuple = () => np.squeeze(s, new[] { 0 });
            tuple.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");

            var row = np.matrix(np.array(new double[,] { { 1, 2, 3 } }));
            var m = np.squeeze(row, new[] { 0 });
            (m is NDMatrix).Should().BeTrue();
            m.shape.Should().Equal(1L, 3L);
            Action both = () => np.squeeze(row, (0, 1));
            both.Should().Throw<ValueError>().WithMessage("cannot select an axis to squeeze out which has size not equal to one");

            var text = np.@char.array(np.array(new[] { "a", "b" }).reshape(1, 2));
            var c = np.squeeze(text, new[] { 0 });
            (c is NDCharArray).Should().BeTrue();
            c.shape.Should().Equal(2L);
        }

        // ------------------------------------------------------------------ fix 6: all / any

        /// <summary>One axis signature each, beside the kept <c>bool all(a)</c>; NumPy: <c>all(b, (0, 2))</c> is
        ///     <c>[False, False, True]</c>, with keepdims <c>(1, 3, 1)</c>; <c>any(b, None)</c> is <c>np.True_</c>; the ufunc
        ///     reduction's repeat text is <c>duplicate value in 'axis'</c>.</summary>
        [TestMethod]
        public void AllAny_OneAxisSignature_Tuple_AsNumPy()
        {
            AxisForms("all").Should().HaveCount(1);
            AxisForms("any").Should().HaveCount(1);
            NpForms("all").Should().HaveCount(2, "bool all(a) stays beside the axis signature");
            var b = A > 5;
            np.all(b, (0, 2)).ToArray<bool>().Should().Equal(false, false, true);
            b.all((0, 2)).ToArray<bool>().Should().Equal(false, false, true);
            np.all(b, (0, 2), keepdims: true).shape.Should().Equal(1L, 3L, 1L);
            var none = np.any(b, null);
            none.ndim.Should().Be(0);
            ((bool)none).Should().BeTrue();
            none.Shape.IsWriteable.Should().BeFalse("NumPy's np.True_ scalar");
            np.all(b, new int[0])[0, 0].ToArray<bool>().Should().Equal(false, false, false, false);
            Action duplicate = () => np.all(b, (1, 0, 0));
            duplicate.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            ((bool)np.all(np.array(5.0), 0)).Should().BeTrue("a 0-d operand takes the integer axis 0");
        }

        /// <summary>
        ///     The owner's question on fix 6 (why not <c>NDArray&lt;bool&gt;</c>): NumPy returns the <c>out</c> array ITSELF,
        ///     of ITS dtype (<c>np.all(b, 0, out=int64)</c> is that int64 array holding 0 / 1, <c>np.any(b, 1, out=float32)</c>
        ///     a float32 one), so the declared type is <see cref="NDArray"/>; without <c>out</c> the instance is the bool
        ///     <c>NDArray&lt;bool&gt;</c> and <c>AsGeneric&lt;bool&gt;()</c> is its typed view. A C# bool in the out slot is
        ///     NumPy's <c>TypeError: return arrays must be of ArrayType</c>.
        /// </summary>
        [TestMethod]
        public void AllAny_ReturnType_IsOutsDtype_AsNumPy()
        {
            var b = np.array(new[,] { { true, false }, { true, true } });
            var o = np.zeros(new Shape(2), np.int64);
            var r = np.all(b, 0, o);
            ReferenceEquals(r, o).Should().BeTrue();
            r.ToArray<long>().Should().Equal(1L, 0L);
            var f = np.zeros(new Shape(2), np.float32);
            np.any(b, 1, f).ToArray<float>().Should().Equal(1f, 1f);
            NDArray<bool> typed = np.all(b, 0).AsGeneric<bool>();
            typed.Should().NotBeNull();
            typed.ToArray<bool>().Should().Equal(true, false);
            Action positional = () => np.all(b, 0, true);
            positional.Should().Throw<TypeError>().WithMessage("return arrays must be of ArrayType");
        }

        /// <summary>A matrix operand gets <c>matrix.all</c>'s answer for every axis spelling: the 1 x 3 row for axis 0, the
        ///     2 x 1 column for axis 1 and <c>(1,)</c>, the 1 x 1 matrix for <c>(0, 1)</c>, the bool matrix for <c>()</c>, the
        ///     scalar for None, and NumPy's TypeError for keepdims. The typed <c>(nd, int axis)</c> overload returned a plain
        ///     array here.</summary>
        [TestMethod]
        public void AllAny_MatrixOperand_EveryAxisSpelling_AsNumPy()
        {
            void Check(NDArray r, long rows, long cols, params bool[] values)
            {
                (r is NDMatrix).Should().BeTrue();
                r.shape.Should().Equal(rows, cols);
                r.ToArray<bool>().Should().Equal(values);
            }

            Check(np.all(MB, 0), 1, 3, true, false, true);
            Check(np.all(MB, 1), 2, 1, false, true);
            Check(np.all(MB, -1), 2, 1, false, true);
            Check(np.all(MB, new[] { 1 }), 2, 1, false, true);
            Check(np.all(MB, (0, 1)), 1, 1, false);
            Check(np.all(MB, new int[0]), 2, 3, true, false, true, true, true, true);
            Check(np.any(MB, 0), 1, 3, true, true, true);
            Check(np.any(MB, (0, 1)), 1, 1, true);
            ((bool)np.all(MB, null)).Should().BeFalse();
            Action keepdims = () => np.any(MB, (0, 1), keepdims: true);
            keepdims.Should().Throw<TypeError>().WithMessage("matrix.any() got an unexpected keyword argument 'keepdims'");
            Check(((NDMatrix)MB).all((0, 1)), 1, 1, false);
        }

        // ------------------------------------------------------------------ fix 7: count_nonzero

        /// <summary>One axis signature beside the kept <c>long count_nonzero(a)</c>; NumPy: <c>count_nonzero(a, (0, 2))</c>
        ///     is <c>[7, 8, 8]</c> (int64), <c>(a, None, keepdims=True)</c> <c>[[[23]]]</c>, <c>(a, None)</c>
        ///     <c>np.int64(23)</c>, the repeat <c>duplicate value in 'axis'</c>, <c>(0, 0, 5)</c> the AxisError of 5.</summary>
        [TestMethod]
        public void CountNonzero_TupleNoneKeepdims_AsNumPy()
        {
            AxisForms("count_nonzero").Should().HaveCount(1);
            var a = A;
            var t = np.count_nonzero(a, (0, 2));
            t.dtype.Should().Be(np.int64);
            t.ToArray<long>().Should().Equal(7L, 8L, 8L);
            np.count_nonzero(a, (2, 0)).ToArray<long>().Should().Equal(7L, 8L, 8L);
            np.count_nonzero(a, (0, 2), true).shape.Should().Equal(1L, 3L, 1L);
            var kept = np.count_nonzero(a, null, true);
            kept.shape.Should().Equal(1L, 1L, 1L);
            kept.ToArray<long>().Should().Equal(23L);
            var flat = np.count_nonzero(a, null);
            flat.ndim.Should().Be(0);
            ((long)flat).Should().Be(23L);
            flat.Shape.IsWriteable.Should().BeFalse("NumPy's np.int64 scalar");
            np.count_nonzero(a, (0, 1, 2)).ndim.Should().Be(0);
            Action duplicate = () => np.count_nonzero(a, (0, -3));
            duplicate.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            Action range = () => np.count_nonzero(a, (0, 0, 5));
            range.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 3*");
        }

        /// <summary>NumPy: the empty tuple counts each element (int64, C-contiguous), a 0-d operand takes <c>()</c> and the
        ///     integer 0 but not <c>(0,)</c>, an F-ordered operand keeps F order (<c>(1, 3)</c> of a (2, 3, 4, 5) F array is an
        ///     F-contiguous (2, 4)), empty data counts zeros, and text counts its non-empty strings.</summary>
        [TestMethod]
        public void CountNonzero_EmptyTupleZeroDLayoutText_AsNumPy()
        {
            var a = A;
            a[0, 0] = np.zeros(4);
            var each = np.count_nonzero(a, new int[0]);
            each.shape.Should().Equal(2L, 3L, 4L);
            each.Shape.IsContiguous.Should().BeTrue();
            each[0, 0].ToArray<long>().Should().Equal(0L, 0L, 0L, 0L);
            each[0, 1].ToArray<long>().Should().Equal(1L, 1L, 1L, 1L);

            var s = np.array(5.0);
            ((long)np.count_nonzero(s, new int[0])).Should().Be(1L);
            ((long)np.count_nonzero(s, 0)).Should().Be(1L);
            ((long)np.count_nonzero(s, null, true)).Should().Be(1L);
            Action tuple = () => np.count_nonzero(s, new[] { 0 });
            tuple.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");

            var f = np.asfortranarray(np.arange(120.0).reshape(2, 3, 4, 5) % 3);
            var fr = np.count_nonzero(f, (1, 3));
            fr.Shape.IsFContiguous.Should().BeTrue();
            fr.Shape.IsContiguous.Should().BeFalse();
            fr.ToArray<long>().Should().AllBeEquivalentTo(10L);
            np.count_nonzero(f, (0, 2), true).shape.Should().Equal(1L, 3L, 1L, 5L);

            np.count_nonzero(np.empty(new Shape(2, 0, 3)), (0, 1)).ToArray<long>().Should().Equal(0L, 0L, 0L);
            np.count_nonzero(np.empty(new Shape(2, 0, 3)), (0, 2)).shape.Should().Equal(0L);
            var u = np.array(new[] { "a", "", "b", "c" }).reshape(2, 2);
            ((long)np.count_nonzero(u, (0, 1))).Should().Be(3L);
            np.count_nonzero(u, null, true).ToArray<long>().Should().Equal(3L);
        }

        // ------------------------------------------------------------------ fix 8: ptp

        /// <summary>One signature; NumPy: <c>ptp(a, None)</c> is 23.0, <c>ptp(a, (0, 2))</c> <c>[15, 15, 15]</c> (keepdims
        ///     <c>(1, 3, 1)</c>), <c>()</c> zeros, <c>(0, 0)</c> <c>duplicate value in 'axis'</c>; a matrix's <c>(0, 1)</c> is
        ///     the 1 x 1 matrix.</summary>
        [TestMethod]
        public void Ptp_OneSignature_NoneTuple_AsNumPy()
        {
            AxisForms("ptp").Should().HaveCount(1);
            ((double)np.ptp(A, null)).Should().Be(23.0);
            np.ptp(A, (0, 2)).ToArray<double>().Should().Equal(15.0, 15.0, 15.0);
            np.ptp(A, (0, 2), keepdims: true).shape.Should().Equal(1L, 3L, 1L);
            np.ptp(A, new int[0])[0, 0].ToArray<double>().Should().Equal(0.0, 0.0, 0.0, 0.0);
            Action duplicate = () => np.ptp(A, (0, 0));
            duplicate.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            var m = np.ptp(M, (0, 1));
            (m is NDMatrix).Should().BeTrue();
            m.shape.Should().Equal(1L, 1L);
            m.ToArray<double>().Should().Equal(6.0);
        }

        // ------------------------------------------------------------------ fix 9: median / nanmedian

        /// <summary>One signature each; NumPy: <c>median(a, None)</c> is 11.5, <c>nanmedian(a, (0, 2))</c>
        ///     <c>[7.5, 11.5, 15.5]</c> (keepdims <c>(1, 3, 1)</c>), <c>()</c> reduces nothing, <c>(0, 0)</c> is
        ///     <c>repeated axis</c> (normalize_axis_tuple's text, not the ufunc's).</summary>
        [TestMethod]
        public void Median_OneSignature_NoneTuple_AsNumPy()
        {
            AxisForms("median").Should().HaveCount(1);
            AxisForms("nanmedian").Should().HaveCount(1);
            ((double)np.median(A, null)).Should().Be(11.5);
            np.nanmedian(A, (0, 2)).ToArray<double>().Should().Equal(7.5, 11.5, 15.5);
            np.median(A, (0, 2), keepdims: true).shape.Should().Equal(1L, 3L, 1L);
            np.median(A, new int[0])[0, 1].ToArray<double>().Should().Equal(4.0, 5.0, 6.0, 7.0);
            Action repeat = () => np.median(A, (0, 0));
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis");
            Action nanRepeat = () => np.nanmedian(A, (0, 0));
            nanRepeat.Should().Throw<ValueError>().WithMessage("repeated axis");
            Action range = () => np.median(A, (0, 5));
            range.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 3*");
        }

        // ------------------------------------------------------------------ fixes 10 / 11: percentile / quantile families

        /// <summary>One signature each, q as one array; NumPy: <c>percentile(a, 50, None)</c> is 11.5,
        ///     <c>nanpercentile(a, 50, (0, 2))</c> <c>[7.5, 11.5, 15.5]</c>, <c>percentile(a, [25, 75], (0, 2))</c> of shape
        ///     <c>(2, 3)</c>, <c>quantile(a, 0.5, None)</c> 11.5, <c>nanquantile(a, 0.5, (0, 2))</c>
        ///     <c>[7.5, 11.5, 15.5]</c>, a repeated tuple axis <c>repeated axis</c>.</summary>
        [TestMethod]
        public void QuantileFamily_OneSignature_NoneTuple_AsNumPy()
        {
            foreach (var name in new[] { "percentile", "nanpercentile", "quantile", "nanquantile" })
            {
                var forms = NpForms(name);
                forms.Should().HaveCount(1, name);
                forms[0].GetParameters().Select(p => p.Name).Should()
                    .Equal(new[] { "a", "q", "axis", "out", "overwrite_input", "method", "keepdims", "weights" }, name);
                forms[0].GetParameters()[1].ParameterType.Should().Be(typeof(NDArray), name);
            }
            ((double)np.percentile(A, 50, null)).Should().Be(11.5);
            np.nanpercentile(A, 50, (0, 2)).ToArray<double>().Should().Equal(7.5, 11.5, 15.5);
            np.percentile(A, new[] { 25.0, 75.0 }, (0, 2)).shape.Should().Equal(2L, 3L);
            ((double)np.quantile(A, 0.5, null)).Should().Be(11.5);
            np.nanquantile(A, 0.5, (0, 2)).ToArray<double>().Should().Equal(7.5, 11.5, 15.5);
            Action repeat = () => np.percentile(A, 50, (0, 0));
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis");
        }

        /// <summary>
        ///     Every former q spelling keeps NumPy's reading of it: a C# number is a Python number, weak for the result dtype
        ///     (<c>percentile(float32, 50.0, 1)</c> is float32 <c>[2, 8]</c>, <c>quantile(float32, 0.25)</c> the float32 2.25), a
        ///     <c>double[]</c> a list (float64, q's axis first: <c>(1, 2)</c>), an explicit 0-d array a STRONG float64 q, and an
        ///     integer literal NumPy's Python int, whose 'linear' index is taken (<c>quantile(int32, 1, 1)</c> is the int32
        ///     <c>[3, 9]</c>; <c>quantile(int32, 0.5, 1)</c> interpolates in float64).
        /// </summary>
        [TestMethod]
        public void QuantileFamily_QForms_KeepNumPysReading()
        {
            var f32 = np.array(new float[,] { { 3, 1, 2 }, { 9, 7, 8 } });
            var i32 = np.array(new int[,] { { 3, 1, 2 }, { 9, 7, 8 } });

            var weak = np.percentile(f32, 50.0, 1);
            weak.dtype.Should().Be(np.float32);
            weak.ToArray<float>().Should().Equal(2f, 8f);
            np.percentile(f32, 50).dtype.Should().Be(np.float32);
            var list = np.percentile(f32, new[] { 50.0 }, 1);
            list.dtype.Should().Be(np.float64);
            list.shape.Should().Equal(1L, 2L);
            np.percentile(f32, np.array(50.0), 1).dtype.Should().Be(np.float64);
            ((float)np.quantile(f32, 0.25)).Should().Be(2.25f);

            var taken = np.quantile(i32, 1, 1);
            taken.dtype.Should().Be(np.int32);
            taken.ToArray<int>().Should().Equal(3, 9);
            np.quantile(i32, 0.5, 1).dtype.Should().Be(np.float64);
            ((int)np.nanquantile(i32, 1)).Should().Be(9);
        }

        // ------------------------------------------------------------------ fix 12: average

        /// <summary>One signature each for average and average_returned; NumPy: <c>average(a, None)</c> is 11.5,
        ///     <c>average(a, (0, 2))</c> <c>[7.5, 11.5, 15.5]</c>, <c>(0, 0)</c> <c>repeated axis in `axis` argument</c>; a
        ///     matrix's <c>(0, 1)</c> is the 1 x 1 matrix <c>[[3.0]]</c> and its None the scalar 3.0.</summary>
        [TestMethod]
        public void Average_OneSignature_NoneTuple_AsNumPy()
        {
            AxisForms("average").Should().HaveCount(1);
            AxisForms("average_returned").Should().HaveCount(1);
            ((double)np.average(A, null)).Should().Be(11.5);
            np.average(A, (0, 2)).ToArray<double>().Should().Equal(7.5, 11.5, 15.5);
            var (avg, sum) = np.average_returned(A, (0, 2));
            avg.ToArray<double>().Should().Equal(7.5, 11.5, 15.5);
            sum.ToArray<double>().Should().Equal(8.0, 8.0, 8.0);
            Action repeat = () => np.average(A, (0, 0));
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis in `axis` argument");
            var m = np.average(M, (0, 1));
            (m is NDMatrix).Should().BeTrue();
            m.shape.Should().Equal(1L, 1L);
            m.ToArray<double>().Should().Equal(3.0);
            var scalar = np.average(M, null);
            (scalar is NDMatrix).Should().BeFalse();
            ((double)scalar).Should().Be(3.0);
        }

        // ------------------------------------------------------------------ fix 13: trim_zeros

        /// <summary>One signature; NumPy: <c>trim_zeros(t, 'fb', None)</c> and <c>(0, 1)</c> are <c>[[1]]</c>, <c>0</c>
        ///     <c>[[0, 1, 0]]</c>, <c>()</c> the input, <c>(0, 0)</c> <c>repeated axis in `axis` argument</c>, <c>(0, 5)</c>
        ///     <c>axis: axis 5 is out of bounds for array of dimension 2</c>.</summary>
        [TestMethod]
        public void TrimZeros_OneSignature_NoneIntTuple_AsNumPy()
        {
            AxisForms("trim_zeros").Should().HaveCount(1);
            var t = np.array(new long[,] { { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 } });
            np.trim_zeros(t, "fb", null).ToArray<long>().Should().Equal(1L);
            np.trim_zeros(t, "fb", (0, 1)).shape.Should().Equal(1L, 1L);
            np.trim_zeros(t, "fb", 0).ToArray<long>().Should().Equal(0L, 1L, 0L);
            np.trim_zeros(t, "fb", new int[0]).shape.Should().Equal(3L, 3L);
            Action repeat = () => np.trim_zeros(t, "fb", (0, 0));
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis in `axis` argument");
            Action range = () => np.trim_zeros(t, "fb", (0, 5));
            range.Should().Throw<AxisError>().WithMessage("axis: axis 5 is out of bounds for array of dimension 2*");
        }

        // ------------------------------------------------------------------ fix 14: size

        /// <summary>NumPy: <c>size(a (2, 3, 4), (0, 2))</c> is 8, <c>(-1, 0)</c> 8, <c>()</c> 1, <c>(0, 0)</c>
        ///     <c>repeated axis</c>; a 0-d operand takes only <c>()</c>.</summary>
        [TestMethod]
        public void Size_Tuple_AsNumPy()
        {
            NpForms("size").Should().HaveCount(1);
            np.size(A, (0, 2)).Should().Be(8L);
            np.size(A, (-1, 0)).Should().Be(8L);
            np.size(A, new int[0]).Should().Be(1L);
            np.size(A, -1).Should().Be(4L);
            Action repeat = () => np.size(A, (0, 0));
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis");
            np.size(np.array(5.0), new int[0]).Should().Be(1L);
            Action zeroD = () => np.size(np.array(5.0), 0);
            zeroD.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");
        }

        // ------------------------------------------------------------------ fix 15: gradient

        /// <summary>NumPy: <c>gradient(f, axis=(0, 1))</c> needs no spacing (<c>[[3, 3, 3] x2]</c>, <c>[[1, 1, 1] x2]</c>),
        ///     <c>axis=(1,)</c> is the single array, <c>axis=()</c> no component, <c>(0, 0)</c> <c>repeated axis</c>,
        ///     <c>gradient(f, 2.0, axis=(0, 1))</c> halves both, and two spacings with one axis are <c>TypeError: invalid
        ///     number of arguments</c>.</summary>
        [TestMethod]
        public void Gradient_AxisForms_AsNumPy()
        {
            AxisForms("gradient").Select(m => m.GetParameters().Single(p => p.Name == "axis").ParameterType)
                .Should().AllBeEquivalentTo(typeof(NDAxis?));
            var f = np.arange(6.0).reshape(2, 3);
            var (gy, gx) = np.gradient(f, axis: (0, 1));
            gy.ToArray<double>().Should().AllBeEquivalentTo(3.0);
            gx.ToArray<double>().Should().AllBeEquivalentTo(1.0);
            var single = np.gradient(f, axis: new[] { 1 });
            single.IsSingle.Should().BeTrue();
            ((NDArray)single).ToArray<double>().Should().AllBeEquivalentTo(1.0);
            np.gradient(f, axis: new int[0]).Length.Should().Be(0);
            Action repeat = () => np.gradient(f, axis: (0, 0));
            repeat.Should().Throw<ValueError>().WithMessage("repeated axis");
            var (hy, hx) = np.gradient(f, 2.0, axis: (0, 1));
            hy.ToArray<double>().Should().AllBeEquivalentTo(1.5);
            hx.ToArray<double>().Should().AllBeEquivalentTo(0.5);
            Action arity = () => np.gradient(f, 2.0, 3.0, axis: 1);
            arity.Should().Throw<TypeError>().WithMessage("invalid number of arguments");
        }
    }
}
