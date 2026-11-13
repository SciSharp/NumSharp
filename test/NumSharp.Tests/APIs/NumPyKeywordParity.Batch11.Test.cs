using System;
using System.Globalization;
using System.Linq;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 11 (the owner approved fixes 1, 2 and 4 and decided D2 and D3): np.intersect1d's
    ///     <see cref="np.Intersect1dResult"/>, np.ma.unique's one typed form, np.ma.mask_rows / mask_cols' deprecated axis,
    ///     the polynomial facades' typed power / tolerance / degree (D2: the object and double forms went), and result
    ///     structs for NumPy's flag-dependent returns (D3: np.average / np.ma.average <c>returned</c>, np.indices
    ///     <c>sparse</c>). Every expected value, dtype and text is NumPy 2.4.2's (scratchpad batch11_numpy_evidence.txt), for
    ///     the call exactly as the C# fix makes it; floating values are compared bit for bit (their shortest round-trip text).
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch11Tests
    {
        /// <summary>Asserts that <paramref name="act"/> throws <typeparamref name="T"/> with NumPy's exact text.</summary>
        /// <typeparam name="T">The exception type NumPy's error maps to.</typeparam>
        /// <param name="act">The call that must fail.</param>
        /// <param name="message">NumPy 2.4.2's text, verbatim.</param>
        /// <exception cref="AssertFailedException">The call does not throw <typeparamref name="T"/> with that text.</exception>
        private static void Throws<T>(Action act, string message) where T : Exception
            => act.Should().Throw<T>().Which.Message.Should().Be(message);

        /// <summary>The values of an array in C order as float64 round-trip texts (Python's <c>repr</c> of each), so a
        ///     comparison sees every bit.</summary>
        /// <param name="a">The array (any real dtype; float16 / float32 widen exactly).</param>
        /// <returns>One text per element.</returns>
        private static string[] Bits(NDArray a)
            => a.astype(np.float64).ToArray<double>().Select(v => double.IsNaN(v) ? "nan" : v.ToString("R", CultureInfo.InvariantCulture)).ToArray();

        /// <summary>The evidence's <c>a23 = np.arange(6.0).reshape(2, 3)</c>.</summary>
        /// <returns>A new (2, 3) float64 array.</returns>
        private static NDArray A23() => np.arange(6.0).reshape(2, 3);

        /// <summary>The evidence's <c>w23 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])</c>.</summary>
        /// <returns>A new (2, 3) float64 array.</returns>
        private static NDArray W23() => np.array(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });

        /// <summary>The evidence's <c>m23 = ma.array(a23, mask=[[0, 1, 0], [0, 0, 1]])</c>.</summary>
        /// <returns>A new (2, 3) masked float64 array.</returns>
        private static NDMaskedArray M23()
            => np.ma.array(A23(), mask: np.array(new[,] { { false, true, false }, { false, false, true } }));

        // ------------------------------------------------------------------------------------------------------------
        // Fix 1: np.intersect1d(ar1, ar2, assume_unique=False, return_indices=False) -> Intersect1dResult
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Without <c>return_indices</c> the result is NumPy's bare array: <c>intersect1d([1,3,4,3], [3,1,2,1])</c>
        ///     is int64 <c>[1, 3]</c>; it converts to <see cref="NDArray"/> and holds one output, so a three-way
        ///     deconstruction is Python's unpacking error.</summary>
        [TestMethod]
        public void Intersect1d_WithoutIndices_IsTheBareArray()
        {
            var r = np.intersect1d(np.array(new long[] { 1, 3, 4, 3 }), np.array(new long[] { 3, 1, 2, 1 }));
            r.Length.Should().Be(1);
            r.comm1.Should().BeNull();
            r.comm2.Should().BeNull();
            NDArray values = r;
            values.dtype.Should().Be(np.int64);
            values.ToArray<long>().Should().Equal(1L, 3L);
            r.shape.Should().Equal(2L);
            Throws<InvalidOperationException>(() => { var (_, _, _) = r; }, "not enough values to unpack (expected 3, got 1)");
        }

        /// <summary>With <c>return_indices</c> it is NumPy's 3-tuple: <c>([1, 3], [0, 1], [1, 0])</c>; with
        ///     <c>assume_unique</c> too, <c>intersect1d([1,2,3], [2,3,4], True, True)</c> is <c>([2, 3], [1, 2], [0, 1])</c>;
        ///     a text operand keeps its dtype (<c>&lt;U1</c>) beside int64 indices.</summary>
        [TestMethod]
        public void Intersect1d_ReturnIndices_IsNumPysTriple()
        {
            var (values, comm1, comm2) = np.intersect1d(np.array(new long[] { 1, 3, 4, 3 }), np.array(new long[] { 3, 1, 2, 1 }),
                return_indices: true);
            values.ToArray<long>().Should().Equal(1L, 3L);
            comm1.dtype.Should().Be(np.int64);
            comm1.ToArray<long>().Should().Equal(0L, 1L);
            comm2.ToArray<long>().Should().Equal(1L, 0L);

            var unique = np.intersect1d(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 2, 3, 4 }), true, true);
            unique.Length.Should().Be(3);
            unique[0].ToArray<long>().Should().Equal(2L, 3L);
            unique[1].ToArray<long>().Should().Equal(1L, 2L);
            unique[2].ToArray<long>().Should().Equal(0L, 1L);
            ((NDArray[])unique).Should().HaveCount(3);

            var (text, i1, i2) = np.intersect1d(np.array(new[] { "b", "a" }), np.array(new[] { "a" }), return_indices: true);
            text.dtype.str.Should().Be("<U1");
            text.GetAtIndex(0).Should().Be("a");
            i1.ToArray<long>().Should().Equal(1L);
            i2.ToArray<long>().Should().Equal(0L);
        }

        /// <summary>The one public form: <c>intersect1d(NDArray, NDArray, bool, bool)</c> returning the result struct
        ///     (NumPy's names and defaults).</summary>
        [TestMethod]
        public void Intersect1d_IsOneSignature()
        {
            var forms = typeof(np).GetMethods().Where(m => m.Name == "intersect1d").ToArray();
            forms.Should().ContainSingle();
            forms[0].ReturnType.Should().Be(typeof(np.Intersect1dResult));
            forms[0].GetParameters().Select(p => p.Name).Should().Equal("ar1", "ar2", "assume_unique", "return_indices");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 2: np.ma.unique(ar1, return_index=False, return_inverse=False) - one typed form
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The flag-less call reads as NumPy's masked array: <c>ma.unique(ma.array([3, 1, 3, 2, 1], mask=[0, 0, 1,
        ///     0, 0]))</c> is int64 data <c>[1, 2, 3, 3]</c>, mask <c>[F, F, F, T]</c>; with <c>return_index</c> the pair
        ///     adds <c>[1, 3, 0, 2]</c>.</summary>
        [TestMethod]
        public void MaUnique_OneTypedForm_ReadsAsTheMaskedArray()
        {
            var mu = np.ma.array(np.array(new long[] { 3, 1, 3, 2, 1 }), mask: np.array(new[] { false, false, true, false, false }));
            var u = np.ma.unique(mu);
            u.Length.Should().Be(1);
            u.dtype.Should().Be(np.int64);
            u.shape.Should().Equal(4L);
            u.data.ToArray<long>().Should().Equal(1L, 2L, 3L, 3L);
            u.mask.ToArray<bool>().Should().Equal(false, false, false, true);
            NDMaskedArray masked = u;
            masked.data.ToArray<long>().Should().Equal(1L, 2L, 3L, 3L);

            var (values, index) = np.ma.unique(mu, true);
            values.mask.ToArray<bool>().Should().Equal(false, false, false, true);
            index.ToArray<long>().Should().Equal(1L, 3L, 0L, 2L);

            typeof(MaskedArrayModule).GetMethods().Where(m => m.Name == "unique").Should().ContainSingle();
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 4: np.ma.mask_rows(a, axis) / mask_cols(a, axis) - NumPy's deprecated, ignored axis
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy ignores <c>axis</c> (with a DeprecationWarning): <c>mask_rows(x, axis=1)</c> masks row 1 as
        ///     <c>mask_rows(x)</c> does, and <c>mask_cols(x, axis=0)</c> / <c>mask_cols(x, None)</c> mask column 1.</summary>
        [TestMethod]
        public void MaskRowsCols_AxisIsIgnored()
        {
            NDMaskedArray X() => np.ma.array(np.zeros(new Shape(3, 3), np.int64),
                mask: np.array(new[,] { { false, false, false }, { false, true, false }, { false, false, false } }));
            var rows = new[] { false, false, false, true, true, true, false, false, false };
            var cols = new[] { false, true, false, false, true, false, false, true, false };
            np.ma.mask_rows(X()).mask.ToArray<bool>().Should().Equal(rows);
            np.ma.mask_rows(X(), 1).mask.ToArray<bool>().Should().Equal(rows);
            np.ma.mask_cols(X()).mask.ToArray<bool>().Should().Equal(cols);
            np.ma.mask_cols(X(), 0).mask.ToArray<bool>().Should().Equal(cols);
            np.ma.mask_cols(X(), null).mask.ToArray<bool>().Should().Equal(cols);
        }

        // ------------------------------------------------------------------------------------------------------------
        // D2: the polynomial facades' typed power, tolerance and degree
        // ------------------------------------------------------------------------------------------------------------

        /// <summary><c>chebpow(c, int pow, int? maxpower = 16)</c>: <c>chebpow([1, 2], 2)</c> is <c>[3, 4, 2]</c>; 17 is
        ///     past the default limit; <c>maxpower=None</c> lifts it (shape <c>(18,)</c>); 3 is past <c>maxpower=2</c>; -1 is
        ///     not a non-negative integer.</summary>
        [TestMethod]
        public void Pow_TypedPowerAndLimit_AsNumPy()
        {
            var cheb = np.polynomial.chebyshev;
            var c = np.array(new[] { 1.0, 2.0 });
            Bits(cheb.chebpow(c, 2)).Should().Equal("3", "4", "2");
            Throws<ValueError>(() => cheb.chebpow(c, 17), "Power is too large");
            cheb.chebpow(c, 17, null).shape.Should().Equal(18L);
            Throws<ValueError>(() => cheb.chebpow(c, 3, maxpower: 2), "Power is too large");
            Throws<ValueError>(() => cheb.chebpow(c, -1), "Power must be a non-negative integer.");
        }

        /// <summary><c>{p}trim(c, double tol = 0)</c> / <c>trimcoef</c>: <c>chebtrim([1, 0.001, 0, 0])</c> is <c>[1,
        ///     0.001]</c>, with 0.01 <c>[1.0]</c>, with 1e-4 <c>[1, 0.001]</c>; a float32 series keeps its dtype; an all-zero
        ///     series is <c>[0.0]</c>; a negative tolerance is NumPy's ValueError.</summary>
        [TestMethod]
        public void Trim_TypedTolerance_AsNumPy()
        {
            var cheb = np.polynomial.chebyshev;
            var pu = np.polynomial.polyutils;
            var c = np.array(new[] { 1.0, 0.001, 0.0, 0.0 });
            Bits(cheb.chebtrim(c)).Should().Equal("1", "0.001");
            Bits(cheb.chebtrim(c, 0.01)).Should().Equal("1");
            Bits(cheb.chebtrim(c, tol: 1e-4)).Should().Equal("1", "0.001");
            Throws<ValueError>(() => cheb.chebtrim(np.array(new[] { 1.0, 2.0 }), -1), "tol must be non-negative");
            var f32 = cheb.chebtrim(np.array(new[] { 1.0f, 0.5f }), 0.5);
            f32.dtype.Should().Be(np.float32);
            Bits(f32).Should().Equal("1");
            Bits(pu.trimcoef(np.array(new[] { 0.0, 0.0, 0.0 }))).Should().Equal("0");
            Bits(pu.trimcoef(np.array(new[] { 1.0, 2.0, 0.1 }), 0.2)).Should().Equal("1", "2");
        }

        /// <summary><c>{p}vander(x, int deg)</c>: <c>chebvander([0.5, 1], 2)</c> is <c>[[1, 0.5, -0.5], [1, 1, 1]]</c>; a
        ///     negative degree is NumPy's ValueError.</summary>
        [TestMethod]
        public void Vander_TypedDegree_AsNumPy()
        {
            var cheb = np.polynomial.chebyshev;
            var v = cheb.chebvander(np.array(new[] { 0.5, 1.0 }), 2);
            v.shape.Should().Equal(2L, 3L);
            Bits(v).Should().Equal("1", "0.5", "-0.5", "1", "1", "1");
            Throws<ValueError>(() => cheb.chebvander(np.array(new[] { 0.5, 1.0 }), -1), "deg must be non-negative");
        }

        /// <summary>Every basis keeps one typed signature per function: <c>{p}pow(object, int, int?)</c>,
        ///     <c>{p}trim(object, double)</c>, <c>{p}vander(object, int)</c>, and <c>trimcoef(object, double)</c>.</summary>
        [TestMethod]
        public void Polynomial_TypedFacades_AreOneSignature()
        {
            foreach (var (module, p) in new[]
                     {
                         (typeof(PowerSeriesModule), "poly"), (typeof(ChebyshevModule), "cheb"), (typeof(LegendreModule), "leg"),
                         (typeof(LaguerreModule), "lag"), (typeof(HermiteModule), "herm"), (typeof(HermiteEModule), "herme"),
                     })
            {
                Types(module, p + "pow").Should().Equal(new[] { typeof(object), typeof(int), typeof(int?) }, p);
                Types(module, p + "trim").Should().Equal(new[] { typeof(object), typeof(double) }, p);
                Types(module, p + "vander").Should().Equal(new[] { typeof(object), typeof(int) }, p);
            }
            Types(typeof(PolyUtilsModule), "trimcoef").Should().Equal(typeof(object), typeof(double));

            static Type[] Types(Type module, string name)
            {
                var forms = module.GetMethods().Where(m => m.Name == name).ToArray();
                forms.Should().ContainSingle(name);
                return forms[0].GetParameters().Select(p => p.ParameterType).ToArray();
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // D3: result structs - np.average / np.ma.average returned, np.indices sparse
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>np.average's bare and tuple returns as one <see cref="np.AverageResult"/>: <c>average(a23)</c> is
        ///     2.5; with <c>returned</c> <c>(2.5, 6.0)</c>; <c>axis=0, weights=w23</c> <c>([2.4, 3.142857142857143, 4.0],
        ///     [5, 7, 9])</c>; <c>axis=1</c> <c>([1, 4], [3, 3])</c>; <c>axis=1, weights=[1, 2, 3], keepdims</c>
        ///     <c>([[1.3333333333333333], [4.333333333333333]], [[6], [6]])</c>; an int64 operand <c>(1.5, 4.0)</c>.</summary>
        [TestMethod]
        public void Average_Returned_IsNumPysPair()
        {
            ((double)np.average(A23())).Should().Be(2.5);
            np.average(A23()).Length.Should().Be(1);
            np.average(A23()).sum_of_weights.Should().BeNull();

            var (avg, sum) = np.average(A23(), returned: true);
            avg.ndim.Should().Be(0);
            Bits(avg).Should().Equal("2.5");
            Bits(sum).Should().Equal("6");

            var (avg0, sum0) = np.average(A23(), 0, W23(), returned: true);
            Bits(avg0).Should().Equal("2.4", "3.142857142857143", "4");
            Bits(sum0).Should().Equal("5", "7", "9");

            var (avg1, sum1) = np.average(A23(), 1, returned: true);
            Bits(avg1).Should().Equal("1", "4");
            Bits(sum1).Should().Equal("3", "3");

            var (avgK, sumK) = np.average(A23(), 1, np.array(new[] { 1.0, 2.0, 3.0 }), returned: true, keepdims: true);
            avgK.shape.Should().Equal(2L, 1L);
            Bits(avgK).Should().Equal("1.3333333333333333", "4.333333333333333");
            sumK.shape.Should().Equal(2L, 1L);
            Bits(sumK).Should().Equal("6", "6");

            var (avgI, sumI) = np.average(np.arange(4L), returned: true);
            avgI.dtype.Should().Be(np.float64);
            Bits(avgI).Should().Equal("1.5");
            Bits(sumI).Should().Equal("4");

            Throws<InvalidOperationException>(() => { var (_, _) = np.average(A23()); }, "not enough values to unpack (expected 2, got 1)");
        }

        /// <summary>The result stands in for the NDArray it replaced: <c>(double)</c> / <c>(float)</c> casts read a 0-d
        ///     average, <c>flags</c> and <c>GetAtIndex&lt;T&gt;</c> forward, it disposes, and a non-0-d cast is NumPy's
        ///     TypeError.</summary>
        [TestMethod]
        public void AverageResult_ForwardsTheBareArray()
        {
            using var r = np.average(A23());
            ((float)r).Should().Be(2.5f);
            ((System.Numerics.Complex)r).Should().Be(new System.Numerics.Complex(2.5, 0));
            r.GetAtIndex<double>(0).Should().Be(2.5);
            r.flags.owndata.Should().BeTrue();
            using var rows = np.average(A23(), 1);
            Throws<TypeError>(() => _ = (double)rows, "only 0-dimensional arrays can be converted to Python scalars");
        }

        /// <summary>np.ma.average's <c>returned</c>: <c>ma.average(m23)</c> is 2.25; with <c>returned</c>
        ///     <c>(2.25, 4.0)</c>; <c>axis=0, weights=w23</c> data <c>[2.4, 4.0, 2.0]</c> beside <c>[5, 5, 3]</c>;
        ///     <c>axis=1</c> data <c>[1.0, 3.5]</c> beside <c>[2, 2]</c>.</summary>
        [TestMethod]
        public void MaAverage_Returned_IsNumPysPair()
        {
            var bare = np.ma.average(M23());
            bare.Length.Should().Be(1);
            Bits(bare.data).Should().Equal("2.25");

            var (avg, sum) = np.ma.average(M23(), returned: true);
            Bits(avg.data).Should().Equal("2.25");
            Bits(sum).Should().Equal("4");

            var (avg0, sum0) = np.ma.average(M23(), 0, W23(), returned: true);
            Bits(avg0.data).Should().Equal("2.4", "4", "2");
            Bits(sum0).Should().Equal("5", "5", "3");

            var (avg1, sum1) = np.ma.average(M23(), 1, returned: true);
            Bits(avg1.data).Should().Equal("1", "3.5");
            Bits(sum1).Should().Equal("2", "2");
        }

        /// <summary>np.indices' dense grid and sparse tuple as one <see cref="np.IndicesResult"/>: <c>indices((2, 3))</c>
        ///     is the int64 (2, 2, 3) grid; <c>sparse=True</c> the (2, 1) and (1, 3) arrays; <c>indices((2, 3), np.int8,
        ///     True)</c> int8 ones; <c>indices((), sparse=True)</c> the empty tuple and <c>sparse=False</c> an empty (0,)
        ///     int64 array; <c>indices((0, 2), sparse=True)</c> a (0, 1) and a (1, 2) array.</summary>
        [TestMethod]
        public void Indices_DenseAndSparse_AsNumPy()
        {
            using var grid = np.indices((2, 3));
            grid.sparse.Should().BeFalse();
            grid.shape.Should().Equal(2L, 2L, 3L);
            grid.dtype.Should().Be(np.int64);
            grid.ToArray<long>().Should().Equal(0L, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 2);

            var (i, j) = np.indices((2, 3), sparse: true);
            i.shape.Should().Equal(2L, 1L);
            i.ToArray<long>().Should().Equal(0L, 1L);
            j.shape.Should().Equal(1L, 3L);
            j.ToArray<long>().Should().Equal(0L, 1L, 2L);

            var bytes = np.indices((2, 3), np.int8, true);
            bytes.Length.Should().Be(2);
            bytes[0].dtype.Should().Be(np.int8);
            bytes[1].dtype.Should().Be(np.int8);

            np.indices(new long[0], sparse: true).ToArray().Should().BeEmpty();
            using var none = np.indices(new long[0]);
            none.shape.Should().Equal(0L);
            none.dtype.Should().Be(np.int64);

            var empty = np.indices((0, 2), sparse: true);
            empty[0].shape.Should().Equal(0L, 1L);
            empty[1].shape.Should().Equal(1L, 2L);
            empty[1].ToArray<long>().Should().Equal(0L, 1L);
        }

        /// <summary>
        ///     A result struct's <c>Deconstruct</c> writes the caller's variables whatever they held before: the C# compiler
        ///     reuses a deconstruction's temporaries, so after a deconstruction of a matrix average they still hold matrices,
        ///     and the subclass weaver used to read the <c>out</c> parameters as operands - the subclass path then wrote its own
        ///     locals and the caller kept the previous statement's arrays (NumPy: <c>np.average(m, returned=True)</c> of the
        ///     2 x 3 matrix is the scalar pair <c>(3.5, 6.0)</c>, after the axis-0 pair of (1, 3) arrays).
        /// </summary>
        [TestMethod]
        public void ResultDeconstruct_AfterAMatrixDeconstruction_WritesTheCallersVariables()
        {
            static NDArray M() => np.matrix(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }));
            var (avg0, sum0) = np.average(M(), 0, returned: true);
            (avg0 is NDMatrix).Should().BeTrue();
            sum0.shape.Should().Equal(1L, 3L);
            var (avg, sum) = np.average(M(), returned: true);
            avg.ndim.Should().Be(0);
            ((double)avg).Should().Be(3.5);
            ((double)sum).Should().Be(6.0);
            var (values, comm1, comm2) = np.intersect1d(np.array(new long[] { 1, 3 }), np.array(new long[] { 3 }), return_indices: true);
            values.ToArray<long>().Should().Equal(3L);
            comm1.ToArray<long>().Should().Equal(1L);
            comm2.ToArray<long>().Should().Equal(0L);
            var (i, j) = np.indices((2, 3), sparse: true);
            i.shape.Should().Equal(2L, 1L);
            j.shape.Should().Equal(1L, 3L);
        }

        /// <summary>A sparse result has no single-array form: converting it to <see cref="NDArray"/> or reading a dense
        ///     member raises; its arrays are read by index, deconstruction or the <see cref="NDArray"/><c>[]</c>
        ///     conversion.</summary>
        [TestMethod]
        public void Indices_Sparse_HasNoArrayForm()
        {
            var sparse = np.indices((2, 3), sparse: true);
            sparse.sparse.Should().BeTrue();
            sparse.values.Should().BeNull();
            Action toArray = () => { NDArray _ = sparse; };
            toArray.Should().Throw<InvalidCastException>();
            Action shape = () => _ = sparse.shape;
            shape.Should().Throw<InvalidCastException>();
            ((NDArray[])sparse).Select(a => a.ndim).Should().Equal(2, 2);
        }
    }
}
