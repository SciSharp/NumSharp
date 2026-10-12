using System;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Matrixlib
{
    /// <summary>
    ///     Unit tests of <see cref="NDMatrix"/> (NumPy's <c>numpy.matrix</c>). The first half ports NumPy 2.4.2's
    ///     own <c>numpy/matrixlib/tests/test_defmatrix.py</c>; the second half pins what is specific to NumSharp's
    ///     design: members and operators follow the STATIC type while the <c>np.*</c> functions follow the object
    ///     (<c>NDMatrixBoundaryTests</c> holds the function rules), a C# number is a 0-d matrix scalar, and every
    ///     member result has NumPy's class, shape and ownership. Every expected value is NumPy 2.4.2's answer for the
    ///     same call. The differential gate with 26,000 cases is <c>FuzzCorpusTests.Matrix</c>.
    /// </summary>
    [TestClass]
    public class NDMatrixTests
    {
        /// <summary>The 2 x 2 integer array NumPy's tests build most matrices from.</summary>
        /// <returns>A new array <c>[[1, 2], [3, 4]]</c> (int64).</returns>
        private static NDArray A22() => np.array(new long[,] { { 1, 2 }, { 3, 4 } });

        /// <summary>The 2 x 2 float array of NumPy's algebra tests.</summary>
        /// <returns>A new array <c>[[1., 2.], [3., 4.]]</c>.</returns>
        private static NDArray F22() => np.array(new double[,] { { 1, 2 }, { 3, 4 } });

        /// <summary>Whether a result is a matrix OBJECT. (The array assertions' BeOfType tests the dtype.)</summary>
        /// <param name="result">The result.</param>
        /// <returns><c>true</c> for an <see cref="NDMatrix"/> instance.</returns>
        private static bool IsMatrix(object result) => result is NDMatrix;

        // ================================================================================================
        // numpy/matrixlib/tests/test_defmatrix.py
        // ================================================================================================

        /// <summary>TestCtor.test_basic: a matrix of an array holds the array's values, bmat tiles blocks, and a
        ///     vector becomes a row.</summary>
        [TestMethod]
        public void Ctor_Basic()
        {
            NDArray a = A22();
            NDMatrix mA = np.matrix(a);
            np.array_equal(mA.A, a).Should().BeTrue();

            NDMatrix c = np.bmat(new[] { new[] { a, a }, new[] { a, a } });
            c.Should().BeOfValues(1L, 2L, 1L, 2L, 3L, 4L, 3L, 4L, 1L, 2L, 1L, 2L, 3L, 4L, 3L, 4L);
            c.shape.Should().Equal(4L, 4L);

            NDArray e = np.array(new long[,] { { 5, 6 }, { 7, 8 } });
            NDMatrix ae = np.bmat(new[] { a, e });
            ae.Should().BeOfValues(1L, 2L, 5L, 6L, 3L, 4L, 7L, 8L);

            NDMatrix mvec = np.matrix(np.arange(5));
            mvec.shape.Should().Equal(1L, 5L);
        }

        /// <summary>TestCtor.test_exceptions: a string that is not a matrix literal raises ValueError.</summary>
        [TestMethod]
        public void Ctor_InvalidString_Raises()
        {
            Action act = () => np.matrix("invalid");
            act.Should().Throw<ValueError>();
        }

        /// <summary>TestProperties.test_sum: the axis sums keep their orientation; no axis gives a scalar.</summary>
        [TestMethod]
        public void Sum_KeepsOrientation()
        {
            NDMatrix m = np.matrix(np.array(new long[,] { { 1, 2, 0, 0 }, { 3, 4, 0, 0 }, { 1, 2, 1, 2 }, { 3, 4, 3, 4 } }));
            NDMatrix sum0 = m.sum(0);
            sum0.shape.Should().Equal(1L, 4L);
            sum0.Should().BeOfValues(8L, 12L, 4L, 6L);

            NDMatrix sum1 = m.sum(1);
            sum1.shape.Should().Equal(4L, 1L);
            sum1.Should().BeOfValues(3L, 7L, 6L, 14L);

            NDArray all = m.sum();
            all.ndim.Should().Be(0);
            ((long)all).Should().Be(30L);
        }

        /// <summary>TestProperties.test_prod / test_max / test_min.</summary>
        [TestMethod]
        public void Prod_Max_Min()
        {
            NDMatrix x = np.matrix(np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }));
            ((long)x.prod()).Should().Be(720L);
            x.prod(0).Should().BeOfValues(4L, 10L, 18L);
            x.prod(0).shape.Should().Equal(1L, 3L);
            x.prod(1).Should().BeOfValues(6L, 120L);
            x.prod(1).shape.Should().Equal(2L, 1L);

            ((long)x.max()).Should().Be(6L);
            x.max(0).Should().BeOfValues(4L, 5L, 6L);
            x.max(1).shape.Should().Equal(2L, 1L);
            x.max(1).Should().BeOfValues(3L, 6L);

            ((long)x.min()).Should().Be(1L);
            x.min(0).Should().BeOfValues(1L, 2L, 3L);
            x.min(1).Should().BeOfValues(1L, 4L);

            ((long)np.matrix(np.array(new long[] { 0, 1, 3 })).prod()).Should().Be(0L);
        }

        /// <summary>TestProperties.test_ptp: the range along an axis is a row for axis 0 and a column for axis 1.</summary>
        [TestMethod]
        public void Ptp()
        {
            NDMatrix mx = np.asmatrix(np.arange(4).reshape(2, 2));
            ((long)mx.ptp()).Should().Be(3L);
            mx.ptp(0).shape.Should().Equal(1L, 2L);
            mx.ptp(0).Should().BeOfValues(2L, 2L);
            mx.ptp(1).shape.Should().Equal(2L, 1L);
            mx.ptp(1).Should().BeOfValues(1L, 1L);
        }

        /// <summary>TestProperties.test_var: the matrix variance equals the array variance.</summary>
        [TestMethod]
        public void Var_EqualsTheArrayVariance()
        {
            NDArray x = np.arange(9).reshape(3, 3);
            NDMatrix mx = np.asmatrix(x);
            ((double)mx.var(ddof: 0)).Should().Be((double)np.var(x, ddof: 0));
            ((double)mx.var(ddof: 1)).Should().Be((double)np.var(x, ddof: 1));
        }

        /// <summary>TestProperties.test_basic: T, H, A and I of a real and a complex matrix.</summary>
        [TestMethod]
        public void Transpose_Hermitian_Inverse()
        {
            NDArray a = F22();
            NDMatrix mA = np.matrix(a);
            np.allclose(np.linalg.inv(a), mA.I).Should().BeTrue();
            np.array_equal(np.transpose(a), mA.T.A).Should().BeTrue();
            np.array_equal(np.transpose(a), mA.H.A).Should().BeTrue();
            np.array_equal(a, mA.A).Should().BeTrue();

            NDArray b = a.astype(np.complex128) * new Complex(1, 2);
            NDMatrix mB = np.matrix(b);
            np.allclose(np.linalg.inv(b), mB.I).Should().BeTrue();
            np.array_equal(np.transpose(b), mB.T.A).Should().BeTrue();
            np.array_equal(np.conjugate(np.transpose(b)), mB.H.A).Should().BeTrue();
        }

        /// <summary>TestProperties.test_comparisons: the comparisons are elementwise and return bool matrices.</summary>
        [TestMethod]
        public void Comparisons()
        {
            NDArray a = np.arange(100).reshape(10, 10);
            NDMatrix mA = np.matrix(a);
            NDMatrix mB = np.matrix(a) + 0.1;

            NDMatrix eq = mB == np.matrix(a + 0.1);
            eq.dtype.Should().Be(np.@bool);
            eq.shape.Should().Equal(10L, 10L);
            ((bool)np.all(eq)).Should().BeTrue();
            ((bool)np.any(mB == np.matrix(a - 0.1))).Should().BeFalse();
            ((bool)np.all(mA < mB)).Should().BeTrue();
            ((bool)np.all(mA <= mB)).Should().BeTrue();
            ((bool)np.all(mA <= mA)).Should().BeTrue();
            ((bool)np.any(mA < mA)).Should().BeFalse();
            ((bool)np.any(mB < mA)).Should().BeFalse();
            ((bool)np.all(mB >= mA)).Should().BeTrue();
            ((bool)np.any(mB > mB)).Should().BeFalse();
            ((bool)np.all(mA == mA)).Should().BeTrue();
            ((bool)np.any(mA == mB)).Should().BeFalse();
            ((bool)np.all(mB != mA)).Should().BeTrue();
        }

        /// <summary>TestProperties.test_asmatrix: asmatrix is a view; a write to the array shows in the matrix.</summary>
        [TestMethod]
        public void AsMatrix_IsAView()
        {
            NDArray a = np.arange(100).reshape(10, 10);
            NDMatrix mA = np.asmatrix(a);
            a.SetInt64(-10, 0, 0);
            ((long)mA[0, 0]).Should().Be(-10L);
            mA.flags.owndata.Should().BeFalse();
        }

        /// <summary>TestProperties.test_noaxis: reductions without an axis are scalars.</summary>
        [TestMethod]
        public void NoAxis_GivesScalars()
        {
            NDMatrix a = np.matrix(np.array(new long[,] { { 1, 0 }, { 0, 1 } }));
            ((long)a.sum()).Should().Be(2L);
            ((double)a.mean()).Should().Be(0.5);
            a.mean().ndim.Should().Be(0);
        }

        /// <summary>TestProperties.test_repr: the repr names the class and indents the continuation rows.</summary>
        [TestMethod]
        public void Repr()
        {
            NDMatrix a = np.matrix(np.array(new long[,] { { 1, 0 }, { 0, 1 } }));
            a.ToString(true).Should().Be("matrix([[1, 0],\n        [0, 1]])");
            a.ToString(false).Should().Be("[[1 0]\n [0 1]]");
        }

        /// <summary>TestProperties.test_make_bool_matrix_from_str: a literal of bools is a bool column.</summary>
        [TestMethod]
        public void Literal_OfBools()
        {
            NDMatrix a = np.matrix("True; True; False");
            a.dtype.Should().Be(np.@bool);
            a.shape.Should().Equal(3L, 1L);
            a.Should().BeOfValues(true, true, false);
        }

        /// <summary>TestCasting.test_basic: adding a float array gives float64, multiplying by a complex one gives
        ///     complex128 (the matrix product).</summary>
        [TestMethod]
        public void Casting()
        {
            NDArray a = np.arange(100).reshape(10, 10);
            NDMatrix mA = np.matrix(a);

            NDMatrix mB = mA.copy() + np.ones(new Shape(10, 10), np.float64) * 0.1;
            mB.dtype.Should().Be(np.float64);
            ((bool)np.all(mA != mB)).Should().BeTrue();
            ((bool)np.all(mB == mA + 0.1)).Should().BeTrue();

            NDMatrix mC = mA.copy() * np.ones(new Shape(10, 10), np.complex128);
            mC.dtype.Should().Be(np.complex128);
        }

        /// <summary>TestAlgebra.test_basic: positive and negative powers, the product, the sum, scaling, and the
        ///     in-place product.</summary>
        [TestMethod]
        public void Algebra()
        {
            NDArray a = F22();
            NDMatrix mA = np.matrix(a);

            NDArray b = np.identity(2);
            for (int i = 0; i < 6; i++)
            {
                np.allclose(mA.__pow__(i).A, b).Should().BeTrue($"m ** {i}");
                b = np.dot(b, a);
            }

            NDArray inverse = np.linalg.inv(a);
            b = np.identity(2);
            for (int i = 0; i < 6; i++)
            {
                np.allclose(mA.__pow__(-i).A, b).Should().BeTrue($"m ** -{i}");
                b = np.dot(b, inverse);
            }

            np.allclose((mA * mA).A, np.dot(a, a)).Should().BeTrue();
            np.allclose((mA + mA).A, a + a).Should().BeTrue();
            np.allclose((3 * mA).A, a * 3).Should().BeTrue();

            NDMatrix mA2 = np.matrix(a);
            mA2.__imul__(3).Should().BeSameAs(mA2);
            np.allclose(mA2.A, a * 3).Should().BeTrue();
        }

        /// <summary>TestAlgebra.test_pow: the in-place power stores into the matrix.</summary>
        [TestMethod]
        public void Pow_InPlace()
        {
            NDMatrix m = np.matrix("1. 2.; 3. 4.");
            NDMatrix m2 = m.copy();
            m2.__ipow__(2);
            NDMatrix mi = m.copy();
            mi.__ipow__(-1);
            NDMatrix m4 = m2.copy();
            m4.__ipow__(2);

            np.allclose(m2.A, m.__pow__(2).A).Should().BeTrue();
            np.allclose(m4.A, np.dot(m2, m2)).Should().BeTrue();
            np.allclose(np.dot(mi, m), np.eye(2)).Should().BeTrue();
        }

        /// <summary>TestAlgebra.test_scalar_type_pow: an integer scalar of any width is a power.</summary>
        [TestMethod]
        public void Pow_ScalarTypes()
        {
            NDMatrix m = np.matrix(A22());
            NDMatrix squared = m.__pow__(2);
            np.array_equal(squared.A, m.__pow__((sbyte)2).A).Should().BeTrue();
            np.array_equal(squared.A, m.__pow__((byte)2).A).Should().BeTrue();
            np.array_equal(squared.A, m.__pow__(NDArray.Scalar(2L)).A).Should().BeTrue();
        }

        /// <summary>TestAlgebra.test_notimplemented: a number to the power of a matrix is a TypeError, and so is a
        ///     float exponent.</summary>
        [TestMethod]
        public void NotImplemented_Operations_Raise()
        {
            NDMatrix a = np.matrix(F22());
            Action rpow = () => a.__rpow__(1.0);
            rpow.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for ** or pow(): 'float' and 'matrix'");

            Action fractional = () => a.__pow__(2.0);
            fractional.Should().Throw<TypeError>().WithMessage("exponent must be an integer");
        }

        /// <summary>TestMatrixReturn.test_instance_methods: the methods a matrix inherits return matrices.</summary>
        [TestMethod]
        public void InheritedMethods_ReturnMatrices()
        {
            NDMatrix a = np.matrix(np.array(new double[] { 1.0 }));
            IsMatrix(a.astype(np.int32)).Should().BeTrue();
            IsMatrix(a.clip(0.0, 1.0)).Should().BeTrue();
            IsMatrix(a.compress(np.array(new[] { true }))).Should().BeTrue();
            IsMatrix(a.repeat(1)).Should().BeTrue();
            IsMatrix(a.reshape(1)).Should().BeTrue();
            IsMatrix(a.swapaxes(0, 0)).Should().BeTrue();
            IsMatrix(a.dot(np.array(new double[] { 1.0 }))).Should().BeTrue();
            IsMatrix(a.copy()).Should().BeTrue();
            IsMatrix(a.conj()).Should().BeTrue();
            IsMatrix(a.cumsum()).Should().BeTrue();
            IsMatrix(a.diagonal()).Should().BeTrue();
            IsMatrix(a.flatten()).Should().BeTrue();
            IsMatrix(a.ravel()).Should().BeTrue();
            IsMatrix(a.round()).Should().BeTrue();
            IsMatrix(a.squeeze()).Should().BeTrue();
            IsMatrix(a.transpose()).Should().BeTrue();
            IsMatrix(a.trace()).Should().BeTrue();
            IsMatrix(a.real).Should().BeTrue();
            IsMatrix(a.imag).Should().BeTrue();

            // nonzero is not re-declared: its index arrays are plain arrays, as in NumPy.
            NDArray[] nz = np.matrix(np.array(new double[] { 0.0 })).nonzero();
            nz.Length.Should().Be(2);
            IsMatrix(nz[0]).Should().BeFalse();
        }

        /// <summary>TestIndexing.test_basic: a column is assigned from a (3, 1) array.</summary>
        [TestMethod]
        public void Indexing_AssignAColumn()
        {
            NDMatrix x = np.asmatrix(np.zeros(new Shape(3, 2), np.float64));
            NDArray y = np.array(new double[,] { { 0.8 }, { 0.2 }, { 0.3 } });
            x.__setitem__(new object[] { Slice.All, 1 }, y > 0.5);
            x.Should().BeOfValues(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        }

        /// <summary>TestNewScalarIndexing: a row keeps two axes, index arrays give matrices, and an element is a
        ///     scalar.</summary>
        [TestMethod]
        public void ScalarAndFancyIndexing()
        {
            NDMatrix a = np.matrix(A22());
            a[0].ndim.Should().Be(2);

            NDMatrix x = a[1, new[] { 0, 1, 0 }];
            x.shape.Should().Equal(1L, 3L);
            x.Should().BeOfValues(3L, 4L, 3L);

            x = a[new object[] { new[] { 1, 0 } }];
            x.shape.Should().Equal(2L, 2L);
            x.Should().BeOfValues(3L, 4L, 1L, 2L);

            x = a[np.array(new long[,] { { 1 }, { 0 } }), np.array(new long[,] { { 1, 0 }, { 0, 1 } })];
            x.shape.Should().Equal(2L, 2L);
            x.Should().BeOfValues(4L, 3L, 1L, 2L);
        }

        /// <summary>TestNewScalarIndexing.test_matrix_element / test_scalar_indexing: indexing a row again gives
        ///     the row; a column is (N, 1); m[i, j] is a scalar.</summary>
        [TestMethod]
        public void MatrixElement()
        {
            NDMatrix x = np.matrix(np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }));
            x[0][0].shape.Should().Equal(1L, 3L);
            x[0][0].Should().BeOfValues(1L, 2L, 3L);
            x[0].shape.Should().Equal(1L, 3L);
            x[":, 0"].shape.Should().Equal(2L, 1L);

            NDMatrix zero = np.matrix(0L);
            zero.shape.Should().Equal(1L, 1L);
            zero[0, 0].ndim.Should().Be(0);
            ((long)zero[0, 0]).Should().Be(0L);
            zero[":, 0"].shape.Should().Equal(1L, 1L);

            NDMatrix z = np.asmatrix(np.zeros(new Shape(3, 2), np.float64));
            ((double)z[0, 0]).Should().Be((double)z[0][0, 0]);
        }

        /// <summary>TestNewScalarIndexing.test_row_column_indexing / test_boolean_indexing / test_list_indexing.</summary>
        [TestMethod]
        public void RowColumnBooleanAndListIndexing()
        {
            NDMatrix x = np.asmatrix(np.eye(2));
            x["0, :"].shape.Should().Equal(1L, 2L);
            x["0, :"].Should().BeOfValues(1.0, 0.0);
            x["1, :"].Should().BeOfValues(0.0, 1.0);
            x[":, 0"].shape.Should().Equal(2L, 1L);
            x[":, 0"].Should().BeOfValues(1.0, 0.0);
            x[":, 1"].Should().BeOfValues(0.0, 1.0);

            NDMatrix m = np.asmatrix(np.arange(6).reshape(3, 2));
            NDMatrix byMask = m[Slice.All, np.array(new[] { true, false })];
            np.array_equal(byMask.A, m[":, 0"].A).Should().BeTrue();
            NDMatrix rowMask = m[np.array(new[] { true, false, false }), Slice.All];
            np.array_equal(rowMask.A, m["0, :"].A).Should().BeTrue();

            np.array_equal(m[Slice.All, new[] { 1, 0 }].A, m[":, ::-1"].A).Should().BeTrue();
            np.array_equal(m[new[] { 2, 1, 0 }, Slice.All].A, m["::-1, :"].A).Should().BeTrue();
        }

        /// <summary>TestShape: ravel and flatten of a matrix are rows; ravel shares memory, flatten does not.</summary>
        [TestMethod]
        public void RavelAndFlatten()
        {
            NDMatrix m = np.matrix(np.array(new long[,] { { 1 }, { 2 } }));
            m.shape.Should().Equal(2L, 1L);
            m.ravel().shape.Should().Equal(1L, 2L);
            m.flatten().shape.Should().Equal(1L, 2L);
            np.ravel(m).shape.Should().Equal(2L);

            NDMatrix x = np.matrix(np.array(new long[,] { { 1, 2, 3 }, { 4, 5, 6 } }));
            x.ravel().Should().BeOfValues(1L, 2L, 3L, 4L, 5L, 6L);
            x.ravel('F').Should().BeOfValues(1L, 4L, 2L, 5L, 3L, 6L);
            x.T.ravel().Should().BeOfValues(1L, 4L, 2L, 5L, 3L, 6L);
            x.T.ravel('A').Should().BeOfValues(1L, 2L, 3L, 4L, 5L, 6L);

            np.may_share_memory(m, m.ravel()).Should().BeTrue();
            np.may_share_memory(m, m.flatten()).Should().BeFalse();
        }

        // ================================================================================================
        // NumSharp's design: the matrix semantics follow the static type
        // ================================================================================================

        /// <summary>
        ///     The operators and the members follow the STATIC type: the same object is a matrix through an
        ///     <see cref="NDMatrix"/> reference and a plain 2-D array through an <see cref="NDArray"/> one. The
        ///     public <c>np.*</c> functions follow the OBJECT: they return what NumPy returns for a matrix
        ///     operand, typed <see cref="NDArray"/> (<see cref="NDMatrixBoundary"/>).
        /// </summary>
        [TestMethod]
        public void Semantics_OperatorsAndMembersFollowTheStaticType_FunctionsFollowTheObject()
        {
            NDMatrix m = np.matrix(A22());
            NDArray a = m;

            (m * m).Should().BeOfValues(7L, 10L, 15L, 22L);
            (a * a).Should().BeOfValues(1L, 4L, 9L, 16L);
            ReferenceEquals(m, a).Should().BeTrue();

            // An operator and a member of the NDArray reference keep the plain-array meaning.
            IsMatrix(a + a).Should().BeFalse();
            NDArray member = a.sum(0);
            IsMatrix(member).Should().BeFalse();
            member.shape.Should().Equal(2L);

            // The function looks at the object: NumPy's np.sum(m, 0) is m.sum(0), a 1 x 2 matrix.
            NDArray reduced = np.sum(a, 0);
            IsMatrix(reduced).Should().BeTrue();
            reduced.shape.Should().Equal(1L, 2L);
            reduced.Should().BeOfValues(4L, 6L);

            // The upcast result is the matrix object: a cast gives the matrix members back without a copy.
            NDMatrix back = (NDMatrix)reduced;
            back.T.shape.Should().Equal(2L, 1L);

            // A is the plain array of the same memory.
            IsMatrix(m.A).Should().BeFalse();
            m.A1.shape.Should().Equal(4L);
        }

        /// <summary>A C# number is a 0-d matrix: a scalar that scales, in np.dot's dtype (the two operand dtypes
        ///     promoted).</summary>
        [TestMethod]
        public void ANumberIsAZeroDScalar()
        {
            NDMatrix two = 2L;
            two.ndim.Should().Be(0);

            NDMatrix m = np.matrix(np.array(new sbyte[,] { { 1, 2 }, { 3, 4 } }));
            NDMatrix scaled = m * 2L;
            scaled.dtype.Should().Be(np.int64);
            scaled.Should().BeOfValues(2L, 4L, 6L, 8L);
            (2L * m).Should().BeOfValues(2L, 4L, 6L, 8L);

            NDMatrix halves = m * 0.5;
            halves.dtype.Should().Be(np.float64);
            halves.Should().BeOfValues(0.5, 1.0, 1.5, 2.0);
        }

        /// <summary>
        ///     <c>np.dot</c>'s scalar route for a float result: each element is <c>0 + scalar * x</c>, so a product
        ///     of -0.0 is +0.0; a result of ONE element is the direct product and keeps -0.0. The result is a new
        ///     C-contiguous matrix that owns its data.
        /// </summary>
        [TestMethod]
        public void Scaling_FollowsNpDot()
        {
            NDMatrix zeros = np.matrix(np.zeros(new Shape(2, 2), np.float64));
            NDMatrix scaled = zeros * -0.5;
            scaled.flags.owndata.Should().BeTrue();
            for (int i = 0; i < 4; i++)
                double.IsNegative(scaled.GetAtIndex<double>(i)).Should().BeFalse("np.dot accumulates into zeros");

            NDMatrix one = np.matrix(np.zeros(new Shape(1, 1), np.float64));
            double.IsNegative((one * -0.5).GetAtIndex<double>(0)).Should().BeTrue("a one-element result is the direct product");

            NDMatrix transposed = np.matrix(np.arange(6.0).reshape(3, 2)).T;
            transposed.flags.c_contiguous.Should().BeFalse();
            NDMatrix doubled = transposed * 2.0;
            doubled.flags.c_contiguous.Should().BeTrue();
            doubled.Should().BeOfValues(0.0, 4.0, 8.0, 2.0, 6.0, 10.0);

            // An integer result is the multiply ufunc: a view of the hidden result.
            (np.matrix(A22()) * 2L).flags.owndata.Should().BeFalse();
        }

        /// <summary>
        ///     A float scaling has three routes (the direct pass over a C-contiguous operand, the fused pass, the
        ///     multiply ufunc plus an in-place addition of zero). At a size past every threshold (70 x 70) each
        ///     must give the bits of the ufunc reference <c>np.add(np.multiply(x, s, dtype: loop), 0)</c>, for
        ///     float64, float32 and complex128, a C-contiguous, a transposed and a stepped operand, special values
        ///     included; the counters prove which route production takes.
        /// </summary>
        [TestMethod]
        public void Scaling_EveryRoute_GivesTheUfuncBits()
        {
            const int side = 70;
            var values = new double[side * side];
            double[] specials = { -0.0, 0.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.Epsilon, -1e-310, 1e300 };
            for (int i = 0; i < values.Length; i++)
                values[i] = i % 7 == 3 ? specials[(i / 7) % specials.Length] : (i % 13 - 6) * 0.37 + i * 1e-3;

            NDArray f64 = np.array(values).reshape(side, side);
            NDArray c128 = f64.astype(np.complex128) * NDArray.Scalar(new Complex(1, -0.5));
            var operands = new (NDArray basis, NDMatrix scalar, NDArray plain, NDArray zero, DType loop)[]
            {
                (f64, -2.5, NDArray.Scalar(-2.5), NDArray.Scalar(0.0), np.float64),
                (f64.astype(np.float32), -2.5f, NDArray.Scalar(-2.5f), NDArray.Scalar(0f), np.float32),
                (c128, -2.5, NDArray.Scalar(-2.5), NDArray.Scalar(Complex.Zero), np.complex128),
            };

            try
            {
                foreach (var (basis, scalar, plain, zero, loop) in operands)
                foreach (string layout in new[] { "C", "T", "reversed", "stepped" })
                {
                    // "stepped" has 2,450 elements: under the float threshold of the fused pass (4,096), over the
                    // complex one (512). The other three have 4,900.
                    NDArray view = layout switch
                    {
                        "C" => basis,
                        "T" => basis.T,
                        "reversed" => basis["::-1, :"],
                        _ => basis["::-1, ::2"],
                    };
                    NDMatrix m = np.asmatrix(view);
                    NDArray expected = np.ascontiguousarray(np.add(np.multiply(view, plain, dtype: loop), zero));
                    bool complex = loop == np.complex128;

                    foreach (string route in new[] { "production", "fused", "two-pass" })
                    foreach (bool scalarFirst in new[] { false, true })
                    {
                        NDMatrix.DisableDirectScale = route != "production";
                        NDMatrix.FusedScaleMinElementsTestOverride = route == "fused" ? 2 : route == "two-pass" ? long.MaxValue : 0;
                        int direct = NDMatrix.DirectScaleRuns, fused = NDMatrix.FusedScaleRuns;
                        NDMatrix got = scalarFirst ? scalar * m : m * scalar;
                        string where = $"{loop} {layout} {route} scalarFirst={scalarFirst}";

                        // Which route ran: production takes the direct pass for a C-contiguous float operand,
                        // the fused pass for a float view of 4,096 elements or more and a complex operand of
                        // 512 or more, and the two ufunc passes for a smaller one.
                        bool expectDirect = route == "production" && !complex && layout == "C";
                        bool expectFused = route == "fused"
                                           || (route == "production" && !expectDirect && view.size >= (complex ? 512 : 4096));
                        (NDMatrix.DirectScaleRuns - direct).Should().Be(expectDirect ? 1 : 0, where);
                        (NDMatrix.FusedScaleRuns - fused).Should().Be(expectFused ? 1 : 0, where);

                        got.dtype.Should().Be(loop);
                        got.shape.Should().Equal(view.shape);
                        got.flags.c_contiguous.Should().BeTrue(where);
                        got.flags.owndata.Should().BeTrue(where);
                        AssertSameBits(got, expected, where);
                    }
                }
            }
            finally
            {
                NDMatrix.DisableDirectScale = false;
                NDMatrix.FusedScaleMinElementsTestOverride = 0;
            }
        }

        /// <summary>Asserts that two arrays of one float dtype and shape hold the same bit patterns, element by
        ///     element in logical order (so a zero's sign and a NaN's payload count).</summary>
        /// <param name="actual">The array under test: float32, float64 or complex128.</param>
        /// <param name="expected">The reference, of the same dtype and size.</param>
        /// <param name="where">The case description for a failure message.</param>
        private static void AssertSameBits(NDArray actual, NDArray expected, string where)
        {
            actual.size.Should().Be(expected.size, where);
            for (long i = 0; i < expected.size; i++)
            {
                bool same;
                if (expected.typecode == NPTypeCode.Single)
                    same = BitConverter.SingleToInt32Bits(actual.GetAtIndex<float>(i)) == BitConverter.SingleToInt32Bits(expected.GetAtIndex<float>(i));
                else if (expected.typecode == NPTypeCode.Double)
                    same = BitConverter.DoubleToInt64Bits(actual.GetAtIndex<double>(i)) == BitConverter.DoubleToInt64Bits(expected.GetAtIndex<double>(i));
                else
                {
                    Complex a = actual.GetAtIndex<Complex>(i), e = expected.GetAtIndex<Complex>(i);
                    same = BitConverter.DoubleToInt64Bits(a.Real) == BitConverter.DoubleToInt64Bits(e.Real)
                           && BitConverter.DoubleToInt64Bits(a.Imaginary) == BitConverter.DoubleToInt64Bits(e.Imaginary);
                }

                if (!same)
                    Assert.Fail($"{where}: element {i} differs in its bits");
            }
        }

        /// <summary>
        ///     An explicit 0-d ARRAY (<c>np.array(2)</c>, <c>np.asarray(2)</c>) is an ndarray to
        ///     <c>matrix.__mul__</c>, not a scalar: on the right it is the product with a 1 x 1 matrix, which raises
        ///     unless the matrix has one column; on the left it is <c>np.dot</c>'s scalar route. Every other 0-d
        ///     value (a C# number, <c>NDArray.Scalar</c>, a reduction result, an element, a norm) still scales.
        ///     NumPy 2.4.2: <c>np.matrix(np.arange(6.).reshape(2, 3)) * np.array(2.0)</c> raises
        ///     <c>ValueError: shapes (2,3) and (1,1) not aligned: 3 (dim 1) != 1 (dim 0)</c>.
        /// </summary>
        [TestMethod]
        public void ExplicitZeroDimArray_IsAnArrayOperand_NotAScalar()
        {
            NDArray a = np.arange(6.0).reshape(2, 3);
            NDMatrix m = np.matrix(a);
            const string notAligned = "shapes (2,3) and (1,1) not aligned: 3 (dim 1) != 1 (dim 0)";

            Action viaArray = () => { var _ = m * np.array(2.0); };
            viaArray.Should().Throw<Exception>().WithMessage(notAligned);
            Action viaAsarray = () => { var _ = m * np.asarray(2.0); };
            viaAsarray.Should().Throw<Exception>().WithMessage(notAligned);
            // np.array(nd) of an explicit 0-d array is an explicit 0-d array again.
            Action viaCopy = () => { var _ = m * np.array(np.array(2.0)); };
            viaCopy.Should().Throw<Exception>().WithMessage(notAligned);
            Action viaObject = () => { var _ = m * (object)np.array(2.0); };
            viaObject.Should().Throw<Exception>().WithMessage(notAligned);

            // One column: the product with [[2]] exists.
            NDMatrix column = np.matrix(np.array(new double[,] { { 1 }, { 2 }, { 3 } }));
            NDMatrix product = column * np.array(2.0);
            product.shape.Should().Equal(3L, 1L);
            product.Should().BeOfValues(2.0, 4.0, 6.0);

            // On the left: N.dot(a0, m), which accumulates into zeros. A product of -0.0 comes out +0.0, and the
            // result is a new owner (a NumPy scalar on the left gives the elementwise view, see the next test).
            NDMatrix zeros = np.matrix(np.zeros(new Shape(2, 2), np.float64));
            NDMatrix left = np.array(-0.5) * zeros;
            left.flags.owndata.Should().BeTrue();
            for (int i = 0; i < 4; i++)
                double.IsNegative(left.GetAtIndex<double>(i)).Should().BeFalse();

            // Every other 0-d value scales, as before.
            NDMatrix[] scaled =
            {
                m * 2.0,
                m * NDArray.Scalar(2.0),
                m * np.sum(np.array(new[] { 0.5, 1.5 })),                                   // a reduction result
                m * a[0, 2],                                                                // an element (2.0)
                m * np.linalg.norm(np.array(new[] { 2.0 })),                                // a norm
                m * np.dot(np.array(new[] { 1.0, 1.0 }), np.array(new[] { 1.0, 1.0 })),     // a vector dot
            };
            foreach (NDMatrix result in scaled)
            {
                result.shape.Should().Equal(2L, 3L);
                result.Should().BeOfValues(0.0, 2.0, 4.0, 6.0, 8.0, 10.0);
            }

            // The mark belongs to the object np.array returned: a copy made another way is a plain 0-d value.
            np.array(2.0).IsExplicitZeroDimArray.Should().BeTrue();
            np.asarray(2.0).IsExplicitZeroDimArray.Should().BeTrue();
            np.array(2.0).copy().IsExplicitZeroDimArray.Should().BeFalse();
            NDArray.Scalar(2.0).IsExplicitZeroDimArray.Should().BeFalse();
            np.array(new[] { 2.0 }).IsExplicitZeroDimArray.Should().BeFalse("the mark is for 0-d arrays only");
        }

        /// <summary>A plain 0-d array on the LEFT is a NumPy scalar: the elementwise product, which keeps the sign
        ///     of a zero and is wrapped as a view.</summary>
        [TestMethod]
        public void NumPyScalarOnTheLeft_IsElementwise()
        {
            NDMatrix zeros = np.matrix(np.zeros(new Shape(2, 2), np.float64));
            NDMatrix product = NDArray.Scalar(-0.5) * zeros;
            product.flags.owndata.Should().BeFalse();
            for (int i = 0; i < 4; i++)
                double.IsNegative(product.GetAtIndex<double>(i)).Should().BeTrue();
        }

        /// <summary>The product of a matrix with a 1-d array sees the array as a ROW first; with the array on the
        ///     left it is np.dot as it stands.</summary>
        [TestMethod]
        public void Product_WithAVector()
        {
            NDMatrix m = np.matrix(np.arange(6).reshape(3, 2));
            NDArray v2 = np.array(new long[] { 1, 2 });
            Action rowTimes = () => { var _ = m * v2; };
            rowTimes.Should().Throw<Exception>().WithMessage("shapes (3,2) and (1,2) not aligned: 2 (dim 1) != 1 (dim 0)");

            NDArray v3 = np.array(new long[] { 1, 2, 3 });
            NDMatrix left = v3 * m;
            left.shape.Should().Equal(1L, 2L);
            left.Should().BeOfValues(16L, 22L);
            left.flags.owndata.Should().BeTrue();

            NDMatrix viaDot = m.dot(v2);
            viaDot.shape.Should().Equal(1L, 3L);
            viaDot.Should().BeOfValues(2L, 8L, 14L);
            viaDot.flags.owndata.Should().BeTrue();
        }

        /// <summary>Ownership follows NumPy: a view for asmatrix, T, a row and every ufunc result; an owner for
        ///     copy, astype and the matrix product.</summary>
        [TestMethod]
        public void Ownership()
        {
            NDArray a = F22();
            np.asmatrix(a).flags.owndata.Should().BeFalse();
            NDMatrix m = np.matrix(a);
            m.flags.owndata.Should().BeTrue();
            m.T.flags.owndata.Should().BeFalse();
            m[0].flags.owndata.Should().BeFalse();
            (m + m).flags.owndata.Should().BeFalse();
            (-m).flags.owndata.Should().BeFalse();
            (m * m).flags.owndata.Should().BeTrue();
            m.copy().flags.owndata.Should().BeTrue();
            m.astype(np.float32).flags.owndata.Should().BeTrue();
            m.sum(0).flags.owndata.Should().BeFalse();
            m[new object[] { new[] { 1, 0 } }].flags.owndata.Should().BeFalse();
            // A 0-d integer ARRAY index is NumPy's HAS_SCALAR_ARRAY: a copy of the selected row.
            m[new object[] { np.array(1L) }].flags.owndata.Should().BeTrue();
        }

        /// <summary>Every result is folded onto two axes; one that cannot be raises NumPy's text.</summary>
        [TestMethod]
        public void TheFinalizeRule()
        {
            np.matrix(np.arange(6)).shape.Should().Equal(1L, 6L);
            np.matrix(NDArray.Scalar(5.0)).shape.Should().Equal(1L, 1L);
            np.asmatrix(np.zeros(new Shape(1, 3, 1, 4, 1), np.float64)).shape.Should().Equal(3L, 4L);

            Action tooLarge = () => np.asmatrix(np.zeros(new Shape(2, 3, 4), np.float64));
            tooLarge.Should().Throw<ValueError>().WithMessage("shape too large to be a matrix.");

            NDMatrix m = np.matrix(np.arange(12).reshape(3, 4));
            m.reshape(12).shape.Should().Equal(1L, 12L);
            m.reshape(1, 2, 6).shape.Should().Equal(2L, 6L);
            m.take(np.array(5L)).shape.Should().Equal(1L, 1L);
            m.diagonal().shape.Should().Equal(1L, 3L);
            m.cumsum().shape.Should().Equal(1L, 12L);

            Action broadcast = () => { var _ = m + np.zeros(new Shape(2, 3, 4), np.int64); };
            broadcast.Should().Throw<ValueError>().WithMessage("shape too large to be a matrix.");
        }

        /// <summary>Index errors carry NumPy's texts, and the flat take names axis 1 of the raveled row.</summary>
        [TestMethod]
        public void IndexErrors()
        {
            NDMatrix m = np.matrix(np.arange(12).reshape(3, 4));
            Action row = () => { var _ = m[3]; };
            row.Should().Throw<IndexError>().WithMessage("index 3 is out of bounds for axis 0 with size 3");
            Action element = () => { var _ = m[0, 9]; };
            element.Should().Throw<IndexError>().WithMessage("index 9 is out of bounds for axis 1 with size 4");
            Action tooMany = () => { var _ = m["0, 0, 0"]; };
            tooMany.Should().Throw<IndexError>().WithMessage("too many indices for array: array is 2-dimensional, but 3 were indexed");
            Action ellipses = () => { var _ = m["..., ..."]; };
            ellipses.Should().Throw<IndexError>().WithMessage("an index can only have a single ellipsis ('...')");
            // np.take is PyArray_TakeFrom: NumPy's IndexError, type and text (NumPy 2.4.2: m.take(np.array([12])) raises
            // "IndexError: index 12 is out of bounds for axis 1 with size 12").
            Action take = () => m.take(np.array(new long[] { 12 }));
            take.Should().Throw<IndexError>().WithMessage("index 12 is out of bounds for axis 1 with size 12");
        }

        /// <summary>
        ///     Assignment: an index of integers and slices assigns into the row or COLUMN the matrix returns, so a
        ///     1-d value does not fit a column; one element refuses an array.
        /// </summary>
        [TestMethod]
        public void Assignment_ThroughTheMatrixView()
        {
            NDMatrix m = np.matrix(np.zeros(new Shape(3, 2), np.int64));
            m.__setitem__(new object[] { Slice.All, 0 }, np.array(new long[,] { { 7 }, { 8 }, { 9 } }));
            m.Should().BeOfValues(7L, 0L, 8L, 0L, 9L, 0L);

            Action flat = () => m.__setitem__(new object[] { Slice.All, 0 }, np.array(new long[] { 7, 8, 9 }));
            flat.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (3,) into shape (3,1)");

            m[1] = 5L;
            m.Should().BeOfValues(7L, 0L, 5L, 5L, 9L, 0L);
            m[0, 1] = 4L;
            ((long)m[0, 1]).Should().Be(4L);

            Action sequence = () => m[0, 0] = np.array(new long[] { 1 });
            sequence.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence.");

            NDMatrix readOnly = np.asmatrix(np.broadcast_to(np.arange(2), new Shape(3, 2)));
            Action write = () => readOnly[0] = 1L;
            write.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
        }

        /// <summary>
        ///     NumPy's value rule for an index-array assignment: a value with more axes than the selection loses
        ///     its leading axes, so a matrix ROW fits a 1-d selection. A value that does not fit raises NumPy's
        ///     text and writes nothing. (A fix of the array setter the matrix class needed.)
        /// </summary>
        [TestMethod]
        public void FancyAssignment_TakesAMatrixRow()
        {
            NDMatrix m = np.matrix(np.zeros(new Shape(3, 4), np.int64));
            NDMatrix row = np.matrix(np.array(new long[] { 5, 6 }));
            m[new[] { 0, 2 }, new[] { 1, 3 }] = row;
            m.Should().BeOfValues(0L, 5L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 6L);

            // The same rule on a plain array.
            NDArray a = np.zeros(new Shape(3, 4), np.int64);
            a[np.array(new long[] { 0, 2 }), np.array(new long[] { 1, 3 })] = np.array(new long[,,] { { { 7, 8 } } });
            a.Should().BeOfValues(0L, 7L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 8L);

            Action column = () => m[new[] { 0, 2 }, new[] { 1, 3 }] = np.matrix(np.array(new long[,] { { 5 }, { 6 } }));
            column.Should().Throw<ValueError>().WithMessage(
                "shape mismatch: value array of shape (2,1) could not be broadcast to indexing result of shape (2,)");
            m.Should().BeOfValues(0L, 5L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 6L);
        }

        /// <summary>A mask of the matrix's own shape takes a scalar or a 1-d value: a matrix value is two-dimensional
        ///     and NumPy refuses it.</summary>
        [TestMethod]
        public void MaskAssignment()
        {
            NDMatrix m = np.matrix(np.arange(12).reshape(3, 4) - 5);
            NDMatrix mask = m > 0;
            m[mask] = 7L;
            m.Should().BeOfValues(-5L, -4L, -3L, -2L, -1L, 0L, 7L, 7L, 7L, 7L, 7L, 7L);

            Action matrixValue = () => m[mask] = np.matrix(np.array(new long[] { 1, 2, 3, 4, 5, 6 }));
            matrixValue.Should().Throw<TypeError>().WithMessage(
                "NumPy boolean array indexing assignment requires a 0 or 1-dimensional input, input has 2 dimensions");

            m.__setitem__(mask, np.array(new long[] { 1, 2, 3, 4, 5, 6 }));
            m.Should().BeOfValues(-5L, -4L, -3L, -2L, -1L, 0L, 1L, 2L, 3L, 4L, 5L, 6L);
        }

        /// <summary>
        ///     A reduction with out= runs in <c>promote_types(out.dtype, operand dtype)</c> and stores with an
        ///     unsafe cast: an int8 sum into a bool out is the truth value of the int8 sum, into an int8 out it
        ///     wraps.
        /// </summary>
        [TestMethod]
        public void Reductions_WithOut()
        {
            NDMatrix m = np.matrix(np.array(new sbyte[,] { { -1, 0, 1, 2 }, { 3, -1, 0, 1 }, { 2, 3, -1, 0 } }));
            NDMatrix flags = np.matrix(np.zeros(new Shape(1, 4), np.@bool));
            m.sum(0, @out: flags).Should().BeSameAs(flags);
            flags.Should().BeOfValues(true, true, false, true);

            NDMatrix hundred = np.matrix(np.full(new Shape(3, 4), (sbyte)100));
            NDMatrix narrow = np.matrix(np.zeros(new Shape(1, 4), np.int8));
            hundred.sum(0, @out: narrow);
            narrow.Should().BeOfValues((sbyte)44, (sbyte)44, (sbyte)44, (sbyte)44);

            NDMatrix wide = np.matrix(np.zeros(new Shape(1, 4), np.int16));
            hundred.sum(0, @out: wide);
            wide.Should().BeOfValues((short)300, (short)300, (short)300, (short)300);

            // var hands out to its LAST sum only: float32 squares summed in float64.
            NDMatrix f4 = m.astype(np.float32);
            NDMatrix variance = np.matrix(np.zeros(new Shape(1, 4), np.float64));
            f4.var(0, @out: variance);
            variance.GetAtIndex<double>(0).Should().Be(2.888889064391454);
            variance.GetAtIndex<double>(2).Should().Be(0.6666666666666666);
        }

        /// <summary>A matrix reduction is always NumPy's ARRAY form: std with an integer dtype raises for
        ///     <c>axis: null</c> too, where np.std of the plain array returns a truncated scalar.</summary>
        [TestMethod]
        public void Std_WithAnIntegerDtype_Raises()
        {
            NDMatrix m = np.matrix(np.arange(12).reshape(3, 4));
            Action flat = () => m.std(dtype: np.int64);
            flat.Should().Throw<Exception>().WithMessage(
                "Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
            ((long)np.std((NDArray)m, dtype: np.int64)).Should().Be(3L);
            ((long)m.var(dtype: np.int64)).Should().Be(12L);
        }

        /// <summary>argmax / argmin / ptp accept axis 0 and 1 only: a negative axis is "unsupported axis".</summary>
        [TestMethod]
        public void ArgReductions_AlignTheirResult()
        {
            NDMatrix m = np.matrix(np.array(new long[,] { { 1, 9, 3 }, { 7, 2, 8 } }));
            m.argmax(0).shape.Should().Equal(1L, 3L);
            m.argmax(0).Should().BeOfValues(1L, 0L, 1L);
            m.argmax(1).shape.Should().Equal(2L, 1L);
            m.argmax(1).Should().BeOfValues(1L, 2L);
            m.argmin(1).Should().BeOfValues(0L, 1L);

            Action negative = () => m.argmax(-1);
            negative.Should().Throw<ValueError>().WithMessage("unsupported axis");
            m.sum(-1).shape.Should().Equal(2L, 1L);
        }

        /// <summary>The class of a rounded matrix follows NumPy's route: a matrix for 0 decimals and for an
        ///     integer matrix, a plain array for a float matrix with decimals.</summary>
        [TestMethod]
        public void Round_ResultClass()
        {
            NDMatrix f = np.matrix(np.array(new double[,] { { 1.25, 2.5 }, { 3.75, 4.5 } }));
            f.round().Should().BeOfValues(1.0, 2.0, 4.0, 4.0);
            f.round().flags.owndata.Should().BeFalse();
            IsMatrix(f.round(1)).Should().BeFalse();
            f.round(1).Should().BeOfValues(1.2, 2.5, 3.8, 4.5);

            NDMatrix i = np.matrix(A22());
            IsMatrix(i.round(1)).Should().BeTrue();
            i.round().flags.owndata.Should().BeTrue();
            IsMatrix(i.round(-1)).Should().BeFalse();
        }

        /// <summary>The constructor fills a sequence element by element with the dtype's own conversion: a Python
        ///     int out of range raises, a complex number is refused by a real dtype.</summary>
        [TestMethod]
        public void Ctor_Sequence_WithDtype()
        {
            NDMatrix m = np.matrix(new object[] { 1.5, 300L }, np.float32);
            m.dtype.Should().Be(np.float32);
            m.Should().BeOfValues(1.5f, 300f);

            Action overflow = () => np.matrix(new object[] { 1.5, 300L }, np.int8);
            overflow.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");

            Action complexIntoFloat = () => np.matrix("1+2j 3", np.float32);
            complexIntoFloat.Should().Throw<TypeError>().WithMessage("float() argument must be a string or a real number, not 'complex'");

            np.matrix("1+2j 3", np.@bool).Should().BeOfValues(true, true);
            np.matrix("1.5 2", np.int8).Should().BeOfValues((sbyte)1, (sbyte)2);
        }

        /// <summary>Matrix literals: rows by semicolons, columns by commas or spaces, Python number syntax.</summary>
        [TestMethod]
        public void Literals()
        {
            NDMatrix m = np.matrix("1 2; 3 4");
            m.dtype.Should().Be(np.int64);
            m.shape.Should().Equal(2L, 2L);
            m.Should().BeOfValues(1L, 2L, 3L, 4L);

            np.matrix("1, 2 3;4 5,6").shape.Should().Equal(2L, 3L);
            np.matrix("[1.5 2]").dtype.Should().Be(np.float64);
            np.matrix("0x10 0b11").Should().BeOfValues(16L, 3L);
            np.matrix("1+2j 3").dtype.Should().Be(np.complex128);
            np.matrix("").shape.Should().Equal(1L, 0L);

            Action ragged = () => np.matrix("1 2; 3 4 5");
            ragged.Should().Throw<ValueError>().WithMessage("Rows not the same size.");
        }

        /// <summary>Iterating a matrix yields its rows as (1, N) matrices.</summary>
        [TestMethod]
        public void Iteration_YieldsRows()
        {
            NDMatrix m = np.matrix(np.arange(6).reshape(3, 2));
            int rows = 0;
            foreach (NDMatrix row in m)
            {
                row.shape.Should().Equal(1L, 2L);
                rows++;
            }

            rows.Should().Be(3);
        }
    }
}
