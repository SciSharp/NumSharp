using System;
using System.Numerics;
using NumSharp.Generic;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Matrixlib
{
    /// <summary>
    ///     Unit tests of the matrix boundary (<see cref="NDMatrixBoundary"/>): a public <c>np.*</c> function that
    ///     receives a <see cref="NDMatrix"/> OBJECT through an <see cref="NDArray"/> reference returns the class
    ///     and the shape NumPy 2.4.2 returns for a <c>numpy.matrix</c> operand, while the operators, the indexers
    ///     and the members of the <see cref="NDArray"/> reference keep the plain-array meaning.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The differential gate is the oracle tier <c>matrix_operand.jsonl</c> (4,645 NumPy answers over 260
    ///     functions) and the caller audit <c>MatrixOperand_LibraryNeverReachesAMatrixRule</c>. These tests pin
    ///     what that tier does not reach: the functions whose rule was taken from NumPy's source and a direct
    ///     probe (tuple results, nested operands, the LU family), the mechanics of the boundary (nesting,
    ///     exceptions, <c>out=</c>), and the documented limits.
    ///     </para>
    ///     <para>
    ///     Every expected class and shape was printed by NumPy 2.4.2 for the same call with
    ///     <c>m = np.matrix([[1., 2., 3.], [4., 5., 6.]])</c>, <c>a = np.asarray(m)</c> and
    ///     <c>sq = np.matrix([[2., 1.], [1., 3.]])</c>.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class NDMatrixBoundaryTests
    {
        /// <summary>The probe's 2 x 3 float matrix, as the <see cref="NDArray"/> reference a function receives.</summary>
        /// <returns>A new matrix object <c>[[1, 2, 3], [4, 5, 6]]</c> (float64) typed <see cref="NDArray"/>.</returns>
        private static NDArray M() => np.matrix(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }));

        /// <summary>The same values as a plain array.</summary>
        /// <returns>A new plain array <c>[[1, 2, 3], [4, 5, 6]]</c> (float64).</returns>
        private static NDArray A() => np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        /// <summary>The probe's symmetric positive definite 2 x 2 matrix.</summary>
        /// <returns>A new matrix object <c>[[2, 1], [1, 3]]</c> (float64) typed <see cref="NDArray"/>.</returns>
        private static NDArray Sq() => np.matrix(np.array(new double[,] { { 2, 1 }, { 1, 3 } }));

        /// <summary>Asserts the class and the shape of a result.</summary>
        /// <param name="result">What the function returned.</param>
        /// <param name="matrix">Whether NumPy returns a <c>numpy.matrix</c> (false: an ndarray or a scalar).</param>
        /// <param name="shape">NumPy's shape.</param>
        private static void Check(NDArray result, bool matrix, params long[] shape)
        {
            (result is NDMatrix).Should().Be(matrix, "NumPy returns {0}", matrix ? "a matrix" : "a plain array");
            result.shape.Should().Equal(shape);
        }

        // ================================================================================================
        // Rules NumPy's source and a direct probe gave (not in the oracle tier)
        // ================================================================================================

        /// <summary>Functions that allocate through the operand's class or wrap their result: a matrix comes back,
        ///     and a 1-d result becomes a 1 x N row.</summary>
        [TestMethod]
        public void SubclassResults_AreMatrices()
        {
            Check(np.nan_to_num(M()), true, 2, 3);
            Check(np.rollaxis(M(), 1), true, 3, 2);
            Check(np.empty_like(M()), true, 2, 3);
            Check(np.empty_like(M(), np.int32), true, 2, 3);
            Check(np.full_like(M(), 7), true, 2, 3);
            np.full_like(M(), 7).Should().BeOfValues(7.0, 7.0, 7.0, 7.0, 7.0, 7.0);
            Check(np.partition(M(), 1), true, 2, 3);
            Check(np.partition(M(), 1, axis: null), true, 1, 6);
            Check(np.argpartition(M(), 1), true, 2, 3);
            Check(np.argpartition(M(), 1, axis: null), true, 1, 6);
            Check(np.piecewise(M(), new[] { M() < 3, M() >= 3 }, new object[] { 0, 1 }), true, 2, 3);
            Check(np.reshape(M(), new Shape(6)), true, 1, 6);
            Check(np.reshape(M(), new Shape(1, 2, 3)), true, 2, 3);
            Check(np.reshape(M(), new Shape(2, 1, 3, 1)), true, 2, 3);
        }

        /// <summary>np.apply_along_axis wraps the assembled result in the operand's class (NumPy's conv.wrap).</summary>
        [TestMethod]
        public void ApplyAlongAxis_ReturnsAMatrix()
        {
            NDArray sums = np.apply_along_axis(v => np.sum(v), 0, M());
            Check(sums, true, 1, 3);
            sums.Should().BeOfValues(5.0, 7.0, 9.0);

            NDArray doubled = np.apply_along_axis(v => v * 2, 1, M());
            Check(doubled, true, 2, 3);
            doubled.Should().BeOfValues(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);
        }

        /// <summary>The split family returns a matrix per section (NumPy slices the matrix).</summary>
        [TestMethod]
        public void Split_ReturnsMatrices()
        {
            NDArray[] thirds = np.split(M(), 3, axis: 1);
            thirds.Length.Should().Be(3);
            foreach (NDArray part in thirds)
                Check(part, true, 2, 1);

            NDArray[] rows = np.split(M(), 2);
            Check(rows[0], true, 1, 3);
            Check(rows[1], true, 1, 3);

            NDArray[] uneven = np.array_split(M(), 2, axis: 1);
            Check(uneven[0], true, 2, 2);
            Check(uneven[1], true, 2, 1);

            NDArray[] withEmpty = np.array_split(M(), 3);
            Check(withEmpty[0], true, 1, 3);
            Check(withEmpty[1], true, 1, 3);
            Check(withEmpty[2], true, 0, 3);
        }

        /// <summary>A ufunc with two outputs wraps both (np.divmod, np.frexp, np.modf).</summary>
        [TestMethod]
        public void TwoOutputUfuncs_ReturnTwoMatrices()
        {
            NDArray i = np.matrix(np.array(new long[,] { { 3, 1, 2 }, { 9, 7, 8 } }));

            var (q, r) = np.divmod(i, 2);
            Check(q, true, 2, 3);
            Check(r, true, 2, 3);
            q.Should().BeOfValues(1L, 0L, 1L, 4L, 3L, 4L);
            r.Should().BeOfValues(1L, 1L, 0L, 1L, 1L, 0L);

            // A matrix in the second position decides too: matrix.__array_priority__ is above an array's.
            var (q2, r2) = np.divmod(np.array(new long[,] { { 3, 1, 2 }, { 9, 7, 8 } }), i);
            Check(q2, true, 2, 3);
            Check(r2, true, 2, 3);

            var (mantissa, exponent) = np.frexp(M());
            Check(mantissa, true, 2, 3);
            Check(exponent, true, 2, 3);

            var (frac, whole) = np.modf(M());
            Check(frac, true, 2, 3);
            Check(whole, true, 2, 3);
        }

        /// <summary>np.clip is a ufunc call in NumPy: a matrix operand OR a matrix bound gives a matrix, a matrix
        ///     out is returned itself, and a matrix where= does not change the class.</summary>
        [TestMethod]
        public void Clip_FollowsEveryOperandButTheMask()
        {
            Check(np.clip(M(), 2, 4), true, 2, 3);
            np.clip(M(), 2, 4).Should().BeOfValues(2.0, 2.0, 3.0, 4.0, 4.0, 4.0);
            Check(np.clip(A(), M(), null), true, 2, 3);
            Check(np.clip(A(), null, M()), true, 2, 3);

            NDArray o = np.matrix(np.zeros(new Shape(2, 3)));
            NDArray returned = np.clip(A(), 2, 4, @out: o);
            ReferenceEquals(returned, o).Should().BeTrue("out= is returned itself");
            o.Should().BeOfValues(2.0, 2.0, 3.0, 4.0, 4.0, 4.0);

            NDArray mask = np.matrix(np.array(new bool[,] { { false, false, true }, { true, true, true } }));
            (np.clip(A(), 2, 4, @out: np.zeros(new Shape(2, 3)), where: mask) is NDMatrix).Should().BeFalse();
        }

        /// <summary>Functions that convert their operand with asarray return plain arrays.</summary>
        [TestMethod]
        public void AsArrayInside_ReturnsPlainArrays()
        {
            Check(np.cross(M(), M()), false, 2, 3);
            Check(np.linalg.cross(M(), M()), false, 2, 3);
            Check(np.einsum("ij,kj->ik", M(), M()), false, 2, 2);
            Check(np.einsum("ij->ji", M()), false, 3, 2);
            Check(np.fft.fftshift(M()), false, 2, 3);
            Check(np.fft.ifftshift(M(), 0), false, 2, 3);
            Check(np.fft.hfft(M()), false, 2, 4);
            Check(np.fft.fft(M()), false, 2, 3);
            Check(np.linalg.norm(M(), axis: 0), false, 3);
            Check(np.linalg.norm(M(), axis: 1, keepdims: true), false, 2, 1);
            np.linalg.norm(M()).ndim.Should().Be(0);
            np.linalg.matrix_norm(M()).ndim.Should().Be(0);
            Check(np.argwhere(M()), false, 6, 2);
            Check(np.flatnonzero(M()), false, 6);
            NDArray<long>[] where = np.nonzero(M());
            where.Length.Should().Be(2);
            Check(where[0], false, 6);
            Check(np.where(M() > 2, M(), 0), false, 2, 3);
            Check(np.lexsort(new[] { M(), M() }), false, 2, 3);
        }

        /// <summary>np.iscomplex is <c>x.imag != 0</c> for a complex matrix (a matrix) and a new zeros array for
        ///     any other dtype (plain); np.isreal is <c>imag(x) == 0</c>, a matrix for every dtype.</summary>
        [TestMethod]
        public void IsComplex_IsReal_FollowNumPysTwoBranches()
        {
            NDArray c = np.matrix(np.array(new[,] { { new Complex(1, 2), new Complex(3, 0) } }));
            Check(np.iscomplex(c), true, 1, 2);
            np.iscomplex(c).Should().BeOfValues(true, false);
            Check(np.iscomplex(M()), false, 2, 3);
            Check(np.isreal(c), true, 1, 2);
            Check(np.isreal(M()), true, 2, 3);
        }

        /// <summary>np.average over a tuple of axes keeps every reduced axis (matrix._collapse returns the kept
        ///     result for any axis that is not None); keepdims is refused by the matrix method NumPy calls.</summary>
        [TestMethod]
        public void Average_OverSeveralAxes_KeepsThem()
        {
            NDArray all = np.average(M(), new[] { 0, 1 });
            Check(all, true, 1, 1);
            all.Should().BeOfValues(3.5);

            Check(np.average(M(), new[] { 1 }), true, 2, 1);
            Check(np.average(M(), new[] { 0 }, weights: A()), true, 1, 3);
            np.average(M(), (int[])null).ndim.Should().Be(0);

            Action keep = () => np.average(M(), new[] { 0, 1 }, keepdims: true);
            keep.Should().Throw<TypeError>().WithMessage("matrix.mean() got an unexpected keyword argument 'keepdims'");
            Action keepWeighted = () => np.average(M(), 0, weights: A(), keepdims: true);
            keepWeighted.Should().Throw<TypeError>().WithMessage("matrix.sum() got an unexpected keyword argument 'keepdims'");
        }

        /// <summary>np.average(..., returned=True): the matrix average and the sum of weights as a plain array of
        ///     the average's shape.</summary>
        [TestMethod]
        public void AverageReturned_PairsTheMatrixWithAPlainSumOfWeights()
        {
            var (avg, weights) = np.average_returned(M(), 0);
            Check(avg, true, 1, 3);
            avg.Should().BeOfValues(2.5, 3.5, 4.5);
            Check(weights, false, 1, 3);
            weights.Should().BeOfValues(2.0, 2.0, 2.0);

            var (scalarAvg, scalarWeights) = np.average_returned(M());
            scalarAvg.ndim.Should().Be(0);
            scalarWeights.ndim.Should().Be(0);
            ((double)scalarAvg).Should().Be(3.5);
            ((double)scalarWeights).Should().Be(6.0);
        }

        /// <summary>np.r_ / np.c_ join through np.concatenate, which keeps the matrix class; a matrix entry cannot
        ///     gain an axis from ndmin.</summary>
        [TestMethod]
        public void IndexTricks_JoinMatricesIntoAMatrix()
        {
            NDArray m = M(), a = A();
            Check(np.r_[m, m], true, 4, 3);
            Check(np.r_[m, a], true, 4, 3);
            Check(np.r_[a, m], true, 4, 3);
            Check(np.r_["1", m, m], true, 2, 6);
            Check(np.r_["0,3", m, m], true, 4, 3);
            Check(np.r_["r", m, m], true, 4, 3);
            Check(np.c_[m, m], true, 2, 6);
            Check(np.c_[m, a], true, 2, 6);
            np.r_[m, m].Should().BeOfValues(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

            // NumPy: newobj.transpose(axes) with three axes on the 2-d matrix.
            Action transposed = () => _ = np.r_["0,3,0", m, m];
            transposed.Should().Throw<Exception>().WithMessage("axes don't match array");
        }

        /// <summary>np.block: repeated concatenation returns a matrix, the single-allocation route (more than
        ///     2*512*512 elements) a new ndarray, and a matrix leaf cannot be nested three lists deep.</summary>
        [TestMethod]
        public void Block_ClassFollowsNumPysRoute()
        {
            NDArray m = M(), a = A();
            Check(np.block(new object[] { m, m }), true, 2, 6);
            Check(np.block(new object[] { new object[] { m }, new object[] { m } }), true, 4, 3);
            Check(np.block(new object[] { m, a }), true, 2, 6);
            Check(np.block(new object[] { new object[] { m, a }, new object[] { a, m } }), true, 4, 6);
            Check(np.block(m), true, 2, 3);

            NDArray big = np.matrix(np.ones(new Shape(600, 600)));
            Check(np.block(new object[] { big, big }), false, 600, 1200);

            // NumPy's AxisError; np.concatenate raises its base class here (house convention), with the same text.
            Action deep = () => np.block(new object[] { new object[] { new object[] { m } } });
            deep.Should().Throw<ArgumentOutOfRangeException>().WithMessage("axis -3 is out of bounds for array of dimension 2*");
            Action mixed = () => np.block(new object[] { new object[] { new object[] { m, a } } });
            mixed.Should().Throw<Exception>().WithMessage(
                "all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) " +
                "and the array at index 1 has 3 dimension(s)");
        }

        /// <summary>The LU family (managed fallback): np.linalg.solve takes its class from b, inv and tensorsolve
        ///     from a.</summary>
        [TestMethod]
        public void Linalg_SolveFollowsB_InvFollowsA()
        {
            NDArray sq = Sq();
            NDArray sa = np.array(new double[,] { { 2, 1 }, { 1, 3 } });

            Check(np.linalg.solve(sq, sa), false, 2, 2);
            Check(np.linalg.solve(sa, sq), true, 2, 2);
            Check(np.linalg.solve(sq, sa["0"]), false, 2);
            Check(np.linalg.inv(sq), true, 2, 2);
            Check(np.linalg.tensorsolve(sq, sa["0"]), true, 1, 2);
            np.allclose(np.linalg.solve(sa, sq), np.eye(2)).Should().BeTrue();
        }

        /// <summary>np.real / np.imag of a complex matrix are matrices over the lane (NumPy's val.real / val.imag);
        ///     np.real of a real matrix is the matrix itself.</summary>
        [TestMethod]
        public void RealAndImag_OfAComplexMatrix_AreMatrices()
        {
            NDArray c = np.matrix(np.array(new[,] { { new Complex(1, 2), new Complex(3, 0) }, { new Complex(0, 0), new Complex(0, 1) } }));
            NDArray re = np.real(c);
            Check(re, true, 2, 2);
            re.Should().BeOfValues(1.0, 3.0, 0.0, 0.0);
            NDArray im = np.imag(c);
            Check(im, true, 2, 2);
            im.Should().BeOfValues(2.0, 0.0, 0.0, 1.0);

            // The real lane is a view: a write reaches the complex matrix.
            re.SetDouble(9.0, 0, 0);
            c.GetComplex(0, 0).Should().Be(new Complex(9, 2));

            NDArray m = M();
            ReferenceEquals(np.real(m), m).Should().BeTrue();
        }

        /// <summary>np.unstack iterates np.moveaxis(x, axis, 0), and iterating a matrix yields 1 x N row matrices.</summary>
        [TestMethod]
        public void Unstack_ReturnsRowMatrices()
        {
            NDArray[] rows = np.unstack(M());
            rows.Length.Should().Be(2);
            Check(rows[0], true, 1, 3);
            Check(rows[1], true, 1, 3);
            rows[1].Should().BeOfValues(4.0, 5.0, 6.0);

            NDArray[] columns = np.unstack(M(), 1);
            columns.Length.Should().Be(3);
            foreach (NDArray column in columns)
                Check(column, true, 1, 2);
            columns[2].Should().BeOfValues(3.0, 6.0);
        }

        /// <summary>The overloads that take their operands by reference carry the rule of their by-value twins.</summary>
        [TestMethod]
        public void ByReferenceOverloads_CarryTheRule()
        {
            NDArray m = M(), two = np.array(2.0);
            Check(np.power(in m, in two), true, 2, 3);
            Shape row = new Shape(6);
            Check(np.reshape(m, ref row), true, 1, 6);
        }

        // ================================================================================================
        // Mechanics
        // ================================================================================================

        /// <summary>A function that calls other functions with rules wraps once: the inner calls compute plain.</summary>
        [TestMethod]
        public void NestedCalls_WrapOnce()
        {
            NDArray stacked = np.vstack(new[] { M(), M() });
            Check(stacked, true, 4, 3);

            // np.var runs np.mean, np.subtract, np.multiply and np.true_divide on the operand inside.
            NDArray variance = np.var(M(), 0);
            Check(variance, true, 1, 3);
            variance.Should().BeOfValues(2.25, 2.25, 2.25);

            // np.sum of a bool matrix in the bool loop runs np.any inside the engine. (np.greater returns the
            // matrix; the > operator of an NDArray reference would return a plain array.)
            NDArray flags = np.greater(M(), 2);
            Check(flags, true, 2, 3);
            NDArray counted = np.sum(flags, 0, dtype: np.@bool);
            Check(counted, true, 1, 3);
            counted.Should().BeOfValues(true, true, true);
        }

        /// <summary>An exception inside a rule leaves the boundary open: the next call applies its rule again.</summary>
        [TestMethod]
        public void AnException_DoesNotLeaveTheThreadInsideARegion()
        {
            Action keep = () => np.sum(M(), 0, keepdims: true);
            keep.Should().Throw<TypeError>().WithMessage("matrix.sum() got an unexpected keyword argument 'keepdims'");
            Check(np.sum(M(), 0), true, 1, 3);

            Action broadcast = () => np.add(M(), np.arange(4.0));
            broadcast.Should().Throw<Exception>();
            Check(np.add(M(), M()), true, 2, 3);

            Action stacked = () => np.stack(new[] { M(), M() });
            stacked.Should().Throw<ValueError>().WithMessage("shape too large to be a matrix.");
            Check(np.negative(M()), true, 2, 3);
        }

        /// <summary>out= is returned itself, whatever its class; a plain out with a matrix operand stays plain.</summary>
        [TestMethod]
        public void Out_IsReturnedItself()
        {
            NDArray matrixOut = np.matrix(np.zeros(new Shape(2, 3)));
            ReferenceEquals(np.add(M(), M(), @out: matrixOut), matrixOut).Should().BeTrue();
            matrixOut.Should().BeOfValues(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);

            NDArray plainOut = np.zeros(new Shape(2, 3));
            NDArray returned = np.add(M(), M(), @out: plainOut);
            ReferenceEquals(returned, plainOut).Should().BeTrue();
            (returned is NDMatrix).Should().BeFalse();
        }

        /// <summary>The operators, the indexers and the members of an NDArray reference keep the plain-array
        ///     meaning for a matrix object, including the members that are implemented through np.* functions.</summary>
        [TestMethod]
        public void OperatorsIndexersAndMembers_StayPlain()
        {
            NDArray a = M();
            (a is NDMatrix).Should().BeTrue();

            Check(a + a, false, 2, 3);
            Check(a * a, false, 2, 3);
            (a * a).Should().BeOfValues(1.0, 4.0, 9.0, 16.0, 25.0, 36.0);
            Check(a["0"], false, 3);
            Check(a[":, 1"], false, 2);
            Check(a.sum(0), false, 3);
            Check(a.mean(1), false, 2);
            Check(a.ravel(), false, 6);
            Check(a.clip(2, 4), false, 2, 3);
            Check(a.cumsum(), false, 6);
            Check(a.transpose(), false, 3, 2);
            Check(a.take(np.array(new long[] { 0, 5 })), false, 2);

            NDArray i = np.matrix(np.array(new long[,] { { 1, 2 }, { 3, 4 } }));
            Check(~i, false, 2, 2);
            (~i).Should().BeOfValues(-2L, -3L, -4L, -5L);
        }

        /// <summary>A result is the matrix object upcast: a cast gives the matrix members back without a copy.</summary>
        [TestMethod]
        public void TheResult_IsTheMatrixObject()
        {
            NDArray reduced = np.sum(M(), 1);
            NDMatrix column = (NDMatrix)reduced;
            column.shape.Should().Equal(3L - 1L, 1L);
            column.T.shape.Should().Equal(1L, 2L);
            column.A1.Should().BeOfValues(6.0, 15.0);
        }

        /// <summary>The caller audit is off in production and records nothing for a call from outside the library.</summary>
        [TestMethod]
        public void CallerAudit_RecordsNothingForAUsersCall()
        {
            NDMatrixBoundary.AuditInternalCallers.Should().BeFalse();
            NDMatrixBoundary.InternalCallers.Clear();
            NDMatrixBoundary.AuditInternalCallers = true;
            try
            {
                Check(np.sum(M(), 0), true, 1, 3);
                Check(np.vstack(new[] { M(), M() }), true, 4, 3);
                Check(np.r_[M(), M()], true, 4, 3);
                NDMatrix typed = np.matrix(A());
                _ = typed.sum(0);
                _ = typed * typed.T;
            }
            finally
            {
                NDMatrixBoundary.AuditInternalCallers = false;
            }
            NDMatrixBoundary.InternalCallers.Should().BeEmpty();
        }

        // ================================================================================================
        // Containers of the library hold a matrix as its plain view
        // ================================================================================================

        /// <summary>numpy.ma keeps the matrix class of its data (a "masked_matrix"); NumSharp's masked array has
        ///     no model for that and stores the plain view, so every masked operation is the plain-array one.</summary>
        [TestMethod]
        [Misaligned]
        public void MaskedArray_StoresAMatrixAsItsPlainView()
        {
            NDArray m = M();
            NDMaskedArray masked = np.ma.array(m);
            (masked.data is NDMatrix).Should().BeFalse("NumPy: ma.array(m).data is the matrix");
            masked.data.shape.Should().Equal(2L, 3L);

            // NumPy: ma.array(m).sum(axis=0) has shape (1, 3).
            masked.sum(axis: 0).shape.Should().Equal(3L);

            // The view shares the matrix's buffer: a write through the masked array reaches the matrix.
            masked.data.SetDouble(99.0, 0, 0);
            m.GetDouble(0, 0).Should().Be(99.0);
        }

        /// <summary>The samplers read a matrix parameter as a plain array; NumPy's draws are plain arrays too.</summary>
        [TestMethod]
        public void RandomSamplers_TakeAMatrixParameter()
        {
            NDArray draws = np.random.default_rng(0).normal(M(), np.array(1.0));
            Check(draws, false, 2, 3);
            NDArray expected = np.random.default_rng(0).normal(A(), np.array(1.0));
            np.array_equal(draws, expected).Should().BeTrue();

            NDArray uniform = np.random.default_rng(0).uniform(M(), M() + 1);
            Check(uniform, false, 2, 3);
        }

        // ================================================================================================
        // Documented limits
        // ================================================================================================

        /// <summary>A function whose C# return type is a typed array cannot return a matrix object: np.isclose
        ///     returns the plain boolean array where NumPy returns a matrix (gap MO01 of the oracle tier).</summary>
        [TestMethod]
        [Misaligned]
        public void TypedReturns_CannotCarryTheClass()
        {
            NDArray<bool> close = np.isclose(M(), A());
            ((object)close is NDMatrix).Should().BeFalse("NumPy: np.isclose(m, a) is a matrix");
            close.shape.Should().Equal(2L, 3L);

            // The untyped comparison ufuncs return the matrix.
            Check(np.equal(M(), A()), true, 2, 3);
        }

        /// <summary>Results where matrix semantics leak into NumPy's own Python code are not reproduced: NumSharp
        ///     returns the plain result shaped by the function's matrix rule.</summary>
        [TestMethod]
        [Misaligned]
        public void NumPyArtifacts_AreNotReproduced()
        {
            // NumPy: np.linalg.vector_norm(m) is an ndarray of shape (6,) holding |x| (m.ravel() stays 1 x 6 and the
            // norm then runs along axis 0). NumSharp returns the norm.
            NDArray norm = np.linalg.vector_norm(M());
            norm.ndim.Should().Be(0);
            ((double)norm).Should().BeApproximately(System.Math.Sqrt(91.0), 1e-12);

            // NumPy: np.average(m, axis=1, weights=ndarray) is a 2 x 2 matrix (the 2 x 1 matrix of sums divided by
            // the (2,) weight sums broadcasts). NumSharp returns the 2 x 1 column of averages.
            NDArray weighted = np.average(M(), 1, weights: A());
            Check(weighted, true, 2, 1);
        }

        /// <summary>The polynomial package computes with plain arrays: mapdomain of a matrix is a plain array
        ///     (NumPy's <c>off + scl*x</c> on the matrix is a matrix).</summary>
        [TestMethod]
        [Misaligned]
        public void PolynomialPackage_ComputesPlain()
        {
            NDArray mapped = np.polynomial.polyutils.mapdomain(M(), (0.0, 1.0), (0.0, 2.0));
            (mapped is NDMatrix).Should().BeFalse("NumPy: polyutils.mapdomain(m, ...) is a matrix");
            mapped.shape.Should().Equal(2L, 3L);
            mapped.Should().BeOfValues(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);
        }
    }
}
