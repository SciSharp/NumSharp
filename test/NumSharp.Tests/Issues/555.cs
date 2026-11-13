using System;
using System.Linq;
using System.Reflection;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #555, "[NEP32 NEP34] API Cleanup and Behavioral Changes"
    ///     (https://github.com/SciSharp/NumSharp/issues/555): NEP 34's ragged-sequence refusal, NEP 32's absent financial
    ///     functions, and every row of the issue's dead-code table computing. Expected values are NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class Issue555
    {
        /// <summary>
        ///     NEP 34: a ragged jagged array raises NumPy's ValueError with NumPy's exact text instead of inferring an object
        ///     array.
        /// </summary>
        [TestMethod]
        public void RaggedSequence_RaisesNumPysValueError()
        {
            new Action(() => np.array(new int[][] { new[] { 1, 2 }, new[] { 1 } })).Should().Throw<ValueError>()
                .WithMessage("setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");
        }

        /// <summary>
        ///     NEP 32: the ten financial functions NumPy removed are not part of <c>np</c> (NumPy 2.4.2 has none of them).
        /// </summary>
        [TestMethod]
        public void FinancialFunctions_StayUnimplemented()
        {
            var removed = new[] { "fv", "pv", "npv", "irr", "mirr", "nper", "pmt", "ppmt", "ipmt", "rate" };
            removed.Where(n => typeof(np).GetMember(n, BindingFlags.Public | BindingFlags.Static).Length > 0)
                .Should().BeEmpty("NEP 32 removed them from NumPy and the answer says they stay unimplemented");
        }

        /// <summary>
        ///     The dead-code table's predicates and logic: <c>isnan</c>, <c>isfinite</c>, <c>isclose</c>, <c>&amp;</c>,
        ///     <c>|</c> and <c>any(axis: 0)</c> compute NumPy's values (they returned null or threw when the issue was filed).
        /// </summary>
        [TestMethod]
        public void DeadCodeTable_PredicatesAndLogic_Compute()
        {
            var f = np.array(new[] { 1.0, double.NaN, double.PositiveInfinity });
            np.isnan(f).Should().BeOfValues(false, true, false);
            np.isfinite(f).Should().BeOfValues(true, false, false);
            np.isclose(np.array(new[] { 1.0, 2.0 }), np.array(new[] { 1.0 + 1e-10, 2.1 })).Should().BeOfValues(true, false);

            var p = np.array(new[] { true, false, true });
            var q = np.array(new[] { true, true, false });
            (p & q).Should().BeOfValues(true, false, false);
            (p | q).Should().BeOfValues(true, true, true);
            np.any(np.array(new[,] { { false, true }, { false, false } }), axis: 0).Should().BeOfValues(false, true);
        }

        /// <summary>
        ///     <c>np.roll</c> (once limited to a few types) keeps the dtype and NumPy's values for every dtype the answer names.
        /// </summary>
        [TestMethod]
        public void Roll_EveryDtypeTheAnswerNames()
        {
            foreach (var dt in new[] { np.@bool, np.uint8, np.int16, np.float16, np.float32, np.complex128, np.@decimal, np.@char })
            {
                var rolled = np.roll(np.arange(4).astype(dt), 1);
                rolled.dtype.Should().Be((DType)dt, $"roll keeps {((DType)dt).name}");
                // arange(4) as bool is [F, T, T, T], so its roll by 1 reads [T, F, T, T]; every other dtype holds 0..3.
                long[] expected = (DType)dt == (DType)np.@bool ? new long[] { 1, 0, 1, 1 } : new long[] { 3, 0, 1, 2 };
                rolled.astype(np.int64).ToArray<long>().Should().Equal(expected, $"roll of {((DType)dt).name}");
            }
        }

        /// <summary>
        ///     The boolean-mask setter (NotImplementedException when the issue was filed) writes through the mask.
        /// </summary>
        [TestMethod]
        public void BooleanMaskSetter_Writes()
        {
            var a = np.arange(6);
            a[a > 2] = 0;
            a.Should().BeOfValues(0L, 1L, 2L, 0L, 0L, 0L);
        }

        /// <summary>
        ///     The linalg rows: <c>np.linalg.norm</c> is public and computes, and the null-returning <c>NDArray</c> stubs
        ///     <c>inv</c> / <c>qr</c> / <c>svd</c> / <c>lstsq</c> are gone (those functions live in <c>np.linalg</c>).
        /// </summary>
        [TestMethod]
        public void LinalgRows_NormIsPublic_StubsAreGone()
        {
            np.linalg.norm(np.array(new[] { 3.0, 4.0 })).GetDouble().Should().Be(5.0);
            foreach (var stub in new[] { "inv", "qr", "svd", "lstsq" })
                typeof(NDArray).GetMember(stub, BindingFlags.Public | BindingFlags.Instance).Should().BeEmpty($"NDArray.{stub} was a null-returning stub");
        }
    }
}
