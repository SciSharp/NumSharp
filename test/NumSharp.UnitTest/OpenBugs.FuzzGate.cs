using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Regression pins for the differential-fuzz GATE-TIGHTENING pass (COMPLETENESS_PLAN
    ///     WS-BUGS): every excuse branch in <see cref="NumSharp.UnitTest.Fuzz.MisalignedRegistry"/>
    ///     was narrowed from a blanket to its documented (op, dtype, kind) cell, and each REAL bug
    ///     the tightening exposed is either fixed in src (verified by a normal test here) or pinned
    ///     under [OpenBugs] below. NumPy 2.4.2 is the source of truth for every assertion.
    /// </summary>
    [TestClass]
    public class FuzzGateRegressionTests : TestClass
    {
        // ============================================================================
        //  F16 / B8: np.invert on non-integer dtypes must throw CLEANLY, never crash.
        //
        //  Historically np.invert(float) reached the IL BitwiseNot emitter with a dtype
        //  it has no opcode for and executed an ILLEGAL CPU INSTRUCTION
        //  (ExecutionEngineException — killed the whole test host; documented in
        //  test/oracle/gen_oracle.py's errors tier as the reason the invert spec was
        //  omitted). The loop-resolution guard in Default.Invert.cs now rejects every
        //  non-(bool|integer) loop dtype up front with NumPy's verbatim TypeError:
        //      "ufunc 'invert' not supported for the input types, and the inputs could
        //       not be safely coerced to any supported types according to the casting
        //       rule ''safe''"           (probed identical on NumPy 2.4.2)
        //  UfuncUnaryBatchOutWhereTests.Invert_FloatInputs_NotSupportedText_AndOrder
        //  pins the float64 message/order; these pin the remaining non-integer dtypes
        //  (float64 kept per plan-spec, plus Half / complex128 / decimal — decimal has
        //  no NumPy analog but must take the same clean-throw path, not the crash).
        // ============================================================================

        [TestMethod]
        public void Invert_Double_ThrowsCleanly_NoHostCrash()
        {
            Action act = () => np.invert(np.array(new double[] { 1.5 }));
            act.Should().Throw<TypeError>()
               .WithMessage("ufunc 'invert' not supported for the input types, and the inputs " +
                            "could not be safely coerced to any supported types according to the casting rule ''safe''");
        }

        [TestMethod]
        public void Invert_Half_ThrowsCleanly_NoHostCrash()
        {
            Action act = () => np.invert(np.array(new Half[] { (Half)1.5f }));
            act.Should().Throw<TypeError>().WithMessage("*ufunc 'invert' not supported*");
        }

        [TestMethod]
        public void Invert_Complex_ThrowsCleanly_NoHostCrash()
        {
            Action act = () => np.invert(np.array(new System.Numerics.Complex[] { new(1, 2) }));
            act.Should().Throw<TypeError>().WithMessage("*ufunc 'invert' not supported*");
        }

        [TestMethod]
        public void Invert_Decimal_ThrowsCleanly_NoHostCrash()
        {
            Action act = () => np.invert(np.array(new decimal[] { 1.5m }));
            act.Should().Throw<TypeError>().WithMessage("*ufunc 'invert' not supported*");
        }

        [TestMethod]
        public void InvertOperator_Float_ThrowsCleanly_NoHostCrash()
        {
            var a = np.array(new float[] { 1.5f, 2.5f });
            Action act = () => { var _ = ~a; };
            act.Should().Throw<TypeError>().WithMessage("*ufunc 'invert' not supported*");
        }
    }
}
