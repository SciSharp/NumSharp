using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using FluentAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Bugs 25-63: NumPy 1.x deprecation audit findings.
    ///
    ///     These bugs were found during a systematic audit comparing NumSharp
    ///     against NumPy 2.x (v2.4.2). They represent cases where NumSharp
    ///     follows NumPy 1.x behavior that was changed, removed, or fixed
    ///     in NumPy 2.0+, or dead code that was never implemented.
    ///
    ///     Each test asserts the CORRECT NumPy 2.x behavior.
    ///     Tests FAIL while the bug exists, and PASS when the bug is fixed.
    ///
    ///     Reference: docs/plans/numpy-1x-deprecation-findings.md
    ///
    ///     Categories:
    ///       Bug 27:  np.roll static returns int instead of NDArray
    ///       Bug 28:  floor/ceil on integer arrays casts to Double
    ///       Bug 29:  np.fmax/np.fmin don't ignore NaN (same as max/min)
    ///       Bug 30:  np.fmin docstrings say "maximum" instead of "minimum"
    ///                (no test — documentation-only bug)
    ///       Bug 31:  stardard_normal misspelled (missing 'd')
    ///                (no test — naming-only bug)
    ///       Bug 32:  np.convolve always returns null
    ///       Bug 33:  np.any(axis) has inverted logic (implements all)
    ///       Bug 34:  np.isnan returns null (dead code)
    ///       Bug 35:  np.isfinite returns null (dead code)
    ///       Bug 36:  np.isclose returns null (dead code)
    ///       Bug 37:  operator &amp; (AND) returns null (dead code)
    ///       Bug 38:  operator | (OR) returns null (dead code)
    ///       Bug 39:  nd.delete() returns null (dead code)
    ///       Bug 40:  nd.inv() returns null (dead code)
    ///       Bug 41:  nd.qr() returns default (null, null)
    ///       Bug 42:  nd.svd() returns default (null, null, null)
    ///       Bug 43:  nd.lstqr() returns null + misspelled (should be lstsq)
    ///       Bug 44:  nd.multi_dot() returns null (dead code)
    ///       Bug 45:  nd.roll(shift) no-axis overload returns null
    ///       Bug 46:  Boolean mask setter throws NotImplementedException
    ///       Bug 47:  np.positive() implements abs() instead of +x identity
    ///       Bug 48:  np.negative() only negates positive values
    ///       Bug 49:  np.all(axis) throws InvalidCastException
    ///       Bug 50:  nd.roll(shift, axis) only supports 3 of 12 dtypes
    ///       Bug 51:  np.log1p computes log10(1+x) instead of ln(1+x)
    ///       Bug 52:  np.std/np.var ignore ddof parameter
    ///       Bug 53:  np.searchsorted returns wrong indices
    ///       Bug 54:  np.moveaxis fails with negative axis
    ///       Bug 55:  np.mean crashes on empty arrays
    ///       Bug 56:  np.abs returns Double for integer input
    ///       Bug 57:  np.sum/np.mean crash on boolean arrays
    ///       Bug 58:  astype(int32) rounds instead of truncating
    ///       Bug 59:  >= and &lt;= operators throw IncorrectShapeException
    ///       Bug 60:  np.argmax ignores NaN values
    ///       Bug 61:  np.linspace returns float32 instead of float64
    ///       Bug 62:  Implicit conversion operators crash across dtypes
    ///       Bug 63:  ToString crashes on empty arrays
    /// </summary>
    public partial class OpenBugs
    {
        // BUG 25 (NEP 50) — REMOVED: False positive. The tests used typed scalars
        // (NPTypeCode.Int16/Int32/Int64) which follow standard promotion rules.
        // NEP 50 "weak typing" only applies to Python int literals, which NumSharp
        // doesn't have. NumSharp's _typemap_arr_scalar entries for typed scalars
        // match NumPy 2.4.2 behavior exactly.
        //
        // BUG 26 (bool + bool) — REMOVED: False positive. NumPy 2.4.2 changed
        // bool + bool to stay as dtype=bool (True + True = True). NumSharp's
        // current behavior (bool OR) produces the same results and is correct.
        // Verified: np.add([T,F,T], [T,T,F]) = [T,T,T] dtype=bool in NumPy 2.4.2.

        // ================================================================
        //
        //  BUG 27: np.roll static returns int instead of NDArray
        //
        //  SEVERITY: High — return type is completely wrong.
        //
        //  The static np.roll() in APIs/np.array_manipulation.cs:16 has
        //  return type `int` and casts `nd.roll(shift, axis)` to int.
        //  In NumPy, np.roll always returns an ndarray.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.roll(np.array([1,2,3,4]), 1)
        //    array([4, 1, 2, 3])
        //
        // ================================================================

        /// <summary>
        ///     BUG 27: np.roll returns int instead of NDArray.
        ///     The return type should be NDArray matching NumPy behavior.
        /// </summary>
        [TestMethod]
        public void Bug_NpRoll_ReturnsInt_ShouldReturnNDArray()
        {
            var a = np.array(new[] { 1, 2, 3, 4, 5 });

            // np.roll is declared as returning int, which is wrong.
            // We test the instance method which should return NDArray.
            var result = a.roll(2);

            result.Should().NotBeNull("roll should return an array");
            result.shape.Should().BeEquivalentTo(new[] { 5 });
            result.GetInt32(0).Should().Be(4,
                "NumPy: np.roll([1,2,3,4,5], 2) = [4,5,1,2,3]. " +
                "The static np.roll in APIs/np.array_manipulation.cs:16 " +
                "returns int instead of NDArray.");
            result.GetInt32(1).Should().Be(5);
            result.GetInt32(2).Should().Be(1);
            result.GetInt32(3).Should().Be(2);
            result.GetInt32(4).Should().Be(3);
        }

        // ================================================================
        //
        //  BUG 28: floor/ceil on integer arrays casts to Double
        //
        //  SEVERITY: Medium — changes dtype unnecessarily and wastes memory.
        //
        //  NumPy 2.1+: np.floor(int_array) returns int_array unchanged.
        //  NumSharp: Always casts to Double via GetComputingType() in
        //  NPTypeCode.cs:577, which maps all integer types to Double.
        //  The Default.Floor.cs switch only handles Double/Single/Decimal.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.floor(np.array([1, 2, 3], dtype=np.int32)).dtype
        //    dtype('int32')
        //    >>> np.ceil(np.array([1, 2, 3], dtype=np.int64)).dtype
        //    dtype('int64')
        //
        // ================================================================

        /// <summary>
        ///     BUG 28a: np.floor on int32 array should preserve int32 dtype.
        ///     NumPy 2.1+: int32 -> int32. NumSharp: int32 -> float64.
        /// </summary>
        [TestMethod]
        public void Bug_Floor_IntArray_ShouldPreserveDtype()
        {
            var a = np.array(new[] { 1, 2, 3 }); // int32
            var result = np.floor(a);

            result.dtype.Should().Be(typeof(int),
                "NumPy 2.1+: np.floor(int32_array).dtype = int32 (no-op for integers). " +
                "NumSharp casts to Double because GetComputingType() in NPTypeCode.cs:577 " +
                "maps all integer types to NPTypeCode.Double, and Default.Floor.cs only " +
                "handles Double/Single/Decimal in its switch.");
            result.GetInt32(0).Should().Be(1);
            result.GetInt32(1).Should().Be(2);
            result.GetInt32(2).Should().Be(3);
        }

        /// <summary>
        ///     BUG 28b: np.ceil on int64 array should preserve int64 dtype.
        /// </summary>
        [TestMethod]
        public void Bug_Ceil_IntArray_ShouldPreserveDtype()
        {
            var a = np.array(new long[] { 10, 20, 30 }); // int64
            var result = np.ceil(a);

            result.dtype.Should().Be(typeof(long),
                "NumPy 2.1+: np.ceil(int64_array).dtype = int64. " +
                "NumSharp casts to Double.");
            result.GetInt64(0).Should().Be(10);
            result.GetInt64(1).Should().Be(20);
            result.GetInt64(2).Should().Be(30);
        }

        // ================================================================
        //
        //  BUG 29: np.fmax/np.fmin don't ignore NaN (same as maximum/minimum)
        //
        //  SEVERITY: Medium — semantic difference from NumPy.
        //
        //  NumPy: np.fmax ignores NaN: fmax(NaN, 1) = 1
        //  NumPy: np.maximum propagates NaN: maximum(NaN, 1) = NaN
        //  NumSharp: Both fmax and maximum have identical implementations.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.fmax(np.nan, 1.0)
        //    1.0
        //    >>> np.maximum(np.nan, 1.0)
        //    nan
        //    >>> np.fmin(np.nan, 1.0)
        //    1.0
        //    >>> np.minimum(np.nan, 1.0)
        //    nan
        //
        // ================================================================

        /// <summary>
        ///     BUG 29a: np.fmax should ignore NaN (return the non-NaN value).
        ///     NumPy: fmax(NaN, 1.0) = 1.0. NumSharp: NaN (propagates like maximum).
        /// </summary>
        [TestMethod]
        public void Bug_Fmax_ShouldIgnoreNaN()
        {
            var a = np.array(new[] { double.NaN, 2.0, double.NaN });
            var b = np.array(new[] { 1.0, double.NaN, 3.0 });
            var result = np.fmax(a, b);

            result.GetDouble(0).Should().Be(1.0,
                "NumPy: np.fmax(NaN, 1.0) = 1.0 (NaN ignored). " +
                "NumSharp fmax has identical implementation to maximum, " +
                "so it propagates NaN instead of ignoring it.");
            result.GetDouble(1).Should().Be(2.0, "fmax(2.0, NaN) = 2.0");
            result.GetDouble(2).Should().Be(3.0, "fmax(NaN, 3.0) = 3.0");
        }

        /// <summary>
        ///     BUG 29b: np.fmin should ignore NaN (return the non-NaN value).
        ///     NumPy: fmin(NaN, 1.0) = 1.0. NumSharp: NaN (propagates like minimum).
        /// </summary>
        [TestMethod]
        public void Bug_Fmin_ShouldIgnoreNaN()
        {
            var a = np.array(new[] { double.NaN, 2.0, double.NaN });
            var b = np.array(new[] { 1.0, double.NaN, 3.0 });
            var result = np.fmin(a, b);

            result.GetDouble(0).Should().Be(1.0,
                "NumPy: np.fmin(NaN, 1.0) = 1.0 (NaN ignored). " +
                "NumSharp fmin has identical implementation to minimum.");
            result.GetDouble(1).Should().Be(2.0, "fmin(2.0, NaN) = 2.0");
            result.GetDouble(2).Should().Be(3.0, "fmin(NaN, 3.0) = 3.0");
        }

        // ================================================================
        //
        //  BUG 32: np.convolve always returns null (dead code)
        //
        //  SEVERITY: Medium — public API that silently returns null.
        //
        //  The Regen template in Math/NdArray.Convolve.cs was never generated
        //  into the #else block, so the method always returns null.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.convolve([1,2,3], [0,1,0.5])
        //    array([0. , 1. , 2.5, 4. , 1.5])
        //
        // ================================================================

        /// <summary>
        ///     BUG 32: np.convolve always returns null.
        ///     The Regen template was never generated into the compiled code.
        /// </summary>
        [TestMethod]
        public void Bug_Convolve_AlwaysReturnsNull()
        {
            var a = np.array(new double[] { 1, 2, 3 });
            var v = np.array(new double[] { 0, 1, 0.5 });
            var result = np.convolve(a, v);

            result.Should().NotBeNull(
                "NumPy: np.convolve([1,2,3],[0,1,0.5]) = [0,1,2.5,4,1.5]. " +
                "NumSharp returns null because the Regen template in " +
                "Math/NdArray.Convolve.cs was never generated into the #else block.");
            result.size.Should().Be(5);
            result.GetDouble(0).Should().BeApproximately(0.0, 1e-10);
            result.GetDouble(1).Should().BeApproximately(1.0, 1e-10);
            result.GetDouble(2).Should().BeApproximately(2.5, 1e-10);
            result.GetDouble(3).Should().BeApproximately(4.0, 1e-10);
            result.GetDouble(4).Should().BeApproximately(1.5, 1e-10);
        }

        // ================================================================
        //
        //  BUG 33: np.any(axis) has inverted logic — implements all() instead
        //
        //  SEVERITY: High — completely wrong results if the throw is fixed.
        //
        //  Bug 22 (in OpenBugs.cs) covers the throw. This bug covers the LOGIC:
        //  ComputeAnyPerAxis initializes currentResult=true and sets it to
        //  true on break when finding a zero. This is all() logic, not any().
        //  For any(): should initialize false and set true on non-zero.
        //
        //  NOTE: Bug 22 must be fixed first (the throw) before this logic
        //  bug becomes visible. This test covers the correct final behavior.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.any([[False, False], [False, True]], axis=0)
        //    array([False,  True])
        //    >>> np.any([[False, False], [False, True]], axis=1)
        //    array([False,  True])
        //
        // ================================================================

        /// <summary>
        ///     BUG 33: np.any(axis=0) has inverted logic (implements all, not any).
        ///     Even if Bug 22's throw is fixed, the logic returns wrong results.
        /// </summary>
        [TestMethod]
        public void Bug_Any_Axis0_InvertedLogic_ImplementsAllInsteadOfAny()
        {
            var a = np.array(new bool[,] { { false, false }, { false, true } });

            NDArray result = null;
            new Action(() => result = np.any(a, 0, false))
                .Should().NotThrow("np.any(axis=0) should not throw");

            result.Should().NotBeNull();
            result.shape.Should().BeEquivalentTo(new[] { 2 });

            // any along axis=0: column 0=[F,F]->F, column 1=[F,T]->T
            result.GetBoolean(0).Should().BeFalse(
                "NumPy: np.any([[F,F],[F,T]], axis=0)[0] = False (no True in column 0). " +
                "ComputeAnyPerAxis in Logic/np.any.cs initializes currentResult=true " +
                "and detects zeros, which is all() logic. For any(), it should " +
                "initialize false and set true when finding non-zero.");
            result.GetBoolean(1).Should().BeTrue(
                "np.any axis=0 column 1=[F,T] -> True");
        }

        /// <summary>
        ///     BUG 33b: np.any keepdims=true should preserve dimensionality.
        /// </summary>
        [TestMethod]
        public void Bug_Any_Axis0_Keepdims()
        {
            var a = np.array(new bool[,] { { true, false }, { false, false } });

            NDArray result = null;
            new Action(() => result = np.any(a, 0, true))
                .Should().NotThrow();

            result.Should().NotBeNull();
            result.shape.Should().BeEquivalentTo(new[] { 1, 2 },
                "NumPy: np.any([[T,F],[F,F]], axis=0, keepdims=True).shape = (1,2)");
            result.GetBoolean(0, 0).Should().BeTrue("any of [T,F] along axis 0 -> T");
            result.GetBoolean(0, 1).Should().BeFalse("any of [F,F] along axis 0 -> F");
        }

        // ================================================================
        //
        //  BUG 34: np.isnan returns null (dead code)
        //
        //  SEVERITY: High — public API that crashes with NullReferenceException.
        //
        //  np.isnan calls DefaultEngine.IsNan which returns null.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.isnan([1.0, np.nan, np.inf])
        //    array([False,  True, False])
        //
        // ================================================================

        /// <summary>
        ///     BUG 34: np.isnan returns null. DefaultEngine.IsNan returns null.
        /// </summary>
        [TestMethod]
        public void Bug_IsNan_ReturnsNull()
        {
            var a = np.array(new[] { 1.0, double.NaN, double.PositiveInfinity, 0.0 });

            NDArray result = null;
            new Action(() => result = np.isnan(a))
                .Should().NotThrow(
                    "NumPy: np.isnan([1.0, NaN, inf, 0.0]) = [F, T, F, F]. " +
                    "NumSharp: DefaultEngine.IsNan returns null, causing NullReferenceException.");

            result.Should().NotBeNull();
            result.GetBoolean(0).Should().BeFalse("1.0 is not NaN");
            result.GetBoolean(1).Should().BeTrue("NaN is NaN");
            result.GetBoolean(2).Should().BeFalse("inf is not NaN");
            result.GetBoolean(3).Should().BeFalse("0.0 is not NaN");
        }

        // ================================================================
        //
        //  BUG 35: np.isfinite returns null (dead code)
        //
        //  SEVERITY: High — public API that crashes with NullReferenceException.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.isfinite([1.0, np.nan, np.inf, -np.inf, 0.0])
        //    array([ True, False, False, False,  True])
        //
        // ================================================================

        /// <summary>
        ///     BUG 35: np.isfinite returns null. DefaultEngine.IsFinite returns null.
        /// </summary>
        [TestMethod]
        public void Bug_IsFinite_ReturnsNull()
        {
            var a = np.array(new[] { 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 });

            NDArray result = null;
            new Action(() => result = np.isfinite(a))
                .Should().NotThrow(
                    "NumPy: np.isfinite([1, NaN, inf, -inf, 0]) = [T, F, F, F, T]. " +
                    "NumSharp: DefaultEngine.IsFinite returns null.");

            result.Should().NotBeNull();
            result.GetBoolean(0).Should().BeTrue("1.0 is finite");
            result.GetBoolean(1).Should().BeFalse("NaN is not finite");
            result.GetBoolean(2).Should().BeFalse("inf is not finite");
            result.GetBoolean(3).Should().BeFalse("-inf is not finite");
            result.GetBoolean(4).Should().BeTrue("0.0 is finite");
        }

        // ================================================================
        //
        //  BUG 36: np.isclose returns null (dead code)
        //
        //  SEVERITY: High — blocks np.allclose which depends on it (Bug 7).
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.isclose([1.0, 1.0001], [1.0, 1.0002], atol=1e-3)
        //    array([ True,  True])
        //
        // ================================================================

        /// <summary>
        ///     BUG 36: np.isclose returns null. DefaultEngine.IsClose returns null.
        /// </summary>
        [TestMethod]
        public void Bug_IsClose_ReturnsNull()
        {
            var a = np.array(new[] { 1.0, 1.0001 });
            var b = np.array(new[] { 1.0, 1.0002 });

            NDArray result = null;
            new Action(() => result = np.isclose(a, b, atol: 1e-3))
                .Should().NotThrow(
                    "NumPy: np.isclose([1.0, 1.0001], [1.0, 1.0002], atol=1e-3) = [T, T]. " +
                    "NumSharp: DefaultEngine.IsClose returns null, causing NullReferenceException. " +
                    "This also blocks np.allclose (Bug 7) which depends on isclose.");

            result.Should().NotBeNull();
            result.GetBoolean(0).Should().BeTrue("1.0 ~ 1.0 within atol=1e-3");
            result.GetBoolean(1).Should().BeTrue("1.0001 ~ 1.0002 within atol=1e-3");
        }

        // ================================================================
        //
        //  BUG 37: operator & (AND) returns null (dead code)
        //
        //  SEVERITY: High — public operator that silently returns null.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.array([True, False, True]) & np.array([True, True, False])
        //    array([ True, False, False])
        //
        // ================================================================

        /// <summary>
        ///     BUG 37: NDArray &amp; NDArray returns null.
        ///     The AND operator in NDArray.AND.cs returns null.
        /// </summary>
        [TestMethod]
        public void Bug_AND_Operator_ReturnsNull()
        {
            var a = np.array(new[] { true, false, true });
            var b = np.array(new[] { true, true, false });

            NDArray result = null;
            new Action(() => result = a & b)
                .Should().NotThrow(
                    "NumPy: [T,F,T] & [T,T,F] = [T,F,F]. " +
                    "NumSharp: operator & in NDArray.AND.cs returns null.");

            result.Should().NotBeNull();
            result.GetBoolean(0).Should().BeTrue("T & T = T");
            result.GetBoolean(1).Should().BeFalse("F & T = F");
            result.GetBoolean(2).Should().BeFalse("T & F = F");
        }

        // ================================================================
        //
        //  BUG 38: operator | (OR) returns null (dead code)
        //
        //  SEVERITY: High — public operator that silently returns null.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.array([True, False, False]) | np.array([False, False, True])
        //    array([ True, False,  True])
        //
        // ================================================================

        /// <summary>
        ///     BUG 38: NDArray | NDArray returns null.
        ///     The OR operator in NDArray.OR.cs returns null.
        /// </summary>
        [TestMethod]
        public void Bug_OR_Operator_ReturnsNull()
        {
            var a = np.array(new[] { true, false, false });
            var b = np.array(new[] { false, false, true });

            NDArray result = null;
            new Action(() => result = a | b)
                .Should().NotThrow(
                    "NumPy: [T,F,F] | [F,F,T] = [T,F,T]. " +
                    "NumSharp: operator | in NDArray.OR.cs returns null.");

            result.Should().NotBeNull();
            result.GetBoolean(0).Should().BeTrue("T | F = T");
            result.GetBoolean(1).Should().BeFalse("F | F = F");
            result.GetBoolean(2).Should().BeTrue("F | T = T");
        }

        // ================================================================
        //
        //  BUG 39: nd.delete() returns null (dead code)
        //
        //  SEVERITY: Medium — public method that silently returns null.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.delete(np.array([1,2,3,4,5]), [1,3])
        //    array([1, 3, 5])
        //
        // ================================================================

        /// <summary>
        ///     BUG 39: nd.delete() returns null.
        ///     NdArray.delete.cs always returns null.
        /// </summary>
        [TestMethod]
        public void Bug_Delete_ReturnsNull()
        {
            var a = np.array(new[] { 1, 2, 3, 4, 5 });
            var result = a.delete(new[] { 1 });

            result.Should().NotBeNull(
                "NumPy: np.delete([1,2,3,4,5], [1]) = [1,3,4,5]. " +
                "NumSharp: NdArray.delete.cs always returns null.");
            result.size.Should().Be(4);
            result.GetInt32(0).Should().Be(1);
            result.GetInt32(1).Should().Be(3);
            result.GetInt32(2).Should().Be(4);
            result.GetInt32(3).Should().Be(5);
        }

        // ================================================================
        //
        //  BUG 40: nd.inv() returns null (dead code)
        //
        //  SEVERITY: High — public method that silently returns null.
        //  The entire LAPACK-based implementation is commented out.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.linalg.inv(np.array([[1,2],[3,4]]))
        //    array([[-2. ,  1. ],
        //           [ 1.5, -0.5]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 40: nd.inv() returns null. Implementation commented out.
        /// </summary>
        [TestMethod]
        public void Bug_Inv_ReturnsNull()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var result = a.inv();

            result.Should().NotBeNull(
                "NumPy: np.linalg.inv([[1,2],[3,4]]) = [[-2,1],[1.5,-0.5]]. " +
                "NumSharp: NdArray.Inv.cs always returns null (implementation commented out).");
        }

        // ================================================================
        //
        //  BUG 41: nd.qr() returns default (null, null)
        //
        //  SEVERITY: High — public method returning null tuple.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> q, r = np.linalg.qr(np.array([[1,2],[3,4]]))
        //    >>> q.shape, r.shape
        //    ((2, 2), (2, 2))
        //
        // ================================================================

        /// <summary>
        ///     BUG 41: nd.qr() returns default (null, null).
        /// </summary>
        [TestMethod]
        public void Bug_QR_ReturnsDefault()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var (q, r) = a.qr();

            q.Should().NotBeNull(
                "NumPy: np.linalg.qr([[1,2],[3,4]]) returns (Q, R) matrices. " +
                "NumSharp: NdArray.QR.cs returns default which is (null, null).");
            r.Should().NotBeNull();
        }

        // ================================================================
        //
        //  BUG 42: nd.svd() returns default (null, null, null)
        //
        //  SEVERITY: High — public method returning null tuple.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> u, s, vh = np.linalg.svd(np.array([[1,2],[3,4]]))
        //    >>> u.shape, s.shape, vh.shape
        //    ((2, 2), (2,), (2, 2))
        //
        // ================================================================

        /// <summary>
        ///     BUG 42: nd.svd() returns default (null, null, null).
        /// </summary>
        [TestMethod]
        public void Bug_SVD_ReturnsDefault()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var (u, s, vh) = a.svd();

            u.Should().NotBeNull(
                "NumPy: np.linalg.svd([[1,2],[3,4]]) returns (U, S, Vh). " +
                "NumSharp: NdArray.SVD.cs returns default which is (null, null, null).");
            s.Should().NotBeNull();
            vh.Should().NotBeNull();
        }

        // ================================================================
        //
        //  BUG 43: nd.lstqr() returns null + misspelled (should be lstsq)
        //
        //  SEVERITY: High — public method returns null AND has wrong name.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> a = np.array([[1,1],[1,2],[1,3]])
        //    >>> b = np.array([1,2,3])
        //    >>> np.linalg.lstsq(a, b, rcond=None)[0]
        //    array([0., 1.])
        //
        // ================================================================

        /// <summary>
        ///     BUG 43: nd.lstqr() returns null. Also misspelled: should be lstsq.
        /// </summary>
        [TestMethod]
        public void Bug_Lstsq_ReturnsNull_AndMisspelled()
        {
            var a = np.array(new double[,] { { 1, 1 }, { 1, 2 }, { 1, 3 } });
            var b = np.array(new double[] { 1, 2, 3 });

            // Note: method is misspelled as "lstqr" instead of "lstsq"
            var result = a.lstqr(b);

            result.Should().NotBeNull(
                "NumPy: np.linalg.lstsq(a, b) returns least-squares solution. " +
                "NumSharp: NdArray.LstSq.cs method 'lstqr' (misspelled, should be " +
                "'lstsq') always returns null (implementation commented out).");
        }

        // ================================================================
        //
        //  BUG 44: nd.multi_dot() returns null (dead code)
        //
        //  SEVERITY: Medium — public method returns null.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.linalg.multi_dot([np.eye(2), np.eye(2)])
        //    array([[1., 0.],
        //           [0., 1.]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 44: nd.multi_dot() returns null.
        /// </summary>
        [TestMethod]
        public void Bug_MultiDot_ReturnsNull()
        {
            var a = np.eye(2);
            var b = np.eye(2);
            var result = a.multi_dot(b);

            result.Should().NotBeNull(
                "NumPy: np.linalg.multi_dot([eye(2), eye(2)]) = eye(2). " +
                "NumSharp: NdArray.multi_dot.cs always returns null.");
        }

        // ================================================================
        //
        //  BUG 45: nd.roll(shift) no-axis overload returns null
        //
        //  SEVERITY: Medium — public method returns null.
        //
        //  The no-axis overload at NDArray.roll.cs:70 has its body
        //  commented out and returns null. The with-axis overload
        //  partially works (Int32/Single/Double only).
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.roll(np.array([1,2,3,4,5]), 2)
        //    array([4, 5, 1, 2, 3])
        //
        // ================================================================

        /// <summary>
        ///     BUG 45: nd.roll(shift) without axis returns null.
        ///     The no-axis overload body is commented out.
        /// </summary>
        [TestMethod]
        public void Bug_Roll_NoAxis_ReturnsNull()
        {
            var a = np.array(new[] { 1, 2, 3, 4, 5 });
            var result = a.roll(2);

            result.Should().NotBeNull(
                "NumPy: np.roll([1,2,3,4,5], 2) = [4,5,1,2,3]. " +
                "NumSharp: NDArray.roll(int shift) at NDArray.roll.cs:70 " +
                "has its body commented out and returns null.");
            result.size.Should().Be(5);
            result.GetInt32(0).Should().Be(4);
            result.GetInt32(1).Should().Be(5);
            result.GetInt32(2).Should().Be(1);
        }

        // ================================================================
        //
        //  BUG 46: Boolean mask setter throws NotImplementedException
        //
        //  SEVERITY: High — fundamental indexing operation broken.
        //
        //  a[mask] = value is a core NumPy operation (e.g., a[a > 5] = 0).
        //  The getter works, but the setter at NDArray.Indexing.Masking.cs:26
        //  throws NotImplementedException.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> a = np.array([1,2,3,4,5])
        //    >>> a[a > 3] = 0
        //    >>> a
        //    array([1, 2, 3, 0, 0])
        //
        // ================================================================

        /// <summary>
        ///     BUG 46: Boolean mask setter throws NotImplementedException.
        ///     a[mask] = value is a fundamental NumPy operation.
        /// </summary>
        [TestMethod]
        public void Bug_BooleanMaskSetter_ThrowsNotImplemented()
        {
            var a = np.array(new[] { 1, 2, 3, 4, 5 });
            var mask = a > np.array(new[] { 3, 3, 3, 3, 3 });

            new Action(() => a[mask] = np.array(new[] { 0 }))
                .Should().NotThrow(
                    "NumPy: a = [1,2,3,4,5]; a[a>3] = 0 -> [1,2,3,0,0]. " +
                    "NumSharp: Boolean mask setter at NDArray.Indexing.Masking.cs:26 " +
                    "throws NotImplementedException.");

            a.GetInt32(0).Should().Be(1);
            a.GetInt32(1).Should().Be(2);
            a.GetInt32(2).Should().Be(3);
            a.GetInt32(3).Should().Be(0, "element > 3 should be set to 0");
            a.GetInt32(4).Should().Be(0, "element > 3 should be set to 0");
        }

        // ================================================================
        //
        //  BUG 47: np.positive() implements abs() instead of +x identity
        //
        //  SEVERITY: Medium — wrong semantics, silently returns wrong values.
        //
        //  NumPy: np.positive(x) is equivalent to +x (identity for numeric).
        //  NumSharp: Takes absolute value of negative numbers (implements
        //  abs instead of positive). Also flips booleans (logical NOT).
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.positive(np.array([-3, -1, 0, 2, 5]))
        //    array([-3, -1,  0,  2,  5])
        //
        // ================================================================

        /// <summary>
        ///     BUG 47: np.positive implements abs() instead of +x identity.
        ///     NumPy: positive([-3,-1,0,2,5]) = [-3,-1,0,2,5] (unchanged).
        ///     NumSharp: [3,1,0,2,5] (takes absolute value of negatives).
        /// </summary>
        [TestMethod]
        public void Bug_Positive_ImplementsAbsInsteadOfIdentity()
        {
            var a = np.array(new[] { -3, -1, 0, 2, 5 });
            var result = np.positive(a);

            result.GetInt32(0).Should().Be(-3,
                "NumPy: np.positive([-3,-1,0,2,5])[0] = -3 (identity, unchanged). " +
                "NumSharp returns 3 because NDArray.positive.cs implements abs() " +
                "instead of the identity +x operation. The code has " +
                "'if (val < 0) out_addr[i] = -val' which is absolute value.");
            result.GetInt32(1).Should().Be(-1, "positive(-1) = -1 (identity)");
            result.GetInt32(2).Should().Be(0, "positive(0) = 0");
            result.GetInt32(3).Should().Be(2, "positive(2) = 2");
            result.GetInt32(4).Should().Be(5, "positive(5) = 5");
        }

        // ================================================================
        //
        //  BUG 48: np.negative() only negates positive values
        //
        //  SEVERITY: Medium — wrong semantics, silently returns wrong values.
        //
        //  NumPy: np.negative(x) is -x for ALL elements.
        //  NumSharp: Only negates positive values, leaves negatives unchanged.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.negative(np.array([-3, -1, 0, 2, 5]))
        //    array([ 3,  1,  0, -2, -5])
        //
        // ================================================================

        /// <summary>
        ///     BUG 48: np.negative only negates positive values, not all.
        ///     NumPy: negative([-3,-1,0,2,5]) = [3,1,0,-2,-5].
        ///     NumSharp: [-3,-1,0,-2,-5] (negatives left unchanged).
        /// </summary>
        [TestMethod]
        public void Bug_Negative_OnlyNegatesPositiveValues()
        {
            var a = np.array(new[] { -3, -1, 0, 2, 5 });
            var result = np.negative(a);

            result.GetInt32(0).Should().Be(3,
                "NumPy: np.negative(-3) = 3. NumSharp returns -3 because " +
                "NDArray.negative.cs has 'if (val > 0) out_addr[i] = -val' " +
                "which only negates positive values, leaving negatives unchanged.");
            result.GetInt32(1).Should().Be(1, "negative(-1) = 1");
            result.GetInt32(2).Should().Be(0, "negative(0) = 0");
            result.GetInt32(3).Should().Be(-2, "negative(2) = -2");
            result.GetInt32(4).Should().Be(-5, "negative(5) = -5");
        }

        // ================================================================
        //
        //  BUG 49: np.all(axis) throws InvalidCastException
        //
        //  SEVERITY: High — axis reduction broken due to cast bug.
        //
        //  np.all(NDArray, int axis) in np.all.cs:89 casts
        //  zeros<bool>(outputShape) to NDArray<bool>, but zeros<bool>
        //  returns NDArray (not NDArray<bool>), causing InvalidCastException.
        //  The implementation logic itself (ComputeAllPerAxis) is correct
        //  and handles all 12 dtypes, but can never be reached.
        //
        //  NOTE: DefaultEngine.All(NDArray, int axis) at Default.All.cs:44
        //  also throws NotImplementedException, but np.all(axis) doesn't
        //  call DefaultEngine — it has its own inline implementation.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.all([[True, False], [True, True]], axis=0)
        //    array([ True, False])
        //
        // ================================================================

        /// <summary>
        ///     BUG 49: np.all(axis) throws InvalidCastException.
        ///     The cast (NDArray&lt;bool&gt;)zeros&lt;bool&gt;(outputShape) at np.all.cs:89
        ///     fails because zeros returns NDArray, not NDArray&lt;bool&gt;.
        /// </summary>
        [TestMethod]
        public void Bug_All_WithAxis_ThrowsInvalidCast()
        {
            var a = np.array(new bool[,] { { true, false }, { true, true } });

            NDArray result = null;
            new Action(() => result = np.all(a, 0))
                .Should().NotThrow(
                    "NumPy: np.all([[T,F],[T,T]], axis=0) = [True, False]. " +
                    "NumSharp: np.all(axis) at np.all.cs:89 throws InvalidCastException " +
                    "because (NDArray<bool>)zeros<bool>(outputShape) fails — " +
                    "zeros<bool> returns NDArray, not NDArray<bool>.");

            result.Should().NotBeNull();
            result.shape.Should().BeEquivalentTo(new[] { 2 });
            result.GetBoolean(0).Should().BeTrue("all of [T,T] along axis 0 -> T");
            result.GetBoolean(1).Should().BeFalse("all of [F,T] along axis 0 -> F");
        }

        // ================================================================
        //
        //  BUG 50: nd.roll(shift, axis) only supports 3 of 12 dtypes
        //
        //  SEVERITY: Medium — throws NotImplementedException for most types.
        //
        //  NDArray.roll(int shift, int axis) only handles Int32, Single,
        //  Double. All other 9 supported dtypes (Boolean, Byte, Int16,
        //  UInt16, UInt32, Int64, UInt64, Char, Decimal) throw.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.roll(np.array([1,2,3], dtype=np.int64), 1)
        //    array([3, 1, 2])
        //    >>> np.roll(np.array([1,2,3], dtype=np.uint8), 1)
        //    array([3, 1, 2], dtype=uint8)
        //
        // ================================================================

        /// <summary>
        ///     BUG 50a: nd.roll on int64 throws NotImplementedException.
        /// </summary>
        [TestMethod]
        public void Bug_Roll_Int64_ThrowsNotImplemented()
        {
            var a = np.array(new long[] { 10, 20, 30 });

            NDArray result = null;
            new Action(() => result = a.roll(1, 0))
                .Should().NotThrow(
                    "NumPy: np.roll(int64_array, 1, axis=0) works for all dtypes. " +
                    "NumSharp: NDArray.roll.cs only handles Int32/Single/Double; " +
                    "all other dtypes throw NotImplementedException.");

            result.Should().NotBeNull();
            result.GetInt64(0).Should().Be(30);
            result.GetInt64(1).Should().Be(10);
            result.GetInt64(2).Should().Be(20);
        }

        /// <summary>
        ///     BUG 50b: nd.roll on byte (uint8) throws NotImplementedException.
        /// </summary>
        [TestMethod]
        public void Bug_Roll_Byte_ThrowsNotImplemented()
        {
            var a = np.array(new byte[] { 1, 2, 3 });

            NDArray result = null;
            new Action(() => result = a.roll(1, 0))
                .Should().NotThrow(
                    "NumPy: np.roll(uint8_array, 1, axis=0) works for all dtypes. " +
                    "NumSharp throws NotImplementedException for byte arrays.");

            result.Should().NotBeNull();
            result.GetByte(0).Should().Be(3);
            result.GetByte(1).Should().Be(1);
            result.GetByte(2).Should().Be(2);
        }

        // ================================================================
        //
        //  BUG 51: np.log1p computes log10(1+x) instead of ln(1+x)
        //
        //  SEVERITY: Critical — returns completely wrong mathematical result.
        //
        //  LOCATION: Default.Log1p.cs
        //    Line 10: Log1p(...) => Log10(nd, dtype?.GetTypeCode());
        //    Lines 42-101: All branches use Math.Log10() instead of Math.Log()
        //
        //  The entire implementation delegates to Log10 instead of Log.
        //  log1p(e-1) should return 1.0 (natural log), but returns 0.434
        //  (log base 10).
        //
        //  PYTHON VERIFICATION:
        //    >>> np.log1p(np.e - 1)
        //    1.0
        //    >>> np.log1p(0)
        //    0.0
        //    >>> np.log1p(1)
        //    0.6931471805599453  (ln(2))
        //
        // ================================================================

        /// <summary>
        ///     BUG 51: np.log1p computes log10(1+x) instead of ln(1+x).
        ///
        ///     NumPy:    log1p(e-1) = 1.0
        ///     NumSharp: log1p(e-1) = 0.434 (log10)
        /// </summary>
        [TestMethod]
        public void Bug_Log1p_UsesLog10_InsteadOfNaturalLog()
        {
            var result = np.log1p(np.array(new double[] { Math.E - 1 }));
            result.GetDouble(0).Should().BeApproximately(1.0, 0.001,
                "np.log1p(e-1) = ln(e) = 1.0. NumSharp returns 0.434 because " +
                "Default.Log1p.cs delegates to Log10 instead of Log. " +
                "Line 10: Log1p(...) => Log10(...). All branches use Math.Log10().");
        }

        // ================================================================
        //
        //  BUG 52: np.std and np.var ignore the ddof parameter
        //
        //  SEVERITY: High — sample std/var (ddof=1) always returns
        //  population std/var (ddof=0). Extremely common in statistics.
        //
        //  ddof=1 is the standard unbiased estimator used in almost all
        //  statistical analysis. Returning the population std when the
        //  user explicitly requests sample std produces wrong results.
        //
        //  PYTHON VERIFICATION:
        //    >>> a = np.array([2, 4, 4, 4, 5, 5, 7, 9])
        //    >>> np.std(a, ddof=0)
        //    2.0
        //    >>> np.std(a, ddof=1)
        //    2.1380899352993952
        //    >>> np.var(a, ddof=0)
        //    4.0
        //    >>> np.var(a, ddof=1)
        //    4.571428571428571
        //
        // ================================================================

        /// <summary>
        ///     BUG 52a: std(ddof=1) returns population std instead of sample std.
        ///
        ///     NumPy:    std(ddof=1) = 2.138
        ///     NumSharp: std(ddof=1) = 2.0 (same as ddof=0)
        /// </summary>
        [TestMethod]
        public void Bug_Std_IgnoresDdof()
        {
            var a = np.array(new double[] { 2, 4, 4, 4, 5, 5, 7, 9 });

            var pop = (double)np.std(a);
            pop.Should().BeApproximately(2.0, 0.001, "population std (ddof=0) = 2.0");

            var sample = (double)np.std(a, ddof: 1);
            sample.Should().BeApproximately(2.138, 0.01,
                "sample std (ddof=1) should be 2.138 (divides by N-1=7 not N=8). " +
                "NumSharp returns 2.0 because the ddof parameter is accepted but " +
                "never used in the calculation.");
        }

        /// <summary>
        ///     BUG 52b: var(ddof=1) returns population var instead of sample var.
        /// </summary>
        [TestMethod]
        public void Bug_Var_IgnoresDdof()
        {
            var a = np.array(new double[] { 2, 4, 4, 4, 5, 5, 7, 9 });

            var sample = (double)np.var(a, ddof: 1);
            sample.Should().BeApproximately(4.571, 0.01,
                "sample var (ddof=1) should be 4.571 (divides by N-1=7). " +
                "NumSharp returns 4.0 because ddof is ignored.");
        }

        // ================================================================
        //
        //  BUG 53: np.searchsorted returns wrong insertion indices
        //
        //  SEVERITY: High — the binary search algorithm is fundamentally
        //  broken. Results are wrong, not just off-by-one.
        //
        //  searchsorted([1,3,5,7], [2]) should return [1] (insert at
        //  index 1 to maintain sort order). Returns [3] instead.
        //
        //  Also, side='right' is not implemented.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.searchsorted([1,3,5,7], 2)
        //    1
        //    >>> np.searchsorted([1,3,5,7], [0,1,2,3,4,5,6,7,8])
        //    array([0, 0, 1, 1, 2, 2, 3, 3, 4])
        //
        // ================================================================

        /// <summary>
        ///     BUG 53: searchsorted returns wrong indices.
        ///
        ///     NumPy:    searchsorted([1,3,5,7], [2]) = [1]
        ///     NumSharp: returns [3]
        /// </summary>
        [TestMethod]
        public void Bug_Searchsorted_WrongIndices()
        {
            var sorted = np.array(new int[] { 1, 3, 5, 7 });
            var result = np.searchsorted(sorted, np.array(new int[] { 2 }));

            result.GetInt32(0).Should().Be(1,
                "searchsorted([1,3,5,7], 2) should return 1 (insert before 3). " +
                "NumSharp returns 3. The binary search algorithm is wrong.");
        }

        // ================================================================
        //
        //  BUG 54: np.moveaxis fails with negative axis values
        //
        //  SEVERITY: High — negative axis is a very common Python idiom.
        //
        //  LOCATION: Default.Transpose.cs, normalize_axis_tuple()
        //  The method normalizes axes against axis.Length (which is 1 for
        //  single-axis moveaxis) instead of against ndim. So moveaxis(a, 0, -1)
        //  on a 3D array normalizes -1 against length 1, giving 0, making
        //  the operation a no-op.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.moveaxis(np.arange(24).reshape(2,3,4), 0, -1).shape
        //    (3, 4, 2)
        //    >>> np.moveaxis(np.arange(24).reshape(2,3,4), -1, 0).shape
        //    (4, 2, 3)
        //
        // ================================================================

        /// <summary>
        ///     BUG 54: moveaxis with negative destination is a no-op.
        ///
        ///     NumPy:    moveaxis(shape(2,3,4), 0, -1).shape = (3,4,2)
        ///     NumSharp: returns (2,3,4) unchanged
        /// </summary>
        [TestMethod]
        public void Bug_Moveaxis_NegativeAxis_NoOp()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.moveaxis(a, 0, -1);

            result.shape[0].Should().Be(3,
                "moveaxis(shape(2,3,4), src=0, dst=-1) should move axis 0 to the end. " +
                "Result shape should be (3,4,2). NumSharp returns (2,3,4) because " +
                "normalize_axis_tuple resolves -1 against axis.Length (1) not ndim (3).");
            result.shape[1].Should().Be(4);
            result.shape[2].Should().Be(2);
        }

        // ================================================================
        //
        //  BUG 55: np.mean crashes on empty arrays
        //
        //  SEVERITY: Medium — NumPy returns NaN with a RuntimeWarning.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.mean(np.array([]))
        //    nan  (with RuntimeWarning: Mean of empty slice)
        //
        // ================================================================

        /// <summary>
        ///     BUG 55: np.mean on empty array crashes instead of returning NaN.
        ///
        ///     NumPy:    mean([]) = NaN (with warning)
        ///     NumSharp: InvalidOperationException: Can't construct NDIterator
        /// </summary>
        [TestMethod]
        public void Bug_Mean_EmptyArray_Crashes()
        {
            var empty = np.array(new double[0]);

            NDArray result = null;
            new Action(() => result = np.mean(empty))
                .Should().NotThrow(
                    "NumPy: mean of empty array returns NaN with RuntimeWarning. " +
                    "NumSharp throws InvalidOperationException because NDIterator " +
                    "cannot handle empty shapes.");
        }

        // ================================================================
        //
        //  BUG 56: np.abs returns Double dtype for integer input
        //
        //  SEVERITY: Medium — dtype should be preserved.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.abs(np.array([-1, 2, -3], dtype=np.int32)).dtype
        //    dtype('int32')
        //
        // ================================================================

        /// <summary>
        ///     BUG 56: np.abs returns Double instead of preserving int dtype.
        ///
        ///     NumPy:    abs(int32[]) -> int32
        ///     NumSharp: abs(int32[]) -> Double
        /// </summary>
        [TestMethod]
        public void Bug_Abs_ReturnsDouble_ForIntInput()
        {
            var result = np.abs(np.array(new int[] { -1, 2, -3 }));

            result.GetInt32(0).Should().Be(1, "abs(-1) = 1");
            result.GetInt32(1).Should().Be(2, "abs(2) = 2");
            result.GetInt32(2).Should().Be(3, "abs(-3) = 3");
            result.dtype.Should().Be(typeof(int),
                "NumPy preserves the input dtype. np.abs(int32[]) returns int32. " +
                "NumSharp returns Double, which forces downstream code to cast.");
        }

        // ================================================================
        //
        //  BUG 57: np.sum/np.mean/np.prod crash on boolean arrays
        //
        //  SEVERITY: Medium — boolean reductions are common for counting
        //  elements matching a condition (e.g., np.sum(arr > threshold)).
        //
        //  PYTHON VERIFICATION:
        //    >>> np.sum(np.array([True, False, True, True]))
        //    3
        //    >>> np.mean(np.array([True, False, True, True]))
        //    0.75
        //
        // ================================================================

        /// <summary>
        ///     BUG 57a: np.sum on boolean array throws NotSupportedException.
        ///
        ///     NumPy:    sum([T, F, T, T]) = 3
        ///     NumSharp: NotSupportedException
        /// </summary>
        [TestMethod]
        public void Bug_Sum_BoolArray_Crashes()
        {
            var a = np.array(new bool[] { true, false, true, true });

            NDArray result = null;
            new Action(() => result = np.sum(a))
                .Should().NotThrow(
                    "NumPy: sum([True, False, True, True]) = 3. NumSharp throws " +
                    "NotSupportedException because boolean is not handled in the " +
                    "reduction type switch.");
        }

        /// <summary>
        ///     BUG 57b: np.mean on boolean array crashes.
        ///
        ///     NumPy:    mean([T, F, T, T]) = 0.75
        ///     NumSharp: crashes
        /// </summary>
        [TestMethod]
        public void Bug_Mean_BoolArray_Crashes()
        {
            var a = np.array(new bool[] { true, false, true, true });

            NDArray result = null;
            new Action(() => result = np.mean(a))
                .Should().NotThrow(
                    "NumPy: mean([True, False, True, True]) = 0.75. " +
                    "NumSharp crashes because boolean reductions are not supported.");
        }

        // ================================================================
        //
        //  BUG 58: astype(int32) rounds to nearest instead of truncating
        //
        //  SEVERITY: Medium — NumPy truncates toward zero (C-style cast).
        //  NumSharp uses Convert.ToInt32 which does banker's rounding.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.array([1.7, 2.3, -1.7, -2.3]).astype(np.int32)
        //    array([ 1,  2, -1, -2])
        //
        // ================================================================

        /// <summary>
        ///     BUG 58: astype(int32) rounds instead of truncating toward zero.
        ///
        ///     NumPy:    [1.7, 2.3, -1.7].astype(int32) = [1, 2, -1]
        ///     NumSharp: [2, 2, -2] (banker's rounding)
        /// </summary>
        [TestMethod]
        public void Bug_Astype_Int32_Rounds_InsteadOfTruncating()
        {
            var result = np.array(new double[] { 1.7, 2.3, -1.7, -2.3 }).astype(np.int32);

            result.GetInt32(0).Should().Be(1,
                "NumPy truncates 1.7 toward zero -> 1. " +
                "NumSharp returns 2 (rounds to nearest via Convert.ToInt32).");
            result.GetInt32(1).Should().Be(2, "2.3 truncated -> 2");
            result.GetInt32(2).Should().Be(-1, "-1.7 truncated toward zero -> -1 (not -2)");
            result.GetInt32(3).Should().Be(-2, "-2.3 truncated -> -2");
        }

        // ================================================================
        //
        //  BUG 59: >= and <= operators throw IncorrectShapeException
        //
        //  SEVERITY: High — these basic comparison operators don't work
        //  even with scalar right-hand operands.
        //
        //  Unlike > and < (which work with scalar int), >= and <= throw
        //  IncorrectShapeException for all inputs.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.arange(5) >= 2
        //    array([False, False,  True,  True,  True])
        //    >>> np.arange(5) <= 2
        //    array([ True,  True,  True, False, False])
        //
        // ================================================================

        /// <summary>
        ///     BUG 59a: >= operator throws IncorrectShapeException.
        ///
        ///     NumPy:    arange(5) >= 2 = [F, F, T, T, T]
        ///     NumSharp: IncorrectShapeException
        /// </summary>
        [TestMethod]
        public void Bug_GreaterOrEqual_Scalar_Crashes()
        {
            var a = np.arange(5);

            NDArray result = null;
            new Action(() => result = a >= 2)
                .Should().NotThrow(
                    "NumPy: arange(5) >= 2 returns [F, F, T, T, T]. " +
                    "NumSharp throws IncorrectShapeException.");

            result.Should().NotBeNull();
            result.GetBoolean(0).Should().BeFalse("0 >= 2 is false");
            result.GetBoolean(2).Should().BeTrue("2 >= 2 is true");
            result.GetBoolean(4).Should().BeTrue("4 >= 2 is true");
        }

        /// <summary>
        ///     BUG 59b: &lt;= operator throws IncorrectShapeException.
        ///
        ///     NumPy:    arange(5) &lt;= 2 = [T, T, T, F, F]
        ///     NumSharp: IncorrectShapeException
        /// </summary>
        [TestMethod]
        public void Bug_LessOrEqual_Scalar_Crashes()
        {
            var a = np.arange(5);

            NDArray result = null;
            new Action(() => result = a <= 2)
                .Should().NotThrow(
                    "NumPy: arange(5) <= 2 returns [T, T, T, F, F]. " +
                    "NumSharp throws IncorrectShapeException.");

            result.Should().NotBeNull();
            result.GetBoolean(0).Should().BeTrue("0 <= 2 is true");
            result.GetBoolean(2).Should().BeTrue("2 <= 2 is true");
            result.GetBoolean(4).Should().BeFalse("4 <= 2 is false");
        }

        // ================================================================
        //
        //  BUG 60: np.argmax ignores NaN values
        //
        //  SEVERITY: Medium — NumPy propagates NaN in comparisons, so
        //  argmax returns the index of the first NaN. NumSharp skips NaN
        //  and returns the index of the actual maximum.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.argmax(np.array([1.0, np.nan, 3.0]))
        //    1
        //
        // ================================================================

        /// <summary>
        ///     BUG 60: argmax ignores NaN, should return index of first NaN.
        ///
        ///     NumPy:    argmax([1.0, NaN, 3.0]) = 1 (NaN index)
        ///     NumSharp: returns 2 (index of 3.0)
        /// </summary>
        [TestMethod]
        public void Bug_Argmax_IgnoresNaN()
        {
            var a = np.array(new double[] { 1.0, double.NaN, 3.0 });
            var result = np.argmax(a);

            ((int)result).Should().Be(1,
                "NumPy: argmax([1, NaN, 3]) = 1 because NaN propagates in " +
                "comparisons (NaN > x is always True in NumPy's argmax). " +
                "NumSharp returns 2, ignoring the NaN.");
        }

        // ================================================================
        //
        //  BUG 61: np.linspace returns float32 instead of float64
        //
        //  SEVERITY: Medium — NumPy always returns float64 by default.
        //  NumSharp returns float32 (Single) when given integer arguments.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.linspace(0, 1, 5).dtype
        //    dtype('float64')
        //
        // ================================================================

        /// <summary>
        ///     BUG 61: linspace returns float32 instead of float64.
        ///
        ///     NumPy:    linspace(0, 1, 5).dtype = float64
        ///     NumSharp: System.Single (float32)
        /// </summary>
        [TestMethod]
        public void Bug_Linspace_ReturnsFloat32()
        {
            var result = np.linspace(0, 1, 5);

            result.dtype.Should().Be(typeof(double),
                "NumPy: linspace always returns float64 by default. " +
                "NumSharp returns float32 (System.Single) when given integer " +
                "arguments, which loses precision.");
        }

        // ================================================================
        //
        //  BUG 62: Implicit conversion operators crash across dtypes
        //
        //  SEVERITY: High — (double)ndarray where ndarray is float32
        //  throws IncorrectShapeException instead of converting the value.
        //  The operator uses GetAtIndex<T> which reads raw bytes as type T
        //  without dtype conversion. For scalar arrays (ndim=0), it throws
        //  because NumSharp creates shape (1,) instead of () for scalars.
        //
        //  PYTHON VERIFICATION:
        //    >>> float(np.array(3.14, dtype=np.float32))
        //    3.140000104904175
        //    >>> float(np.array(42, dtype=np.int32))
        //    42.0
        //
        // ================================================================

        /// <summary>
        ///     BUG 62: Implicit conversion (double) on float32 NDArray crashes.
        ///
        ///     NumPy:    float(float32_array) works
        ///     NumSharp: IncorrectShapeException
        /// </summary>
        [TestMethod]
        public void Bug_ImplicitConversion_CrossDtype_Crashes()
        {
            var f32 = np.array(new float[] { 3.14f });

            double val = 0;
            new Action(() => val = (double)(NDArray)f32)
                .Should().NotThrow(
                    "Converting a scalar float32 NDArray to double should work. " +
                    "NumSharp throws IncorrectShapeException because np.array(scalar) " +
                    "creates shape (1,) not (), and the implicit operator requires ndim=0.");
        }

        // ================================================================
        //
        //  BUG 63: ToString crashes on empty arrays
        //
        //  SEVERITY: Low — should print "[]" like NumPy.
        //
        //  PYTHON VERIFICATION:
        //    >>> str(np.array([]))
        //    '[]'
        //
        // ================================================================

        /// <summary>
        ///     BUG 63: ToString on empty array crashes.
        ///
        ///     NumPy:    str(array([])) = "[]"
        ///     NumSharp: InvalidOperationException: Can't construct NDIterator
        /// </summary>
        [TestMethod]
        public void Bug_ToString_EmptyArray_Crashes()
        {
            var empty = np.zeros(new Shape(0));

            string result = null;
            new Action(() => result = empty.ToString())
                .Should().NotThrow(
                    "NumPy: str(array([])) returns '[]'. " +
                    "NumSharp throws InvalidOperationException because NDIterator " +
                    "cannot handle empty shapes.");
        }
    }
}
