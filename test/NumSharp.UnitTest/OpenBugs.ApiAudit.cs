using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Bugs 64-80: API correctness bugs found during memory safety audit.
    ///
    ///     These bugs were discovered as side effects of an exhaustive memory safety
    ///     analysis that tested 129 APIs over 500 iterations each with forced GC.
    ///     Memory management itself was proven safe — these are behavioral differences
    ///     from NumPy 2.x (v2.4.2) uncovered during the testing.
    ///
    ///     Each test asserts the CORRECT NumPy 2.x behavior.
    ///     Tests FAIL while the bug exists, and PASS when the bug is fixed.
    ///
    ///     Categories:
    ///       Bug 64:  np.sign returns Double for integer input (should preserve dtype)
    ///       Bug 65:  REMOVED — duplicate of Bug 10 (np.unique unsorted)
    ///       Bug 66:  FIXED — operator != (NDArray-NDArray) now works via ILKernelGenerator
    ///       Bug 67:  FIXED — operator > (NDArray-NDArray) now works via ILKernelGenerator
    ///       Bug 68:  FIXED — operator < (NDArray-NDArray) now works via ILKernelGenerator
    ///       Bug 69:  Boolean mask getter returns cast mask instead of selected elements
    ///       Bug 70:  Boolean mask setter silently no-ops (different path from Bug 46)
    ///       Bug 71:  REMOVED — false positive (NumPy 2.x also rejects int() on 1-d arrays)
    ///       Bug 72:  (double)int32-scalar-NDArray reinterprets bytes instead of converting
    ///       Bug 73:  NDArray.reshape(Shape()) throws NullReferenceException
    ///       Bug 74:  FIXED — np.argmin now handles NaN correctly (tests in ArgMaxArgMinEdgeCaseTests.cs)
    ///       Bug 75:  FIXED — np.prod on bool now works (converts to int64, same as NumPy)
    ///       Bug 76:  FIXED — np.cumsum on bool now works (converts to int64, same as NumPy)
    ///       Bug 77:  np.sign(NaN array) throws ArithmeticException (should return NaN)
    ///       Bug 78:  np.std/np.var crash on empty arrays (sibling of Bug 55)
    ///       Bug 79:  FIXED — Modulo now uses NumPy/Python semantics (tests in EdgeCaseTests.cs)
    ///       Bug 80:  Fancy indexing (NDArray int indices) setter silently no-ops
    ///
    ///     Overlap notes:
    ///       - Bug 46 (DeprecationAudit) covers bool mask setter throwing NotImplementedException;
    ///         Bug 70 here shows it silently no-ops in a different code path.
    ///       - Bug 59 (DeprecationAudit) covers &gt;=/&lt;= with scalar operands;
    ///         Bugs 67-68 here cover NDArray-NDArray operands (different overloads).
    ///       - Bug 56 (DeprecationAudit) covers np.abs dtype preservation;
    ///         Bug 64 here covers np.sign dtype (same pattern, different function).
    ///       - Bug 62 (DeprecationAudit) covers implicit conversion crashes;
    ///         Bug 72 here adds the specific (double)int32_NDArray byte-reinterpretation.
    ///       - Bug 57 (DeprecationAudit) covers np.sum/np.mean on bool;
    ///         Bugs 75-76 here cover np.prod and np.cumsum on bool (same pattern).
    ///       - Bug 60 (DeprecationAudit) covers np.argmax ignoring NaN;
    ///         Bug 74 here covers np.argmin ignoring NaN (same pattern).
    ///       - Bug 55 (DeprecationAudit) covers np.mean on empty arrays;
    ///         Bug 78 here covers np.std and np.var on empty arrays (same pattern).
    /// </summary>
    public partial class OpenBugs
    {
        // ================================================================
        //
        //  BUG 64: np.sign returns Double dtype for integer input
        //
        //  SEVERITY: Medium — dtype should be preserved (same pattern as Bug 56).
        //
        //  NumPy: np.sign(int32_array) returns int32.
        //  NumSharp: Returns Double because the sign implementation casts
        //  through a floating-point computation path.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.sign(np.array([-3, 0, 5], dtype=np.int32)).dtype
        //    dtype('int32')
        //    >>> np.sign(np.array([-3, 0, 5], dtype=np.int32))
        //    array([-1,  0,  1])
        //
        // ================================================================

        /// <summary>
        ///     BUG 64: np.sign returns Double instead of preserving int32 dtype.
        ///
        ///     NumPy:    sign(int32[]).dtype = int32
        ///     NumSharp: sign(int32[]).dtype = Double
        /// </summary>
        [Test]
        public void Bug_Sign_ReturnsDouble_ForIntInput()
        {
            var a = np.array(new int[] { -3, 0, 5 });
            var result = np.sign(a);

            // Check dtype first — NumSharp returns Double, so GetInt32 would crash
            result.dtype.Should().Be(typeof(int),
                "NumPy preserves the input dtype: np.sign(int32[]).dtype = int32. " +
                "NumSharp returns Double, same pattern as Bug 56 (np.abs). " +
                "The sign implementation routes through a floating-point computation path.");

            // These assertions verify correct values once dtype is fixed
            result.GetInt32(0).Should().Be(-1, "sign(-3) = -1");
            result.GetInt32(1).Should().Be(0, "sign(0) = 0");
            result.GetInt32(2).Should().Be(1, "sign(5) = 1");
        }

        // BUG 65 (np.unique unsorted) — REMOVED: Duplicate of Bug 10 in OpenBugs.cs
        // which already covers np.unique returning first-appearance order instead of sorted.

        // BUG 66 (operator != NDArray-NDArray) — FIXED in ILKernelGenerator.Comparison.cs
        // Tests: ComparisonOpTests.NotEqual_Int32_SameType, np_comparison_Test.not_equal_ArrayArray

        // BUG 67 (operator > NDArray-NDArray) — FIXED in ILKernelGenerator.Comparison.cs
        // Tests: ComparisonOpTests.Greater_Int32_SameType, np_comparison_Test.greater_ArrayArray

        // BUG 68 (operator < NDArray-NDArray) — FIXED in ILKernelGenerator.Comparison.cs
        // Tests: ComparisonOpTests.Less_Int32_SameType, np_comparison_Test.less_ArrayArray

        // ================================================================
        //
        //  BUG 69: Boolean mask getter returns cast mask instead of selection
        //
        //  SEVERITY: Critical — fundamental indexing operation returns wrong data.
        //
        //  a[bool_mask] should return only the elements where mask is True.
        //  Instead, NumSharp returns the mask itself cast to the array's dtype
        //  (e.g., [1,0,1,0,1] for a bool mask on an int array of size 5).
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> a = np.arange(5)
        //    >>> mask = np.array([True, False, True, False, True])
        //    >>> a[mask]
        //    array([0, 2, 4])
        //    >>> a[mask].shape
        //    (3,)
        //
        // ================================================================

        /// <summary>
        ///     BUG 69: Boolean mask getter returns cast mask [1,0,1,0,1] (size=5)
        ///     instead of selected elements [0,2,4] (size=3).
        ///
        ///     NumPy:    arange(5)[mask] = [0, 2, 4]
        ///     NumSharp: returns [1, 0, 1, 0, 1] (mask cast to int)
        /// </summary>
        [Test]
        public void Bug_BooleanMaskGetter_ReturnsMaskInsteadOfSelection()
        {
            var a = np.arange(5); // [0, 1, 2, 3, 4]
            var mask = np.array(new bool[] { true, false, true, false, true });

            var result = a[mask];

            result.size.Should().Be(3,
                "NumPy: arange(5)[[T,F,T,F,T]] selects 3 elements where mask is True. " +
                "NumSharp returns size=5 because it casts the boolean mask to the " +
                "array's dtype and returns [1,0,1,0,1] instead of selecting elements.");
            result.GetInt32(0).Should().Be(0, "a[0] where mask[0]=True");
            result.GetInt32(1).Should().Be(2, "a[2] where mask[2]=True");
            result.GetInt32(2).Should().Be(4, "a[4] where mask[4]=True");
        }

        // ================================================================
        //
        //  BUG 70: Boolean mask setter silently no-ops
        //
        //  SEVERITY: High — fundamental indexing operation silently does nothing.
        //
        //  Bug 46 (DeprecationAudit) covers the code path that throws
        //  NotImplementedException. This bug covers a different code path
        //  where the setter silently does nothing — the array is unchanged
        //  after a[mask] = value.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> a = np.array([10, 20, 30, 40, 50])
        //    >>> mask = np.array([True, False, True, False, True])
        //    >>> a[mask] = 99
        //    >>> a
        //    array([99, 20, 99, 40, 99])
        //
        // ================================================================

        /// <summary>
        ///     BUG 70: Boolean mask setter silently no-ops (different path from Bug 46).
        ///
        ///     NumPy:    a = [10,20,30,40,50]; a[mask] = 99 -> [99,20,99,40,99]
        ///     NumSharp: array unchanged after assignment
        /// </summary>
        [Test]
        public void Bug_BooleanMaskSetter_SilentlyNoOps()
        {
            var a = np.array(new int[] { 10, 20, 30, 40, 50 });
            var mask = np.array(new bool[] { true, false, true, false, true });

            // Use NDArray value to exercise a different code path than Bug 46
            a[mask] = np.array(new int[] { 99, 99, 99 });

            a.GetInt32(0).Should().Be(99,
                "NumPy: a[mask] = 99 sets True-positions to 99. a[0] should be 99. " +
                "NumSharp: the boolean mask setter silently no-ops — the array is " +
                "unchanged after the assignment. Bug 46 covers the code path that " +
                "throws NotImplementedException; this is a different path that " +
                "silently does nothing.");
            a.GetInt32(1).Should().Be(20, "mask[1]=False, unchanged");
            a.GetInt32(2).Should().Be(99, "mask[2]=True, should be 99");
            a.GetInt32(3).Should().Be(40, "mask[3]=False, unchanged");
            a.GetInt32(4).Should().Be(99, "mask[4]=True, should be 99");
        }

        // BUG 71 ((int)NDArray on 1-element array) — REMOVED: False positive.
        // NumPy 2.4.2 also raises TypeError: "only 0-dimensional arrays can be
        // converted to Python scalars" for int(np.array([42])). NumSharp's
        // IncorrectShapeException on shape (1,) is the correct behavior.
        // The implicit conversion operator correctly requires ndim=0.
        // Verified: int(np.array(42)) works in both NumPy and NumSharp (0-d arrays).

        // ================================================================
        //
        //  BUG 72: (double)int32-scalar-NDArray reinterprets bytes → garbage
        //
        //  SEVERITY: Critical — silently returns completely wrong values.
        //
        //  When casting an int32 scalar NDArray (ndim=0) to double via the
        //  implicit conversion operator, NumSharp uses GetAtIndex<double> which
        //  reads the raw 4 bytes of int32 storage as if they were part of an
        //  8-byte double. This produces garbage values (~6.95e-310) instead of
        //  performing numeric conversion.
        //
        //  np.sum(np.arange(10)) returns an int32 scalar NDArray (ndim=0) with
        //  value 45. (int)result works fine (same dtype), but (double)result
        //  returns 6.95e-310 due to byte reinterpretation.
        //
        //  Also affects NDArray.Scalar(42) — (double) cast returns garbage.
        //
        //  VERIFIED IN NUMSHARP:
        //    (int)(NDArray)np.sum(np.arange(10)) = 45  ← correct (same dtype)
        //    (double)(NDArray)np.sum(np.arange(10)) = 6.95e-310  ← GARBAGE
        //    (double)(NDArray)NDArray.Scalar(42) = 6.95e-310  ← GARBAGE
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> float(np.sum(np.arange(10)))
        //    45.0
        //    >>> float(np.array(42, dtype=np.int32))
        //    42.0
        //
        // ================================================================

        /// <summary>
        ///     BUG 72: (double) cast on int32 scalar NDArray reinterprets bytes → garbage.
        ///
        ///     NumPy:    float(np.sum(np.arange(10))) = 45.0
        ///     NumSharp: returns ~6.95e-310 (int32 bytes read as double)
        /// </summary>
        [Test]
        public void Bug_DoubleCast_Int32NDArray_ReturnsGarbage()
        {
            // np.sum returns a scalar NDArray (ndim=0)
            // After BUG-21 fix: int32 input now accumulates to int64 (NumPy 2.x behavior)
            var sum = np.sum(np.arange(10));
            sum.dtype.Should().Be(typeof(long), "NumPy 2.x: int32 sum accumulates to int64");
            sum.ndim.Should().Be(0, "np.sum without axis returns a scalar (ndim=0)");

            // Same-dtype cast works fine (now int64 after BUG-21 fix)
            ((long)(NDArray)sum).Should().Be(45, "(long) cast on int64 scalar works");

            // Cross-dtype cast produces garbage
            double val = 0;
            new Action(() => val = (double)(NDArray)sum)
                .Should().NotThrow(
                    "The cast doesn't throw — it silently returns garbage.");

            val.Should().Be(45.0,
                "NumPy: float(np.sum(np.arange(10))) = 45.0 (numeric conversion). " +
                "NumSharp: the implicit (double) operator uses GetAtIndex<double> " +
                "which reads the raw 4-byte int32 representation of 45 as if it " +
                "were an 8-byte double, producing ~6.95e-310 (garbage). " +
                "Same bug affects NDArray.Scalar(42) cast to double.");
        }

        // ================================================================
        //
        //  BUG 73: NDArray.reshape(Shape()) throws NullReferenceException
        //
        //  SEVERITY: Medium — reshaping to scalar shape crashes.
        //
        //  new Shape() (default constructor) creates a scalar shape.
        //  Passing this to reshape() causes a NullReferenceException because
        //  the scalar Shape has null dimensions.
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.array([42]).reshape(())
        //    array(42)
        //    >>> np.array([42]).reshape(()).shape
        //    ()
        //    >>> np.array([42]).reshape(()).ndim
        //    0
        //
        // ================================================================

        /// <summary>
        ///     BUG 73: reshape to scalar shape (Shape()) throws NullReferenceException.
        ///
        ///     NumPy:    np.array([42]).reshape(()) = array(42), shape=(), ndim=0
        ///     NumSharp: NullReferenceException
        /// </summary>
        [Test]
        public void Bug_Reshape_ScalarShape_ThrowsNullReference()
        {
            var a = np.array(new int[] { 42 }); // shape (1,)

            NDArray result = null;
            new Action(() => result = a.reshape(new Shape()))
                .Should().NotThrow(
                    "NumPy: np.array([42]).reshape(()) creates a scalar-shaped array " +
                    "with shape=() and ndim=0. NumSharp throws NullReferenceException " +
                    "because the default Shape() constructor creates a scalar shape " +
                    "with null dimensions, and reshape doesn't handle this case.");

            result.Should().NotBeNull();
            result.ndim.Should().Be(0, "scalar shape has ndim=0");
            result.GetInt32(0).Should().Be(42, "value should be preserved");
        }

        // ================================================================
        //
        //  BUG 74: FIXED — np.argmin now correctly returns index of first NaN.
        //  Tests moved to NumPyPortedTests/ArgMaxArgMinEdgeCaseTests.cs
        //
        // ================================================================

        // BUG 75 (np.prod on bool) — FIXED and moved to:
        // test/NumSharp.UnitTest/Backends/Unmanaged/Math/np.prod.tests.cs
        // Test: BooleanArray_TreatsAsIntAndReturnsInt64

        // BUG 76 (np.cumsum on bool) — FIXED and moved to:
        // test/NumSharp.UnitTest/Math/NDArray.cumsum.Test.cs
        // Test: BooleanArray_TreatsAsIntAndReturnsInt64

        // ================================================================
        //
        //  BUG 77: np.sign on NaN array throws ArithmeticException
        //
        //  SEVERITY: Medium — sign(NaN) should return NaN, not crash.
        //
        //  The sign implementation calls Math.Sign() which throws
        //  ArithmeticException for NaN input. NumPy returns NaN for
        //  sign(NaN), 1.0 for sign(inf), -1.0 for sign(-inf).
        //
        //  VERIFIED IN NUMSHARP:
        //    np.sign([NaN, 1.0, inf, -inf]) throws ArithmeticException:
        //    "Function does not accept floating point Not-a-Number values."
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.sign(np.array([np.nan, np.inf, -np.inf, 0.0]))
        //    array([nan,  1., -1.,  0.])
        //
        // ================================================================

        /// <summary>
        ///     BUG 77: np.sign on NaN array throws ArithmeticException.
        ///
        ///     NumPy:    sign([NaN, inf, -inf, 0]) = [NaN, 1, -1, 0]
        ///     NumSharp: ArithmeticException from Math.Sign(NaN)
        ///
        ///     FIXED: IL kernel now checks for NaN and returns NaN directly.
        /// </summary>
        [Test]
        public void Bug_Sign_NaN_ThrowsArithmeticException()
        {
            var a = np.array(new double[] { double.NaN, 1.0, double.PositiveInfinity, double.NegativeInfinity, 0.0 });

            NDArray result = null;
            new Action(() => result = np.sign(a))
                .Should().NotThrow(
                    "NumPy: sign([NaN,1,inf,-inf,0]) = [NaN,1,1,-1,0]. " +
                    "NumSharp IL kernel handles NaN by returning NaN directly.");

            result.Should().NotBeNull();
            double.IsNaN(result.GetDouble(0)).Should().BeTrue("sign(NaN) = NaN");
            result.GetDouble(1).Should().Be(1.0, "sign(1.0) = 1.0");
            result.GetDouble(2).Should().Be(1.0, "sign(+inf) = 1.0");
            result.GetDouble(3).Should().Be(-1.0, "sign(-inf) = -1.0");
            result.GetDouble(4).Should().Be(0.0, "sign(0.0) = 0.0");
        }

        // ================================================================
        //
        //  BUG 78: np.std and np.var crash on empty arrays
        //
        //  SEVERITY: Medium — NumPy returns NaN with RuntimeWarning.
        //
        //  Bug 55 (DeprecationAudit) covers np.mean on empty arrays.
        //  np.std and np.var have the same issue — they crash with
        //  InvalidOperationException: "Can't construct NDIterator with
        //  an empty shape."
        //
        //  VERIFIED IN NUMSHARP:
        //    np.std(double[0]) -> InvalidOperationException
        //    np.var(double[0]) -> InvalidOperationException
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.std(np.array([]))
        //    nan  (RuntimeWarning: Degrees of freedom <= 0 for slice)
        //    >>> np.var(np.array([]))
        //    nan  (RuntimeWarning: Degrees of freedom <= 0 for slice)
        //
        // ================================================================

        /// <summary>
        ///     BUG 78a: np.std on empty array crashes instead of returning NaN.
        ///
        ///     NumPy:    std([]) = NaN (with warning)
        ///     NumSharp: InvalidOperationException: Can't construct NDIterator
        /// </summary>
        [Test]
        public void Bug_Std_EmptyArray_Crashes()
        {
            var empty = np.array(new double[0]);

            NDArray result = null;
            new Action(() => result = np.std(empty))
                .Should().NotThrow(
                    "NumPy: std of empty array returns NaN with RuntimeWarning. " +
                    "NumSharp throws InvalidOperationException because NDIterator " +
                    "cannot handle empty shapes. Same pattern as Bug 55 (mean on empty).");
        }

        /// <summary>
        ///     BUG 78b: np.var on empty array crashes instead of returning NaN.
        ///
        ///     NumPy:    var([]) = NaN (with warning)
        ///     NumSharp: InvalidOperationException: Can't construct NDIterator
        /// </summary>
        [Test]
        public void Bug_Var_EmptyArray_Crashes()
        {
            var empty = np.array(new double[0]);

            NDArray result = null;
            new Action(() => result = np.var(empty))
                .Should().NotThrow(
                    "NumPy: var of empty array returns NaN with RuntimeWarning. " +
                    "NumSharp throws InvalidOperationException because NDIterator " +
                    "cannot handle empty shapes. Same pattern as Bug 55 (mean on empty).");
        }

        // ================================================================
        //
        //  BUG 79: FIXED — Modulo now uses NumPy/Python floored division semantics.
        //  Tests moved to Backends/Kernels/EdgeCaseTests.cs
        //
        // ================================================================

        // ================================================================
        //
        //  BUG 80: Fancy indexing (NDArray int indices) setter silently no-ops
        //
        //  SEVERITY: High — fundamental indexing operation silently does nothing.
        //
        //  a[int_array_indices] = values should set elements at the
        //  specified indices. NumSharp silently ignores the assignment —
        //  the array is unchanged after the operation.
        //
        //  This is different from Bug 46/70 (boolean mask setter) — this
        //  uses integer array indices (fancy indexing), not boolean masks.
        //  Manual SetInt32 at the same indices works correctly.
        //
        //  VERIFIED IN NUMSHARP:
        //    a = [10,20,30,40,50]; a[[1,3]] = [99,88] -> still [10,20,30,40,50]
        //    SetInt32(99, 1); SetInt32(88, 3) -> [10,99,30,88,50] (works)
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> a = np.array([10, 20, 30, 40, 50])
        //    >>> a[np.array([1, 3])] = np.array([99, 88])
        //    >>> a
        //    array([10, 99, 30, 88, 50])
        //
        // ================================================================

        /// <summary>
        ///     BUG 80: Fancy indexing setter silently no-ops.
        ///
        ///     NumPy:    a[[1,3]] = [99,88] -> [10,99,30,88,50]
        ///     NumSharp: array unchanged after assignment
        /// </summary>
        [Test]
        public void Bug_FancyIndexSetter_SilentlyNoOps()
        {
            var a = np.array(new int[] { 10, 20, 30, 40, 50 });
            var idx = np.array(new int[] { 1, 3 });

            a[idx] = np.array(new int[] { 99, 88 });

            a.GetInt32(0).Should().Be(10, "index 0 not in idx, unchanged");
            a.GetInt32(1).Should().Be(99,
                "NumPy: a[np.array([1,3])] = [99,88] sets a[1]=99. " +
                "NumSharp: the fancy indexing setter silently no-ops — the array " +
                "is unchanged after assignment. Manual SetInt32 at the same " +
                "indices works correctly, so the underlying storage is fine.");
            a.GetInt32(2).Should().Be(30, "index 2 not in idx, unchanged");
            a.GetInt32(3).Should().Be(88, "a[3] should be 88");
            a.GetInt32(4).Should().Be(50, "index 4 not in idx, unchanged");
        }
    }
}
