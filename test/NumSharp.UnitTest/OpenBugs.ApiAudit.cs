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
    ///     Tests verify the fix works correctly after the bug is resolved.
    /// </summary>
    public partial class OpenBugs
    {
        /// <summary>
        ///     BUG 64: FIXED — np.sign now preserves int32 dtype.
        ///
        ///     NumPy:    sign(int32[]).dtype = int32
        ///     NumSharp: FIXED — sign(int32[]).dtype = int32
        /// </summary>
        [Test]
        public void Bug64_Sign_PreservesDtype()
        {
            var a = np.array(new int[] { -3, 0, 5 });
            var result = np.sign(a);

            result.dtype.Should().Be(typeof(int), "np.sign should preserve input dtype");
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

        /// <summary>
        ///     BUG 69: FIXED — Boolean mask getter now returns selected elements.
        ///
        ///     NumPy:    arange(5)[mask] = [0, 2, 4]
        ///     NumSharp: FIXED — returns [0, 2, 4] (size=3)
        /// </summary>
        [Test]
        public void Bug69_BooleanMaskGetter_ReturnsSelection()
        {
            var a = np.arange(5); // [0, 1, 2, 3, 4]
            var mask = np.array(new bool[] { true, false, true, false, true });

            var result = a[mask];

            result.size.Should().Be(3, "Boolean mask selects 3 elements where mask is True");
            // arange returns Int64
            result.GetInt64(0).Should().Be(0, "a[0] where mask[0]=True");
            result.GetInt64(1).Should().Be(2, "a[2] where mask[2]=True");
            result.GetInt64(2).Should().Be(4, "a[4] where mask[4]=True");
        }

        /// <summary>
        ///     BUG 70: FIXED — Boolean mask setter now modifies elements.
        ///
        ///     NumPy:    a[mask] = 99 -> [99, 20, 99, 40, 99]
        ///     NumSharp: FIXED — correctly sets masked elements
        /// </summary>
        [Test]
        public void Bug70_BooleanMaskSetter_Works()
        {
            var a = np.array(new int[] { 10, 20, 30, 40, 50 });
            var mask = np.array(new bool[] { true, false, true, false, true });

            a[mask] = 99;

            a.GetInt32(0).Should().Be(99, "mask[0]=True, should be 99");
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

        /// <summary>
        ///     BUG 73: FIXED — reshape to scalar Shape() now works.
        ///
        ///     NumPy:    np.array([42]).reshape(()) = array(42), ndim=0
        ///     NumSharp: FIXED — correctly creates scalar array
        /// </summary>
        [Test]
        public void Bug73_Reshape_ScalarShape_Works()
        {
            var a = np.array(new int[] { 42 }); // shape (1,)

            var result = a.reshape(new Shape());

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

        /// <summary>
        ///     BUG 77: FIXED — np.sign on NaN now returns NaN (not throws).
        ///
        ///     NumPy:    sign([NaN, inf, -inf, 0]) = [NaN, 1, -1, 0]
        ///     NumSharp: FIXED — returns correct values including NaN
        /// </summary>
        [Test]
        public void Bug77_Sign_NaN_ReturnsNaN()
        {
            var a = np.array(new double[] { double.NaN, 1.0, double.PositiveInfinity, double.NegativeInfinity, 0.0 });

            var result = np.sign(a);

            double.IsNaN(result.GetDouble(0)).Should().BeTrue("sign(NaN) = NaN");
            result.GetDouble(1).Should().Be(1.0, "sign(1.0) = 1.0");
            result.GetDouble(2).Should().Be(1.0, "sign(+inf) = 1.0");
            result.GetDouble(3).Should().Be(-1.0, "sign(-inf) = -1.0");
            result.GetDouble(4).Should().Be(0.0, "sign(0.0) = 0.0");
        }

        /// <summary>
        ///     BUG 78a: FIXED — np.std on empty array returns NaN.
        ///
        ///     NumPy:    std([]) = NaN (with warning)
        ///     NumSharp: FIXED — returns NaN
        /// </summary>
        [Test]
        public void Bug78a_Std_EmptyArray_ReturnsNaN()
        {
            var empty = np.array(new double[0]);

            var result = np.std(empty);

            double.IsNaN(result.GetDouble()).Should().BeTrue("std of empty array should be NaN");
        }

        /// <summary>
        ///     BUG 78b: FIXED — np.var on empty array returns NaN.
        ///
        ///     NumPy:    var([]) = NaN (with warning)
        ///     NumSharp: FIXED — returns NaN
        /// </summary>
        [Test]
        public void Bug78b_Var_EmptyArray_ReturnsNaN()
        {
            var empty = np.array(new double[0]);

            var result = np.var(empty);

            double.IsNaN(result.GetDouble()).Should().BeTrue("var of empty array should be NaN");
        }

        // ================================================================
        //
        //  BUG 79: FIXED — Modulo now uses NumPy/Python floored division semantics.
        //  Tests moved to Backends/Kernels/EdgeCaseTests.cs
        //
        // ================================================================

        /// <summary>
        ///     BUG 80: FIXED — Fancy indexing setter now works.
        ///
        ///     NumPy:    a[[1,3]] = [99,88] -> [10,99,30,88,50]
        ///     NumSharp: FIXED — correctly sets indexed elements
        /// </summary>
        [Test]
        public void Bug80_FancyIndexSetter_Works()
        {
            var a = np.array(new int[] { 10, 20, 30, 40, 50 });
            var idx = np.array(new int[] { 1, 3 });

            a[idx] = np.array(new int[] { 99, 88 });

            a.GetInt32(0).Should().Be(10, "index 0 not in idx, unchanged");
            a.GetInt32(1).Should().Be(99, "a[1] should be 99");
            a.GetInt32(2).Should().Be(30, "index 2 not in idx, unchanged");
            a.GetInt32(3).Should().Be(88, "a[3] should be 88");
            a.GetInt32(4).Should().Be(50, "index 4 not in idx, unchanged");
        }
    }
}
