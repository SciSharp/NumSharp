using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using FluentAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Master collection of reproducible open bugs found during broadcast testing.
    ///     Each test asserts the CORRECT NumPy behavior as source of truth.
    ///     Tests FAIL while the bug exists, and PASS when the bug is fixed.
    ///
    ///     All NumPy behaviors were verified by running actual Python code against
    ///     NumPy v2.4.2 on 2026-02-07. The Python verification script is documented
    ///     in the session transcript.
    ///
    ///     When a bug is fixed, the test starts passing. Move passing tests to
    ///     the appropriate permanent test class (e.g., NpBroadcastFromNumPyTests).
    ///
    ///     =====================================================================
    ///     ARCHITECTURAL ROOT CAUSE ANALYSIS
    ///     =====================================================================
    ///
    ///     The majority of bugs trace back to a single architectural problem:
    ///     NumSharp has MULTIPLE code paths for traversing array elements, and
    ///     they don't all handle the ViewInfo + BroadcastInfo stride combination
    ///     correctly.
    ///
    ///     Coordinate-based access (Shape.GetOffset(coords), used by GetInt32/
    ///     GetDouble/etc.) is CORRECT — it properly combines the slice ViewInfo
    ///     strides with the broadcast BroadcastInfo zero-strides. But flat/linear
    ///     iteration paths (used by ToString, flatten, concatenate, np.minimum)
    ///     compute offsets differently and produce wrong results for broadcasted
    ///     arrays — especially when the source was sliced before broadcasting.
    ///
    ///     This means: any code that accesses elements by coordinate will see the
    ///     right values, but any code that iterates linearly (for rendering,
    ///     copying, or element-wise operations) may see wrong values.
    ///
    ///     Bug categories:
    ///       - Bugs 1, 11, 12:  Linear iterator disagrees with indexed access
    ///       - Bug 5/9:         np.minimum's specific iteration transposes inputs
    ///       - Bugs 6, 8:       Comparison operators missing NDArray vs NDArray support
    ///       - Bug 7:           np.allclose fundamentally broken (not broadcast-specific)
    ///       - Bugs 2, 3, 4:    broadcast_to semantic/protection issues
    ///       - Bug 10:          np.unique missing sort (not broadcast-specific)
    ///       - Bug 13:          cumsum axis on broadcast reads garbage memory
    ///       - Bug 14:          roll on broadcast produces zeros/wrong values
    ///       - Bug 15:          sum/mean/var/std axis=0 on column-broadcast under-counts
    ///       - Bug 16:          argsort crashes on any 2D array (not broadcast-specific)
    ///
    ///     Total: 16 distinct bugs, 34 test methods.
    /// </summary>
    [TestClass]
    public class OpenBugs : TestClass
    {
        // ================================================================
        //
        //  BUG 1: ToString on broadcasted sliced arrays shows wrong values
        //
        //  SEVERITY: High — affects all debugging/display of broadcast arrays.
        //
        //  PATTERN: The most visible manifestation of the iterator-vs-indexed
        //  access divergence. GetInt32(i,j) uses Shape.GetOffset(coords) which
        //  correctly computes: baseOffset + (coord * viewStride) for each dim,
        //  where viewStride already accounts for the original slice strides.
        //  But ToString uses a linear NDIterator (or similar flat traversal)
        //  that computes offsets differently — it appears to use the broadcast
        //  strides directly against the raw storage pointer without going
        //  through the ViewInfo coordinate-to-offset translation.
        //
        //  The result: GetInt32 returns correct values, but ToString shows
        //  wrong values. This affects reversed slices (negative strides),
        //  step slices (strides > 1), 2D sliced columns, and double-sliced
        //  (slice-of-slice) arrays when any of these are then broadcasted.
        //
        //  EVIDENCE: Four variants tested below. In all cases, coordinate
        //  access (GetInt32) returns the correct NumPy values, but the string
        //  representation is wrong. The specific wrong values vary:
        //    - Reversed slice:  shows forward order [0,1,2] instead of [2,1,0]
        //    - Step slice:      shows scrambled [0,4,2] instead of [0,2,4]
        //    - Column slice:    shows zeros [0,0,0] instead of [9,9,9]
        //    - Double slice:    shows garbage [32,32,32,32] instead of [8,8,8,8]
        //
        //  Note: Simple broadcasts WITHOUT slicing (e.g. broadcast_to([1,2,3],
        //  (2,3))) render correctly in ToString. The bug only manifests when
        //  the source array has non-trivial ViewInfo (was sliced before
        //  broadcasting).
        //
        //  AFFECTED FILES: The rendering path in NDArray.ToString, likely
        //  using NDIterator or a raw pointer walk that doesn't account for
        //  the combined ViewInfo+BroadcastInfo stride calculation.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.broadcast_to(np.arange(3)[::-1], (2,3))
        //    array([[2, 1, 0], [2, 1, 0]])
        //    >>> np.broadcast_to(np.arange(6)[::2], (2,3))
        //    array([[0, 2, 4], [0, 2, 4]])
        //    >>> np.broadcast_to(np.arange(12).reshape(3,4)[:,1:2], (3,3))
        //    array([[1, 1, 1], [5, 5, 5], [9, 9, 9]])
        //    >>> x=np.arange(12).reshape(3,4); np.broadcast_to(x[::2,0:1], (2,4))
        //    array([[0, 0, 0, 0], [8, 8, 8, 8]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 1a: ToString ignores negative strides on broadcasted arrays.
        ///
        ///     Setup: arange(3) = [0,1,2], reversed via [::-1] to [2,1,0],
        ///     then broadcast_to (2,3). The reversed slice has stride=-1 in
        ///     ViewInfo, and the broadcast adds a zero-stride row dimension.
        ///
        ///     NumPy output:  [[2, 1, 0], [2, 1, 0]]
        ///     NumSharp GetInt32: [2,1,0],[2,1,0]  (CORRECT)
        ///     NumSharp ToString: [[0,1,2],[0,1,2]]  (WRONG — forward order)
        ///
        ///     The ToString iterator likely starts at the raw base pointer and
        ///     walks forward, ignoring that the ViewInfo stride is -1, which
        ///     means the logical element order is reversed from the physical
        ///     memory layout.
        /// </summary>
        [TestMethod]
        public void Bug_ToString_ReversedSliceBroadcast()
        {
            var rev = np.arange(3)["::-1"]; // [2, 1, 0]
            var brev = np.broadcast_to(rev, new Shape(2, 3));

            // Coordinate access is correct
            Assert.AreEqual(2, brev.GetInt32(0, 0));
            Assert.AreEqual(1, brev.GetInt32(0, 1));
            Assert.AreEqual(0, brev.GetInt32(0, 2));

            // NumPy: ToString must show "2, 1, 0" (reversed order)
            var str = brev.ToString(false);
            str.Should().Contain("2, 1, 0",
                "NumPy: broadcast_to(arange(3)[::-1], (2,3)) displays [[2,1,0],[2,1,0]]. " +
                "NumSharp shows [[0,1,2],[0,1,2]] — the ToString iterator ignores the " +
                "negative stride from the reversed slice when combined with broadcasting.");
        }

        /// <summary>
        ///     BUG 1b: ToString shows scrambled values for step-sliced broadcast.
        ///
        ///     Setup: arange(6) = [0,1,2,3,4,5], step-sliced via [::2] to
        ///     [0,2,4] (stride=2 in ViewInfo), then broadcast_to (2,3).
        ///
        ///     NumPy output:  [[0, 2, 4], [0, 2, 4]]
        ///     NumSharp GetInt32: [0,2,4],[0,2,4]  (CORRECT)
        ///     NumSharp ToString: [[0,4,2],[0,4,2]]  (WRONG — scrambled)
        ///
        ///     The values 0, 4, 2 suggest the iterator is using stride=2 but
        ///     wrapping around or mis-computing offsets: offset 0→value 0,
        ///     offset 2→value 2 (skipped), offset 4→value 4, then offset
        ///     1→value 1 (but shows 4?). The exact scrambling pattern suggests
        ///     the broadcast stride and the view stride are being multiplied
        ///     or combined incorrectly.
        /// </summary>
        [TestMethod]
        public void Bug_ToString_StepSliceBroadcast()
        {
            var stepped = np.arange(6)["::2"]; // [0, 2, 4]
            var bstepped = np.broadcast_to(stepped, new Shape(2, 3));

            // Coordinate access is correct
            Assert.AreEqual(0, bstepped.GetInt32(0, 0));
            Assert.AreEqual(2, bstepped.GetInt32(0, 1));
            Assert.AreEqual(4, bstepped.GetInt32(0, 2));

            // NumPy: ToString must show "0, 2, 4" (in order)
            var str = bstepped.ToString(false);
            str.Should().Contain("0, 2, 4",
                "NumPy: broadcast_to(arange(6)[::2], (2,3)) displays [[0,2,4],[0,2,4]]. " +
                "NumSharp shows [[0,4,2],[0,4,2]] — the ToString iterator mis-computes " +
                "offsets when the source has stride=2 from step-slicing.");
        }

        /// <summary>
        ///     BUG 1c: ToString shows zeros in last row for 2D sliced column broadcast.
        ///
        ///     Setup: arange(12).reshape(3,4) sliced to column 1 via [:,1:2]
        ///     giving (3,1) array [[1],[5],[9]], then broadcast_to (3,3).
        ///     The column slice has a row stride of 4 (stepping over the full
        ///     4-wide row to get to the next row's column 1).
        ///
        ///     NumPy output:  [[1,1,1], [5,5,5], [9,9,9]]
        ///     NumSharp GetInt32: [[1,1,1],[5,5,5],[9,9,9]]  (CORRECT)
        ///     NumSharp ToString: [[1,1,1],[5,5,5],[0,0,0]]  (WRONG — last row zeros)
        ///
        ///     The zeros in the last row suggest the iterator overflows past
        ///     the allocated storage. The 3x4 array has 12 ints = 48 bytes.
        ///     Column 1 values are at byte offsets 4, 20, 36. The iterator
        ///     appears to walk linearly with some stride that misses offset 36
        ///     and lands in zeroed/garbage memory beyond the allocation.
        /// </summary>
        [TestMethod]
        public void Bug_ToString_SlicedColumnBroadcast()
        {
            var x = np.arange(12).reshape(3, 4);
            var col = x[":, 1:2"]; // (3,1): [[1],[5],[9]]
            var bcol = np.broadcast_to(col, new Shape(3, 3));

            // Coordinate access is correct
            Assert.AreEqual(9, bcol.GetInt32(2, 0));
            Assert.AreEqual(9, bcol.GetInt32(2, 2));

            // NumPy: ToString must show "9, 9, 9" in last row
            var str = bcol.ToString(false);
            str.Should().Contain("9, 9, 9",
                "NumPy: broadcast_to(x[:,1:2], (3,3)) displays [[1,1,1],[5,5,5],[9,9,9]]. " +
                "NumSharp shows [[1,1,1],[5,5,5],[0,0,0]] — the ToString iterator " +
                "overflows past the storage for the last row of the sliced+broadcast array.");
        }

        /// <summary>
        ///     BUG 1d: ToString for double-sliced broadcast shows garbage.
        ///
        ///     Setup: arange(12).reshape(3,4) → x[::2,:] gives rows 0,2 →
        ///     (2,4) view with row stride=8 (skipping row 1). Then [:,0:1]
        ///     gives column 0 → (2,1) with values [[0],[8]]. Then broadcast
        ///     to (2,4).
        ///
        ///     This is the most complex case: a slice-of-a-slice (double
        ///     ViewInfo nesting) combined with broadcasting.
        ///
        ///     NumPy output:  [[0,0,0,0], [8,8,8,8]]
        ///     NumSharp GetInt32: [[0,0,0,0],[8,8,8,8]]  (CORRECT)
        ///     NumSharp ToString: [[0,0,0,0],[32,32,32,32]]  (WRONG — garbage)
        ///
        ///     The value 32 (0x20) in the second row indicates the iterator
        ///     is reading from a completely wrong memory offset. Since the
        ///     source array contains values 0-11 (each 4 bytes), value 32
        ///     cannot come from any valid element — it's reading from beyond
        ///     the array or from uninitialized memory. The double-slicing
        ///     creates a compound stride that the ToString iterator fails
        ///     to resolve correctly.
        /// </summary>
        [TestMethod]
        public void Bug_ToString_DoubleSlicedBroadcast()
        {
            var x = np.arange(12).reshape(3, 4);
            var dslice = x["::2, :"];
            var dslice_col = dslice[":, 0:1"];
            var bdslice = np.broadcast_to(dslice_col, new Shape(2, 4));

            // Coordinate access is correct
            Assert.AreEqual(8, bdslice.GetInt32(1, 0));

            // NumPy: ToString must show "8, 8, 8, 8" in second row
            var str = bdslice.ToString(false);
            str.Should().Contain("8, 8, 8, 8",
                "NumPy: broadcast_to(x[::2,0:1], (2,4)) displays [[0,0,0,0],[8,8,8,8]]. " +
                "NumSharp shows garbage values (e.g. 32) in the second row — the double-" +
                "sliced ViewInfo compound stride is not resolved by the ToString iterator.");
        }

        // ================================================================
        //
        //  BUG 2: broadcast_to allows write-through (no read-only flag)
        //
        //  SEVERITY: Medium — silent data corruption risk.
        //
        //  In NumPy, broadcast_to returns a read-only view:
        //    y = np.broadcast_to(x, (2,4))
        //    y.flags.writeable → False
        //    y[0,0] = 999 → ValueError: assignment destination is read-only
        //
        //  This is critical because broadcast views have zero-stride dimensions.
        //  Multiple logical positions (e.g. y[0,0] and y[1,0]) map to the SAME
        //  physical memory location. Writing to y[0,0] would silently modify
        //  y[1,0] and the original array x[0], violating the broadcasting
        //  contract that each row is an independent "copy" of the source.
        //
        //  NumSharp has no read-only concept on NDArray/UnmanagedStorage.
        //  SetInt32(999, 0, 0) succeeds silently, corrupting the original.
        //
        //  FIX APPROACH: Add a Writeable flag to NDArray or UnmanagedStorage.
        //  broadcast_to should set Writeable=false. All Set* methods should
        //  check this flag and throw InvalidOperationException.
        //
        //  PYTHON VERIFICATION:
        //    >>> x = np.array([1,2,3,4])
        //    >>> y = np.broadcast_to(x, (2,4))
        //    >>> y.flags.writeable
        //    False
        //    >>> y[0,0] = 999
        //    ValueError: assignment destination is read-only
        //
        // ================================================================

        /// <summary>
        ///     BUG 2: broadcast_to result is not read-only. Writing to the
        ///     broadcast view succeeds and modifies the original source array.
        ///
        ///     NumPy: y[0,0] = 999 raises ValueError.
        ///     NumSharp: SetInt32(999, 0, 0) succeeds silently.
        /// </summary>
        [TestMethod]
        public void Bug_BroadcastTo_NoReadOnlyProtection()
        {
            var x = np.array(new int[] { 1, 2, 3, 4 });
            var bx = np.broadcast_to(x, new Shape(2, 4));

            // NumPy: Writing to broadcast result must throw
            Action write = () => bx.SetInt32(999, 0, 0);
            write.Should().Throw<Exception>(
                "NumPy raises ValueError: assignment destination is read-only. " +
                "NumSharp has no read-only flag, so SetInt32 succeeds and silently " +
                "modifies the original array x[0] to 999, corrupting shared memory.");
        }

        // ================================================================
        //
        //  BUG 3: broadcast_to uses bilateral instead of one-directional
        //
        //  SEVERITY: Medium — wrong API semantics, accepts invalid inputs.
        //
        //  NumPy's broadcast_to(array, shape) is STRICTLY one-directional:
        //  it only stretches dimensions of the source array that are size 1
        //  to match the target shape. It NEVER modifies the target shape.
        //
        //  Examples that NumPy rejects:
        //    broadcast_to(ones(3), (1,))     → ValueError (can't shrink 3→1)
        //    broadcast_to(ones(1,2), (2,1))  → ValueError (can't reshape 2→1)
        //
        //  NumSharp's implementation delegates to DefaultEngine.Broadcast(from,
        //  against) which performs MUTUAL/bilateral broadcasting. This means:
        //    broadcast_to(ones(3), (1,))     → returns shape (3,) [target stretched]
        //    broadcast_to(ones(1,2), (2,1))  → returns shape (2,2) [both stretched]
        //
        //  This breaks user code that relies on broadcast_to to validate
        //  shape compatibility in one direction only.
        //
        //  FIX APPROACH: np.broadcast_to.cs should validate that every
        //  dimension of the source either equals the target or is 1. If
        //  any target dimension is smaller than the source, throw.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.broadcast_to(np.ones(3), (1,))
        //    ValueError: operands could not be broadcast together with
        //    remapped shapes [original->remapped]: (3,) and requested shape (1,)
        //    >>> np.broadcast_to(np.ones((1,2)), (2,1))
        //    ValueError: operands could not be broadcast together with
        //    remapped shapes [original->remapped]: (1,2) and requested shape (2,1)
        //
        // ================================================================

        /// <summary>
        ///     BUG 3: broadcast_to uses bilateral broadcasting instead of
        ///     NumPy's one-directional semantics. Cases that should throw
        ///     instead succeed by stretching the target shape.
        ///
        ///     NumPy: broadcast_to(ones(3), (1,)) raises ValueError.
        ///     NumSharp: Returns shape (3,) — stretched the target.
        ///
        ///     NumPy: broadcast_to(ones(1,2), (2,1)) raises ValueError.
        ///     NumSharp: Returns shape (2,2) — stretched both.
        /// </summary>
        [TestMethod]
        public void Bug_BroadcastTo_BilateralSemantics()
        {
            // NumPy: broadcast_to(ones(3), (1,)) must throw
            new Action(() => np.broadcast_to(np.ones(new Shape(3)), new Shape(1)))
                .Should().Throw<Exception>(
                    "NumPy raises ValueError for broadcast_to((3,), (1,)). " +
                    "NumSharp's bilateral broadcast succeeds, returning shape (3,) " +
                    "by stretching the target shape (1,) up to match the source.");

            // NumPy: broadcast_to(ones(1,2), (2,1)) must throw
            new Action(() => np.broadcast_to(np.ones(new Shape(1, 2)), new Shape(2, 1)))
                .Should().Throw<Exception>(
                    "NumPy raises ValueError for broadcast_to((1,2), (2,1)). " +
                    "NumSharp's bilateral broadcast succeeds, returning shape (2,2) " +
                    "by stretching both source dim 0 (1→2) and target dim 1 (1→2).");
        }

        // ================================================================
        //
        //  BUG 4: Re-broadcast inconsistency (IsBroadcasted guard)
        //
        //  SEVERITY: Medium — blocks legitimate chained broadcasting.
        //
        //  NumPy handles re-broadcasting transparently:
        //    bx, _ = broadcast_arrays(ones(1,3), ones(3,1))  # bx is (3,3)
        //    broadcast_to(bx, (2,3,3))  # works fine → (2,3,3)
        //
        //  NumSharp's 2-arg Broadcast(Shape, Shape) at line ~300 of
        //  Default.Broadcasting.cs has an explicit guard:
        //    if (leftShape.IsBroadcasted || rightShape.IsBroadcasted)
        //        throw new NotSupportedException();
        //
        //  This blocks ALL re-broadcasting through the 2-arg path. But the
        //  N-array overload Broadcast(Shape[]) does NOT have this guard,
        //  creating an inconsistency:
        //    broadcast_to(bx, (2,3,3))                  → throws (2-arg path)
        //    broadcast_arrays(bx, ones(2,3,3))           → works (N-arg path)
        //
        //  The guard was likely added as a safety measure to avoid complex
        //  stride-on-stride calculations, but it's overly conservative.
        //
        //  FIX APPROACH: Remove the IsBroadcasted guard or make the 2-arg
        //  Broadcast correctly handle already-broadcasted shapes by resolving
        //  the effective strides before re-broadcasting.
        //
        //  PYTHON VERIFICATION:
        //    >>> x, _ = np.broadcast_arrays(np.ones((1,3)), np.ones((3,1)))
        //    >>> np.broadcast_to(x, (2,3,3)).shape
        //    (2, 3, 3)
        //
        // ================================================================

        /// <summary>
        ///     BUG 4: Re-broadcasting an already-broadcasted array to a larger
        ///     shape throws NotSupportedException. NumPy handles this transparently.
        ///
        ///     Setup: broadcast_arrays produces (3,3) broadcasted result.
        ///     Then broadcast_to that (3,3) to (2,3,3) should just add a
        ///     new leading dimension with stride=0, but the IsBroadcasted
        ///     guard blocks it.
        /// </summary>
        [TestMethod]
        public void Bug_ReBroadcast_Inconsistency()
        {
            var x = np.ones(new Shape(1, 3));
            var y = np.ones(new Shape(3, 1));
            var (bx, _) = np.broadcast_arrays(x, y); // bx is (3,3), IsBroadcasted=true

            // NumPy: Re-broadcasting must work transparently
            NDArray result = null;
            new Action(() => result = np.broadcast_to(bx, new Shape(2, 3, 3)))
                .Should().NotThrow(
                    "NumPy handles re-broadcasting transparently: broadcast_to((3,3)→(2,3,3)) " +
                    "just adds a leading dimension with stride=0. NumSharp's 2-arg " +
                    "Broadcast(Shape,Shape) throws NotSupportedException due to an explicit " +
                    "guard: if (leftShape.IsBroadcasted || rightShape.IsBroadcasted) throw. " +
                    "The N-array overload Broadcast(Shape[]) lacks this guard, creating " +
                    "an inconsistency where broadcast_arrays succeeds but broadcast_to throws.");

            result.Should().NotBeNull();
            result.shape.Should().BeEquivalentTo(new[] { 2, 3, 3 });
        }

        // ================================================================
        //
        //  BUG 5 / BUG 9: np.minimum broadcast produces wrong values
        //
        //  SEVERITY: High — silently returns wrong computation results.
        //
        //  np.minimum with broadcasting produces transposed/swapped values.
        //  The bug affects ALL numeric types: int, float, double, long, byte.
        //
        //  NumPy: minimum([1,5,3], [[2],[4]]) = [[1,2,2],[1,4,3]]
        //  NumSharp: Returns [[1,4,2],[1,2,3]] for ALL types.
        //
        //  The pattern: row 0 should use b[0]=2 for comparisons, but gets
        //  b[1]=4. Row 1 should use b[1]=4, but gets b[0]=2. The b values
        //  are transposed between rows.
        //
        //  CRITICAL OBSERVATION: np.maximum with the EXACT same inputs
        //  returns CORRECT values: [[2,5,3],[4,5,4]]. This proves the
        //  broadcasting infrastructure (shape resolution, stride setup) is
        //  correct — the bug is specifically in minimum's iteration logic.
        //
        //  Most likely: minimum creates its MultiIterator or paired iteration
        //  with the two input arrays in swapped order compared to maximum.
        //  Since minimum(a,b) ≠ minimum(b,a) when both are < and > the
        //  comparison value, swapping the iteration order of the two broadcast
        //  dimensions produces the transposed result.
        //
        //  FIX APPROACH: Check the argument order in np.minimum's call to
        //  the broadcast iteration. Compare with np.maximum which works.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.minimum(np.array([1,5,3]), np.array([[2],[4]]))
        //    array([[1, 2, 2], [1, 4, 3]])
        //    >>> np.minimum(np.array([1.,5.,3.]), np.array([[2.],[4.]]))
        //    array([[1., 2., 2.], [1., 4., 3.]])
        //    >>> np.minimum(np.array([1,5,3], dtype=np.float32),
        //    ...            np.array([[2],[4]], dtype=np.float32))
        //    array([[1., 2., 2.], [1., 4., 3.]], dtype=float32)
        //
        // ================================================================

        /// <summary>
        ///     BUG 5: np.minimum with int32 broadcast returns wrong values.
        ///     Row 0 gets b[1]=4 instead of b[0]=2, and vice versa.
        ///
        ///     NumPy:   [[1, 2, 2], [1, 4, 3]]
        ///     NumSharp: [[1, 4, 2], [1, 2, 3]]  (b values transposed)
        /// </summary>
        [TestMethod]
        public void Bug_Minimum_IntBroadcast_WrongValues()
        {
            var a = np.array(new int[] { 1, 5, 3 });
            var b = np.array(new int[,] { { 2 }, { 4 } });
            var r = np.minimum(a, b);

            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });

            var expected = np.array(new int[,] { { 1, 2, 2 }, { 1, 4, 3 } });
            np.array_equal(r, expected).Should().BeTrue(
                "NumPy: minimum([1,5,3], [[2],[4]]) = [[1,2,2],[1,4,3]]. " +
                "NumSharp returns [[1,4,2],[1,2,3]] — the b column vector values " +
                "are transposed between rows, suggesting the broadcast iteration " +
                "in minimum reads the two inputs in swapped order. np.maximum " +
                "with identical inputs returns correct values.");
        }

        /// <summary>
        ///     BUG 9a: np.minimum with double broadcast — same transposition bug.
        ///     Confirms the bug is type-independent (not an int-specific code path).
        ///
        ///     NumPy:   [[1., 2., 2.], [1., 4., 3.]]
        ///     NumSharp: [[1., 4., 2.], [1., 2., 3.]]
        /// </summary>
        [TestMethod]
        public void Bug_Minimum_DoubleBroadcast_WrongValues()
        {
            var a = np.array(new double[] { 1, 5, 3 });
            var b = np.array(new double[,] { { 2 }, { 4 } });
            var r = np.minimum(a, b);

            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });

            var expected = np.array(new double[,] { { 1, 2, 2 }, { 1, 4, 3 } });
            np.array_equal(r, expected).Should().BeTrue(
                "NumPy: minimum([1.,5.,3.], [[2.],[4.]]) = [[1,2,2],[1,4,3]]. " +
                "NumSharp returns [[1,4,2],[1,2,3]] for double — same transposition " +
                "as int, confirming the bug is in the iteration logic, not type-specific.");
        }

        /// <summary>
        ///     BUG 9b: np.minimum with float32 broadcast — same transposition bug.
        ///
        ///     NumPy:   [[1f, 2f, 2f], [1f, 4f, 3f]]
        ///     NumSharp: [[1f, 4f, 2f], [1f, 2f, 3f]]
        /// </summary>
        [TestMethod]
        public void Bug_Minimum_FloatBroadcast_WrongValues()
        {
            var a = np.array(new float[] { 1f, 5f, 3f });
            var b = np.array(new float[,] { { 2f }, { 4f } });
            var r = np.minimum(a, b);

            var expected = np.array(new float[,] { { 1f, 2f, 2f }, { 1f, 4f, 3f } });
            np.array_equal(r, expected).Should().BeTrue(
                "NumPy: minimum([1,5,3], [[2],[4]]) = [[1,2,2],[1,4,3]] for float32. " +
                "NumSharp returns [[1,4,2],[1,2,3]] — identical transposition to int/double.");
        }

        // ================================================================
        //
        //  BUG 6: != operator throws InvalidCastException (NDArray vs NDArray)
        //
        //  SEVERITY: High — element-wise != between arrays is non-functional.
        //
        //  The != operator (NDArray.NotEquals.cs) has a different bug from
        //  the other comparison operators. While >, <, >=, <= throw
        //  IncorrectShapeException (they don't support NDArray vs NDArray at
        //  all — see Bug 8), != attempts to handle it but goes through the
        //  wrong code path.
        //
        //  The operator signature is: op_Inequality(NDArray np, Object obj)
        //  When the right operand is an NDArray, it gets boxed as Object.
        //  The implementation then tries to convert it via IConvertible
        //  (likely to treat it as a scalar), which fails because NDArray
        //  doesn't implement IConvertible.
        //
        //  The == operator works correctly for NDArray vs NDArray because
        //  it has a separate overload that handles this case with broadcasting.
        //
        //  FIX APPROACH: Add an NDArray-typed overload for !=, similar to ==.
        //  Or fix the Object overload to detect NDArray and dispatch to
        //  element-wise comparison with broadcasting.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.array([1,2,3]) != np.array([[1],[2],[3]])
        //    array([[False,  True,  True],
        //           [ True, False,  True],
        //           [ True,  True, False]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 6: The != operator throws InvalidCastException when both
        ///     operands are NDArrays (even same shape). The implementation
        ///     tries to cast the NDArray RHS to IConvertible (scalar path)
        ///     instead of performing element-wise comparison.
        ///
        ///     NumPy: Returns (3,3) bool with False on diagonal.
        ///     NumSharp: InvalidCastException: Unable to cast 'NDArray' to 'IConvertible'.
        /// </summary>
        [TestMethod]
        public void Bug_NotEquals_NDArrayBroadcast_Throws()
        {
            var a = np.array(new int[] { 1, 2, 3 });
            var b = np.array(new int[,] { { 1 }, { 2 }, { 3 } });

            NDArray result = null;
            new Action(() => { result = (a != b); })
                .Should().NotThrow(
                    "NumPy: array([1,2,3]) != array([[1],[2],[3]]) returns a (3,3) bool array. " +
                    "NumSharp throws InvalidCastException because op_Inequality(NDArray, Object) " +
                    "tries to convert the NDArray RHS via IConvertible instead of dispatching " +
                    "to element-wise broadcast comparison.");

            result.Should().NotBeNull();
            result.shape.Should().BeEquivalentTo(new[] { 3, 3 });

            // Diagonal = False (equal), off-diagonal = True (not equal)
            result.GetBoolean(0, 0).Should().BeFalse("1 == 1");
            result.GetBoolean(1, 1).Should().BeFalse("2 == 2");
            result.GetBoolean(2, 2).Should().BeFalse("3 == 3");
            result.GetBoolean(0, 1).Should().BeTrue("1 != 2");
            result.GetBoolean(1, 0).Should().BeTrue("2 != 1");
        }

        // ================================================================
        //
        //  BUG 7: np.allclose ALWAYS throws NullReferenceException
        //
        //  SEVERITY: Critical — the function is entirely non-functional.
        //
        //  np.allclose throws NullReferenceException for ALL inputs,
        //  including allclose(a, a) with a simple 1D array. This is NOT
        //  a broadcast-specific bug — the function is fundamentally broken.
        //
        //  The crash trace:
        //    np.allclose(a, b)
        //    → DefaultEngine.AllClose(a, b, rtol, atol, equal_nan)
        //      at Default.AllClose.cs:line 23
        //    → np.all(NDArray a)
        //      at np.all.cs:line 29 ← NullReferenceException
        //
        //  The AllClose implementation likely computes abs(a-b) <= atol+rtol*abs(b)
        //  and then calls np.all() on the boolean result. But somewhere in
        //  that chain, a null NDArray is produced (possibly from the <= operator
        //  failing to handle NDArray vs NDArray — see Bug 8) and passed to
        //  np.all(), which dereferences it and throws NullReferenceException.
        //
        //  FIX APPROACH: Debug Default.AllClose.cs line 23. The intermediate
        //  boolean array from the element-wise comparison is likely null.
        //  Fixing Bug 8 (comparison operators) may cascade-fix this.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.allclose(np.array([1.,2.,3.]), np.array([1.,2.,3.]))
        //    True
        //    >>> np.allclose(np.array([1.,2.,3.]),
        //    ...             np.array([[1.,2.,3.],[1.,2.,3.]]))
        //    True
        //
        // ================================================================

        /// <summary>
        ///     BUG 7a: np.allclose crashes even for the simplest case:
        ///     comparing an array to itself.
        ///
        ///     NumPy: allclose(a, a) returns True.
        ///     NumSharp: NullReferenceException at np.all.cs:line 29.
        /// </summary>
        [TestMethod]
        public void Bug_Allclose_AlwaysThrows()
        {
            var a = np.array(new double[] { 1.0, 2.0, 3.0 });

            bool result = false;
            new Action(() => result = np.allclose(a, a))
                .Should().NotThrow(
                    "NumPy: allclose(a, a) returns True — the simplest possible case. " +
                    "NumSharp throws NullReferenceException because the intermediate boolean " +
                    "array (from abs(a-a) <= atol comparison) is null when passed to np.all(). " +
                    "This likely cascades from Bug 8 (<= operator not supporting NDArray vs NDArray).");

            result.Should().BeTrue();
        }

        /// <summary>
        ///     BUG 7b: np.allclose also crashes with broadcast-compatible shapes.
        ///     (Consequence of 7a — documenting separately for completeness.)
        ///
        ///     NumPy: allclose(shape(3,), shape(2,3)) returns True when all elements match.
        ///     NumSharp: NullReferenceException.
        /// </summary>
        [TestMethod]
        public void Bug_Allclose_BroadcastThrows()
        {
            var a = np.array(new double[] { 1.0, 2.0, 3.0 });         // shape (3,)
            var b = np.array(new double[,] { { 1, 2, 3 }, { 1, 2, 3 } }); // shape (2,3)

            bool result = false;
            new Action(() => result = np.allclose(a, b))
                .Should().NotThrow(
                    "NumPy: allclose(array([1,2,3]), array([[1,2,3],[1,2,3]])) returns True. " +
                    "NumSharp throws NullReferenceException — same root cause as Bug 7a.");

            result.Should().BeTrue();
        }

        // ================================================================
        //
        //  BUG 8: >, <, >=, <= throw IncorrectShapeException
        //         for NDArray vs NDArray (even same shape)
        //
        //  SEVERITY: Critical — element-wise comparison between arrays is
        //  entirely missing for 4 of 6 comparison operators.
        //
        //  The comparison operators >, <, >=, <= only support NDArray vs
        //  scalar (e.g., array > 5). When both operands are NDArray — even
        //  with the EXACT SAME shape — they throw IncorrectShapeException:
        //  "This method does not work with this shape or was not already
        //  implemented."
        //
        //  This contrasts with == which correctly supports NDArray vs NDArray
        //  with broadcasting. The != operator attempts it but crashes via
        //  a different path (see Bug 6).
        //
        //  Operator support matrix:
        //    ==  NDArray vs NDArray: WORKS (with broadcasting)
        //    !=  NDArray vs NDArray: CRASHES (InvalidCastException) — Bug 6
        //    >   NDArray vs NDArray: CRASHES (IncorrectShapeException) — Bug 8
        //    <   NDArray vs NDArray: CRASHES (IncorrectShapeException) — Bug 8
        //    >=  NDArray vs NDArray: CRASHES (IncorrectShapeException) — Bug 8
        //    <=  NDArray vs NDArray: CRASHES (IncorrectShapeException) — Bug 8
        //    All NDArray vs scalar: WORKS
        //
        //  This also cascade-causes Bug 7 (np.allclose) because allclose
        //  internally needs to evaluate abs(a-b) <= tolerance, which requires
        //  NDArray <= NDArray to work.
        //
        //  FIX APPROACH: Implement NDArray vs NDArray comparison in
        //  NDArray.Greater.cs and NDArray.Lower.cs, following the pattern
        //  used by NDArray.Equals.cs (which works). Should support both
        //  same-shape and broadcast-compatible shapes.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.array([1,5,3]) > np.array([2,4,3])
        //    array([False,  True, False])
        //    >>> np.array([1,5,3]) > np.array([[2],[4]])
        //    array([[False,  True,  True], [False,  True, False]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 8a: > operator throws for NDArray vs NDArray, same shape.
        ///     Even the simplest case (two 1D arrays of equal length) fails.
        ///
        ///     NumPy: array([1,5,3]) > array([2,4,3]) = [False,True,False]
        ///     NumSharp: IncorrectShapeException
        /// </summary>
        [TestMethod]
        public void Bug_GreaterThan_NDArrayVsNDArray_SameShape()
        {
            var a = np.array(new int[] { 1, 5, 3 });
            var b = np.array(new int[] { 2, 4, 3 });

            NDArray result = null;
            new Action(() => { result = (a > b); })
                .Should().NotThrow(
                    "NumPy: array([1,5,3]) > array([2,4,3]) returns [False,True,False]. " +
                    "NumSharp throws IncorrectShapeException because the > operator only " +
                    "supports NDArray vs scalar, not NDArray vs NDArray.");

            result.Should().NotBeNull();
            result.GetBoolean(0).Should().BeFalse("1 > 2 is False");
            result.GetBoolean(1).Should().BeTrue("5 > 4 is True");
            result.GetBoolean(2).Should().BeFalse("3 > 3 is False");
        }

        /// <summary>
        ///     BUG 8b: &lt; operator throws for NDArray vs NDArray, same shape.
        ///
        ///     NumPy: array([1,5,3]) &lt; array([2,4,3]) = [True,False,False]
        ///     NumSharp: IncorrectShapeException
        /// </summary>
        [TestMethod]
        public void Bug_LessThan_NDArrayVsNDArray_SameShape()
        {
            var a = np.array(new int[] { 1, 5, 3 });
            var b = np.array(new int[] { 2, 4, 3 });

            NDArray result = null;
            new Action(() => { result = (a < b); })
                .Should().NotThrow(
                    "NumPy: array([1,5,3]) < array([2,4,3]) returns [True,False,False]. " +
                    "NumSharp throws IncorrectShapeException — same missing implementation as >.");

            result.Should().NotBeNull();
            result.GetBoolean(0).Should().BeTrue("1 < 2 is True");
            result.GetBoolean(1).Should().BeFalse("5 < 4 is False");
            result.GetBoolean(2).Should().BeFalse("3 < 3 is False");
        }

        /// <summary>
        ///     BUG 8c: > operator throws for NDArray vs NDArray with broadcast.
        ///     Not only is same-shape broken, but broadcast-compatible shapes
        ///     also fail.
        ///
        ///     NumPy: array([1,5,3]) > array([[2],[4]]) = [[F,T,T],[F,T,F]]
        ///     NumSharp: IncorrectShapeException
        /// </summary>
        [TestMethod]
        public void Bug_GreaterThan_NDArrayVsNDArray_Broadcast()
        {
            var a = np.array(new int[] { 1, 5, 3 });        // (3,)
            var b = np.array(new int[,] { { 2 }, { 4 } });  // (2,1)

            NDArray result = null;
            new Action(() => { result = (a > b); })
                .Should().NotThrow(
                    "NumPy: array([1,5,3]) > array([[2],[4]]) returns (2,3) bool " +
                    "[[F,T,T],[F,T,F]]. NumSharp throws IncorrectShapeException.");

            result.Should().NotBeNull();
            result.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            result.GetBoolean(0, 0).Should().BeFalse("1 > 2 is False");
            result.GetBoolean(0, 1).Should().BeTrue("5 > 2 is True");
            result.GetBoolean(0, 2).Should().BeTrue("3 > 2 is True");
            result.GetBoolean(1, 0).Should().BeFalse("1 > 4 is False");
            result.GetBoolean(1, 1).Should().BeTrue("5 > 4 is True");
            result.GetBoolean(1, 2).Should().BeFalse("3 > 4 is False");
        }

        // ================================================================
        //
        //  BUG 10: np.unique returns unsorted results
        //
        //  SEVERITY: Medium — violates NumPy's sorted-output guarantee.
        //
        //  NumPy's np.unique() guarantees sorted output:
        //    "Returns the sorted unique elements of an array."
        //    — numpy.org/doc/stable/reference/generated/numpy.unique.html
        //
        //  NumSharp's np.unique returns elements in first-encounter order,
        //  not sorted. This is NOT a broadcast bug — it affects all arrays.
        //
        //  NumPy:   unique([3,1,2,1,3]) = [1,2,3]  (sorted)
        //  NumSharp: unique([3,1,2,1,3]) = [3,1,2]  (encounter order)
        //
        //  Root cause: The implementation likely uses a HashSet or LinkedHashSet
        //  to collect unique values, which preserves insertion order but does
        //  not sort. NumPy internally sorts the array first, then removes
        //  consecutive duplicates.
        //
        //  FIX APPROACH: Sort the unique values before returning. Either sort
        //  the result array, or use a SortedSet<T> instead of HashSet<T>.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.unique(np.array([3, 1, 2, 1, 3]))
        //    array([1, 2, 3])
        //
        // ================================================================

        /// <summary>
        ///     BUG 10: np.unique returns unsorted results.
        ///
        ///     NumPy: unique([3,1,2,1,3]) = [1,2,3] (always sorted).
        ///     NumSharp: Returns [3,1,2] (encounter/insertion order).
        /// </summary>
        [TestMethod]
        public void Bug_Unique_NotSorted()
        {
            var a = np.array(new int[] { 3, 1, 2, 1, 3 });
            var u = np.unique(a);

            u.size.Should().Be(3, "There are 3 unique values");

            // NumPy guarantees sorted output
            u.GetInt32(0).Should().Be(1,
                "NumPy: unique returns sorted unique values. First element must be 1. " +
                "NumSharp returns 3 (first-encountered value).");
            u.GetInt32(1).Should().Be(2,
                "NumPy: Second unique element must be 2. NumSharp returns 1.");
            u.GetInt32(2).Should().Be(3,
                "NumPy: Third unique element must be 3. NumSharp returns 2.");
        }

        // ================================================================
        //
        //  BUG 11: flatten() on column-broadcast gives wrong order
        //
        //  SEVERITY: High — silently returns wrong element ordering.
        //
        //  flatten() should always return elements in C-order (row-major):
        //  iterate the last dimension fastest. For a (3,3) array:
        //    [0,0] [0,1] [0,2] [1,0] [1,1] [1,2] [2,0] [2,1] [2,2]
        //
        //  For broadcast_to([[1],[2],[3]], (3,3)):
        //    C-order: [1,1,1, 2,2,2, 3,3,3]  (each row repeated)
        //
        //  NumSharp returns: [1,2,3, 1,2,3, 1,2,3]  (column-major order)
        //
        //  This is the flatten/ravel manifestation of Bug 1's iterator
        //  problem. The flatten implementation uses a linear traversal that
        //  doesn't correctly handle zero-stride broadcast dimensions.
        //
        //  When the COLUMN dimension has stride=0 (row-vector broadcast,
        //  e.g. broadcast_to([1,2,3], (3,3))), flatten happens to produce
        //  the correct result: [1,2,3,1,2,3,1,2,3] — because the linear
        //  traversal happens to match C-order for that memory layout.
        //
        //  But when the ROW dimension has stride=0 (column-vector broadcast,
        //  e.g. broadcast_to([[1],[2],[3]], (3,3))), the linear traversal
        //  iterates down columns first (Fortran-order) instead of across
        //  rows first (C-order).
        //
        //  Interestingly, np.ravel() on the same array returns CORRECT
        //  results [1,1,1,2,2,2,3,3,3]. This suggests ravel uses a
        //  different code path than flatten for broadcasted arrays.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.broadcast_to(np.array([[1],[2],[3]]), (3,3)).flatten()
        //    array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        //
        // ================================================================

        /// <summary>
        ///     BUG 11: flatten() on column-broadcast array iterates in
        ///     Fortran-order instead of C-order.
        ///
        ///     NumPy:   broadcast_to([[1],[2],[3]], (3,3)).flatten() = [1,1,1,2,2,2,3,3,3]
        ///     NumSharp: Returns [1,2,3,1,2,3,1,2,3] (column-major iteration).
        ///
        ///     Note: ravel() returns correct results for the same input,
        ///     suggesting it uses a different iteration path than flatten().
        /// </summary>
        [TestMethod]
        public void Bug_Flatten_ColumnBroadcast_WrongOrder()
        {
            var a = np.array(new int[,] { { 1 }, { 2 }, { 3 } }); // (3,1)
            var b = np.broadcast_to(a, new Shape(3, 3));

            var flat = b.flatten();
            flat.size.Should().Be(9);

            // NumPy C-order: row-major → [1,1,1,2,2,2,3,3,3]
            var expected = np.array(new int[] { 1, 1, 1, 2, 2, 2, 3, 3, 3 });
            np.array_equal(flat, expected).Should().BeTrue(
                "NumPy: broadcast_to([[1],[2],[3]], (3,3)).flatten() = [1,1,1,2,2,2,3,3,3] " +
                "(C-order / row-major). NumSharp returns [1,2,3,1,2,3,1,2,3] — iterating " +
                "down columns first (Fortran-order) because the flatten iterator doesn't " +
                "account for the zero-stride row dimension in the column-broadcast layout. " +
                "ravel() on the same array returns correct results via a different code path.");
        }

        // ================================================================
        //
        //  BUG 12: concatenate/hstack/vstack on broadcasted arrays
        //          produces wrong values
        //
        //  SEVERITY: High — silently returns wrong array contents.
        //
        //  np.concatenate (and hstack/vstack which delegate to it) produces
        //  wrong values when any input array is broadcasted. The shape of
        //  the result is correct, but the element values are wrong.
        //
        //  The bug is in the COPY step: when concatenate copies elements
        //  from a broadcasted source array into the destination, it uses
        //  a linear offset calculation that doesn't account for zero-stride
        //  broadcast dimensions. The result is that rows get duplicated
        //  (column-broadcast), shifted (sliced+broadcast), or contain
        //  garbage values (reading past allocation).
        //
        //  WORKAROUND: Call np.copy() on broadcasted arrays before passing
        //  them to concatenate/hstack/vstack. This materializes the broadcast
        //  into contiguous memory, which concatenate can then handle correctly.
        //
        //  Verified: hstack(copy(a), copy(b)) returns correct results.
        //
        //  Three variants tested:
        //    a) hstack with column-broadcast inputs → second row copies first
        //    b) vstack with sliced+broadcast input → rows shifted
        //    c) concatenate axis=0 with sliced+broadcast → garbage/zeros
        //
        //  PYTHON VERIFICATION:
        //    >>> a = np.broadcast_to(np.array([[1],[2]]), (2,2))
        //    >>> b = np.broadcast_to(np.array([[3],[4]]), (2,3))
        //    >>> np.hstack([a, b])
        //    array([[1, 1, 3, 3, 3], [2, 2, 4, 4, 4]])
        //    >>> x = np.arange(6).reshape(3,2)
        //    >>> np.vstack([np.broadcast_to(x[:,0:1],(3,3)), [[10,20,30]]])
        //    array([[ 0,  0,  0], [ 2,  2,  2], [ 4,  4,  4], [10, 20, 30]])
        //    >>> x = np.arange(12).reshape(3,4)
        //    >>> np.concatenate([np.broadcast_to(x[:,1:2],(3,3)), np.ones((1,3),dtype=int)])
        //    array([[1, 1, 1], [5, 5, 5], [9, 9, 9], [1, 1, 1]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 12a: hstack on column-broadcast arrays duplicates row 0
        ///     into row 1.
        ///
        ///     Setup: Two column-vector arrays broadcast to (2,2) and (2,3).
        ///     hstack should produce (2,5) with distinct rows.
        ///
        ///     NumPy:   [[1,1,3,3,3], [2,2,4,4,4]]
        ///     NumSharp: [[1,1,3,3,3], [1,1,3,3,3]]  (row 1 = copy of row 0)
        ///
        ///     The copy routine reads both rows from offset 0 of the broadcast
        ///     source, ignoring that the column dimension has stride=0 while
        ///     the row dimension has the original source stride.
        /// </summary>
        [TestMethod]
        public void Bug_Hstack_Broadcast_WrongValues()
        {
            var a = np.broadcast_to(np.array(new int[,] { { 1 }, { 2 } }), new Shape(2, 2));
            var b = np.broadcast_to(np.array(new int[,] { { 3 }, { 4 } }), new Shape(2, 3));
            var r = np.hstack(a, b);

            r.shape.Should().BeEquivalentTo(new[] { 2, 5 });

            var expected = np.array(new int[,] { { 1, 1, 3, 3, 3 }, { 2, 2, 4, 4, 4 } });
            np.array_equal(r, expected).Should().BeTrue(
                "NumPy: hstack(broadcast[[1],[2]]→(2,2), broadcast[[3],[4]]→(2,3)) = " +
                "[[1,1,3,3,3],[2,2,4,4,4]]. NumSharp returns [[1,1,3,3,3],[1,1,3,3,3]] — " +
                "row 1 is a copy of row 0 because the concatenate copy routine iterates " +
                "the broadcast source with a linear offset that doesn't account for the " +
                "zero-stride column dimension. Workaround: np.copy() inputs first.");
        }

        /// <summary>
        ///     BUG 12b: vstack with sliced+broadcast input shifts row values.
        ///
        ///     Setup: arange(6).reshape(3,2)[:,0:1] → (3,1): [[0],[2],[4]],
        ///     then broadcast_to (3,3), then vstack with [[10,20,30]].
        ///
        ///     NumPy:   [[0,0,0], [2,2,2], [4,4,4], [10,20,30]]
        ///     NumSharp: [[0,0,0], [0,0,0], [2,2,2], [10,20,30]]  (shifted)
        ///
        ///     Row 1 should be [2,2,2] but shows [0,0,0] (row 0's value).
        ///     Row 2 should be [4,4,4] but shows [2,2,2] (row 1's value).
        ///     The iteration is off by one row because the sliced source's
        ///     row stride is not correctly applied during the copy.
        /// </summary>
        [TestMethod]
        public void Bug_Vstack_SlicedBroadcast_WrongValues()
        {
            var x = np.arange(6).reshape(3, 2);
            var col = x[":, 0:1"]; // (3,1): [[0],[2],[4]]
            var bcol = np.broadcast_to(col, new Shape(3, 3));
            var other = np.array(new int[,] { { 10, 20, 30 } });

            var r = np.vstack(bcol, other);
            r.shape.Should().BeEquivalentTo(new[] { 4, 3 });

            r.GetInt32(0, 0).Should().Be(0, "Row 0 should be [0,0,0]");
            r.GetInt32(1, 0).Should().Be(2,
                "NumPy: Row 1 should be [2,2,2]. NumSharp returns [0,0,0] — " +
                "the concatenate copy routine is off by one row because the sliced " +
                "source's row stride (2 elements, skipping column 1) is not correctly " +
                "applied during linear iteration of the broadcast array.");
            r.GetInt32(2, 0).Should().Be(4, "Row 2 should be [4,4,4]");
            r.GetInt32(3, 0).Should().Be(10, "Row 3 should be [10,20,30]");
        }

        /// <summary>
        ///     BUG 12c: concatenate axis=0 with sliced+broadcast reads garbage.
        ///
        ///     Setup: arange(12).reshape(3,4)[:,1:2] → (3,1): [[1],[5],[9]],
        ///     then broadcast_to (3,3), then concatenate with ones(1,3).
        ///
        ///     NumPy:   [[1,1,1], [5,5,5], [9,9,9], [1,1,1]]
        ///     NumSharp: [[1,1,1], [5,5,5], [garbage], [1,1,1]]
        ///
        ///     Row 2 should be [9,9,9] but contains garbage values (e.g.
        ///     32765, 0, or other values depending on memory state). The
        ///     sliced column has row stride=4 (stepping over 4-wide rows).
        ///     The concatenate copy routine doesn't apply this stride,
        ///     reading from wrong memory offsets for the last row.
        /// </summary>
        [TestMethod]
        public void Bug_Concatenate_SlicedBroadcast_WrongValues()
        {
            var x = np.arange(12).reshape(3, 4);
            var col = x[":, 1:2"]; // (3,1): [[1],[5],[9]]
            var bcol = np.broadcast_to(col, new Shape(3, 3));
            var other = np.ones(new Shape(1, 3), np.int32);

            var r = np.concatenate(new NDArray[] { bcol, other }, axis: 0);
            r.shape.Should().BeEquivalentTo(new[] { 4, 3 });

            r.GetInt32(0, 0).Should().Be(1, "Row 0 should be [1,1,1]");
            r.GetInt32(1, 0).Should().Be(5, "Row 1 should be [5,5,5]");
            r.GetInt32(2, 0).Should().Be(9,
                "NumPy: Row 2 should be [9,9,9]. NumSharp reads garbage values because " +
                "the concatenate copy routine doesn't apply the sliced column's row stride " +
                "(4 elements per row in the source) when iterating the broadcast array. " +
                "The linear offset overshoots and reads from beyond the valid data.");
            r.GetInt32(3, 0).Should().Be(1, "Row 3 should be [1,1,1]");
        }

        // ================================================================
        //
        //  BUG 13: cumsum with axis on broadcast arrays reads garbage memory
        //
        //  SEVERITY: Critical — returns uninitialized memory values.
        //
        //  np.cumsum(broadcast_array, axis=0) returns garbage values
        //  (uninitialized memory like -1564032936, 32765, etc.) when
        //  reducing along the non-broadcast axis of a broadcast array.
        //
        //  The bug specifically manifests when axis=0 is the reduction
        //  axis and the broadcast was along axis=0 (row-broadcast) OR
        //  when axis=0 reduces a column-broadcast. The key pattern is:
        //  cumsum reads with wrong strides and accesses memory outside
        //  the source array's actual data region.
        //
        //  cumsum with axis=1 on row-broadcast: CORRECT
        //  cumsum with axis=0 on row-broadcast: WRONG (garbage)
        //  cumsum with axis=0 on col-broadcast: WRONG (garbage)
        //  cumsum with axis=1 on col-broadcast: WRONG (wrong accumulation)
        //  cumsum no-axis (flatten first): CORRECT
        //
        //  FIX APPROACH: The cumsum implementation likely uses linear
        //  pointer iteration along the reduction axis instead of
        //  coordinate-based access. It should use GetOffset(coords)
        //  to properly resolve broadcast zero-strides.
        //
        //  PYTHON VERIFICATION:
        //    >>> a = np.broadcast_to(np.array([1,2,3]), (3,3))
        //    >>> np.cumsum(a, axis=0)
        //    array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        //    >>> b = np.broadcast_to(np.array([[1],[2],[3]]), (3,3))
        //    >>> np.cumsum(b, axis=0)
        //    array([[1, 1, 1], [3, 3, 3], [6, 6, 6]])
        //    >>> np.cumsum(b, axis=1)
        //    array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 13a: cumsum axis=0 on row-broadcast returns garbage.
        ///
        ///     Setup: broadcast_to([1,2,3], (3,3)) — each row is [1,2,3].
        ///     cumsum(axis=0) should accumulate down columns: row 0 = [1,2,3],
        ///     row 1 = [2,4,6], row 2 = [3,6,9].
        ///
        ///     NumPy:   [[1,2,3], [2,4,6], [3,6,9]]
        ///     NumSharp: [[garbage, garbage, garbage], ...] (uninitialized memory)
        /// </summary>
        [TestMethod]
        public void Bug_Cumsum_Axis0_RowBroadcast_Garbage()
        {
            var a = np.broadcast_to(np.array(new int[] { 1, 2, 3 }), new Shape(3, 3));
            var result = np.cumsum(a, 0);

            result.shape.Should().BeEquivalentTo(new[] { 3, 3 });

            // Row 0 should be the original values
            result.GetInt32(0, 0).Should().Be(1, "cumsum axis=0 row 0 col 0");
            result.GetInt32(0, 1).Should().Be(2, "cumsum axis=0 row 0 col 1");
            result.GetInt32(0, 2).Should().Be(3, "cumsum axis=0 row 0 col 2");

            // Row 1 should be cumulative sum of 2 identical rows
            result.GetInt32(1, 0).Should().Be(2,
                "NumPy: cumsum(broadcast_to([1,2,3],(3,3)), axis=0)[1,0] = 2 (1+1). " +
                "NumSharp returns garbage values (e.g. 32765) because cumsum's axis " +
                "iteration reads from uninitialized memory instead of correctly " +
                "traversing the broadcast zero-stride dimension.");
            result.GetInt32(1, 1).Should().Be(4, "cumsum axis=0 row 1 col 1 = 2+2");
            result.GetInt32(1, 2).Should().Be(6, "cumsum axis=0 row 1 col 2 = 3+3");

            // Row 2 = sum of 3 identical rows
            result.GetInt32(2, 0).Should().Be(3, "cumsum axis=0 row 2 col 0 = 1+1+1");
            result.GetInt32(2, 1).Should().Be(6, "cumsum axis=0 row 2 col 1 = 2+2+2");
            result.GetInt32(2, 2).Should().Be(9, "cumsum axis=0 row 2 col 2 = 3+3+3");
        }

        /// <summary>
        ///     BUG 13b: cumsum axis=0 on column-broadcast returns garbage.
        ///
        ///     Setup: broadcast_to([[1],[2],[3]], (3,3)) — each row is constant.
        ///     cumsum(axis=0) should accumulate: [1,1,1] → [3,3,3] → [6,6,6].
        ///
        ///     NumPy:   [[1,1,1], [3,3,3], [6,6,6]]
        ///     NumSharp: [[garbage, garbage, garbage], ...]
        /// </summary>
        [TestMethod]
        public void Bug_Cumsum_Axis0_ColBroadcast_Garbage()
        {
            var col = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var result = np.cumsum(a, 0);

            result.shape.Should().BeEquivalentTo(new[] { 3, 3 });

            result.GetInt32(0, 0).Should().Be(1, "cumsum axis=0 row 0 = [1,1,1]");
            result.GetInt32(0, 1).Should().Be(1);
            result.GetInt32(0, 2).Should().Be(1);

            result.GetInt32(1, 0).Should().Be(3,
                "NumPy: cumsum(broadcast_to([[1],[2],[3]],(3,3)), axis=0)[1,0] = 3 (1+2). " +
                "NumSharp returns garbage values because the axis=0 iteration reads " +
                "memory at wrong offsets for column-broadcast arrays.");
            result.GetInt32(1, 1).Should().Be(3);
            result.GetInt32(1, 2).Should().Be(3);

            result.GetInt32(2, 0).Should().Be(6, "cumsum axis=0 row 2 = [6,6,6]");
            result.GetInt32(2, 1).Should().Be(6);
            result.GetInt32(2, 2).Should().Be(6);
        }

        /// <summary>
        ///     BUG 13c: cumsum axis=1 on column-broadcast returns wrong accumulation.
        ///
        ///     Setup: broadcast_to([[1],[2],[3]], (3,3)) — rows are [1,1,1], [2,2,2], [3,3,3].
        ///     cumsum(axis=1) should accumulate across each row:
        ///       row 0: [1, 2, 3]  (cumsum of [1,1,1])
        ///       row 1: [2, 4, 6]  (cumsum of [2,2,2])
        ///       row 2: [3, 6, 9]  (cumsum of [3,3,3])
        ///
        ///     NumPy:   [[1,2,3], [2,4,6], [3,6,9]]
        ///     NumSharp: [[3,3,3], [6,6,6], [9,9,9]]
        ///
        ///     The values suggest cumsum reads with wrong strides — it appears
        ///     to be accumulating along axis=0 instead of axis=1.
        /// </summary>
        [TestMethod]
        public void Bug_Cumsum_Axis1_ColBroadcast_Wrong()
        {
            var col = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var result = np.cumsum(a, 1);

            result.shape.Should().BeEquivalentTo(new[] { 3, 3 });

            // Row 0: cumsum of [1,1,1] = [1, 2, 3]
            result.GetInt32(0, 0).Should().Be(1, "cumsum([1,1,1])[0] = 1");
            result.GetInt32(0, 1).Should().Be(2,
                "NumPy: cumsum(axis=1) on column-broadcast row [1,1,1] gives [1,2,3]. " +
                "NumSharp returns [3,3,3] — it appears to accumulate along the wrong " +
                "axis because the zero-stride column dimension confuses the iteration.");
            result.GetInt32(0, 2).Should().Be(3, "cumsum([1,1,1])[2] = 3");

            // Row 1: cumsum of [2,2,2] = [2, 4, 6]
            result.GetInt32(1, 0).Should().Be(2);
            result.GetInt32(1, 1).Should().Be(4);
            result.GetInt32(1, 2).Should().Be(6);

            // Row 2: cumsum of [3,3,3] = [3, 6, 9]
            result.GetInt32(2, 0).Should().Be(3);
            result.GetInt32(2, 1).Should().Be(6);
            result.GetInt32(2, 2).Should().Be(9);
        }

        // ================================================================
        //
        //  BUG 14: roll on broadcast arrays produces wrong values
        //
        //  SEVERITY: High — silently returns zeros/wrong values.
        //
        //  NDArray.roll(shift, axis) on broadcast arrays produces wrong
        //  values. The first row may be correct, but subsequent rows
        //  contain zeros or garbage. Non-broadcast arrays work correctly.
        //
        //  The bug is in the roll implementation's element copy loop,
        //  which uses linear memory access that doesn't account for
        //  broadcast zero-strides. When reading source elements to
        //  shift them, it reads from wrong memory offsets beyond row 0.
        //
        //  row-broadcast (2,3) roll(1, axis=1):
        //    NumPy:   [[3,1,2], [3,1,2]]
        //    NumSharp: [[3,1,2], [0,0,0]]  (row 1 = zeros)
        //
        //  col-broadcast (3,3) roll(1, axis=0):
        //    NumPy:   [[30,30,30], [10,10,10], [20,20,20]]
        //    NumSharp: [[30,30,30], [0,0,0], [0,0,0]]
        //
        //  PYTHON VERIFICATION:
        //    >>> np.roll(np.broadcast_to(np.array([1,2,3]), (2,3)), 1, axis=1)
        //    array([[3, 1, 2], [3, 1, 2]])
        //    >>> a = np.broadcast_to(np.array([[10],[20],[30]]), (3,3))
        //    >>> np.roll(a, 1, axis=0)
        //    array([[30, 30, 30], [10, 10, 10], [20, 20, 20]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 14a: roll(shift=1, axis=1) on row-broadcast produces zeros
        ///     in the second row.
        ///
        ///     Setup: broadcast_to([1,2,3], (2,3)) — both rows are [1,2,3].
        ///     roll(1, axis=1) shifts columns right by 1.
        ///
        ///     NumPy:   [[3,1,2], [3,1,2]]
        ///     NumSharp: [[3,1,2], [0,0,0]]
        /// </summary>
        [TestMethod]
        public void Bug_Roll_RowBroadcast_ZerosInSecondRow()
        {
            var a = np.broadcast_to(np.array(new int[] { 1, 2, 3 }), new Shape(2, 3));
            var result = a.roll(1, 1);

            result.shape.Should().BeEquivalentTo(new[] { 2, 3 });

            // Row 0 happens to be correct
            result.GetInt32(0, 0).Should().Be(3, "roll shift=1: last element wraps to front");
            result.GetInt32(0, 1).Should().Be(1);
            result.GetInt32(0, 2).Should().Be(2);

            // Row 1 should be identical (same source data)
            result.GetInt32(1, 0).Should().Be(3,
                "NumPy: roll(broadcast_to([1,2,3],(2,3)), 1, axis=1) row 1 = [3,1,2]. " +
                "NumSharp returns [0,0,0] because the roll implementation reads row 1's " +
                "source data using linear offset that lands in zeroed memory instead of " +
                "the broadcast zero-stride repeated row.");
            result.GetInt32(1, 1).Should().Be(1);
            result.GetInt32(1, 2).Should().Be(2);
        }

        /// <summary>
        ///     BUG 14b: roll(shift=1, axis=0) on column-broadcast produces zeros.
        ///
        ///     Setup: broadcast_to([[10],[20],[30]], (3,3)).
        ///     roll(1, axis=0) shifts rows down by 1.
        ///
        ///     NumPy:   [[30,30,30], [10,10,10], [20,20,20]]
        ///     NumSharp: [[30,30,30], [0,0,0], [0,0,0]]
        /// </summary>
        [TestMethod]
        public void Bug_Roll_ColBroadcast_ZerosAfterFirstRow()
        {
            var col = np.array(new int[,] { { 10 }, { 20 }, { 30 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var result = a.roll(1, 0);

            result.shape.Should().BeEquivalentTo(new[] { 3, 3 });

            // Row 0 = last original row (30) shifted to front
            result.GetInt32(0, 0).Should().Be(30, "roll shift=1 axis=0: last row wraps to front");
            result.GetInt32(0, 1).Should().Be(30);
            result.GetInt32(0, 2).Should().Be(30);

            // Row 1 = original row 0 (10)
            result.GetInt32(1, 0).Should().Be(10,
                "NumPy: roll(broadcast_to([[10],[20],[30]],(3,3)), 1, axis=0)[1] = [10,10,10]. " +
                "NumSharp returns [0,0,0] because the roll implementation reads with " +
                "linear offsets that don't account for the column-broadcast zero-strides.");
            result.GetInt32(1, 1).Should().Be(10);
            result.GetInt32(1, 2).Should().Be(10);

            // Row 2 = original row 1 (20)
            result.GetInt32(2, 0).Should().Be(20);
            result.GetInt32(2, 1).Should().Be(20);
            result.GetInt32(2, 2).Should().Be(20);
        }

        // ================================================================
        //
        //  BUG 15: sum/mean/var/std with axis=0 on column-broadcast
        //          returns wrong values (under-counts rows)
        //
        //  SEVERITY: Critical — silently returns wrong computation results.
        //
        //  When reducing along axis=0 on a column-broadcast array, the
        //  reduction functions (sum, mean, var, std) produce wrong values.
        //  The sum appears to read fewer elements than the actual number
        //  of rows, and since mean/var/std all depend on sum, they are
        //  all affected.
        //
        //  The bug pattern:
        //    - Row-broadcast + axis=0: CORRECT
        //    - Row-broadcast + axis=1: CORRECT
        //    - Col-broadcast + axis=1: CORRECT
        //    - Col-broadcast + axis=0: WRONG ← the buggy combination
        //
        //  This suggests the reduction's axis=0 iteration uses strides
        //  that are wrong for the column-broadcast memory layout. The
        //  iteration likely uses the physical stride (which is the
        //  original column source's row stride) instead of the broadcast
        //  shape's row stride.
        //
        //  Examples:
        //    broadcast_to([[100],[200],[300]], (3,3)):
        //      sum(axis=0) = [300,300,300] instead of [600,600,600]
        //      (sums to 100+200=300 instead of 100+200+300=600)
        //
        //    broadcast_to([[1],[2],[3],[4],[5]], (5,3)):
        //      sum(axis=0) = [7,7,7] instead of [15,15,15]
        //
        //  sum no-axis (flattens first): CORRECT → the flatten path handles
        //  broadcast correctly for the total sum, but the axis reduction path
        //  doesn't.
        //
        //  PYTHON VERIFICATION:
        //    >>> a = np.broadcast_to(np.array([[100],[200],[300]]), (3,3))
        //    >>> np.sum(a, axis=0)
        //    array([600, 600, 600])
        //    >>> np.mean(a.astype(float), axis=0)
        //    array([200., 200., 200.])
        //    >>> b = np.broadcast_to(np.array([[1.],[2.],[3.]]), (3,3))
        //    >>> np.var(b, axis=0)
        //    array([0.66666667, 0.66666667, 0.66666667])
        //    >>> np.std(b, axis=0)
        //    array([0.81649658, 0.81649658, 0.81649658])
        //
        // ================================================================

        /// <summary>
        ///     BUG 15a: sum(axis=0) on column-broadcast returns wrong totals.
        ///
        ///     Setup: broadcast_to([[100],[200],[300]], (3,3)).
        ///     Each column should sum to 100+200+300=600.
        ///
        ///     NumPy:   [600, 600, 600]
        ///     NumSharp: [300, 300, 300]  (under-counts, appears to miss row 2)
        /// </summary>
        [TestMethod]
        public void Bug_Sum_Axis0_ColBroadcast_WrongValues()
        {
            var col = np.array(new int[,] { { 100 }, { 200 }, { 300 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var result = np.sum(a, 0);

            result.shape.Should().BeEquivalentTo(new[] { 3 });

            result.GetInt32(0).Should().Be(600,
                "NumPy: sum(broadcast_to([[100],[200],[300]],(3,3)), axis=0) = [600,600,600]. " +
                "NumSharp returns [300,300,300] — only summing 2 of 3 rows (100+200=300). " +
                "The axis=0 reduction iteration uses wrong strides for column-broadcast arrays, " +
                "causing it to read fewer rows than exist in the broadcast shape.");
            result.GetInt32(1).Should().Be(600);
            result.GetInt32(2).Should().Be(600);
        }

        /// <summary>
        ///     BUG 15b: mean(axis=0) on column-broadcast returns wrong average.
        ///     Cascades from Bug 15a (sum is wrong → mean = sum/count is wrong).
        ///
        ///     NumPy:   mean = [2.0, 2.0, 2.0]
        ///     NumSharp: mean = [1.0, 1.0, 1.0]
        /// </summary>
        [TestMethod]
        public void Bug_Mean_Axis0_ColBroadcast_WrongValues()
        {
            var col = np.array(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var result = np.mean(a, 0);

            result.shape.Should().BeEquivalentTo(new[] { 3 });

            result.GetDouble(0).Should().BeApproximately(2.0, 1e-10,
                "NumPy: mean(broadcast_to([[1],[2],[3]],(3,3)), axis=0) = [2,2,2]. " +
                "NumSharp returns [1,1,1] because the underlying sum is wrong (Bug 15a), " +
                "giving sum=3 instead of sum=6, so mean=3/3=1 instead of 6/3=2.");
            result.GetDouble(1).Should().BeApproximately(2.0, 1e-10);
            result.GetDouble(2).Should().BeApproximately(2.0, 1e-10);
        }

        /// <summary>
        ///     BUG 15c: var(axis=0) on column-broadcast returns wrong variance.
        ///
        ///     NumPy:   var = [0.6667, 0.6667, 0.6667]
        ///     NumSharp: var = [0.0, 0.0, 0.0]
        /// </summary>
        [TestMethod]
        public void Bug_Var_Axis0_ColBroadcast_WrongValues()
        {
            var col = np.array(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var result = np.var(a, 0);

            result.shape.Should().BeEquivalentTo(new[] { 3 });

            result.GetDouble(0).Should().BeApproximately(2.0 / 3.0, 1e-10,
                "NumPy: var(broadcast_to([[1],[2],[3]],(3,3)), axis=0) = [0.667,0.667,0.667]. " +
                "NumSharp returns [0,0,0] because the wrong mean (Bug 15b) cascades: " +
                "mean=1 makes each deviation wrong, producing incorrect variance.");
            result.GetDouble(1).Should().BeApproximately(2.0 / 3.0, 1e-10);
            result.GetDouble(2).Should().BeApproximately(2.0 / 3.0, 1e-10);
        }

        /// <summary>
        ///     BUG 15d: std(axis=0) on column-broadcast returns wrong std deviation.
        ///
        ///     NumPy:   std = [0.8165, 0.8165, 0.8165]
        ///     NumSharp: std = [0.0, 0.0, 0.0]
        /// </summary>
        [TestMethod]
        public void Bug_Std_Axis0_ColBroadcast_WrongValues()
        {
            var col = np.array(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } });
            var a = np.broadcast_to(col, new Shape(3, 3));
            var result = np.std(a, 0);

            result.shape.Should().BeEquivalentTo(new[] { 3 });

            var expected_std = Math.Sqrt(2.0 / 3.0);
            result.GetDouble(0).Should().BeApproximately(expected_std, 1e-10,
                "NumPy: std(broadcast_to([[1],[2],[3]],(3,3)), axis=0) = [0.8165,...]. " +
                "NumSharp returns [0,0,0]. Root cause: sum(axis=0) is wrong (Bug 15a), " +
                "which cascades through mean → var → std.");
            result.GetDouble(1).Should().BeApproximately(expected_std, 1e-10);
            result.GetDouble(2).Should().BeApproximately(expected_std, 1e-10);
        }

        /// <summary>
        ///     BUG 15e: sum(axis=0) on a larger column-broadcast (5,3) returns wrong totals.
        ///     Verifies the bug scales with array size (not just 3x3).
        ///
        ///     NumPy:   sum(axis=0) = [15, 15, 15]  (1+2+3+4+5)
        ///     NumSharp: sum(axis=0) = [7, 7, 7]
        /// </summary>
        [TestMethod]
        public void Bug_Sum_Axis0_ColBroadcast_5x3_WrongValues()
        {
            var col = np.array(new int[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
            var a = np.broadcast_to(col, new Shape(5, 3));
            var result = np.sum(a, 0);

            result.shape.Should().BeEquivalentTo(new[] { 3 });

            result.GetInt32(0).Should().Be(15,
                "NumPy: sum(broadcast_to([[1],[2],[3],[4],[5]],(5,3)), axis=0) = [15,15,15]. " +
                "NumSharp returns [7,7,7]. The wrong value (7 instead of 15) confirms the " +
                "axis=0 reduction iteration reads wrong memory offsets for column-broadcast.");
            result.GetInt32(1).Should().Be(15);
            result.GetInt32(2).Should().Be(15);
        }

        // ================================================================
        //
        //  BUG 16: argsort crashes on any 2D array (not broadcast-specific)
        //
        //  SEVERITY: High — the function is non-functional for 2D+.
        //
        //  NDArray.argsort<T>() throws InvalidOperationException:
        //  "Failed to compare two elements in the array" for ANY 2D
        //  array, whether broadcast or not. The function works correctly
        //  for 1D arrays.
        //
        //  This is NOT a broadcast-specific bug. It affects all 2D arrays.
        //  The root cause is likely that the internal comparison function
        //  used for sorting doesn't handle multi-dimensional data correctly
        //  — it may be comparing elements across rows instead of within
        //  each row (axis=-1 default).
        //
        //  PYTHON VERIFICATION:
        //    >>> np.argsort(np.array([[3,1,2],[6,4,5]]))
        //    array([[1, 2, 0], [1, 2, 0]])
        //    >>> np.argsort(np.array([[3,1,2],[6,4,5]]), axis=0)
        //    array([[0, 0, 0], [1, 1, 1]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 16a: argsort crashes on any 2D int array.
        ///
        ///     NumPy:   argsort([[3,1,2],[6,4,5]]) = [[1,2,0],[1,2,0]]
        ///     NumSharp: InvalidOperationException: Failed to compare two elements
        /// </summary>
        [TestMethod]
        public void Bug_Argsort_2D_Crashes()
        {
            var a = np.array(new int[,] { { 3, 1, 2 }, { 6, 4, 5 } });

            NDArray result = null;
            new Action(() => result = a.argsort<int>())
                .Should().NotThrow(
                    "NumPy: argsort([[3,1,2],[6,4,5]]) returns [[1,2,0],[1,2,0]]. " +
                    "NumSharp throws InvalidOperationException: 'Failed to compare two " +
                    "elements in the array' for ANY 2D array. The comparison function " +
                    "used in the internal sort doesn't handle multi-dimensional indexing.");

            result.Should().NotBeNull();
            result.shape.Should().BeEquivalentTo(new[] { 2, 3 });

            // Row 0: argsort of [3,1,2] = [1,2,0]
            result.GetInt32(0, 0).Should().Be(1, "index of min(3,1,2)=1 is at position 1");
            result.GetInt32(0, 1).Should().Be(2, "index of 2 is at position 2");
            result.GetInt32(0, 2).Should().Be(0, "index of max(3,1,2)=3 is at position 0");

            // Row 1: argsort of [6,4,5] = [1,2,0]
            result.GetInt32(1, 0).Should().Be(1, "index of min(6,4,5)=4 is at position 1");
            result.GetInt32(1, 1).Should().Be(2, "index of 5 is at position 2");
            result.GetInt32(1, 2).Should().Be(0, "index of max(6,4,5)=6 is at position 0");
        }

        /// <summary>
        ///     BUG 16b: argsort crashes on 2D double array too.
        ///     Confirms the bug is type-independent.
        ///
        ///     NumPy:   argsort([[3.0,1.0,2.0]]) = [[1,2,0]]
        ///     NumSharp: InvalidOperationException
        /// </summary>
        [TestMethod]
        public void Bug_Argsort_2D_Double_Crashes()
        {
            var a = np.array(new double[,] { { 3.0, 1.0, 2.0 } });

            NDArray result = null;
            new Action(() => result = a.argsort<double>())
                .Should().NotThrow(
                    "NumPy: argsort([[3.0,1.0,2.0]]) returns [[1,2,0]]. " +
                    "NumSharp throws InvalidOperationException for any 2D array, " +
                    "regardless of dtype.");

            result.Should().NotBeNull();
            result.GetInt32(0, 0).Should().Be(1);
            result.GetInt32(0, 1).Should().Be(2);
            result.GetInt32(0, 2).Should().Be(0);
        }

        // ================================================================
        //
        //  BUG 4 VARIANT: np.clip on broadcast throws NotSupportedException
        //
        //  Same root cause as Bug 4 — the IsBroadcasted guard in
        //  Broadcast(Shape, Shape) blocks the operation. Clip internally
        //  broadcasts the input with the min/max bounds, hitting the guard.
        //
        //  Documented here as a test case but not a new distinct bug.
        //
        //  PYTHON VERIFICATION:
        //    >>> np.clip(np.broadcast_to(np.array([1.,5.,9.]), (2,3)), 2., 7.)
        //    array([[2., 5., 7.], [2., 5., 7.]])
        //
        // ================================================================

        /// <summary>
        ///     BUG 4 VARIANT: np.clip on broadcast array throws because clip
        ///     internally broadcasts and hits the IsBroadcasted guard.
        ///
        ///     NumPy:   clip(broadcast, 2, 7) = [[2,5,7],[2,5,7]]
        ///     NumSharp: NotSupportedException: Unable to broadcast already broadcasted shape.
        /// </summary>
        [TestMethod]
        public void Bug_Clip_Broadcast_ThrowsNotSupported()
        {
            var a = np.broadcast_to(np.array(new double[] { 1.0, 5.0, 9.0 }), new Shape(2, 3));

            NDArray result = null;
            new Action(() => result = np.clip(a, 2.0, 7.0))
                .Should().NotThrow(
                    "NumPy: clip(broadcast_to([1,5,9],(2,3)), 2, 7) = [[2,5,7],[2,5,7]]. " +
                    "NumSharp throws NotSupportedException: 'Unable to broadcast already " +
                    "broadcasted shape' — same IsBroadcasted guard as Bug 4. Clip internally " +
                    "broadcasts the input to apply min/max, hitting the guard.");

            result.Should().NotBeNull();
            result.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            result.GetDouble(0, 0).Should().Be(2.0);
            result.GetDouble(0, 1).Should().Be(5.0);
            result.GetDouble(0, 2).Should().Be(7.0);
        }
    }
}
