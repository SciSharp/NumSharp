using System;
using System.Diagnostics;

namespace NumSharp.Tests.AuditV2
{
    /// <summary>
    ///     Audit v2 — Group 4: Logic + Shape + Storage.
    ///     Each test reproduces a real correctness gap documented in
    ///     <c>docs/plans/audit_v2/04_logic_shape_storage.md</c>.
    ///     Tests are marked [OpenBugs] so CI skips them until the underlying
    ///     defect is fixed.
    /// </summary>
    [TestClass]
    public class AuditV2_LogicShapeStorage
    {
        // ============================================================================
        // T1.10 — Shape is a `readonly struct` with a mutating set indexer.
        // File: src/NumSharp.Core/View/Shape.cs
        //
        // FIXED (2026-10-05, docs/plans/view-shape-open-bugs.md bug 2a): the indexer
        // has no setter any more. It used to write into the referenced
        // `long[] dimensions` array, leaving `_flags`, `size` and `_hashCode` stale
        // (Equals/GetHashCode/Size/IsContiguous silently wrong), and because shapes
        // and arrays share dimension arrays, it also changed every array sharing it.
        // ============================================================================
        /// <summary>
        /// T1.10 — FIXED: the <see cref="Shape"/> indexer is read-only, so a shape cannot be mutated through it and its
        /// cached size, hash and equality always agree with its dimensions (NumPy's shape is an immutable tuple).
        /// </summary>
        [TestMethod]
        public void T1_10_Shape_Indexer_IsReadOnly()
        {
            var indexer = typeof(Shape).GetProperty("Item", new[] { typeof(int) });
            indexer.Should().NotBeNull("Shape keeps its read accessor shape[i]");
            indexer!.CanWrite.Should().BeFalse("Shape is immutable, like the tuple NumPy hands out");

            var shape = new Shape(new long[] { 4, 5, 6 });
            shape[0].Should().Be(4);
            shape[-1].Should().Be(6, "a negative axis counts from the end");
            shape.Size.Should().Be(120);

            var rebuilt = new Shape(new long[] { 4, 5, 6 });
            shape.GetHashCode().Should().Be(rebuilt.GetHashCode());
            shape.Equals(rebuilt).Should().BeTrue();
        }

        // ============================================================================
        // T1.13 / T1.57 — UnmanagedStorage.SetValue(object, ...) and CopyTo paths
        // miss SByte / Half / Complex on the #else branches.
        // File: src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Setters.cs
        //       (SetValue(object,int[]) lines 161-218, SetValue(object,long[]) 229-273)
        // File: src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.cs
        //       (CopyTo(void*) lines 1245-1350, CopyTo(IMemoryBlock) lines 1356-1467)
        // The expanded switches cover only 12 dtypes (no SByte/Half/Complex) and
        // fall through to `default: throw new NotSupportedException();`.
        // ============================================================================
        /// <summary>
        /// T1.13 — UnmanagedStorage.SetValue(object, long[]) throws on SByte/Half/Complex.
        /// Generic typed `SetValue&lt;T&gt;` works; only the boxing-object overload is broken.
        /// </summary>
        [TestMethod]
        public void T1_13_SetValue_Object_LongIndices_Missing_SByte_Half_Complex()
        {
            // SByte
            var sb = np.zeros(new Shape(3), typeof(sbyte));
            Action setSByte = () => sb.Storage.SetValue((object)(sbyte)5, (long)0);
            setSByte.Should().NotThrow("SetValue(object, long[]) must support SByte");

            // Half
            var hf = np.zeros(new Shape(3), typeof(Half));
            Action setHalf = () => hf.Storage.SetValue((object)(Half)2.5f, (long)0);
            setHalf.Should().NotThrow("SetValue(object, long[]) must support Half");

            // Complex
            var cx = np.zeros(new Shape(3), typeof(System.Numerics.Complex));
            Action setComplex = () => cx.Storage.SetValue((object)new System.Numerics.Complex(1, 2), (long)0);
            setComplex.Should().NotThrow("SetValue(object, long[]) must support Complex");
        }

        /// <summary>
        /// T1.57 — UnmanagedStorage.SetValue(object, int[]) throws on SByte/Half/Complex.
        /// Same root cause as T1.13 but on the int[]-index overload.
        /// </summary>
        [TestMethod]
        public void T1_57_SetValue_Object_IntIndices_Missing_SByte_Half_Complex()
        {
            var sb = np.zeros(new Shape(3), typeof(sbyte));
            Action setSByte = () => sb.Storage.SetValue((object)(sbyte)5, new int[] { 0 });
            setSByte.Should().NotThrow("SetValue(object, int[]) must support SByte");

            var hf = np.zeros(new Shape(3), typeof(Half));
            Action setHalf = () => hf.Storage.SetValue((object)(Half)2.5f, new int[] { 0 });
            setHalf.Should().NotThrow("SetValue(object, int[]) must support Half");

            var cx = np.zeros(new Shape(3), typeof(System.Numerics.Complex));
            Action setComplex = () => cx.Storage.SetValue((object)new System.Numerics.Complex(1, 2), new int[] { 0 });
            setComplex.Should().NotThrow("SetValue(object, int[]) must support Complex");
        }

        /// <summary>
        /// T1.13 (CopyTo) — UnmanagedStorage.CopyTo(void*) throws on SByte/Half/Complex.
        /// </summary>
        [TestMethod]
        public unsafe void T1_13_CopyTo_VoidPtr_Missing_SByte_Half_Complex()
        {
            // SByte
            {
                var arr = np.ones(new Shape(3), typeof(sbyte));
                var dst = new sbyte[3];
                Action act = () => { fixed (sbyte* p = dst) { arr.Storage.CopyTo((void*)p); } };
                act.Should().NotThrow("CopyTo(void*) must support SByte");
            }

            // Half
            {
                var arr = np.ones(new Shape(3), typeof(Half));
                var dst = new Half[3];
                Action act = () => { fixed (Half* p = dst) { arr.Storage.CopyTo((void*)p); } };
                act.Should().NotThrow("CopyTo(void*) must support Half");
            }

            // Complex
            {
                var arr = np.ones(new Shape(3), typeof(System.Numerics.Complex));
                var dst = new System.Numerics.Complex[3];
                Action act = () => { fixed (System.Numerics.Complex* p = dst) { arr.Storage.CopyTo((void*)p); } };
                act.Should().NotThrow("CopyTo(void*) must support Complex");
            }
        }

        // ============================================================================
        // T1.29 — FIXED: the OWNDATA bit (0x0004) is now maintained by
        // UnmanagedStorage.OnReshaped() at every storage funnel (allocating
        // ctors/_Allocate set it, Alias/wrap/_baseStorage/ExternalBase transitions
        // clear it, in-place shape swaps reconcile), mirroring
        // UnmanagedStorage.OwnsData — NumPy ctors.c: fa->flags |= NPY_ARRAY_OWNDATA
        // when allocating, cleared when data is passed in. Kept as a regression pin;
        // the extensive probed-vs-NumPy matrix lives in Backends/OwnDataFlagTests.cs.
        // ============================================================================
        /// <summary>
        /// T1.29 — `Shape.OwnsData` reports true for arrays that own their buffer
        /// (NumPy: OWNDATA=True for every allocating constructor).
        /// </summary>
        [TestMethod]
        public void T1_29_OWNDATA_Flag_Never_Set()
        {
            // Freshly allocated arrays own their data per NumPy semantics.
            var fresh = np.arange(10);
            fresh.Shape.OwnsData.Should().BeTrue("freshly allocated arrays own their data (NumPy: OWNDATA=True)");

            var zeros = np.zeros(new Shape(5));
            zeros.Shape.OwnsData.Should().BeTrue("np.zeros result owns its data (NumPy: OWNDATA=True)");

            var copy = fresh.copy();
            copy.Shape.OwnsData.Should().BeTrue(".copy() returns an owning array (NumPy: OWNDATA=True)");
        }

        // ============================================================================
        // T1.42 — Shape.Equals / operator== compare only `dimensions`, not
        // strides / offset / bufferSize.
        // File: src/NumSharp.Core/View/Shape.cs (lines 1353-1377 operator==,
        //       lines 1397-1418 Equals)
        // Two semantically different shapes (C- vs F-contig, different offsets,
        // different bufferSize) compare equal and hash equal.
        // ============================================================================
        /// <summary>
        /// T1.42 — Two shapes with identical dimensions but different memory
        /// layouts (strides), offsets, or bufferSize are reported as equal.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.42")]
        public void T1_42_Shape_Equals_Ignores_Strides_Offset_BufferSize()
        {
            // C-contiguous (row-major) and F-contiguous (column-major) for the
            // same logical dimensions have different memory layouts.
            var cContig = new Shape(new long[] { 3, 4 }, new long[] { 4, 1 }, 0, 12);
            var fContig = new Shape(new long[] { 3, 4 }, new long[] { 1, 3 }, 0, 12);

            cContig.IsContiguous.Should().BeTrue();
            fContig.IsFContiguous.Should().BeTrue();
            fContig.IsContiguous.Should().BeFalse();

            cContig.Equals(fContig).Should().BeFalse(
                "C-contig and F-contig shapes must not compare equal — they describe different memory layouts");

            // Same dims, different offsets (two windows into the same buffer).
            var noOffset = new Shape(new long[] { 3, 4 }, new long[] { 4, 1 }, 0, 20);
            var withOffset = new Shape(new long[] { 3, 4 }, new long[] { 4, 1 }, 8, 20);
            noOffset.Equals(withOffset).Should().BeFalse(
                "shapes with different offsets describe different views and must not compare equal");

            // Same dims, different bufferSize.
            var bufferA = new Shape(new long[] { 3, 4 }, new long[] { 4, 1 }, 0, 12);
            var bufferB = new Shape(new long[] { 3, 4 }, new long[] { 4, 1 }, 0, 100);
            bufferA.Equals(bufferB).Should().BeFalse(
                "shapes with different bufferSize describe different storages and must not compare equal");
        }

        // ============================================================================
        // T1.64 — FIXED alongside T1.29: nd.Shape.Flags carries OWNDATA for owners
        // and drops it on views; nd.flags.owndata reads this Shape bit, so the whole
        // flags oracle (flags_oracle.jsonl) gates the mirror. Regression pin.
        // ============================================================================
        /// <summary>
        /// T1.64 — `nd.Shape.Flags` has OWNDATA set exactly for owning arrays.
        /// NumPy: `np.arange(10).flags.owndata` is True;
        /// `np.arange(10)[1:5].flags.owndata` is False.
        /// </summary>
        [TestMethod]
        public void T1_64_Flags_OWNDATA_Always_False()
        {
            var fresh = np.arange(10);
            ((int)fresh.Shape.Flags & (int)ArrayFlags.OWNDATA).Should().NotBe(0,
                "np.arange owns its data (NumPy: flags.owndata == True)");

            var slice = fresh["1:5"];
            ((int)slice.Shape.Flags & (int)ArrayFlags.OWNDATA).Should().Be(0,
                "a slice view does not own its data (NumPy: flags.owndata == False)");
        }

        // ============================================================================
        // Tier 2 — Performance gap >2× confirmed (not exact ratio).
        // np.all / np.any contiguous int32: ~3× slower than NumPy on 1M elements
        // np.nonzero: ~17× slower than NumPy on 1M elements
        // ============================================================================
        /// <summary>
        /// T2 perf — `np.all` on a contiguous all-true int32 1M array is much
        /// slower than NumPy. The audit's ~13× ratio depends on hardware; we
        /// assert only that the gap exceeds 2× (which is well above the
        /// SIMD/short-circuit headroom available in `NDAllKernel&lt;T&gt;`).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "audit-v2-T2-all-perf")]
        public void T2_All_Contiguous_Is_Much_Slower_Than_NumPy()
        {
            // NumPy baseline (measured on the same hardware): ~21ms / 100 iters
            // for 1M-element int32. NumSharp on same workload: 60-270ms.
            const long numpyBaselineMs = 21;
            int n = 1_000_000;
            var arr = np.arange(1, n + 1); // all-true to force full scan

            // Warmup
            for (int i = 0; i < 5; i++) np.all(arr);
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 100; i++) np.all(arr);
            sw.Stop();

            sw.ElapsedMilliseconds.Should().BeLessThan(2 * numpyBaselineMs,
                "np.all on contiguous int32 1M must be within 2× of NumPy (currently ~3×, audit reports up to 13×)");
        }

        /// <summary>
        /// T2 perf — `np.nonzero` is dramatically slower than NumPy because of
        /// per-element `long[ndim]` allocation in the masking kernel.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "audit-v2-T2-nonzero-perf")]
        public void T2_NonZero_Is_Much_Slower_Than_NumPy()
        {
            const long numpyBaselineMs = 16;
            int n = 1_000_000;
            var arr = np.arange(n) % 2;

            for (int i = 0; i < 3; i++) np.nonzero(arr);
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 10; i++) np.nonzero(arr);
            sw.Stop();

            sw.ElapsedMilliseconds.Should().BeLessThan(2 * numpyBaselineMs,
                "np.nonzero on 1M int32 must be within 2× of NumPy (currently ~17×, audit reports up to 29×)");
        }
    }
}
