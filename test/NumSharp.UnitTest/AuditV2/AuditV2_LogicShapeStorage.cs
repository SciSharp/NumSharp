using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.AuditV2
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
        // File: src/NumSharp.Core/View/Shape.cs (lines 754-760)
        //
        // Although Shape is declared `public readonly partial struct Shape`, the
        // indexer at line 754-760 exposes a `set` accessor that mutates the
        // referenced `long[] dimensions` array.  Because `_flags`, `size`, and
        // `_hashCode` are computed once at construction, mutating `dimensions[i]`
        // via `shape[i] = x` leaves the cached fields stale and silently breaks
        // any downstream check (Equals/GetHashCode/Size/IsContiguous).
        // ============================================================================
        /// <summary>
        /// T1.10 — Mutating the Shape indexer breaks immutability and invalidates
        /// the cached _flags / size / _hashCode (Shape is supposed to be immutable
        /// per NumPy semantics and per the struct's `readonly` modifier).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.10")]
        public void T1_10_Shape_SetIndexer_Mutates_Readonly_Struct()
        {
            var shape = new Shape(new long[] { 4, 5, 6 });

            // Capture initial state
            long originalSize = shape.Size;        // 4 * 5 * 6 = 120
            int originalHash = shape.GetHashCode();
            int originalFlags = shape._flags;

            originalSize.Should().Be(120);

            // Mutate via set indexer (this should NOT compile, or it should be
            // a no-op, on a readonly struct that promises immutability).
            shape[0] = 99;

            // The dimensions array IS mutated (proving mutation occurred):
            shape[0].Should().Be(99);
            shape.Dimensions[0].Should().Be(99);

            // BUG: Cached size is stale — still reports the pre-mutation value.
            // Expected: 99 * 5 * 6 = 2970 (or, ideally, the mutation should not be possible).
            shape.Size.Should().Be(99L * 5 * 6, "Shape.Size must reflect actual dimensions or mutation must be disallowed");

            // BUG: Cached hash code is stale (Shape contract: equal shapes -> equal hash).
            var rebuilt = new Shape(new long[] { 99, 5, 6 });
            shape.GetHashCode().Should().Be(rebuilt.GetHashCode(),
                "GetHashCode must agree with an equivalently-constructed shape");

            // BUG: Equality compares using cached size + dims, so even though
            // dimensions now match (99,5,6), Equals returns false (size mismatch).
            shape.Equals(rebuilt).Should().BeTrue("two shapes with the same dimensions must be equal");
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
        // T1.29 — Shape.OWNDATA flag declared at 0x0004 but never set anywhere.
        // File: src/NumSharp.Core/View/Shape.cs (line 27 enum, line 358-362 getter,
        //       line 127 ComputeFlagsStatic — no OR with OWNDATA).
        // ============================================================================
        /// <summary>
        /// T1.29 — `Shape.OwnsData` always returns false because no code path
        /// ever sets the OWNDATA flag (0x0004). NumPy reports OWNDATA=True for
        /// any array that owns its data buffer.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.29")]
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
        // T1.64 — np.arr.flags.OWNDATA always reports False; NumPy reports True
        // for arrays that own their data. Surface-level manifestation of T1.29.
        // ============================================================================
        /// <summary>
        /// T1.64 — `nd.Shape.Flags` never has OWNDATA set, regardless of how the
        /// array was created. NumPy: `np.arange(10).flags.owndata` is True;
        /// `np.arange(10)[1:5].flags.owndata` is False.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.64")]
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
        /// SIMD/short-circuit headroom available in `NpyAllKernel&lt;T&gt;`).
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
