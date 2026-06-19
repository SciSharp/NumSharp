using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the same-dtype non-contiguous copy path (<c>UnmanagedStorage.Clone</c> →
/// <c>CloneData</c> → <c>NpyIter.Copy</c> → <c>TryCopySameType</c>).
///
/// The SIMD <c>StridedCastKernel</c> rejects the five Vector-less dtypes
/// (Boolean/Char/Half/Decimal/Complex), so their strided/broadcast clones take the
/// <c>CopyGeneralSameType</c> fallback. That fallback used to reconstruct the full
/// N-dim coordinate with a div+mod per axis <b>per element</b>, which ran ~16-33×
/// slower than NumPy's typed strided copy (e.g. 1M strided Char/Half clone ≈ 0.03×
/// NumPy). It now walks outer axes by incremental stride-add + carry and moves each
/// contiguous inner run with <see cref="Buffer.MemoryCopy"/> — dtype-agnostic byte
/// movement that matches the fast kernel's wall time (now ~1.5-4× NumPy).
///
/// These tests guard the <b>correctness</b> of that fast path: a clone must be a
/// bit-exact, contiguous, independent materialization of the view's logical element
/// order, for every dtype and every awkward layout — especially the negative-stride
/// inner axis (forces the scalar inner walk, never memcpy) and the negative-stride
/// outer axis (forces the per-row memcpy to march backwards through the buffer).
/// </summary>
[TestClass]
public class StridedCopySameTypeParityTests
{
    private static readonly NPTypeCode[] AllDtypes =
    {
        NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
        NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
        NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
        NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
    };

    // (name, builder) for the awkward layouts the fast path must absorb.
    private static (string name, Func<NDArray, NDArray> make)[] Layouts(int r, int c) => new (string, Func<NDArray, NDArray>)[]
    {
        ("F",        b => b.copy(order: 'F')),
        ("T",        b => b.T),
        ("strided",  b => b[":, ::2"]),                 // strided inner axis (no memcpy)
        ("sliced",   b => b["1:" + (r - 1) + ", 1:" + (c - 1)]), // offset != 0, positive strides
        ("negrow",   b => b["::-1, :"]),                // negative OUTER stride → backward row memcpy
        ("negcol",   b => b[":, ::-1"]),                // negative INNER stride → scalar inner walk
        ("negboth",  b => b["::-1, ::-1"]),             // both negative
        ("bcast",    b => np.broadcast_to(b["0:1, :"], new Shape(r, c))), // stride-0 outer axis
    };

    [DataTestMethod]
    [DynamicData(nameof(DtypeRows), DynamicDataSourceType.Method)]
    public void Clone_OfNonContiguousView_IsBitExactContiguousMaterialization(NPTypeCode dt)
    {
        const int R = 7, C = 9;
        // distinct positive values 1..(R*C mod 13)+1 — keeps Boolean meaningful (mix of 0/1)
        // and every other dtype's cells individually identifiable.
        var baseArr = ((np.arange(R * C) % 13) + 1).astype(dt).reshape(R, C);

        foreach (var (name, make) in Layouts(R, C))
        {
            var view = make(baseArr);

            // copy() and (where it has a loop) positive() both route through the same
            // same-type clone — assert both materialize the view's logical order bit-exactly.
            AssertCloneMatchesView(view, view.copy(), $"{dt}/{name}/copy");
            if (dt != NPTypeCode.Boolean) // positive has no Boolean loop (matches NumPy)
                AssertCloneMatchesView(view, np.positive(view), $"{dt}/{name}/positive");
        }
    }

    private static void AssertCloneMatchesView(NDArray view, NDArray clone, string ctx)
    {
        Assert.AreEqual(view.size, clone.size, $"{ctx}: size");
        // A clone is a dense, owning materialization — C-contiguous for copy() and for
        // strided-view positive(), F-contiguous for positive() of an already-F-contiguous
        // view (layout-preserving whole-buffer memcpy). Either proves "no gaps / strides /
        // broadcast"; what must never happen is a strided or broadcast result.
        Assert.IsTrue(clone.Shape.IsContiguous || clone.Shape.IsFContiguous, $"{ctx}: clone must be contiguous (C or F)");

        for (long i = 0; i < view.size; i++)
        {
            // Bit-exact: a clone is pure byte movement, no rounding. Boxed Equals is exact
            // for every value type incl. Half/Complex/Decimal/Char/Boolean. clone reads
            // contiguous slot i; view computes the strided offset independently — so a kernel
            // that writes the wrong cell (or buffer-base garbage) is caught here.
            Assert.AreEqual(view.GetAtIndex(i), clone.GetAtIndex(i), $"{ctx}[{i}]");
        }
    }

    [DataTestMethod]
    [DynamicData(nameof(DtypeRows), DynamicDataSourceType.Method)]
    public void Clone_IsIndependentOfSource(NPTypeCode dt)
    {
        // A clone must own its data: mutating the source must not bleed into the clone.
        var baseArr = ((np.arange(6 * 6) % 11) + 1).astype(dt).reshape(6, 6);
        var view = baseArr["::-1, ::-1"]; // negative-stride view, the trickiest base-pointer case
        var clone = view.copy();

        var before = clone.GetAtIndex(0);
        // overwrite the entire source through a fresh value
        baseArr[":"] = np.zeros(new Shape(6, 6), dt);
        Assert.AreEqual(before, clone.GetAtIndex(0), $"{dt}: clone aliased its source");
    }

    public static System.Collections.Generic.IEnumerable<object[]> DtypeRows()
    {
        foreach (var dt in AllDtypes)
            yield return new object[] { dt };
    }
}
