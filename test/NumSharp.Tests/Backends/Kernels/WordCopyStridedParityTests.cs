using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
///     Pins the 4-byte / 8-byte same-type STRIDED copy kernel (<c>Cast.WordCopy.cs</c>): the SIMD reverse
///     (stride −1), deinterleave (stride 2) and AVX2-gather (any other stride, broadcast included) inner loops, with
///     per-row memcpy for a unit stride. A copy moves bits, so the result must be the view's logical elements bit for
///     bit — NaN payloads and signalling NaNs included, which a float move through arithmetic would disturb. Rows are
///     long enough (1,000+ elements) for the vector bodies to run, and odd so the scalar remainders run too; the
///     small-array clone tests (<see cref="StridedCopySameTypeParityTests"/>) only reach the scalar loop.
/// </summary>
[TestClass]
public class WordCopyStridedParityTests
{
    /// <summary>The 4/8-byte dtypes the kernel serves.</summary>
    private static readonly NPTypeCode[] Dtypes =
    {
        NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Single, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Double,
    };

    /// <summary>
    ///     Copies of every inner-stride case (±1, ±2, ±3, 7, broadcast 0) over a 2-D base with an outer axis, for every
    ///     4/8-byte dtype, equal the element-by-element read of the view — compared as raw bits.
    /// </summary>
    [TestMethod]
    public unsafe void StridedCopies_AreBitExact_ForEveryInnerStride()
    {
        const int rows = 3, cols = 7 * 1001 + 5;
        foreach (var dt in Dtypes)
        {
            var baseArr = Patterned(dt, rows * cols).reshape(rows, cols);
            foreach (var spec in new[] { ":, ::1", ":, ::-1", ":, ::2", ":, 1::2", ":, ::-2", ":, ::3", ":, ::-3", ":, ::7", "::-1, ::2", "::2, ::-1" })
            {
                var view = baseArr[spec];
                AssertBitExactCopy(view, $"{dt} [{spec}]");
            }
            // A stride-0 inner axis: every element of the row is the same source element (the gather replicates it).
            var bcast = np.broadcast_to(baseArr[":, 5:6"], new Shape(rows, 1003));
            AssertBitExactCopy(bcast, $"{dt} broadcast");
        }
    }

    /// <summary>
    ///     The same-size integer reinterprets (int32↔uint32, int64↔uint64) use the kernel as a bit copy; a float↔int
    ///     pair of the same size must NOT (it is a value cast), so the resolver refuses it.
    /// </summary>
    [TestMethod]
    public void Resolver_AdmitsBitCopiesOnly()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            Assert.Inconclusive("AVX2 kernels are not available on this host");
        DirectILKernelGenerator.TryGetWordCopyStridedKernel(NPTypeCode.Single, NPTypeCode.Single).Should().NotBeNull();
        DirectILKernelGenerator.TryGetWordCopyStridedKernel(NPTypeCode.Int32, NPTypeCode.UInt32).Should().NotBeNull();
        DirectILKernelGenerator.TryGetWordCopyStridedKernel(NPTypeCode.UInt64, NPTypeCode.Int64).Should().NotBeNull();
        DirectILKernelGenerator.TryGetWordCopyStridedKernel(NPTypeCode.Single, NPTypeCode.Int32).Should().BeNull("float -> int is a value cast");
        DirectILKernelGenerator.TryGetWordCopyStridedKernel(NPTypeCode.Int64, NPTypeCode.Double).Should().BeNull("int -> float is a value cast");
        DirectILKernelGenerator.TryGetWordCopyStridedKernel(NPTypeCode.Int16, NPTypeCode.Int16).Should().BeNull("2-byte copies are SubwordCopy's");
        DirectILKernelGenerator.TryGetWordCopyStridedKernel(NPTypeCode.Complex, NPTypeCode.Complex).Should().BeNull("16-byte elements are not served");
    }

    /// <summary>
    ///     <paramref name="n"/> elements of <paramref name="dt"/> with pseudo-random bits that include, for the float
    ///     dtypes, signalling and negative quiet NaNs with payloads, negative zero and (8-byte) negative infinity —
    ///     the patterns a copy through float arithmetic would quiet, canonicalise or flush.
    /// </summary>
    /// <param name="dt">A 4- or 8-byte dtype.</param>
    /// <param name="n">Element count.</param>
    /// <returns>A new C-contiguous 1-D array.</returns>
    private static NDArray Patterned(NPTypeCode dt, int n)
    {
        if (DirectILKernelGenerator.GetTypeSize(dt) == 4)
        {
            var bits = new uint[n];
            for (int k = 0; k < n; k++)
                bits[k] = (k % 11) switch
                {
                    0 => 0x7f800001u + (uint)k,          // signalling NaN, payload k+1 (float32 view)
                    1 => 0xffc00000u | (uint)k,          // negative quiet NaN with payload
                    2 => 0x80000000u,                    // -0.0
                    _ => 0x3f800000u + (uint)(k * 2654435761u >> 9),
                };
            return np.array(bits).view(dt);
        }
        var bits8 = new ulong[n];
        for (int k = 0; k < n; k++)
            bits8[k] = (k % 11) switch
            {
                0 => 0x7ff0000000000001UL + (ulong)k,    // signalling NaN (float64 view)
                1 => 0xfff8000000000000UL | (ulong)k,    // negative quiet NaN
                2 => 0x8000000000000000UL,               // -0.0
                3 => 0xfff0000000000000UL,               // -inf
                _ => 0x3ff0000000000000UL + (ulong)k * 0x9e3779b97f4a7c15UL % 0x000fffffffffffffUL,
            };
        return np.array(bits8).view(dt);
    }

    /// <summary>Asserts <c>view.copy()</c> equals the element-by-element read of the view, compared as raw bits.</summary>
    /// <param name="view">The strided view.</param>
    /// <param name="what">Label for the failure message.</param>
    private static void AssertBitExactCopy(NDArray view, string what)
    {
        var copy = view.copy();
        copy.Shape.IsContiguous.Should().BeTrue($"{what}: a copy is C-contiguous");
        bool eight = view.dtypesize == 8;
        // Flattened: the coordinate getters read a single index on a 2-D array as a ROW coordinate, not a flat index.
        var copyBits = copy.view(eight ? np.uint64 : np.uint32).reshape(-1);
        long i = 0;
        foreach (var idx in np.ndindex(view.shape))
        {
            var coords = Array.ConvertAll(idx, x => (int)x);
            ulong want = eight ? BitConverter.ToUInt64(BytesOf(view, coords), 0) : BitConverter.ToUInt32(BytesOf(view, coords), 0);
            ulong got = eight ? copyBits.GetUInt64((int)i) : copyBits.GetUInt32((int)i);
            got.Should().Be(want, $"{what}: element {i}");
            i++;
        }
    }

    /// <summary>The raw bytes of one element of <paramref name="a"/> at <paramref name="coords"/> (read through its strides).</summary>
    /// <param name="a">The array.</param>
    /// <param name="coords">The element's coordinates.</param>
    /// <returns>The element's bytes.</returns>
    private static unsafe byte[] BytesOf(NDArray a, int[] coords)
    {
        long off = a.Shape.offset;
        for (int d = 0; d < coords.Length; d++)
            off += coords[d] * a.Shape.strides[d];
        var bytes = new byte[a.dtypesize];
        byte* p = (byte*)a.Storage.Address + off * a.dtypesize;
        for (int b = 0; b < bytes.Length; b++)
            bytes[b] = p[b];
        return bytes;
    }
}
