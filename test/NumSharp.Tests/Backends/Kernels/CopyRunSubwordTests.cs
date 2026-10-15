using System.Collections.Generic;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     <see cref="NumPyMinMaxReduce.CopyRun{T}"/>'s sub-word vector body (1- and 2-byte elements at a source stride of
    ///     2 or -1 into a contiguous destination — the staging copy of np.sum / np.mean's cast fills and of np.max's
    ///     buffer fills) against a plain element loop, for every run length 0..200 and both widths, plus the strides that
    ///     keep the scalar loop.
    /// </summary>
    /// <remarks>
    ///     The vector body copies blocks of 8 bytes / 4 words (stride 2) or 16 bytes / 8 words (reversed) and leaves the
    ///     rest to the scalar loop; a block boundary or a tail off by one would leave a destination element unwritten (the
    ///     destination is pre-filled with a sentinel) or write a wrong one.
    /// </remarks>
    [TestClass]
    public unsafe class CopyRunSubwordTests
    {
        [TestMethod]
        public void CopyRun_SubwordElements_CopyEveryElementInOrder()
        {
            var fails = new List<string>();
            foreach (long xs in new long[] { 2, -1, 3, -2, 1 })
            {
                for (long n = 0; n <= 200; n++)
                {
                    long extent = n == 0 ? 1 : (n - 1) * System.Math.Abs(xs) + 1;

                    byte* b = (byte*)NativeMemory.Alloc((nuint)extent);
                    byte* ob = (byte*)NativeMemory.Alloc((nuint)(n + 1));
                    ushort* w = (ushort*)NativeMemory.Alloc((nuint)(extent * 2));
                    ushort* ow = (ushort*)NativeMemory.Alloc((nuint)((n + 1) * 2));
                    try
                    {
                        for (long i = 0; i < extent; i++) { b[i] = (byte)(i * 7 + 3); w[i] = (ushort)(i * 977 + 5); }
                        for (long i = 0; i <= n; i++) { ob[i] = 0xEE; ow[i] = 0xEEEE; }
                        byte* xb = xs > 0 ? b : b + extent - 1;
                        ushort* xw = xs > 0 ? w : w + extent - 1;

                        NumPyMinMaxReduce.CopyRun(ob, 1, xb, xs, n);
                        NumPyMinMaxReduce.CopyRun(ow, 1, xw, xs, n);

                        for (long i = 0; i < n; i++)
                        {
                            if (ob[i] != xb[i * xs]) { fails.Add($"byte xs={xs} n={n} at {i}: {ob[i]} vs {xb[i * xs]}"); break; }
                            if (ow[i] != xw[i * xs]) { fails.Add($"word xs={xs} n={n} at {i}: {ow[i]} vs {xw[i * xs]}"); break; }
                        }
                        if (ob[n] != 0xEE || ow[n] != 0xEEEE)
                            fails.Add($"xs={xs} n={n}: wrote past the run's end");
                    }
                    finally
                    {
                        NativeMemory.Free(b); NativeMemory.Free(ob); NativeMemory.Free(w); NativeMemory.Free(ow);
                    }
                }
            }

            fails.Should().BeEmpty();
        }
    }
}
