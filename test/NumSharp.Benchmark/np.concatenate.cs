using System;
using System.Collections.Generic;
using System.Diagnostics;
using NumSharp;

namespace NumSharp.Benchmark
{
    /// <summary>
    ///     Manual Stopwatch-based benchmark suite for <c>np.concatenate</c>.
    ///     Covers every relevant variation: dtype, layout, size, array count,
    ///     promotion paths, and the kwarg surface (<c>out=</c>, <c>dtype=</c>,
    ///     <c>axis=None</c>, <c>casting=</c>).
    ///
    ///     <para>
    ///     Run from the benchmark project root via:
    ///         <code>dotnet run -c Release -- concat</code>
    ///     The Program.cs dispatcher routes the <c>concat</c> argument to
    ///     <see cref="RunAll"/>. Pass an additional section name to limit
    ///     the run (e.g. <c>dotnet run -c Release -- concat layout</c>).
    ///     </para>
    ///
    ///     <para>
    ///     A Stopwatch-based harness sidesteps BenchmarkDotNet 0.12.1's
    ///     incompatibility with LangVersion=latest under .NET 10 SDKs.
    ///     The format mirrors the Python NumPy reference benchmark so
    ///     numbers can be diff-compared directly.
    ///     </para>
    /// </summary>
    public static class npconcatenate
    {
        // -----------------------------------------------------------------
        // Harness primitives
        // -----------------------------------------------------------------

        private const int DefaultWarmup = 20;
        private const int DefaultReps = 100;

        /// <summary>
        ///     Run <paramref name="work"/> a few warmup times, then time
        ///     <paramref name="reps"/> iterations. Reports the median to
        ///     suppress outliers (matches the Python harness).
        /// </summary>
        private static double BenchMedianMs(string label, Func<NDArray> work, int warmup = DefaultWarmup, int reps = DefaultReps)
        {
            for (int i = 0; i < warmup; i++) work();
            var ts = new double[reps];
            for (int i = 0; i < reps; i++)
            {
                var sw = Stopwatch.StartNew();
                work();
                sw.Stop();
                ts[i] = sw.Elapsed.TotalMilliseconds;
            }
            Array.Sort(ts);
            var median = ts[reps / 2];
            Console.WriteLine($"  {label,-44}  {median,9:F3} ms");
            return median;
        }

        private static void Header(string section)
        {
            Console.WriteLine();
            Console.WriteLine($"== {section} ==");
        }

        // -----------------------------------------------------------------
        // Entry points
        // -----------------------------------------------------------------

        public static void RunAll(string section = null)
        {
            Console.WriteLine("=== np.concatenate variation sweep ===");
            Console.WriteLine($"Runtime: {Environment.Version} on {Environment.OSVersion.VersionString}");
            Console.WriteLine($"Warmup={DefaultWarmup} iters, median-of-{DefaultReps}.\n");

            bool runAll = string.IsNullOrEmpty(section);

            if (runAll || section == "dtype")       RunDtypeSweep();
            if (runAll || section == "layout")      RunLayoutSweep();
            if (runAll || section == "size")        RunSizeSweep();
            if (runAll || section == "count")       RunCountSweep();
            if (runAll || section == "promotion")   RunPromotionSweep();
            if (runAll || section == "kwargs")      RunKwargsSweep();
        }

        // -----------------------------------------------------------------
        // 1. Dtype sweep — 1M+1M same-dtype, C-contig, axis=0.
        //    Every supported NPTypeCode exercised through the same fast path.
        // -----------------------------------------------------------------

        private static void RunDtypeSweep()
        {
            Header("DTYPE SWEEP (1M+1M, same dtype, contig, axis=0)");
            var dtypes = new (string, NPTypeCode)[]
            {
                ("bool",       NPTypeCode.Boolean),
                ("int8",       NPTypeCode.SByte),
                ("uint8",      NPTypeCode.Byte),
                ("int16",      NPTypeCode.Int16),
                ("uint16",     NPTypeCode.UInt16),
                ("int32",      NPTypeCode.Int32),
                ("uint32",     NPTypeCode.UInt32),
                ("int64",      NPTypeCode.Int64),
                ("uint64",     NPTypeCode.UInt64),
                ("char",       NPTypeCode.Char),
                ("float16",    NPTypeCode.Half),
                ("float32",    NPTypeCode.Single),
                ("float64",    NPTypeCode.Double),
                ("decimal",    NPTypeCode.Decimal),
                ("complex128", NPTypeCode.Complex),
            };
            foreach (var (name, tc) in dtypes)
            {
                var a = np.ones(new Shape(1_000_000), tc);
                var b = np.ones(new Shape(1_000_000), tc);
                BenchMedianMs($"dtype_{name}_1M", () => np.concatenate(new[] { a, b }, 0));
            }
        }

        // -----------------------------------------------------------------
        // 2. Layout sweep — int32, 1M elements total per source.
        //    Validates: C-contig 1D/2D/3D, F-contig, strided, transposed,
        //    sliced, broadcast (read-only stride-0 axis).
        // -----------------------------------------------------------------

        private static void RunLayoutSweep()
        {
            Header("LAYOUT SWEEP (axis varies, int32, 1M elements/src)");

            // C-contig 1D
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Int32);
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32);
                BenchMedianMs("layout_c_contig_1d", () => np.concatenate(new[] { a, b }, 0));
            }
            // C-contig 2D (1000x1000) along both axes
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000);
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000);
                BenchMedianMs("layout_c_contig_2d_axis0", () => np.concatenate(new[] { a, b }, 0));
                BenchMedianMs("layout_c_contig_2d_axis1", () => np.concatenate(new[] { a, b }, 1));
            }
            // C-contig 3D (100x100x100)
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(100, 100, 100);
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(100, 100, 100);
                BenchMedianMs("layout_c_contig_3d_axis0", () => np.concatenate(new[] { a, b }, 0));
                BenchMedianMs("layout_c_contig_3d_axis2", () => np.concatenate(new[] { a, b }, 2));
            }
            // F-contig 2D — exercises the F-contig fast path branch.
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000).copy('F');
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000).copy('F');
                BenchMedianMs("layout_f_contig_2d_axis0", () => np.concatenate(new[] { a, b }, 0));
                BenchMedianMs("layout_f_contig_2d_axis1", () => np.concatenate(new[] { a, b }, 1));
            }
            // Strided view — every other row of a 2x-larger backing array.
            {
                var big = np.arange(2_000_000).astype(NPTypeCode.Int32).reshape(2000, 1000);
                var a = big["::2"];
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000);
                BenchMedianMs("layout_strided_2d_axis0", () => np.concatenate(new[] { a, b }, 0));
            }
            // Transposed (F-contig view of C-contig) — needs F-fast path.
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000).T;
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000).T;
                BenchMedianMs("layout_transposed_2d_axis0", () => np.concatenate(new[] { a, b }, 0));
            }
            // Simple sliced view (offset, contig).
            {
                var big = np.arange(2_000_000).astype(NPTypeCode.Int32).reshape(2000, 1000);
                var a = big["500:1500, :"];
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000);
                BenchMedianMs("layout_sliced_2d_axis0", () => np.concatenate(new[] { a, b }, 0));
            }
            // Broadcast — (1, 1000) -> (1000, 1000) read-only stride-0 axis.
            {
                var a = np.broadcast_to(
                    np.arange(1000).astype(NPTypeCode.Int32).reshape(1, 1000),
                    new Shape(1000, 1000));
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000);
                BenchMedianMs("layout_broadcast_2d_axis0", () => np.concatenate(new[] { a, b }, 0));
            }
        }

        // -----------------------------------------------------------------
        // 3. Size sweep — 1D int32, 2 arrays of N elements each.
        //    Shows fixed per-call overhead amortizing into bandwidth limit.
        // -----------------------------------------------------------------

        private static void RunSizeSweep()
        {
            Header("SIZE SWEEP (1D int32, 2 arrays of N elements)");
            foreach (var n in new[] { 100L, 1_000L, 10_000L, 100_000L, 1_000_000L, 10_000_000L })
            {
                var a = np.arange(n).astype(NPTypeCode.Int32);
                var b = np.arange(n).astype(NPTypeCode.Int32);
                int reps = n < 100_000 ? 500 : (n < 1_000_000 ? 200 : 50);
                BenchMedianMs($"size_{n}", () => np.concatenate(new[] { a, b }, 0), reps: reps);
            }
        }

        // -----------------------------------------------------------------
        // 4. Array-count sweep — N arrays of 100k int32 each, axis=0.
        //    Stresses per-source dispatch overhead.
        // -----------------------------------------------------------------

        private static void RunCountSweep()
        {
            Header("ARRAY COUNT SWEEP (each 100k int32, axis=0)");
            foreach (var n in new[] { 2, 4, 8, 16, 64, 256, 1024 })
            {
                var arrs = new NDArray[n];
                for (int i = 0; i < n; i++)
                    arrs[i] = np.arange(100_000).astype(NPTypeCode.Int32);
                int reps = n <= 64 ? 100 : 30;
                BenchMedianMs($"count_{n}", () => np.concatenate(arrs, 0), reps: reps);
            }
        }

        // -----------------------------------------------------------------
        // 5. Promotion sweep — 1M+1M, mixed dtype pairs, axis=0.
        //    Verifies NEP50 promotion (T1.8 regression guard) and exercises
        //    the IL cast kernels for each (src,result) pair.
        // -----------------------------------------------------------------

        private static void RunPromotionSweep()
        {
            Header("PROMOTION SWEEP (1M+1M, mixed dtypes)");

            NDArray Make(NPTypeCode tc) => np.ones(new Shape(1_000_000), tc);

            void Pair(string name, NPTypeCode A, NPTypeCode B)
            {
                var a = Make(A); var b = Make(B);
                BenchMedianMs($"prom_{name}", () => np.concatenate(new[] { a, b }, 0));
            }

            // Integer-only
            Pair("int8_int16",        NPTypeCode.SByte, NPTypeCode.Int16);
            Pair("int8_uint8",        NPTypeCode.SByte, NPTypeCode.Byte);
            Pair("int32_int64",       NPTypeCode.Int32, NPTypeCode.Int64);
            Pair("int32_uint32",      NPTypeCode.Int32, NPTypeCode.UInt32);

            // Integer x Float
            Pair("int32_float32",     NPTypeCode.Int32,  NPTypeCode.Single);
            Pair("int32_float64",     NPTypeCode.Int32,  NPTypeCode.Double);
            Pair("int64_float64",     NPTypeCode.Int64,  NPTypeCode.Double);

            // Float x Float
            Pair("half_single",       NPTypeCode.Half,   NPTypeCode.Single);
            Pair("float32_float64",   NPTypeCode.Single, NPTypeCode.Double);

            // Anything x Complex (currently the slowest path; scalar IL kernels).
            Pair("float64_complex",   NPTypeCode.Double, NPTypeCode.Complex);
            Pair("int32_complex",     NPTypeCode.Int32,  NPTypeCode.Complex);
        }

        // -----------------------------------------------------------------
        // 6. Kwarg surface — out=, dtype=, axis=None, casting=.
        // -----------------------------------------------------------------

        private static void RunKwargsSweep()
        {
            Header("KWARG SURFACE (out=, dtype=, axis=None, casting=)");

            // out=  same dtype — pure memcpy into provided buffer.
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Int32);
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32);
                var dst = new NDArray(NPTypeCode.Int32, new Shape(2_000_000), fillZeros: false);
                BenchMedianMs("out_int32_1M", () => np.concatenate(new[] { a, b }, 0, @out: dst));
            }

            // out=  cross-dtype — casts each source into a Double buffer.
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Single);
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32);
                var dst = new NDArray(NPTypeCode.Double, new Shape(2_000_000), fillZeros: false);
                BenchMedianMs("out_mixed_to_float64", () => np.concatenate(new[] { a, b }, 0, @out: dst, casting: "unsafe"));
            }

            // dtype= override — forces result dtype before promotion would pick.
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Int32);
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32);
                BenchMedianMs("dtype_override_int32_to_float64", () => np.concatenate(new[] { a, b }, 0, dtype: NPTypeCode.Double));
            }

            // axis=None — flatten then concat.
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000);
                var b = np.arange(1_000_000).astype(NPTypeCode.Int32).reshape(1000, 1000);
                BenchMedianMs("axis_none_2x_1M_2D", () => np.concatenate(new[] { a, b }, axis: null));
            }

            // casting='unsafe' downcast (int64 -> int32) — exercises the
            // narrowing IL cast kernel.
            {
                var a = np.arange(1_000_000).astype(NPTypeCode.Int64);
                var b = np.arange(1_000_000).astype(NPTypeCode.Int64);
                BenchMedianMs("casting_unsafe_int64_to_int32", () => np.concatenate(new[] { a, b }, 0, dtype: NPTypeCode.Int32, casting: "unsafe"));
            }
        }
    }
}
