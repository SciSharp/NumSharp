#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaltext_arch2_probe.cs — decision D1 of docs/plans/evaluate-text.md, second round: the staged composite against
// whole-array stages ("lite", the design E1 implements), on the tree, kinds, layouts and sizes of evaltext_arch_probe.cs.
//   materialize np.equal into a full bool temporary, then np.evaluate of the numeric rest (the reference)
//   staged      one composite NDInnerLoopFunc (text kernel per 4,096-element block into an L1 scratch, then the fused
//               numeric kernel)
//   lite        the stage's own IL text kernel over the whole operands into a pooled bool temporary — the trivial loop,
//               the rows kernel for narrow S / U rows, or NDIter, as the unfused comparison routes — then the numeric
//               remainder's program over [temp, age] through TensorEngine.Evaluate(program, ...), what a lowered
//               program runs per call
// Every executor is verified equal to the materialized route before it is timed.
//
// A prototype over NumSharp internals (ILKernelGenerator's text kernels and rows kernel, VString allocators,
// NDExprProgram, NDIterRef's rows driver): it was measured against 78b28c2b and may need updating to compile against a
// later tree. The numbers are in the plan (§3).
// Run (Release only): dotnet run -c Release benchmark/fusion/probes/evaltext_arch2_probe.cs
// Pinned to 0x14 (two P-cores) unless NS_PROBE_AFFINITY (hex) says otherwise; best of 7 after a paused double warm-up,
// two interleaved rounds per executor.
// =============================================================================
using System.Diagnostics;
using System.Runtime.InteropServices;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

unsafe
{
    // The pin is set before any array exists, so the tier-1 compiler threads inherit it with the rest of the process.
    string affinity = Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } mask ? mask : "14";
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(affinity, 16);
    // The staged block: 4,096 bool results fit L1 beside the operands' cache lines, so the numeric kernel reads them hot.
    const int B = 4096;
    byte* scratch = (byte*)NativeMemory.AlignedAlloc(B, 64);
    var words = new[] { "paris", "london", "berlin", "madrid", "rome", "vienna", "oslo", "lisbon" };

    // The numeric remainder `in0 & (in1 > 30)` over (bool, int64) as a program: its kernel feeds the staged composite,
    // the program itself is what lite hands to TensorEngine.Evaluate.
    var island = NDExprProgram.Build(NDExpr.BitwiseAnd(NDExpr.Input(0), NDExpr.Greater(NDExpr.Input(1), NDExpr.Const(30))),
                                     new[] { NPTypeCode.Boolean, NPTypeCode.Int64 });
    var numSimd = island.Kernel;

    // The staged composite over operands (names, literal, age, result): per block, `text` writes the block's bools into
    // the scratch, then `numeric` reads the scratch and age and writes the result. Returns one inner loop an NDIter (or a
    // trivial call) drives like any fused kernel.
    NDInnerLoopFunc Composite(NDInnerLoopFunc text, NDInnerLoopFunc numeric)
        => (dataptrs, strides, count, aux) =>
        {
            void** p = stackalloc void*[3];
            long* s = stackalloc long[3];
            byte* n0 = (byte*)dataptrs[0], l0 = (byte*)dataptrs[1], a0 = (byte*)dataptrs[2], o0 = (byte*)dataptrs[3];
            long sn = strides[0], sl = strides[1], sa = strides[2], so = strides[3];
            for (long done = 0; done < count; done += B)
            {
                long c = Math.Min(B, count - done);
                p[0] = n0 + done * sn; p[1] = l0 + done * sl; p[2] = scratch;
                s[0] = sn; s[1] = sl; s[2] = 1;
                text(p, s, c, aux);
                p[0] = scratch; p[1] = a0 + done * sa; p[2] = o0 + done * so;
                s[0] = 1; s[1] = sa; s[2] = so;
                numeric(p, s, c, null);
            }
        };

    foreach (string kind in new[] { "U", "S", "T" })
    foreach (string layout in new[] { "contig", "strided", "rows8" })
    foreach (int n in new[] { 1_000, 100_000, 1_000_000 })
    {
        // ---- data: n logical elements in every layout (strided reads every 2nd of 2n, rows8 8 of 64 per row) ----
        var rnd = new Random(42);
        int total = layout == "contig" ? n : n * (layout == "strided" ? 2 : 8);
        var strs = new string[total];
        for (int i = 0; i < total; i++) strs[i] = words[rnd.Next(words.Length)];
        NDArray baseNames = kind switch
        {
            "U" => np.array(strs),
            "S" => np.array(strs).astype(np.dtype("S6")),
            _ => np.array(strs).astype(np.dtypes.StringDType()),
        };
        NDArray baseAge = np.arange(total).astype(NPTypeCode.Int64) % 97;
        NDArray names, age;
        if (layout == "contig") { names = baseNames; age = baseAge; }
        else if (layout == "strided") { names = baseNames["::2"]; age = baseAge["::2"]; }
        else
        {
            names = baseNames.reshape(n / 8, 64)[":, :8"];
            age = baseAge.reshape(n / 8, 64)[":, :8"];
        }
        // S compares with a bytes literal (a 0-d S5), U / T with a str (a C# string).
        NDArray lit = kind == "S" ? np.array(new[] { "paris" }).astype(np.dtype("S5")).reshape(Shape.NewScalar()) : (NDArray)"paris";
        bool vstring = kind == "T";
        NDInnerLoopFunc text = vstring
            ? ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.Equal, 0)
            : ILKernelGenerator.GetTextCompareKernel(TextCompareOp.Equal, kind == "U" ? 4 : 1, names.dtype.itemsize, lit.dtype.itemsize, 0);
        var staged = Composite(text, numSimd);
        // StringDType has no rows kernel: its narrow rows run the chunk kernel under NDIter.
        NDInnerLoop2DFunc? textRows = vstring ? null
            : ILKernelGenerator.GetTextCompareRowsKernel(TextCompareOp.Equal, kind == "U" ? 4 : 1, names.dtype.itemsize, lit.dtype.itemsize, 0, masked: false);
        NDArray dummyBool = np.zeros(new Shape(1), NPTypeCode.Boolean);
        var textFlags = new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.NO_BROADCAST };
        var allFlags = new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.NO_BROADCAST };

        // The per-call context both executors pay: the program lookup of a real np.evaluate (a numeric tree of the same
        // size stands in), and for StringDType the literal promoted to StringDType with both allocators locked for the
        // kernel's arena reads (released, and the promoted literal disposed, when `body` returns). body gets the literal
        // to pass and the kernel's aux pointer (zero for S / U); returns body's result.
        NDArray RunWithContext(Func<NDArray, IntPtr, NDArray> body)
        {
            var proxy = NDExpr.BitwiseAnd(NDExpr.Arr(dummyBool), NDExpr.Greater(age, NDExpr.Const(30)));
            _ = proxy.GetProgram(null, out _);
            if (!vstring)
                return body(lit, IntPtr.Zero);
            var tlit = NDVStringComparison.ToDefaultVString(lit);
            var a1 = names.Storage.VStringSlice.Allocator;
            var a2 = tlit.Storage.VStringSlice.Allocator;
            VStringAllocator.acquire_allocators(a1, a2);
            try
            {
                VStringPairAux pair = default;
                pair.ArenaA = a1.ArenaBase;
                pair.ArenaB = a2.ArenaBase;
                pair.Rule = NDVStringCells.RuleOf(names.dtype);
                return body(tlit, (IntPtr)(&pair));
            }
            finally
            {
                VStringAllocator.release_allocators(a1, a2);
                tlit.Dispose();
            }
        }

        // One staged call: the composite once (contiguous) or under NDIter. Returns the fresh bool result.
        NDArray RunStaged() => RunWithContext((l, aux) =>
        {
            var result = new NDArray(NPTypeCode.Boolean, new Shape((long[])names.Shape.dimensions.Clone(), copy: false), false);
            if (layout == "contig")
            {
                void** p = stackalloc void*[4];
                long* s = stackalloc long[4];
                p[0] = (byte*)names.Address; p[1] = (byte*)l.Address; p[2] = (byte*)age.Address; p[3] = (byte*)result.Address;
                s[0] = vstring ? 16 : names.dtype.itemsize; s[1] = 0; s[2] = 8; s[3] = 1;
                staged(p, s, n, (void*)aux);
            }
            else
            {
                var descrs = new DType[4];
                descrs[3] = DType.Boolean;
                using var iter = NDIterRef.AdvancedNew(4, new[] { names, l, age, result },
                    NDIterGlobalFlags.EXTERNAL_LOOP | (vstring ? NDIterGlobalFlags.REFS_OK : 0),
                    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING, allFlags, opDescrs: descrs);
                iter.ForEach(staged, (void*)aux);
            }
            return result;
        });

        // One lite call: the text kernel over the whole operands into a temporary (released when the call ends), then
        // the remainder's program over [temp, age]. Returns the remainder's fresh result.
        NDArray RunLite() => RunWithContext((l, aux) =>
        {
            using var temp = new NDArray(NPTypeCode.Boolean, new Shape((long[])names.Shape.dimensions.Clone(), copy: false), false);
            if (layout == "contig")
            {
                void** p = stackalloc void*[3];
                long* s = stackalloc long[3];
                p[0] = (byte*)names.Address; p[1] = (byte*)l.Address; p[2] = (byte*)temp.Address;
                s[0] = vstring ? 16 : names.dtype.itemsize; s[1] = 0; s[2] = 1;
                text(p, s, n, (void*)aux);
            }
            else
            {
                var descrs = new DType[3];
                descrs[2] = DType.Boolean;
                using var iter = NDIterRef.AdvancedNew(3, new[] { names, l, temp },
                    NDIterGlobalFlags.EXTERNAL_LOOP | (vstring ? NDIterGlobalFlags.REFS_OK : 0),
                    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING, textFlags, opDescrs: descrs);
                // The unfused comparison's own route: blocks of narrow rows go to the rows kernel (S / U only).
                if (!vstring && iter.CanRunRows)
                    iter.ForEachRows(textRows!, (void*)aux);
                else
                    iter.ForEach(text, (void*)aux);
            }
            return age.TensorEngine.Evaluate(island, new[] { temp, age }, null);
        });

        // The materialized route, the reference: np.equal into a temporary, then np.evaluate of the numeric rest.
        NDArray RunMaterialized()
        {
            using var t = np.equal(names, lit);
            return np.evaluate(NDExpr.BitwiseAnd(t, NDExpr.Greater(age, NDExpr.Const(30))));
        }

        // ---- verify: every executor equals the materialized route before anything is timed ------------------------
        using (var r0 = RunMaterialized())
        using (var r1 = RunStaged())
        using (var r2 = RunLite())
            if (!np.array_equal(r0, r1) || !np.array_equal(r0, r2))
                throw new Exception($"mismatch {kind} {layout} {n}");

        // Best per-call time of f in microseconds: a warm-up burst queues the hot methods for tier-1, the pause lets the
        // background compiler install them, a second burst runs the tier-1 code once; then the minimum over 7 rounds.
        double Time(Func<NDArray> f)
        {
            for (int w = 0; w < 20; w++) f().Dispose();
            Thread.Sleep(100);
            for (int w = 0; w < 20; w++) f().Dispose();
            int reps = n <= 1000 ? 3000 : n <= 100_000 ? 200 : 25;
            double best = double.MaxValue;
            for (int r = 0; r < 7; r++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < reps; i++) f().Dispose();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000 / reps);
            }
            return best;
        }

        // Two interleaved rounds so host drift hits the three executors alike.
        double tM = double.MaxValue, tS = double.MaxValue, tL = double.MaxValue;
        for (int round = 0; round < 2; round++)
        {
            tM = Math.Min(tM, Time(RunMaterialized));
            tS = Math.Min(tS, Time(RunStaged));
            tL = Math.Min(tL, Time(RunLite));
        }
        Console.WriteLine($"{kind} {layout,-8} n={n,9}  materialize {tM,9:F1}  lite {tL,9:F1}  staged {tS,9:F1} us   lite/staged={tL / tS:F2}  M/staged={tM / tS:F2}");
    }
}
