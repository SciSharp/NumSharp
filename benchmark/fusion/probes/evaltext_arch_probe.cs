#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaltext_arch_probe.cs — decision D1 of docs/plans/evaluate-text.md, first round: how a text node should run inside
// np.evaluate, measured on (names == "paris") & (age > 30) with every executor doing its real per-call work, each
// verified equal to the unfused operators before it is timed.
//   unfused     np.equal, age > 30, then &: three whole-array calls (NumSharp's own statements)
//   materialize np.equal into a full bool temporary, then np.evaluate of the numeric rest
//   staged      ONE composite NDInnerLoopFunc: per block of 4,096 elements the text kernel writes an L1 scratch, then
//               the fused numeric kernel reads it; run once on a trivially iterable call or per NDIter chunk
//   inline~     staged with a forced-scalar numeric kernel: a proxy for emitting the text compare inside the fused
//               kernel, where a text element is not a SIMD lane and the numeric part loses its vectors
// Kinds U6 / S6 / T (StringDType); layouts contiguous 1-D, strided 1-D (x[::2]), narrow rows (x2d[:, :8]); n = 1K /
// 100K / 1M. evaltext_arch2_probe.cs is the second round (whole-array stages, "lite", against staged).
//
// A prototype over NumSharp internals (ILKernelGenerator's text kernels, VString allocators, NDExprProgram, NDIterRef):
// it was measured against 78b28c2b and may need updating to compile against a later tree. The numbers are in the plan.
// Run (Release only): dotnet run -c Release benchmark/fusion/probes/evaltext_arch_probe.cs
// Pinned to 0x14 (two P-cores) unless NS_PROBE_AFFINITY (hex) says otherwise; best of 7 after a paused double warm-up.
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

    // The fused kernel of the numeric remainder `in0 & (in1 > 30)` over (bool, int64); scalar = true builds it with
    // NDExpr.ForceScalar, the inline proxy's kernel. Returns the kernel's inner-loop delegate.
    NDInnerLoopFunc NumericKernel(bool scalar)
    {
        NDExpr.ForceScalar = scalar;
        try
        {
            var e = NDExpr.BitwiseAnd(NDExpr.Input(0), NDExpr.Greater(NDExpr.Input(1), NDExpr.Const(30)));
            return NDExprProgram.Build(e, new[] { NPTypeCode.Boolean, NPTypeCode.Int64 }).Kernel;
        }
        finally { NDExpr.ForceScalar = false; }
    }
    var numSimd = NumericKernel(false);
    var numScalar = NumericKernel(true);

    // The staged composite over operands (names, literal, age, result): per block, `text` writes the block's bools into
    // the scratch (names / literal strides as given, scratch stride 1), then `numeric` reads the scratch and age and
    // writes the result. Returns one inner loop an NDIter (or a trivial call) drives like any fused kernel.
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

        // ---- the staged executor ---------------------------------------------------------------------------------
        NDInnerLoopFunc text = vstring
            ? ILKernelGenerator.GetVStringCompareKernel(TextCompareOp.Equal, 0)
            : ILKernelGenerator.GetTextCompareKernel(TextCompareOp.Equal, kind == "U" ? 4 : 1, names.dtype.itemsize, lit.dtype.itemsize, 0);
        var stagedSimd = Composite(text, numSimd);
        var stagedScalar = Composite(text, numScalar);
        NDArray dummyBool = np.zeros(new Shape(1), NPTypeCode.Boolean);

        // One staged call: the per-call program lookup a real np.evaluate pays (a numeric tree of the same size stands
        // in for it), the StringDType context (the literal promoted to StringDType, both allocators locked for the
        // kernel's arena reads, released after), then the composite once (contiguous) or under NDIter. Returns the
        // fresh bool result.
        NDArray RunStaged(NDInnerLoopFunc composite)
        {
            var proxy = NDExpr.BitwiseAnd(NDExpr.Arr(dummyBool), NDExpr.Greater(age, NDExpr.Const(30)));
            _ = proxy.GetProgram(null, out _);
            var dims = (long[])names.Shape.dimensions.Clone();
            var result = new NDArray(NPTypeCode.Boolean, new Shape(dims, copy: false), false);
            // Set only for StringDType (the promoted literal and the two locked allocators); null for S / U.
            NDArray? tlit = null;
            VStringAllocator? a1 = null, a2 = null;
            VStringPairAux pair = default;
            void* aux = null;
            if (vstring)
            {
                tlit = NDVStringComparison.ToDefaultVString(lit);
                a1 = names.Storage.VStringSlice.Allocator;
                a2 = tlit.Storage.VStringSlice.Allocator;
                VStringAllocator.acquire_allocators(a1, a2);
                pair.ArenaA = a1.ArenaBase;
                pair.ArenaB = a2.ArenaBase;
                pair.Rule = NDVStringCells.RuleOf(names.dtype);
                aux = &pair;
            }
            try
            {
                NDArray l = tlit ?? lit;
                if (layout == "contig")
                {
                    void** p = stackalloc void*[4];
                    long* s = stackalloc long[4];
                    p[0] = (byte*)names.Address; p[1] = (byte*)l.Address; p[2] = (byte*)age.Address; p[3] = (byte*)result.Address;
                    s[0] = vstring ? 16 : names.dtype.itemsize; s[1] = 0; s[2] = 8; s[3] = 1;
                    composite(p, s, n, aux);
                }
                else
                {
                    var flags = new[]
                    {
                        NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY,
                        NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.NO_BROADCAST,
                    };
                    var descrs = new DType[4];
                    descrs[3] = DType.Boolean;
                    using var iter = NDIterRef.AdvancedNew(4, new[] { names, l, age, result },
                        NDIterGlobalFlags.EXTERNAL_LOOP | (vstring ? NDIterGlobalFlags.REFS_OK : 0),
                        NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING, flags, opDescrs: descrs);
                    iter.ForEach(composite, aux);
                }
                return result;
            }
            finally
            {
                if (vstring)
                {
                    VStringAllocator.release_allocators(a1!, a2!);
                    tlit!.Dispose();
                }
            }
        }

        // The materialized route: np.equal into a temporary, then np.evaluate of the numeric rest. Returns the result.
        NDArray RunMaterialized()
        {
            using var t = np.equal(names, lit);
            return np.evaluate(NDExpr.BitwiseAnd(t, NDExpr.Greater(age, NDExpr.Const(30))));
        }

        // NumSharp's own statements, the reference every executor is checked against. Returns the result.
        NDArray RunUnfused()
        {
            using var m1 = np.equal(names, lit);
            using var m2 = age > 30;
            return m1 & m2;
        }

        // ---- verify: every executor equals the unfused statements before anything is timed ------------------------
        using (var r0 = RunUnfused())
        using (var r1 = RunMaterialized())
        using (var r2 = RunStaged(stagedSimd))
        using (var r3 = RunStaged(stagedScalar))
        {
            if (!np.array_equal(r0, r1) || !np.array_equal(r0, r2) || !np.array_equal(r0, r3))
                throw new Exception($"mismatch {kind} {layout} {n}");
        }

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

        // Interleave the executors per round so host drift hits them alike; materialize and staged get a second round.
        double tU = Time(RunUnfused), tM = Time(RunMaterialized), tS = Time(() => RunStaged(stagedSimd)), tI = Time(() => RunStaged(stagedScalar));
        double tM2 = Time(RunMaterialized), tS2 = Time(() => RunStaged(stagedSimd));
        tM = Math.Min(tM, tM2); tS = Math.Min(tS, tS2);
        Console.WriteLine($"{kind} {layout,-8} n={n,9}  unfused {tU,9:F1}  materialize {tM,9:F1}  staged {tS,9:F1}  inline~ {tI,9:F1} us   S/M={tM / tS:F2}x");
    }
}
