#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property TargetFramework=net10.0
// Review the REAL np.argsort path (not the POC kernel): decompose where time goes.
using System;
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends.Sorting;

static double Best(int rounds, Action a){ double b=1e18; for(int i=0;i<rounds;i++){ var sw=Stopwatch.StartNew(); a(); sw.Stop(); b=Math.Min(b, sw.Elapsed.TotalMilliseconds);} return b; }
const int R = 9;

int[] sizes = { 1_000_000, 4_000_000 };
var rng = new Random(42);

Console.WriteLine("== REAL np.argsort path decomposition (Release, best-of-9) ==");
Console.WriteLine("NumPy 2.4.2 ref @i9-13900K: int32 1M=45 M/s 4M=28 | int64 1M=31 4M=20 | f64 1M=40 4M=24\n");

foreach (int N in sizes)
{
    // ---------- int32 ----------
    var d32 = new int[N]; for (int i=0;i<N;i++) d32[i]=rng.Next(int.MinValue,int.MaxValue);
    var a32 = np.array(d32);
    np.argsort(a32); // warmup/JIT

    double full32 = Best(R, () => { var g = np.argsort(a32); GC.KeepAlive(g); });

    // sub-step: the 6 scratch arrays ArgSortInto allocates per call
    double scratch32 = Best(R, () => {
        var k32=new uint[N]; var t32=new uint[N]; var k64=new ulong[N]; var t64=new ulong[N];
        var idx=new long[N]; var it=new long[N];
        GC.KeepAlive(k32); GC.KeepAlive(t32); GC.KeepAlive(k64); GC.KeepAlive(t64); GC.KeepAlive(idx); GC.KeepAlive(it);
    });
    // sub-step: only the buffers a 32-bit dtype actually USES (k32,t32,idx,it)
    double scratchUsed32 = Best(R, () => {
        var k32=new uint[N]; var t32=new uint[N]; var idx=new long[N]; var it=new long[N];
        GC.KeepAlive(k32); GC.KeepAlive(t32); GC.KeepAlive(idx); GC.KeepAlive(it);
    });
    // sub-step: output NDArray alloc (int64, N)
    double outAlloc = Best(R, () => { var o = new NDArray(NPTypeCode.Int64, new Shape(N), false); GC.KeepAlive(o); });

    // sub-step: pure kernel on pre-allocated/pre-fixed buffers (POC-equivalent)
    double kern32 = KernelOnly32(d32, N, R);

    Console.WriteLine($"int32 N={N,8}:");
    Console.WriteLine($"   np.argsort FULL    {full32,7:F2}ms  ({N/full32/1e3,5:F0} M/s)");
    Console.WriteLine($"   scratch alloc(6)   {scratch32,7:F2}ms  ({100*scratch32/full32,4:F0}% of full)  [40N bytes]");
    Console.WriteLine($"   scratch used(4)    {scratchUsed32,7:F2}ms  ({100*scratchUsed32/full32,4:F0}% of full)  [24N bytes]");
    Console.WriteLine($"   output alloc       {outAlloc,7:F2}ms  ({100*outAlloc/full32,4:F0}% of full)");
    Console.WriteLine($"   KERNEL only        {kern32,7:F2}ms  ({N/kern32/1e3,5:F0} M/s)  ({100*kern32/full32,4:F0}% of full)");
    Console.WriteLine($"   -> overhead        {full32-kern32,7:F2}ms  ({100*(full32-kern32)/full32,4:F0}% of full)");

    // ---------- int64 ----------
    var d64 = new long[N]; for (int i=0;i<N;i++) d64[i]=((long)rng.Next()<<32)|(uint)rng.Next();
    var a64 = np.array(d64);
    np.argsort(a64);
    double full64 = Best(R, () => { var g = np.argsort(a64); GC.KeepAlive(g); });
    double kern64 = KernelOnly64(d64, N, R);
    Console.WriteLine($"int64 N={N,8}:");
    Console.WriteLine($"   np.argsort FULL    {full64,7:F2}ms  ({N/full64/1e3,5:F0} M/s)");
    Console.WriteLine($"   KERNEL only        {kern64,7:F2}ms  ({N/kern64/1e3,5:F0} M/s)  ({100*kern64/full64,4:F0}% of full)");
    Console.WriteLine($"   -> overhead        {full64-kern64,7:F2}ms  ({100*(full64-kern64)/full64,4:F0}% of full)\n");
}

static unsafe double KernelOnly32(int[] data, int N, int R)
{
    var k=new uint[N]; var t=new uint[N]; var idx=new long[N]; var it=new long[N]; var cnt=new int[256];
    double b=1e18;
    fixed(uint* pk=k, pt=t) fixed(long* pi=idx, pit=it) fixed(int* pc=cnt)
    {
        for(int r=0;r<R;r++)
        {
            // fill keys+indices fresh each round (part of every real line call)
            for(int i=0;i<N;i++){ pk[i]=(uint)data[i]^0x80000000u; pi[i]=i; }
            var sw=Stopwatch.StartNew();
            long* res = RadixSort.ArgSortU32(pk, pt, pi, pit, N, 4, pc);
            sw.Stop(); b=Math.Min(b, sw.Elapsed.TotalMilliseconds);
            GC.KeepAlive((IntPtr)res);
        }
    }
    return b;
}
static unsafe double KernelOnly64(long[] data, int N, int R)
{
    var k=new ulong[N]; var t=new ulong[N]; var idx=new long[N]; var it=new long[N]; var cnt=new int[256];
    double b=1e18;
    fixed(ulong* pk=k, pt=t) fixed(long* pi=idx, pit=it) fixed(int* pc=cnt)
    {
        for(int r=0;r<R;r++)
        {
            for(int i=0;i<N;i++){ pk[i]=(ulong)data[i]^0x8000000000000000UL; pi[i]=i; }
            var sw=Stopwatch.StartNew();
            long* res = RadixSort.ArgSortU64(pk, pt, pi, pit, N, pc);
            sw.Stop(); b=Math.Min(b, sw.Elapsed.TotalMilliseconds);
            GC.KeepAlive((IntPtr)res);
        }
    }
    return b;
}
