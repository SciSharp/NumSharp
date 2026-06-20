#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// asiter_half_wall.cs — is Half sum compute-bound on the f16->f64 conversion?
// Compare across sizes (L2..RAM): Half->double, Half->float, plain double (mem ref).
// Single-call latency methodology (reps kept low so independent calls don't overlap).
//
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using NumSharp;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("FATAL: Debug build."); return; }

const int ROUNDS = 11;
int[] SIZES = { 262144, 1048576, 4194304, 16777216 };
var rng = new Random(7);

nint AllocHalf(int n){ unsafe { var p=(Half*)NativeMemory.Alloc((nuint)(n*2)); for(int i=0;i<n;i++) p[i]=(Half)rng.NextDouble(); return (nint)p; } }
nint AllocDbl(int n){ unsafe { var p=(double*)NativeMemory.Alloc((nuint)(n*8)); for(int i=0;i<n;i++) p[i]=rng.NextDouble(); return (nint)p; } }
void Free(nint a){ unsafe { NativeMemory.Free((void*)a); } }

// single-call latency: warm once, time ONE call per round (best-of), to avoid cross-call overlap
double Bench1(Func<double> op)
{
    for(int i=0;i<3;i++) op();
    double best=double.MaxValue;
    for(int r=0;r<ROUNDS;r++){ var sw=Stopwatch.StartNew(); op(); sw.Stop(); best=Math.Min(best, sw.Elapsed.TotalMilliseconds); }
    return best;
}

Console.WriteLine("Single-call latency. half bytes=2N, double bytes=8N. GB/s = bytes/time.\n");
Console.WriteLine($"{"N",10} | {"H->f64 ms",10} {"GB/s",6} | {"H->f32 ms",10} {"GB/s",6} | {"f64 ms",9} {"GB/s",6}  (f64=mem ref)");
foreach (int n in SIZES)
{
    nint hp=AllocHalf(n), dp=AllocDbl(n);
    double rd=Bench1(()=>HalfSumD(hp,n)), rf=Bench1(()=>HalfSumF(hp,n)), rr=Bench1(()=>DblSum(dp,n));
    Console.WriteLine($"{n,10} | {rd,10:F4} {(double)n*2/rd/1e6,6:F1} | {rf,10:F4} {(double)n*2/rf/1e6,6:F1} | {rr,9:F4} {(double)n*8/rr/1e6,6:F1}");
    Free(hp); Free(dp);
}

static unsafe double HalfSumD(nint a, long n){ Half* p=(Half*)a; double s=0; for(long i=0;i<n;i++) s+=(double)p[i]; return s; }
static unsafe double HalfSumF(nint a, long n){ Half* p=(Half*)a; float s=0; for(long i=0;i<n;i++) s+=(float)p[i]; return s; }
static unsafe double DblSum(nint a, long n){ double* p=(double*)a; double s=0; for(long i=0;i<n;i++) s+=p[i]; return s; }
