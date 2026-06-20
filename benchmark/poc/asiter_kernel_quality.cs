#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// asiter_kernel_quality.cs — find the BEST inner-loop kernel per reduction.
// Isolates kernel quality on CONTIGUOUS data (the chunk an EXLOOP iterator
// hands the kernel) across cache tiers, reporting GB/s to expose whether each
// case is compute-bound (SIMD/unroll helps) or memory-bound (already optimal).
//
//   Half.sum     : 1-acc / 4-acc / 8-acc scalar (no packed f16->f32 on AVX2)
//   Complex.sum  : 1-acc / 4-acc scalar / Vector256<double> double-view SIMD
//   Half.max     : 1-acc / 4-acc scalar (NaN-propagate)
//   Complex.max  : 1-acc / 2-acc lexicographic
//
// Run:  dotnet run -c Release - < benchmark/poc/asiter_kernel_quality.cs
//
using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using NumSharp;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("FATAL: Debug build."); return; }

const int ROUNDS = 9;
int[] SIZES = { 1024, 65536, 4_000_000 };
var rng = new Random(2024);

nint AllocHalf(int n) { unsafe { var p=(Half*)NativeMemory.Alloc((nuint)(n*2)); for(int i=0;i<n;i++) p[i]=(Half)rng.NextDouble(); return (nint)p; } }
nint AllocCplx(int n) { unsafe { var p=(Complex*)NativeMemory.Alloc((nuint)(n*16)); for(int i=0;i<n;i++) p[i]=new Complex(rng.NextDouble(),rng.NextDouble()); return (nint)p; } }
void Free(nint a) { unsafe { NativeMemory.Free((void*)a); } }

double Bench(Func<double> op, long n)
{
    int reps = (int)Math.Max(1, 16_000_000L / n);
    for (int i=0;i<3;i++) op();
    double best=double.MaxValue;
    for (int r=0;r<ROUNDS;r++){ var sw=Stopwatch.StartNew(); for(int i=0;i<reps;i++) op(); sw.Stop(); best=Math.Min(best, sw.Elapsed.TotalMilliseconds/reps); }
    return best;
}
string GBs(double ms, long bytes) => $"{bytes/ms/1e6,6:F1}";

Console.WriteLine("Contiguous inner-loop kernel quality. ms=best-of-9; GB/s=bytes_read/time; spd=vs variant #1.\n");

Console.WriteLine($"{"Half.sum",-12} {"N",9} | {"1acc ms",9} {"GB/s",6} | {"4acc ms",9} {"GB/s",6} {"spd",5} | {"8acc ms",9} {"GB/s",6} {"spd",5}");
foreach (int n in SIZES)
{
    nint p=AllocHalf(n); long bytes=(long)n*2;
    double r1=Bench(()=>HalfSum1(p,n),n), r4=Bench(()=>HalfSum4(p,n),n), r8=Bench(()=>HalfSum8(p,n),n);
    double v1=HalfSum1(p,n),v4=HalfSum4(p,n),v8=HalfSum8(p,n);
    string chk=(Math.Abs(v1-v4)<1e-3*(1+Math.Abs(v1))&&Math.Abs(v1-v8)<1e-3*(1+Math.Abs(v1)))?"":" CHK!";
    Console.WriteLine($"{"",-12} {n,9} | {r1,9:F4} {GBs(r1,bytes),6} | {r4,9:F4} {GBs(r4,bytes),6} {r1/r4,5:F2} | {r8,9:F4} {GBs(r8,bytes),6} {r1/r8,5:F2}{chk}");
    Free(p);
}
Console.WriteLine();

Console.WriteLine($"{"Complex.sum",-12} {"N",9} | {"1acc ms",9} {"GB/s",6} | {"4acc ms",9} {"GB/s",6} {"spd",5} | {"SIMD ms",9} {"GB/s",6} {"spd",5}");
foreach (int n in SIZES)
{
    nint p=AllocCplx(n); long bytes=(long)n*16;
    double r1=Bench(()=>CplxSum1(p,n),n), r4=Bench(()=>CplxSum4(p,n),n), rS=Bench(()=>CplxSumSimd(p,n),n);
    double v1=CplxSum1(p,n),v4=CplxSum4(p,n),vS=CplxSumSimd(p,n);
    string chk=(Math.Abs(v1-v4)<1e-3*(1+Math.Abs(v1))&&Math.Abs(v1-vS)<1e-3*(1+Math.Abs(v1)))?"":" CHK!";
    Console.WriteLine($"{"",-12} {n,9} | {r1,9:F4} {GBs(r1,bytes),6} | {r4,9:F4} {GBs(r4,bytes),6} {r1/r4,5:F2} | {rS,9:F4} {GBs(rS,bytes),6} {r1/rS,5:F2}{chk}");
    Free(p);
}
Console.WriteLine();

Console.WriteLine($"{"Half.max",-12} {"N",9} | {"1acc ms",9} {"GB/s",6} | {"4acc ms",9} {"GB/s",6} {"spd",5}");
foreach (int n in SIZES)
{
    nint p=AllocHalf(n); long bytes=(long)n*2;
    double r1=Bench(()=>HalfMax1(p,n),n), r4=Bench(()=>HalfMax4(p,n),n);
    string chk=HalfMax1(p,n)==HalfMax4(p,n)?"":" CHK!";
    Console.WriteLine($"{"",-12} {n,9} | {r1,9:F4} {GBs(r1,bytes),6} | {r4,9:F4} {GBs(r4,bytes),6} {r1/r4,5:F2}{chk}");
    Free(p);
}
Console.WriteLine();

Console.WriteLine($"{"Complex.max",-12} {"N",9} | {"1acc ms",9} {"GB/s",6} | {"2acc ms",9} {"GB/s",6} {"spd",5}");
foreach (int n in SIZES)
{
    nint p=AllocCplx(n); long bytes=(long)n*16;
    double r1=Bench(()=>CplxMax1(p,n),n), r2=Bench(()=>CplxMax2(p,n),n);
    string chk=CplxMax1(p,n)==CplxMax2(p,n)?"":" CHK!";
    Console.WriteLine($"{"",-12} {n,9} | {r1,9:F4} {GBs(r1,bytes),6} | {r2,9:F4} {GBs(r2,bytes),6} {r1/r2,5:F2}{chk}");
    Free(p);
}

static unsafe double HalfSum1(nint a, long n){ Half* p=(Half*)a; double s=0; for(long i=0;i<n;i++) s+=(double)p[i]; return s; }
static unsafe double HalfSum4(nint a, long n){ Half* p=(Half*)a; double s0=0,s1=0,s2=0,s3=0; long i=0; for(;i+4<=n;i+=4){ s0+=(double)p[i]; s1+=(double)p[i+1]; s2+=(double)p[i+2]; s3+=(double)p[i+3]; } double s=s0+s1+s2+s3; for(;i<n;i++) s+=(double)p[i]; return s; }
static unsafe double HalfSum8(nint a, long n){ Half* p=(Half*)a; double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0; long i=0; for(;i+8<=n;i+=8){ s0+=(double)p[i];s1+=(double)p[i+1];s2+=(double)p[i+2];s3+=(double)p[i+3];s4+=(double)p[i+4];s5+=(double)p[i+5];s6+=(double)p[i+6];s7+=(double)p[i+7]; } double s=s0+s1+s2+s3+s4+s5+s6+s7; for(;i<n;i++) s+=(double)p[i]; return s; }

static unsafe double CplxSum1(nint a, long n){ Complex* p=(Complex*)a; Complex s=Complex.Zero; for(long i=0;i<n;i++) s+=p[i]; return s.Real+s.Imaginary; }
static unsafe double CplxSum4(nint a, long n){ Complex* p=(Complex*)a; Complex w=Complex.Zero,x=Complex.Zero,y=Complex.Zero,z=Complex.Zero; long i=0; for(;i+4<=n;i+=4){ w+=p[i]; x+=p[i+1]; y+=p[i+2]; z+=p[i+3]; } Complex s=w+x+y+z; for(;i<n;i++) s+=p[i]; return s.Real+s.Imaginary; }
static unsafe double CplxSumSimd(nint a, long n){
    double* d=(double*)a; long m=n*2;
    Vector256<double> v0=Vector256<double>.Zero, v1=Vector256<double>.Zero; long i=0;
    for(;i+8<=m;i+=8){ v0+=Vector256.Load(d+i); v1+=Vector256.Load(d+i+4); }
    var v=v0+v1; double real=v.GetElement(0)+v.GetElement(2), imag=v.GetElement(1)+v.GetElement(3);
    for(;i<m;i+=2){ real+=d[i]; imag+=d[i+1]; }
    return real+imag;
}

static unsafe double HalfMax1(nint a, long n){ Half* p=(Half*)a; double best=double.NegativeInfinity; for(long i=0;i<n;i++){ double v=(double)p[i]; if(double.IsNaN(v)) return double.NaN; if(v>best) best=v; } return best; }
static unsafe double HalfMax4(nint a, long n){ Half* p=(Half*)a; double b0=double.NegativeInfinity,b1=b0,b2=b0,b3=b0; bool nan=false; long i=0; for(;i+4<=n;i+=4){ double v0=(double)p[i],v1=(double)p[i+1],v2=(double)p[i+2],v3=(double)p[i+3]; nan|=double.IsNaN(v0)|double.IsNaN(v1)|double.IsNaN(v2)|double.IsNaN(v3); if(v0>b0)b0=v0; if(v1>b1)b1=v1; if(v2>b2)b2=v2; if(v3>b3)b3=v3; } double best=Math.Max(Math.Max(b0,b1),Math.Max(b2,b3)); for(;i<n;i++){ double v=(double)p[i]; if(double.IsNaN(v))return double.NaN; if(v>best)best=v; } return nan?double.NaN:best; }

static unsafe double CplxMax1(nint a, long n){ Complex* p=(Complex*)a; Complex best=p[0]; for(long i=1;i<n;i++){ var v=p[i]; if(v.Real>best.Real||(v.Real==best.Real&&v.Imaginary>best.Imaginary)) best=v; } return best.Real+best.Imaginary; }
static unsafe double CplxMax2(nint a, long n){ Complex* p=(Complex*)a; Complex b0=p[0]; Complex b1=n>1?p[1]:p[0]; long i=2; for(;i+2<=n;i+=2){ var v0=p[i]; var v1=p[i+1]; if(v0.Real>b0.Real||(v0.Real==b0.Real&&v0.Imaginary>b0.Imaginary)) b0=v0; if(v1.Real>b1.Real||(v1.Real==b1.Real&&v1.Imaginary>b1.Imaginary)) b1=v1; } Complex best=(b0.Real>b1.Real||(b0.Real==b1.Real&&b0.Imaginary>b1.Imaginary))?b0:b1; for(;i<n;i++){ var v=p[i]; if(v.Real>best.Real||(v.Real==best.Real&&v.Imaginary>best.Imaginary)) best=v; } return best.Real+best.Imaginary; }
