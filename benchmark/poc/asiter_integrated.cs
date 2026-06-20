#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// asiter_integrated.cs — confirm the BEST kernels deliver END-TO-END through
// NpyIter.ExecuteReducing (stride-checked SIMD/unroll inner loop), vs the plain
// scalar struct kernel, vs the AsIterator baseline. contig + strided, 4M.
//
//   Complex.sum : scalar-struct  vs  SIMD-in-chunk (Vector256<double> when stride==16)
//   Half.max    : scalar-struct  vs  4acc-in-chunk (when stride==2)
//
using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("FATAL: Debug build."); return; }

const int ROUNDS = 9;
var rng = new Random(31);
NDArray CplxArr(int n){ var a=new Complex[n]; for(int i=0;i<n;i++) a[i]=new Complex(rng.NextDouble(),rng.NextDouble()); return np.array(a); }
NDArray HalfArr(int n){ var a=new double[n]; for(int i=0;i<n;i++) a[i]=rng.NextDouble(); return np.array(a).astype(NPTypeCode.Half); }
NDArray Strided(NDArray f){ int n=(int)f.size; int r=(int)Math.Sqrt(n); while(r>1&&n%r!=0)r--; return f.reshape(r,n/r).T; }

double Bench(Func<double> op){ for(int i=0;i<3;i++) op(); double best=double.MaxValue; for(int r=0;r<ROUNDS;r++){ var sw=Stopwatch.StartNew(); op(); sw.Stop(); best=Math.Min(best,sw.Elapsed.TotalMilliseconds);} return best; }
bool Close(double a,double b)=>Math.Abs(a-b)<=1e-6*(1+Math.Abs(b));

int n=4_000_000;
var cC=CplxArr(n); var cS=Strided(CplxArr(n));
var hC=HalfArr(n); var hS=Strided(HalfArr(n));

Console.WriteLine($"4M elements, best-of-{ROUNDS}. spd = baseline_AsIter / candidate.\n");
Console.WriteLine($"{"case",-20} | {"AsIter ms",10} | {"Exec-scalar",12} {"spd",5} | {"Exec-opt",10} {"spd",5} | chk");
Console.WriteLine(new string('-',86));

foreach (var (lay,arr) in new[]{("contig",cC),("strided",cS)})
{
    double a=Bench(()=>CplxSum_AsIter(arr));
    double b=Bench(()=>CplxSum_ExecScalar(arr));
    double c=Bench(()=>CplxSum_ExecSimd(arr));
    double rf=CplxSum_AsIter(arr);
    Console.WriteLine($"{$"Complex.sum {lay}",-20} | {a,10:F4} | {b,12:F4} {a/b,5:F2} | {c,10:F4} {a/c,5:F2} | {(Close(CplxSum_ExecSimd(arr),rf)?"ok":"DIFF")}");
}
foreach (var (lay,arr) in new[]{("contig",hC),("strided",hS)})
{
    double a=Bench(()=>HalfMax_AsIter(arr));
    double b=Bench(()=>HalfMax_ExecScalar(arr));
    double c=Bench(()=>HalfMax_Exec4(arr));
    double rf=HalfMax_AsIter(arr);
    Console.WriteLine($"{$"Half.max {lay}",-20} | {a,10:F4} | {b,12:F4} {a/b,5:F2} | {c,10:F4} {a/c,5:F2} | {(Close(HalfMax_Exec4(arr),rf)?"ok":"DIFF")}");
}

// ---- Complex.sum ----
static double CplxSum_AsIter(NDArray arr){ var it=arr.AsIterator<Complex>(); Complex s=Complex.Zero; while(it.HasNext()) s+=it.MoveNext(); return s.Real+s.Imaginary; }
static unsafe double CplxSum_ExecScalar(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); var s=it.ExecuteReducing<CSumScalar,Complex>(default,Complex.Zero); return s.Real+s.Imaginary; }
static unsafe double CplxSum_ExecSimd(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); var s=it.ExecuteReducing<CSumSimd,Complex>(default,Complex.Zero); return s.Real+s.Imaginary; }

// ---- Half.max ----
static double HalfMax_AsIter(NDArray arr){ var it=arr.AsIterator<Half>(); double best=double.NegativeInfinity; while(it.HasNext()){ double v=(double)it.MoveNext(); if(double.IsNaN(v))return double.NaN; if(v>best)best=v; } return best; }
static unsafe double HalfMax_ExecScalar(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); var a=it.ExecuteReducing<HMaxScalar,MMAcc>(default,new MMAcc{Best=double.NegativeInfinity}); return a.SawNaN?double.NaN:a.Best; }
static unsafe double HalfMax_Exec4(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); var a=it.ExecuteReducing<HMax4,MMAcc>(default,new MMAcc{Best=double.NegativeInfinity}); return a.SawNaN?double.NaN:a.Best; }

public struct MMAcc { public double Best; public bool Seen; public bool SawNaN; }

public readonly struct CSumScalar : INpyReducingInnerLoop<Complex>
{ public unsafe bool Execute(void** dp,long* st,long count,ref Complex sum){ byte* p=(byte*)dp[0]; long s=st[0]; Complex acc=sum; for(long i=0;i<count;i++) acc+=*(Complex*)(p+i*s); sum=acc; return true; } }

public readonly struct CSumSimd : INpyReducingInnerLoop<Complex>
{
    public unsafe bool Execute(void** dp, long* st, long count, ref Complex sum)
    {
        byte* p=(byte*)dp[0]; long s=st[0];
        if (s == 16)
        {
            double* d=(double*)p; long m=count*2;
            Vector256<double> v0=Vector256<double>.Zero, v1=Vector256<double>.Zero; long i=0;
            for(;i+8<=m;i+=8){ v0+=Vector256.Load(d+i); v1+=Vector256.Load(d+i+4); }
            var v=v0+v1; double re=sum.Real+v.GetElement(0)+v.GetElement(2), im=sum.Imaginary+v.GetElement(1)+v.GetElement(3);
            for(;i<m;i+=2){ re+=d[i]; im+=d[i+1]; }
            sum=new Complex(re,im);
        }
        else { Complex acc=sum; for(long i=0;i<count;i++) acc+=*(Complex*)(p+i*s); sum=acc; }
        return true;
    }
}

public readonly struct HMaxScalar : INpyReducingInnerLoop<MMAcc>
{ public unsafe bool Execute(void** dp,long* st,long count,ref MMAcc a){ byte* p=(byte*)dp[0]; long s=st[0]; double best=a.Best; for(long i=0;i<count;i++){ double v=(double)*(Half*)(p+i*s); if(double.IsNaN(v)){a.SawNaN=true;return false;} if(v>best)best=v; } a.Best=best; a.Seen=true; return true; } }

public readonly struct HMax4 : INpyReducingInnerLoop<MMAcc>
{
    public unsafe bool Execute(void** dp, long* st, long count, ref MMAcc a)
    {
        byte* p=(byte*)dp[0]; long s=st[0]; double best=a.Best;
        if (s == 2)
        {
            Half* h=(Half*)p; double b0=best,b1=double.NegativeInfinity,b2=b1,b3=b1; bool nan=false; long i=0;
            for(;i+4<=count;i+=4){ double v0=(double)h[i],v1=(double)h[i+1],v2=(double)h[i+2],v3=(double)h[i+3]; nan|=double.IsNaN(v0)|double.IsNaN(v1)|double.IsNaN(v2)|double.IsNaN(v3); if(v0>b0)b0=v0; if(v1>b1)b1=v1; if(v2>b2)b2=v2; if(v3>b3)b3=v3; }
            best=Math.Max(Math.Max(b0,b1),Math.Max(b2,b3));
            for(;i<count;i++){ double v=(double)h[i]; if(double.IsNaN(v)){a.SawNaN=true;return false;} if(v>best)best=v; }
            if(nan){a.SawNaN=true;return false;}
        }
        else { for(long i=0;i<count;i++){ double v=(double)*(Half*)(p+i*s); if(double.IsNaN(v)){a.SawNaN=true;return false;} if(v>best)best=v; } }
        a.Best=best; a.Seen=true; return true;
    }
}
