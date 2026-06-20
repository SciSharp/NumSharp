#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// asiter_foreach_vs_struct.cs — the REAL comparison for the wiring decision.
// Production already migrated sum/prod/mean + Half min/max to NpyIter via the
// ForEach(delegate) helpers (ComplexReduceViaNpyIter / HalfReduceViaNpyIter /
// HalfMinMaxViaNpyIter). The plan upgrades those to struct-generic
// ExecuteReducing (devirtualized + inlined + SIMD/unroll-in-chunk).
//
// This measures ForEach-lambda (current) vs ExecuteReducing-struct (proposed),
// so we only convert where there is a real win and LEAVE compute-bound paths.
//
//   Complex.sum  : ForEach scalar-add   vs  ExecuteReducing Vector256<double> SIMD
//   Complex.prod : ForEach scalar-mul   vs  ExecuteReducing scalar (devirt only)
//   Half.sum     : ForEach scalar f64   vs  ExecuteReducing scalar f64 (devirt only)
//   Half.max     : ForEach scalar       vs  ExecuteReducing 4-acc-in-chunk
//
// Run:  dotnet run -c Release - < benchmark/poc/asiter_foreach_vs_struct.cs
//
using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("FATAL: Debug build. Use -c Release."); return; }

const int ROUNDS = 9;
var rng = new Random(31);
NDArray CplxArr(int n){ var a=new Complex[n]; for(int i=0;i<n;i++) a[i]=new Complex(rng.NextDouble(),rng.NextDouble()); return np.array(a); }
NDArray HalfArr(int n){ var a=new double[n]; for(int i=0;i<n;i++) a[i]=rng.NextDouble(); return np.array(a).astype(NPTypeCode.Half); }
NDArray Strided(NDArray f){ int n=(int)f.size; int r=(int)Math.Sqrt(n); while(r>1&&n%r!=0)r--; return f.reshape(r,n/r).T; }

double Bench(Func<double> op){ for(int i=0;i<3;i++) op(); double best=double.MaxValue; for(int r=0;r<ROUNDS;r++){ var sw=Stopwatch.StartNew(); op(); sw.Stop(); best=Math.Min(best,sw.Elapsed.TotalMilliseconds);} return best; }
bool Close(double a,double b)=>(double.IsNaN(a)&&double.IsNaN(b))||Math.Abs(a-b)<=1e-5*(1+Math.Abs(b));

int n=4_000_000;
var cC=CplxArr(n); var cS=Strided(CplxArr(n));
var hC=HalfArr(n); var hS=Strided(HalfArr(n));

Console.WriteLine($"4M elements, best-of-{ROUNDS}. spd = ForEach_ms / Struct_ms  (>1 => struct faster).\n");
Console.WriteLine($"{"case",-20} | {"ForEach ms",10} | {"Struct ms",10} {"spd",6} | chk");
Console.WriteLine(new string('-',60));

foreach (var (lay,arr) in new[]{("contig",cC),("strided",cS)})
{ double a=Bench(()=>CSum_FE(arr)); double b=Bench(()=>CSum_K(arr)); Console.WriteLine($"{$"Complex.sum {lay}",-20} | {a,10:F4} | {b,10:F4} {a/b,6:F2} | {(Close(CSum_K(arr),CSum_FE(arr))?"ok":"DIFF")}"); }
foreach (var (lay,arr) in new[]{("contig",cC),("strided",cS)})
{ double a=Bench(()=>CProd_FE(arr)); double b=Bench(()=>CProd_K(arr)); Console.WriteLine($"{$"Complex.prod {lay}",-20} | {a,10:F4} | {b,10:F4} {a/b,6:F2} | {(Close(CProd_K(arr),CProd_FE(arr))?"ok":"DIFF")}"); }
foreach (var (lay,arr) in new[]{("contig",hC),("strided",hS)})
{ double a=Bench(()=>HSum_FE(arr)); double b=Bench(()=>HSum_K(arr)); Console.WriteLine($"{$"Half.sum {lay}",-20} | {a,10:F4} | {b,10:F4} {a/b,6:F2} | {(Close(HSum_K(arr),HSum_FE(arr))?"ok":"DIFF")}"); }
foreach (var (lay,arr) in new[]{("contig",hC),("strided",hS)})
{ double a=Bench(()=>HMax_FE(arr)); double b=Bench(()=>HMax_K(arr)); Console.WriteLine($"{$"Half.max {lay}",-20} | {a,10:F4} | {b,10:F4} {a/b,6:F2} | {(Close(HMax_K(arr),HMax_FE(arr))?"ok":"DIFF")}"); }

// ---- current production: ForEach(delegate) ----
static unsafe double CSum_FE(NDArray arr){ var acc=Complex.Zero; using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); it.ForEach((dp,st,c,aux)=>{ byte* p=(byte*)dp[0]; long s=st[0]; var a=(Complex*)aux; for(long i=0;i<c;i++) *a+=*(Complex*)(p+i*s); }, &acc); return acc.Real+acc.Imaginary; }
static unsafe double CProd_FE(NDArray arr){ var acc=Complex.One; using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); it.ForEach((dp,st,c,aux)=>{ byte* p=(byte*)dp[0]; long s=st[0]; var a=(Complex*)aux; for(long i=0;i<c;i++) *a*=*(Complex*)(p+i*s); }, &acc); return acc.Real+acc.Imaginary; }
static unsafe double HSum_FE(NDArray arr){ double acc=0; using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); it.ForEach((dp,st,c,aux)=>{ byte* p=(byte*)dp[0]; long s=st[0]; double* a=(double*)aux; for(long i=0;i<c;i++) *a+=(double)*(Half*)(p+i*s); }, &acc); return acc; }
static unsafe double HMax_FE(NDArray arr){ HE acc=default; using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); it.ForEach((dp,st,c,aux)=>{ byte* p=(byte*)dp[0]; long s=st[0]; var a=(HE*)aux; for(long i=0;i<c;i++){ double v=(double)*(Half*)(p+i*s); if(double.IsNaN(v))a->nan=true; else if(!a->seen||v>a->best){a->best=v;a->seen=true;} } }, &acc); return acc.nan?double.NaN:acc.best; }

// ---- proposed: ExecuteReducing(struct) ----
static unsafe double CSum_K(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); var s=it.ExecuteReducing<CSumK,Complex>(default,Complex.Zero); return s.Real+s.Imaginary; }
static unsafe double CProd_K(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); var s=it.ExecuteReducing<CProdK,Complex>(default,Complex.One); return s.Real+s.Imaginary; }
static unsafe double HSum_K(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); return it.ExecuteReducing<HSumK,double>(default,0.0); }
static unsafe double HMax_K(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); var a=it.ExecuteReducing<HMaxK,HE>(default,default); return a.nan?double.NaN:(a.seen?a.best:double.NegativeInfinity); }

public struct HE { public double best; public bool seen; public bool nan; }

public readonly struct CSumK : INpyReducingInnerLoop<Complex>
{ public unsafe bool Execute(void** dp,long* st,long count,ref Complex sum){ byte* p=(byte*)dp[0]; long s=st[0];
    if(s==16 && Vector256.IsHardwareAccelerated){ double* d=(double*)p; long m=count*2; Vector256<double> v0=Vector256<double>.Zero,v1=Vector256<double>.Zero; long i=0; for(;i+8<=m;i+=8){ v0+=Vector256.Load(d+i); v1+=Vector256.Load(d+i+4);} var v=v0+v1; double re=sum.Real+v.GetElement(0)+v.GetElement(2),im=sum.Imaginary+v.GetElement(1)+v.GetElement(3); for(;i<m;i+=2){re+=d[i];im+=d[i+1];} sum=new Complex(re,im);} else { Complex acc=sum; for(long i=0;i<count;i++) acc+=*(Complex*)(p+i*s); sum=acc; } return true; } }
public readonly struct CProdK : INpyReducingInnerLoop<Complex>
{ public unsafe bool Execute(void** dp,long* st,long count,ref Complex prod){ byte* p=(byte*)dp[0]; long s=st[0]; Complex acc=prod; for(long i=0;i<count;i++) acc*=*(Complex*)(p+i*s); prod=acc; return true; } }
public readonly struct HSumK : INpyReducingInnerLoop<double>
{ public unsafe bool Execute(void** dp,long* st,long count,ref double sum){ byte* p=(byte*)dp[0]; long s=st[0]; double acc=sum; for(long i=0;i<count;i++) acc+=(double)*(Half*)(p+i*s); sum=acc; return true; } }
public readonly struct HMaxK : INpyReducingInnerLoop<HE>
{ public unsafe bool Execute(void** dp,long* st,long count,ref HE a){ byte* p=(byte*)dp[0]; long s=st[0]; double best=a.seen?a.best:double.NegativeInfinity; bool seen=a.seen;
    if(s==2){ Half* h=(Half*)p; double b0=best,b1=double.NegativeInfinity,b2=b1,b3=b1; bool nan=false; long i=0; for(;i+4<=count;i+=4){ double v0=(double)h[i],v1=(double)h[i+1],v2=(double)h[i+2],v3=(double)h[i+3]; nan|=double.IsNaN(v0)|double.IsNaN(v1)|double.IsNaN(v2)|double.IsNaN(v3); if(v0>b0)b0=v0; if(v1>b1)b1=v1; if(v2>b2)b2=v2; if(v3>b3)b3=v3; } best=Math.Max(Math.Max(b0,b1),Math.Max(b2,b3)); seen|=count>0; for(;i<count;i++){ double v=(double)h[i]; if(double.IsNaN(v)){a.nan=true;a.best=best;a.seen=true;return false;} if(v>best)best=v; } if(nan){a.nan=true;a.best=best;a.seen=seen;return false;} }
    else { for(long i=0;i<count;i++){ double v=(double)*(Half*)(p+i*s); if(double.IsNaN(v)){a.nan=true;a.seen=true;return false;} if(!seen||v>best){best=v;seen=true;} } }
    a.best=best; a.seen=seen; return true; } }
