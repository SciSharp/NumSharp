#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// asiter_nan_probe.cs — confirm NaN-aware Half/Complex reductions (the remaining
// AsIterator NaN sites) get the same win from ExecuteReducing. ~12% NaN planted.
//
// Run:  dotnet run -c Release - < benchmark/poc/asiter_nan_probe.cs
//
using System;
using System.Diagnostics;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
if (dbgCore?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("FATAL: Debug build."); return; }

const int ROUNDS = 7;
var rng = new Random(99);

NDArray HalfArr(int n)
{
    var a = new double[n];
    for (int i = 0; i < n; i++) a[i] = (rng.Next(8) == 0) ? double.NaN : rng.NextDouble();
    return np.array(a).astype(NPTypeCode.Half);
}
NDArray CplxArr(int n)
{
    var a = new Complex[n];
    for (int i = 0; i < n; i++) a[i] = (rng.Next(8) == 0) ? new Complex(double.NaN, 0) : new Complex(rng.NextDouble(), rng.NextDouble());
    return np.array(a);
}
NDArray Strided(NDArray flat) { int n=(int)flat.size; int r=(int)Math.Sqrt(n); while(r>1 && n%r!=0) r--; return flat.reshape(r, n/r).T; }

double Bench(Func<double> op, long n, out double res)
{
    int reps = (int)Math.Max(1, 8_000_000L / n);
    for (int i = 0; i < 2; i++) res = op(); res = op();
    double best = double.MaxValue;
    for (int round = 0; round < ROUNDS; round++)
    { var sw = Stopwatch.StartNew(); double last=0; for (int i=0;i<reps;i++) last=op(); sw.Stop(); best=Math.Min(best, sw.Elapsed.TotalMilliseconds/reps); res=last; }
    return best;
}
bool Close(double a,double b)=> double.IsNaN(a)&&double.IsNaN(b) || Math.Abs(a-b)<=1e-6*(1+Math.Abs(b));

Console.WriteLine($"{"op",-14} {"layout",-9} {"N",9} | A(ms) AsIter | B(ms) Exec  speedup | chk");
Console.WriteLine(new string('-', 78));
foreach (int n in new[] { 4_000_000 })
{
    var hC=HalfArr(n); var hS=Strided(HalfArr(n)); var cC=CplxArr(n); var cS=Strided(CplxArr(n));
    foreach (var (lay,arr) in new[]{("contig",hC),("strided",hS)})
    { double ra=Bench(()=>HalfNanSum_AsIter(arr),n,out _); double rb=Bench(()=>HalfNanSum_Exec(arr),n,out var vb); double rf=HalfNanSum_AsIter(arr);
      Console.WriteLine($"{"Half.nansum",-14} {lay,-9} {n,9} | {ra,11:F4} | {rb,10:F4} {ra/rb,5:F2}x | {(Close(vb,rf)?"ok":"DIFF")}"); }
    foreach (var (lay,arr) in new[]{("contig",cC),("strided",cS)})
    { double ra=Bench(()=>CplxNanSum_AsIter(arr),n,out _); double rb=Bench(()=>CplxNanSum_Exec(arr),n,out var vb); double rf=CplxNanSum_AsIter(arr);
      Console.WriteLine($"{"Complex.nansum",-14} {lay,-9} {n,9} | {ra,11:F4} | {rb,10:F4} {ra/rb,5:F2}x | {(Close(vb,rf)?"ok":"DIFF")}"); }
}

static double HalfNanSum_AsIter(NDArray arr){ double s=0; var it=arr.AsIterator<Half>(); while(it.HasNext()){ Half v=it.MoveNext(); if(!Half.IsNaN(v)) s+=(double)v; } return s; }
static unsafe double HalfNanSum_Exec(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); return it.ExecuteReducing<HalfNanSumK,double>(default,0.0); }
static double CplxNanSum_AsIter(NDArray arr){ var s=Complex.Zero; var it=arr.AsIterator<Complex>(); while(it.HasNext()){ var v=it.MoveNext(); if(!double.IsNaN(v.Real)&&!double.IsNaN(v.Imaginary)) s+=v; } return s.Real+s.Imaginary; }
static unsafe double CplxNanSum_Exec(NDArray arr){ using var it=NpyIterRef.New(arr,NpyIterGlobalFlags.EXTERNAL_LOOP); var s=it.ExecuteReducing<CplxNanSumK,Complex>(default,Complex.Zero); return s.Real+s.Imaginary; }

public readonly struct HalfNanSumK : INpyReducingInnerLoop<double>
{ public unsafe bool Execute(void** dp,long* st,long count,ref double sum){ byte* p=(byte*)dp[0]; long s=st[0]; double a=sum; for(long i=0;i<count;i++){ double v=(double)*(Half*)(p+i*s); if(!double.IsNaN(v)) a+=v; } sum=a; return true; } }
public readonly struct CplxNanSumK : INpyReducingInnerLoop<Complex>
{ public unsafe bool Execute(void** dp,long* st,long count,ref Complex sum){ byte* p=(byte*)dp[0]; long s=st[0]; Complex a=sum; for(long i=0;i<count;i++){ var v=*(Complex*)(p+i*s); if(!double.IsNaN(v.Real)&&!double.IsNaN(v.Imaginary)) a+=v; } sum=a; return true; } }
