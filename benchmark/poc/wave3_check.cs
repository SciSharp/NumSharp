#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// wave3_check.cs — NumPy parity for Half sum/prod/mean/min/max/argmin/argmax
// after the struct-kernel upgrade (sum/prod/mean/min/max) + AsIterator removal
// (argmin/argmax). min/max/arg are exact; sum/prod/mean use f16-aware tolerance
// (NumSharp double-accumulates then narrows — a documented, more-accurate
// divergence from NumPy's f16 pairwise; large sums still saturate to ±inf).
//
using System;
using System.Text.Json;
using NumSharp;

var R = JsonDocument.Parse(System.IO.File.ReadAllText("benchmark/poc/wave3_ref.json")).RootElement;
double Dv(JsonElement e){ if(e.ValueKind==JsonValueKind.String){ var s=e.GetString(); return s=="nan"?double.NaN:(s=="inf"?double.PositiveInfinity:(s=="-inf"?double.NegativeInfinity:double.Parse(s))); } return e.GetDouble(); }
double RefD(string k)=>Dv(R.GetProperty(k));
long RefI(string k)=>R.GetProperty(k).GetInt64();
double HVal(NDArray nd)=>(double)(Half)nd.GetAtIndex(0);
long ArgVal(NDArray nd)=>Convert.ToInt64(nd.GetAtIndex(0));
NDArray H(params double[] v)=>np.array(v).astype(NPTypeCode.Half);
NDArray Hrange(int n){ var a=new double[n]; for(int i=0;i<n;i++)a[i]=i; return np.array(a).astype(NPTypeCode.Half); }
bool Same(double a,double b,double tol){ if(double.IsNaN(a)&&double.IsNaN(b))return true; if(double.IsInfinity(a)||double.IsInfinity(b))return a==b; return Math.Abs(a-b)<=tol*(1+Math.Abs(b)); }

int pass=0, fail=0;
void Ck(string k, double got, double tol){ var e=RefD(k); bool ok=Same(got,e,tol); if(ok)pass++; else fail++; Console.WriteLine($"  [{(ok?"PASS":"FAIL")}] {k,-13} got={got}  exp={e}"); }
void CkI(string k, long got){ var e=RefI(k); bool ok=got==e; if(ok)pass++; else fail++; Console.WriteLine($"  [{(ok?"PASS":"FAIL")}] {k,-13} got={got}  exp={e}"); }

Console.WriteLine("Half sum/prod/mean (f16-tolerance):");
var h=H(1.5,2.25,-0.5,3.0,4.5,-2.0,0.25,1.0);
Ck("sum",HVal(np.sum(h)),0.02); Ck("mean",HVal(np.mean(h)),0.02);
Ck("prod",HVal(np.prod(H(1.5,2.0,0.5,1.25,2.0))),0.02);
var m=Hrange(24).reshape(4,6).T;
Ck("sum_T",HVal(np.sum(m)),0.02); Ck("mean_T",HVal(np.mean(m)),0.02);
Ck("sum_slice",HVal(np.sum(Hrange(30)["::3"])),0.02);
var big=H(new double[1]); { var arr=new double[100000]; for(int i=0;i<100000;i++)arr[i]=2.5; big=np.array(arr).astype(NPTypeCode.Half); }
Ck("sum_inf",HVal(np.sum(big)),0.02);

Console.WriteLine("Half min/max/arg (exact):");
var v=H(3.0,1.5,4.5,1.5,-2.0,4.5,0.0);
Ck("max",HVal(np.max(v)),1e-6); Ck("min",HVal(np.min(v)),1e-6); CkI("argmax",ArgVal(np.argmax(v))); CkI("argmin",ArgVal(np.argmin(v)));
var mm=H(5,2,8,1,9,3,7,4,6,0,11,10).reshape(3,4).T;
Ck("max_T",HVal(np.max(mm)),1e-6); Ck("min_T",HVal(np.min(mm)),1e-6); CkI("argmax_T",ArgVal(np.argmax(mm))); CkI("argmin_T",ArgVal(np.argmin(mm)));
var ss=H(2,9,1,8,3,7,4,6,0,5)["::2"];
CkI("argmax_slice",ArgVal(np.argmax(ss))); CkI("argmin_slice",ArgVal(np.argmin(ss)));

Console.WriteLine("Half NaN (propagate; arg → first-NaN C-order index):");
var n=H(1.0,2.0,double.NaN,3.0);
Ck("max_nan",HVal(np.max(n)),1e-6); Ck("min_nan",HVal(np.min(n)),1e-6); CkI("argmax_nan",ArgVal(np.argmax(n))); CkI("argmin_nan",ArgVal(np.argmin(n)));
var nt=H(1,2,3,double.NaN,5,6).reshape(2,3).T;
CkI("argmax_nanT",ArgVal(np.argmax(nt))); Ck("max_nanT",HVal(np.max(nt)),1e-6);

Console.WriteLine($"\n{pass} passed, {fail} failed.");
