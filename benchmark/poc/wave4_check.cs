#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// wave4_check.cs — NumPy parity for Half/Complex NaN reductions after the
// struct-kernel wiring. nansum/nanprod/nanmin/nanmax/nanmean exact-ish; Half
// nanvar/nanstd use a looser tol (NumSharp double-accumulates vs NumPy's f16
// pairwise — documented divergence); Complex nanvar/nanstd are float64 → tight.
//
using System;
using System.Numerics;
using System.Text.Json;
using NumSharp;

var R = JsonDocument.Parse(System.IO.File.ReadAllText("benchmark/poc/wave4_ref.json")).RootElement;
double Dv(JsonElement e){ if(e.ValueKind==JsonValueKind.String){ var s=e.GetString(); return s=="nan"?double.NaN:(s=="inf"?double.PositiveInfinity:(s=="-inf"?double.NegativeInfinity:double.Parse(s))); } return e.GetDouble(); }
double RefD(string k)=>Dv(R.GetProperty(k));
Complex RefC(string k){ var e=R.GetProperty(k); return new Complex(Dv(e[0]),Dv(e[1])); }
double HVal(NDArray nd)=>(double)(Half)nd.GetAtIndex(0);
double DVal(NDArray nd)=>Convert.ToDouble(nd.GetAtIndex(0));
Complex CVal(NDArray nd)=>(Complex)nd.GetAtIndex(0);
NDArray H(double[] v)=>np.array(v).astype(NPTypeCode.Half);
double nan=double.NaN;
double[] hg(int n){ var a=new double[n]; for(int i=0;i<n;i++) a[i]= i%5==2 ? nan : (i%7)-3; return a; }
Complex[] cg(int n){ var a=new Complex[n]; for(int i=0;i<n;i++) a[i]= i%5==2 ? new Complex(nan,(i%4)-1) : new Complex((i%7)-3,(i%4)-1); return a; }
bool SameD(double a,double b,double tol){ if(double.IsNaN(a)&&double.IsNaN(b))return true; if(double.IsInfinity(a)||double.IsInfinity(b))return a==b; return Math.Abs(a-b)<=tol*(1+Math.Abs(b)); }
bool SameC(Complex a,Complex b,double tol){ bool re=(double.IsNaN(a.Real)&&double.IsNaN(b.Real))||Math.Abs(a.Real-b.Real)<=tol*(1+Math.Abs(b.Real)); bool im=(double.IsNaN(a.Imaginary)&&double.IsNaN(b.Imaginary))||Math.Abs(a.Imaginary-b.Imaginary)<=tol*(1+Math.Abs(b.Imaginary)); return re&&im; }

int pass=0, fail=0;
void Ck(string k, double got, double tol){ var e=RefD(k); bool ok=SameD(got,e,tol); if(ok)pass++; else fail++; Console.WriteLine($"  [{(ok?"PASS":"FAIL")}] {k,-12} got={got}  exp={e}"); }
void CkC(string k, Complex got, double tol){ var e=RefC(k); bool ok=SameC(got,e,tol); if(ok)pass++; else fail++; Console.WriteLine($"  [{(ok?"PASS":"FAIL")}] {k,-12} got={got}  exp={e}"); }

Console.WriteLine("Half NaN reductions:");
var h=H(hg(1000));
Ck("h_nansum",HVal(np.nansum(h)),1e-3); Ck("h_nanmin",HVal(np.nanmin(h)),1e-6); Ck("h_nanmax",HVal(np.nanmax(h)),1e-6);
Ck("h_nanmean",HVal(np.nanmean(h)),0.02); Ck("h_nanvar",HVal(np.nanvar(h)),0.05); Ck("h_nanstd",HVal(np.nanstd(h)),0.05);
Ck("h_nanvar1",HVal(np.nanvar(h,ddof:1)),0.05);
var hp=new double[12]; for(int i=0;i<12;i++) hp[i]= i%5==2 ? nan : (i%4)+1;
Ck("h_nanprod",HVal(np.nanprod(H(hp))),1e-3);
var hT=H(hg(120)).reshape(10,12).T;
Ck("hT_nansum",HVal(np.nansum(hT)),1e-3); Ck("hT_nanmean",HVal(np.nanmean(hT)),0.02); Ck("hT_nanmin",HVal(np.nanmin(hT)),1e-6);

Console.WriteLine("Complex NaN reductions:");
var c=np.array(cg(1000));
CkC("c_nansum",CVal(np.nansum(c)),1e-9); CkC("c_nanmean",CVal(np.nanmean(c)),1e-9);
Ck("c_nanvar",DVal(np.nanvar(c)),1e-6); Ck("c_nanstd",DVal(np.nanstd(c)),1e-6); Ck("c_nanvar1",DVal(np.nanvar(c,ddof:1)),1e-6);
var cT=np.array(cg(120)).reshape(10,12).T;
CkC("cT_nansum",CVal(np.nansum(cT)),1e-9); CkC("cT_nanmean",CVal(np.nanmean(cT)),1e-9);

Console.WriteLine("All-NaN slices:");
var hn=H(new double[]{nan,nan,nan,nan,nan,nan,nan,nan});
Ck("hn_nanmin",HVal(np.nanmin(hn)),1e-6); Ck("hn_nanmean",HVal(np.nanmean(hn)),1e-6); Ck("hn_nanvar",HVal(np.nanvar(hn)),1e-6);
var cnArr=new Complex[8]; for(int i=0;i<8;i++) cnArr[i]=new Complex(nan,0); var cn=np.array(cnArr);
CkC("cn_nanmean",CVal(np.nanmean(cn)),1e-9); Ck("cn_nanvar",DVal(np.nanvar(cn)),1e-6);

Console.WriteLine($"\n{pass} passed, {fail} failed.");
