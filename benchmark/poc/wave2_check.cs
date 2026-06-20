#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// wave2_check.cs — NumPy parity for Complex min/max/argmin/argmax after the
// struct-kernel + KEEPORDER(min/max)/CORDER(arg) wiring. Includes the
// transposed-multi-NaN bug-fix case (NumPy returns the memory-order-first NaN).
//
using System;
using System.Numerics;
using System.Text.Json;
using NumSharp;

var R = JsonDocument.Parse(System.IO.File.ReadAllText("benchmark/poc/wave2_ref.json")).RootElement;
double D(JsonElement e)=> e.ValueKind==JsonValueKind.String ? double.NaN : e.GetDouble();
Complex RefC(string k){ var e=R.GetProperty(k); return new Complex(D(e[0]), D(e[1])); }
long RefI(string k)=> R.GetProperty(k).GetInt64();
Complex Val(NDArray nd)=>(Complex)nd.GetAtIndex(0);
long ArgVal(NDArray nd)=>Convert.ToInt64(nd.GetAtIndex(0));
bool CClose(Complex a,Complex b){ bool re=(double.IsNaN(a.Real)&&double.IsNaN(b.Real))||Math.Abs(a.Real-b.Real)<=1e-9*(1+Math.Abs(b.Real)); bool im=(double.IsNaN(a.Imaginary)&&double.IsNaN(b.Imaginary))||Math.Abs(a.Imaginary-b.Imaginary)<=1e-9*(1+Math.Abs(b.Imaginary)); return re&&im; }

int pass=0, fail=0;
void Ck(string k, Complex got){ var e=RefC(k); bool ok=CClose(got,e); if(ok)pass++; else fail++; Console.WriteLine($"  [{(ok?"PASS":"FAIL")}] {k,-14} got={got}  exp={e}"); }
void CkI(string k, long got){ var e=RefI(k); bool ok=got==e; if(ok)pass++; else fail++; Console.WriteLine($"  [{(ok?"PASS":"FAIL")}] {k,-14} got={got}  exp={e}"); }

NDArray Arr(params Complex[] a)=>np.array(a);
Complex C(double r,double i)=>new Complex(r,i);

var c = Arr(C(3,1),C(3,9),C(3,2),C(1,5),C(3,9));
Ck("max_lex",Val(np.max(c))); Ck("min_lex",Val(np.min(c))); CkI("argmax_lex",ArgVal(np.argmax(c))); CkI("argmin_lex",ArgVal(np.argmin(c)));

var marr=new Complex[12]; for(int i=0;i<12;i++) marr[i]=C(i,11-i);
var m=np.array(marr).reshape(3,4).T;
Ck("max_T",Val(np.max(m))); Ck("min_T",Val(np.min(m))); CkI("argmax_T",ArgVal(np.argmax(m))); CkI("argmin_T",ArgVal(np.argmin(m)));

var sarr=new Complex[20]; for(int i=0;i<20;i++) sarr[i]=C(i,(i*7)%11);
var s=np.array(sarr)["::3"];
Ck("max_slice",Val(np.max(s))); Ck("min_slice",Val(np.min(s))); CkI("argmax_slice",ArgVal(np.argmax(s))); CkI("argmin_slice",ArgVal(np.argmin(s)));

// bug-fix case
double nan=double.NaN;
var aflat=Arr(C(1,1),C(nan,5),C(3,3),C(nan,7),C(2,2),C(4,4)).reshape(2,3);
var aT=aflat.T;
Console.WriteLine("-- transposed multi-NaN (bug-fix): NumPy returns memory-order-first NaN (nan,5) --");
Ck("min_aT",Val(np.min(aT))); Ck("max_aT",Val(np.max(aT))); CkI("argmin_aT",ArgVal(np.argmin(aT))); CkI("argmax_aT",ArgVal(np.argmax(aT)));
Ck("min_a",Val(np.min(aflat))); Ck("max_a",Val(np.max(aflat))); CkI("argmin_a",ArgVal(np.argmin(aflat))); CkI("argmax_a",ArgVal(np.argmax(aflat)));

var b=Arr(C(1,1),C(nan,0),C(2,2));
Ck("min_1nan",Val(np.min(b))); CkI("argmin_1nan",ArgVal(np.argmin(b)));

Console.WriteLine($"\n{pass} passed, {fail} failed.");
