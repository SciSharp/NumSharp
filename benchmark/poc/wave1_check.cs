#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// wave1_check.cs — NumPy parity for Complex sum/prod/mean after the struct-kernel
// upgrade. Builds the SAME deterministic arrays as wave1_ref.py and compares.
//
using System;
using System.Numerics;
using System.Text.Json;
using NumSharp;

var refJson = System.IO.File.ReadAllText("benchmark/poc/wave1_ref.json");
using var doc = JsonDocument.Parse(refJson);
var R = doc.RootElement;

Complex Ref(string k){ var e=R.GetProperty(k); return new Complex(e[0].GetDouble(), e[1].GetDouble()); }
Complex Gen(int i)=>new Complex((i%7)-3, (i%5)-2);
Complex Gp(int i)=>new Complex((i%4)+1, (i%3)-1);
NDArray ArrGen(int n){ var a=new Complex[n]; for(int i=0;i<n;i++) a[i]=Gen(i); return np.array(a); }
NDArray ArrGp(int n){ var a=new Complex[n]; for(int i=0;i<n;i++) a[i]=Gp(i); return np.array(a); }
Complex Val(NDArray nd)=>(Complex)nd.GetAtIndex(0);
bool Close(Complex a,Complex b)=>Math.Abs(a.Real-b.Real)<=1e-6*(1+Math.Abs(b.Real)) && Math.Abs(a.Imaginary-b.Imaginary)<=1e-6*(1+Math.Abs(b.Imaginary));

int pass=0, fail=0;
void Check(string k, Complex got){ var exp=Ref(k); bool ok=Close(got,exp); if(ok)pass++; else fail++; Console.WriteLine($"  [{(ok?"PASS":"FAIL")}] {k,-14} got={got}  exp={exp}"); }

Console.WriteLine("Complex sum/mean:");
foreach(int n in new[]{1,2,1000,100000}){ var a=ArrGen(n); Check($"sum_{n}",Val(np.sum(a))); Check($"mean_{n}",Val(np.mean(a))); }

var m=ArrGen(10000).reshape(100,100);
Check("sum_T",Val(np.sum(m.T)));
Check("mean_T",Val(np.mean(m.T)));

var flat=ArrGen(10000);
Check("sum_slice",Val(np.sum(flat["::3"])));
Check("mean_slice",Val(np.mean(flat["::3"])));

var b=np.broadcast_to(ArrGen(8).reshape(1,8), new Shape(1000,8));
Check("sum_bcast",Val(np.sum(b)));
Check("mean_bcast",Val(np.mean(b)));

Console.WriteLine("Complex prod:");
Check("prod_10",Val(np.prod(ArrGp(10))));
Check("prod_T",Val(np.prod(ArrGp(16).reshape(4,4).T)));
Check("prod_slice",Val(np.prod(ArrGp(20)["::2"])));

Console.WriteLine($"\n{pass} passed, {fail} failed.");
