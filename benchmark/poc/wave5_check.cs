#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// wave5_check.cs — All/Any (Half/Complex) on contiguous AND strided (transposed)
// layouts must agree and match NumPy; bool max==any, min==all.
//
using System;
using System.Numerics;
using NumSharp;

int pass=0, fail=0;
void Ck(string k, bool got, bool exp){ bool ok=got==exp; if(ok)pass++; else fail++; Console.WriteLine($"  [{(ok?"PASS":"FAIL")}] {k,-26} got={got} exp={exp}"); }

NDArray H(double[] v)=>np.array(v).astype(NPTypeCode.Half);
double nan=double.NaN;

// (label, values, expect any, expect all) — Half
var halfCases = new (string,double[],bool,bool)[]{
    ("half[1,2,3]",   new double[]{1,2,3},   true,  true),
    ("half[1,0,3]",   new double[]{1,0,3},   true,  false),
    ("half[0,0,0]",   new double[]{0,0,0},   false, false),
    ("half[0,nan,0]", new double[]{0,nan,0}, true,  false),
    ("half[1,nan,2]", new double[]{1,nan,2}, true,  true),
};
Console.WriteLine("Half any/all (contig + strided):");
foreach(var (lbl,v,ea,el) in halfCases){
    var a=H(v);
    Ck($"{lbl} any",        np.any(a), ea);
    Ck($"{lbl} all",        np.all(a), el);
    // strided: tile into 2 rows then transpose → non-contiguous, same multiset
    var big=new double[v.Length*2]; for(int i=0;i<v.Length;i++){ big[i]=v[i]; big[v.Length+i]=v[i]; }
    var s=H(big).reshape(2,v.Length).T;   // shape (len,2), F-contiguous view
    Ck($"{lbl} any/strided", np.any(s), ea);
    Ck($"{lbl} all/strided", np.all(s), el);
}

Complex C(double r,double i)=>new Complex(r,i);
var cplxCases = new (string,Complex[],bool,bool)[]{
    ("cplx[(1,0),(0,2)]",     new[]{C(1,0),C(0,2)},     true,  true),
    ("cplx[(0,0),(1,1)]",     new[]{C(0,0),C(1,1)},     true,  false),
    ("cplx[(0,0),(0,0)]",     new[]{C(0,0),C(0,0)},     false, false),
    ("cplx[(nan,0),(1,1)]",   new[]{C(nan,0),C(1,1)},   true,  true),
};
Console.WriteLine("Complex any/all (contig + strided):");
foreach(var (lbl,v,ea,el) in cplxCases){
    var a=np.array(v);
    Ck($"{lbl} any", np.any(a), ea);
    Ck($"{lbl} all", np.all(a), el);
    var big=new Complex[v.Length*2]; for(int i=0;i<v.Length;i++){ big[i]=v[i]; big[v.Length+i]=v[i]; }
    var s=np.array(big).reshape(2,v.Length).T;
    Ck($"{lbl} any/strided", np.any(s), ea);
    Ck($"{lbl} all/strided", np.all(s), el);
}

Console.WriteLine("bool max(=any)/min(=all):");
bool BV(NDArray nd)=>(bool)nd.GetAtIndex(0);
var b1=np.array(new[]{true,false,true});
Ck("bool[T,F,T] max", BV(np.max(b1)), true);  Ck("bool[T,F,T] min", BV(np.min(b1)), false);
var b2=np.array(new[]{true,true,true});
Ck("bool[T,T,T] max", BV(np.max(b2)), true);  Ck("bool[T,T,T] min", BV(np.min(b2)), true);
var b3=np.array(new[]{false,false});
Ck("bool[F,F] max",   BV(np.max(b3)), false); Ck("bool[F,F] min",   BV(np.min(b3)), false);
// strided bool
var b4=np.array(new[]{true,false,true,true,false,true}).reshape(2,3).T;
Ck("bool strided max", BV(np.max(b4)), true); Ck("bool strided min", BV(np.min(b4)), false);

Console.WriteLine($"\n{pass} passed, {fail} failed.");
