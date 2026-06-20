#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

const int N = 4_000_000;
unsafe static void F32Bool(float* s, bool* d){
    int i=0,n=4_000_000; var z=Vector256<float>.Zero; var one=Vector256.Create(1); var perm=Vector256.Create(0,4,1,5,2,6,3,7);
    Vector256<int> M(int k){ var eq=Avx.Compare(Avx.LoadVector256(s+k),z,FloatComparisonMode.OrderedEqualNonSignaling); return Avx2.AndNot(eq.AsInt32(),one); }
    for(;i+32<=n;i+=32){
        var w0=Avx2.PackUnsignedSaturate(M(i+0),M(i+8)); var w1=Avx2.PackUnsignedSaturate(M(i+16),M(i+24));
        var b=Avx2.PackUnsignedSaturate(w0.AsInt16(),w1.AsInt16());
        Avx.Store((byte*)(d+i), Avx2.PermuteVar8x32(b.AsInt32(),perm).AsByte());
    }
    for(;i<n;i++) d[i]=Converts.ToBoolean(s[i]);
}
unsafe static void F64Bool(double* s, bool* d){
    int i=0,n=4_000_000; var z=Vector256<double>.Zero; var one=Vector256.Create(1L);
    var pick=Vector256.Create(0,2,4,6,0,2,4,6); var perm=Vector256.Create(0,4,1,5,2,6,3,7);
    Vector128<int> M(int k){ var eq=Avx.Compare(Avx.LoadVector256(s+k),z,FloatComparisonMode.OrderedEqualNonSignaling); var bm=Avx2.AndNot(eq.AsInt64(),one); return Avx2.PermuteVar8x32(bm.AsInt32(),pick).GetLower(); }
    for(;i+32<=n;i+=32){
        var i0=Vector256.Create(M(i+0),M(i+4)); var i1=Vector256.Create(M(i+8),M(i+12));
        var i2=Vector256.Create(M(i+16),M(i+20)); var i3=Vector256.Create(M(i+24),M(i+28));
        var w0=Avx2.PackUnsignedSaturate(i0,i1); var w1=Avx2.PackUnsignedSaturate(i2,i3);
        var b=Avx2.PackUnsignedSaturate(w0.AsInt16(),w1.AsInt16());
        Avx.Store((byte*)(d+i), Avx2.PermuteVar8x32(b.AsInt32(),perm).AsByte());
    }
    for(;i<n;i++) d[i]=Converts.ToBoolean(s[i]);
}
unsafe static void I32Bool(int* s, bool* d){
    int i=0,n=4_000_000; var z=Vector256<int>.Zero; var one=Vector256.Create(1); var perm=Vector256.Create(0,4,1,5,2,6,3,7);
    Vector256<int> M(int k){ return Avx2.AndNot(Avx2.CompareEqual(Avx.LoadVector256(s+k),z),one); }
    for(;i+32<=n;i+=32){
        var w0=Avx2.PackUnsignedSaturate(M(i+0),M(i+8)); var w1=Avx2.PackUnsignedSaturate(M(i+16),M(i+24));
        var b=Avx2.PackUnsignedSaturate(w0.AsInt16(),w1.AsInt16());
        Avx.Store((byte*)(d+i), Avx2.PermuteVar8x32(b.AsInt32(),perm).AsByte());
    }
    for(;i<n;i++) d[i]=Converts.ToBoolean(s[i]);
}

var rnd=new Random(5);
var sf=new float[N]; var sd=new double[N]; var si=new int[N];
float[] spf={0f,-0f,1.5f,float.NaN,float.PositiveInfinity,-0f,1e-40f};
for(int i=0;i<N;i++){ sf[i]= rnd.Next(100)<20? spf[rnd.Next(spf.Length)] : (float)(rnd.NextDouble()*2-1);
                      sd[i]= rnd.Next(100)<20? (double)spf[rnd.Next(spf.Length)] : rnd.NextDouble()*2-1;
                      si[i]= rnd.Next(100)<30? 0 : rnd.Next(-5,5); }
var rf=new bool[N]; var bf=new bool[N]; var rd=new bool[N]; var bd=new bool[N]; var ri=new bool[N]; var bi=new bool[N];
unsafe {
    fixed(float* pf=sf) fixed(double* pd=sd) fixed(int* pi=si)
    fixed(bool* prf=rf,pbf=bf,prd=rd,pbd=bd,pri=ri,pbi=bi){
        for(int i=0;i<N;i++){ prf[i]=Converts.ToBoolean(pf[i]); prd[i]=Converts.ToBoolean(pd[i]); pri[i]=Converts.ToBoolean(pi[i]); }
        F32Bool(pf,pbf); F64Bool(pd,pbd); I32Bool(pi,pbi);
        long ef=0,ed=0,ei=0; for(int i=0;i<N;i++){ if(pbf[i]!=prf[i])ef++; if(pbd[i]!=prd[i])ed++; if(pbi[i]!=pri[i])ei++; }
        Console.WriteLine($"correctness diffs: f32->bool={ef} f64->bool={ed} i32->bool={ei}");
        static double Bf(delegate*<float*,bool*,void> f,float* s,bool* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        static double Bd2(delegate*<double*,bool*,void> f,double* s,bool* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        static double Bi(delegate*<int*,bool*,void> f,int* s,bool* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        double tf=Bf(&F32Bool,pf,pbf), td=Bd2(&F64Bool,pd,pbd), ti=Bi(&I32Bool,pi,pbi);
        Console.WriteLine($"f32->bool NS {tf:F3}  NumPy 1.927  NPY/NS {1.927/tf:F2}");
        Console.WriteLine($"f64->bool NS {td:F3}  NumPy 2.356  NPY/NS {2.356/td:F2}");
        Console.WriteLine($"i32->bool NS {ti:F3}  NumPy 1.135  NPY/NS {1.135/ti:F2}");
    }
}
