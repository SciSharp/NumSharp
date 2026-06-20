#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

const int N = 4_000_000;

// c128->i32: deinterleave reals (UnpackLow+Permute4x64) + cvttpd2dq
unsafe static void C2I32(Complex* s, int* d){
    int i=0,n=4_000_000; double* p=(double*)s;
    for(;i+4<=n;i+=4){
        var a=Avx.LoadVector256(p+2*i);    // re0 im0 re1 im1
        var b=Avx.LoadVector256(p+2*i+4);  // re2 im2 re3 im3
        var reals=Avx2.Permute4x64(Avx.UnpackLow(a,b),0xD8); // re0 re1 re2 re3
        Sse2.Store(d+i, Avx.ConvertToVector128Int32WithTruncation(reals));
    }
    for(;i<n;i++) d[i]=Converts.ToInt32(s[i]);
}
// c128->i8: deinterleave+cvtt -> 8x Vector128<int> -> combine to 4x V256 -> mask+pack+vpermd
unsafe static void C2I8(Complex* s, sbyte* d){
    int i=0,n=4_000_000; double* p=(double*)s; var mask=Vector256.Create(0xFF); var perm=Vector256.Create(0,4,1,5,2,6,3,7);
    for(;i+32<=n;i+=32){
        Vector128<int> R(int k){
            var a=Avx.LoadVector256(p+2*(i+k));
            var b=Avx.LoadVector256(p+2*(i+k)+4);
            return Avx.ConvertToVector128Int32WithTruncation(Avx2.Permute4x64(Avx.UnpackLow(a,b),0xD8));
        }
        var i0=Avx2.And(Vector256.Create(R(0),R(4)),mask);
        var i1=Avx2.And(Vector256.Create(R(8),R(12)),mask);
        var i2=Avx2.And(Vector256.Create(R(16),R(20)),mask);
        var i3=Avx2.And(Vector256.Create(R(24),R(28)),mask);
        var w0=Avx2.PackUnsignedSaturate(i0,i1); var w1=Avx2.PackUnsignedSaturate(i2,i3);
        var bb=Avx2.PackUnsignedSaturate(w0.AsInt16(),w1.AsInt16());
        Avx.Store((byte*)(d+i), Avx2.PermuteVar8x32(bb.AsInt32(),perm).AsByte());
    }
    for(;i<n;i++) d[i]=Converts.ToSByte(s[i]);
}

var src=new Complex[N]; var rnd=new Random(11);
(double,double)[] sp={(3,4),(3e9,1),(128.5,-9),(-300,0),(double.NaN,2),(double.PositiveInfinity,-1),(2147483653.0,0),(-128.6,7)};
for(int i=0;i<N;i++){int r=rnd.Next(100); if(r<10){var t=sp[rnd.Next(sp.Length)]; src[i]=new Complex(t.Item1,t.Item2);} else src[i]=new Complex(rnd.NextDouble()*120000-60000, rnd.NextDouble()*10);}
var ri=new int[N]; var bi=new int[N]; var rb=new sbyte[N]; var bb=new sbyte[N];

unsafe {
    fixed(Complex* s=src) fixed(int* pri=ri,pbi=bi) fixed(sbyte* prb=rb,pbb=bb){
        for(int i=0;i<N;i++){ pri[i]=Converts.ToInt32(s[i]); prb[i]=Converts.ToSByte(s[i]); }
        C2I32(s,pbi); C2I8(s,pbb);
        long e32=0,e8=0; for(int i=0;i<N;i++){ if(pbi[i]!=pri[i])e32++; if(pbb[i]!=prb[i])e8++; }
        Console.WriteLine($"correctness diffs: c128->i32={e32}  c128->i8={e8}");
        static double B32(delegate*<Complex*,int*,void> f,Complex* s,int* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        static double B8(delegate*<Complex*,sbyte*,void> f,Complex* s,sbyte* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        // scalar baselines
        static void S32(Complex* s,int* d){for(int i=0;i<4_000_000;i++)d[i]=Converts.ToInt32(s[i]);}
        static void S8(Complex* s,sbyte* d){for(int i=0;i<4_000_000;i++)d[i]=Converts.ToSByte(s[i]);}
        double ts32=B32(&S32,s,pri), ts8=B8(&S8,s,prb);
        double t32=B32(&C2I32,s,pbi), t8=B8(&C2I8,s,pbb);
        Console.WriteLine($"c128->i32  scalar {ts32:F3}  SIMD {t32:F3}  ({ts32/t32:F1}x)  NumPy 5.030  NPY/NS {5.030/t32:F2}");
        Console.WriteLine($"c128->i8   scalar {ts8:F3}  SIMD {t8:F3}  ({ts8/t8:F1}x)  NumPy 3.459  NPY/NS {3.459/t8:F2}");
    }
}
