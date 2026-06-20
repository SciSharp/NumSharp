#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;
using Half = System.Half;

const int N = 4_000_000;

// Giesen branchless half->float (exact for finite; inf/nan via cmpgt) over Vector256
static Vector256<float> H2F(Vector256<int> h){
    var maskNoSign=Vector256.Create(0x7fff);
    var magic=Vector256.Create((254-15)<<23).AsSingle();
    var wasInfNan=Vector256.Create(0x7bff);
    var expInfNan=Vector256.Create(255<<23).AsSingle();
    var expmant=Avx2.And(maskNoSign,h);
    var scaled=Avx.Multiply(Avx2.ShiftLeftLogical(expmant,13).AsSingle(),magic);
    var infnan=Avx2.CompareGreaterThan(expmant,wasInfNan);
    var sign=Avx2.ShiftLeftLogical(Avx2.AndNot(maskNoSign,h),16);
    var signInf=Avx.Or(sign.AsSingle(),Avx.And(infnan.AsSingle(),expInfNan));
    return Avx.Or(scaled,signInf);
}

// V1 scalar: current Converts
unsafe static void V1_I32(Half* s, int* d){ for(int i=0;i<4_000_000;i++) d[i]=Converts.ToInt32(s[i]); }
// V2 scalar (float)half + (int) cvtt via Converts.ToInt32(float)
unsafe static void V2_I32(Half* s, int* d){ for(int i=0;i<4_000_000;i++) d[i]=Converts.ToInt32((float)s[i]); }
// V3 SIMD: vpmovzxwd + Giesen + cvttps2dq
unsafe static void V3_I32(Half* s, int* d){
    int i=0,n=4_000_000; ushort* p=(ushort*)s;
    for(;i+8<=n;i+=8){
        var h=Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p+i)); // zero-extend 8 u16->8 i32
        Avx.Store(d+i, Avx.ConvertToVector256Int32WithTruncation(H2F(h)));
    }
    for(;i<n;i++) d[i]=Converts.ToInt32((float)s[i]);
}
// f16->f32 SIMD
unsafe static void V3_F32(Half* s, float* d){
    int i=0,n=4_000_000; ushort* p=(ushort*)s;
    for(;i+8<=n;i+=8){ var h=Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p+i)); Avx.Store(d+i, H2F(h)); }
    for(;i<n;i++) d[i]=(float)s[i];
}
// f16->i8 SIMD: Giesen+cvtt -> 4x V256<int> -> mask+pack+vpermd
unsafe static void V3_I8(Half* s, sbyte* d){
    int i=0,n=4_000_000; ushort* p=(ushort*)s; var mask=Vector256.Create(0xFF); var perm=Vector256.Create(0,4,1,5,2,6,3,7);
    Vector256<int> C(int k){ return Avx.ConvertToVector256Int32WithTruncation(H2F(Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p+k)))); }
    for(;i+32<=n;i+=32){
        var i0=Avx2.And(C(i+0),mask); var i1=Avx2.And(C(i+8),mask); var i2=Avx2.And(C(i+16),mask); var i3=Avx2.And(C(i+24),mask);
        var w0=Avx2.PackUnsignedSaturate(i0,i1); var w1=Avx2.PackUnsignedSaturate(i2,i3);
        var b=Avx2.PackUnsignedSaturate(w0.AsInt16(),w1.AsInt16());
        Avx.Store((byte*)(d+i), Avx2.PermuteVar8x32(b.AsInt32(),perm).AsByte());
    }
    for(;i<n;i++) d[i]=Converts.ToSByte((float)s[i]);
}

var rnd=new Random(7); var src=new Half[N];
float[] sp={0f,-0f,1.5f,300f,128.5f,-300f,65504f,float.NaN,float.PositiveInfinity,float.NegativeInfinity,6e-5f,-6e-8f};
for(int i=0;i<N;i++){ src[i]= rnd.Next(100)<15? (Half)sp[rnd.Next(sp.Length)] : (Half)(float)(rnd.NextDouble()*600-300); }
var ri=new int[N]; var b1=new int[N]; var b2=new int[N]; var b3=new int[N];
var rf=new float[N]; var bf=new float[N]; var r8=new sbyte[N]; var b8=new sbyte[N];
unsafe {
    fixed(Half* s=src) fixed(int* pri=ri,pb1=b1,pb2=b2,pb3=b3) fixed(float* prf=rf,pbf=bf) fixed(sbyte* pr8=r8,pb8=b8){
        for(int i=0;i<N;i++){ pri[i]=Converts.ToInt32((float)s[i]); prf[i]=(float)s[i]; pr8[i]=Converts.ToSByte((float)s[i]); }
        V1_I32(s,pb1); V2_I32(s,pb2); V3_I32(s,pb3); V3_F32(s,pbf); V3_I8(s,pb8);
        long e3=0,ef=0,e8=0,efNanOnly=0; for(int i=0;i<N;i++){ if(pb3[i]!=pri[i])e3++; if(pbf[i]!=prf[i]){ if(float.IsNaN(pbf[i])&&float.IsNaN(prf[i])) efNanOnly++; else ef++; } if(pb8[i]!=pr8[i])e8++; }
        Console.WriteLine($"correctness diffs: f16->i32={e3}  f16->f32={ef} (nan-payload-only={efNanOnly})  f16->i8={e8}");
        static double Bi(delegate*<Half*,int*,void> f,Half* s,int* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        static double Bf(delegate*<Half*,float*,void> f,Half* s,float* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        static double B8(delegate*<Half*,sbyte*,void> f,Half* s,sbyte* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        double t1=Bi(&V1_I32,s,pb1), t2=Bi(&V2_I32,s,pb2), t3=Bi(&V3_I32,s,pb3), tf=Bf(&V3_F32,s,pbf), t8=B8(&V3_I8,s,pb8);
        Console.WriteLine($"f16->i32  V1 scalar {t1:F3}  V2 (float)h {t2:F3}  V3 Giesen {t3:F3}   NumPy 5.100  NPY/NS(V3) {5.100/t3:F2}");
        Console.WriteLine($"f16->f32  V3 Giesen {tf:F3}   NumPy 4.119  NPY/NS {4.119/tf:F2}");
        Console.WriteLine($"f16->i8   V3 Giesen {t8:F3}   NumPy 4.037  NPY/NS {4.037/t8:F2}");
    }
}
