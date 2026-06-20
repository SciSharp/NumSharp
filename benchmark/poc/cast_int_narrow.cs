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
// i32->i8: mask0xFF + 2x packus + vpermd  (no cvtt)
unsafe static void I32_I8(int* s, sbyte* d){
    int i=0,n=4_000_000; var mask=Vector256.Create(0xFF); var perm=Vector256.Create(0,4,1,5,2,6,3,7);
    for(;i+32<=n;i+=32){
        var i0=Avx2.And(Avx.LoadVector256(s+i+0),mask); var i1=Avx2.And(Avx.LoadVector256(s+i+8),mask);
        var i2=Avx2.And(Avx.LoadVector256(s+i+16),mask); var i3=Avx2.And(Avx.LoadVector256(s+i+24),mask);
        var w0=Avx2.PackUnsignedSaturate(i0,i1); var w1=Avx2.PackUnsignedSaturate(i2,i3);
        var b=Avx2.PackUnsignedSaturate(w0.AsInt16(),w1.AsInt16());
        Avx.Store((byte*)(d+i), Avx2.PermuteVar8x32(b.AsInt32(),perm).AsByte());
    }
    for(;i<n;i++) d[i]=Converts.ToSByte(s[i]);
}
// i32->i16: mask0xFFFF + 1x packus + vpermq 0xD8
unsafe static void I32_I16(int* s, short* d){
    int i=0,n=4_000_000; var mask=Vector256.Create(0xFFFF);
    for(;i+16<=n;i+=16){
        var i0=Avx2.And(Avx.LoadVector256(s+i+0),mask); var i1=Avx2.And(Avx.LoadVector256(s+i+8),mask);
        Avx.Store((ushort*)(d+i), Avx2.Permute4x64(Avx2.PackUnsignedSaturate(i0,i1).AsUInt64(),0xD8).AsUInt16());
    }
    for(;i<n;i++) d[i]=Converts.ToInt16(s[i]);
}
// i64->i32: extract low dword of each long via vpermd [0,2,4,6]
unsafe static void I64_I32(long* s, int* d){
    int i=0,n=4_000_000; var pick=Vector256.Create(0,2,4,6,0,2,4,6);
    for(;i+8<=n;i+=8){
        var a=Avx2.PermuteVar8x32(Avx.LoadVector256(s+i+0).AsInt32(),pick).GetLower(); // 4 i32
        var b=Avx2.PermuteVar8x32(Avx.LoadVector256(s+i+4).AsInt32(),pick).GetLower();
        Sse2.Store(d+i+0,a); Sse2.Store(d+i+4,b);
    }
    for(;i<n;i++) d[i]=Converts.ToInt32(s[i]);
}
// i64->i16: low word of each long. mask low16 then 2-step pack via vpermd gather of words.
unsafe static void I64_I16(long* s, short* d){
    int i=0,n=4_000_000; var pick=Vector256.Create(0,2,4,6,0,2,4,6);
    for(;i+8<=n;i+=8){
        // i64->i32 (low dword), then i32->i16 low word
        var a=Avx2.PermuteVar8x32(Avx.LoadVector256(s+i+0).AsInt32(),pick).GetLower(); // 4 i32
        var b=Avx2.PermuteVar8x32(Avx.LoadVector256(s+i+4).AsInt32(),pick).GetLower();
        var lo=Sse2.And(a,Vector128.Create(0xFFFF)); var hi=Sse2.And(b,Vector128.Create(0xFFFF));
        Sse2.Store((ushort*)(d+i), Sse41.PackUnsignedSaturate(lo,hi)); // 8 u16
    }
    for(;i<n;i++) d[i]=Converts.ToInt16(s[i]);
}

var rnd=new Random(11);
var s32=new int[N]; var s64=new long[N];
int[] sp32={300,-1,128,256,-129,70000,int.MaxValue,int.MinValue};
long[] sp64={300,-1,256,128,65535,-129,(1L<<40)+5,long.MinValue};
for(int i=0;i<N;i++){ s32[i]= rnd.Next(100)<10? sp32[rnd.Next(sp32.Length)] : rnd.Next(-200000,200000);
                      s64[i]= rnd.Next(100)<10? sp64[rnd.Next(sp64.Length)] : ((long)rnd.Next()<<20)^rnd.Next(); }

var r1=new sbyte[N]; var b1=new sbyte[N]; var r2=new short[N]; var b2=new short[N];
var r3=new int[N]; var b3=new int[N]; var r4=new short[N]; var b4=new short[N];
unsafe {
    fixed(int* p32=s32) fixed(long* p64=s64)
    fixed(sbyte* pr1=r1,pb1=b1) fixed(short* pr2=r2,pb2=b2) fixed(int* pr3=r3,pb3=b3) fixed(short* pr4=r4,pb4=b4){
        for(int i=0;i<N;i++){ pr1[i]=Converts.ToSByte(p32[i]); pr2[i]=Converts.ToInt16(p32[i]); pr3[i]=Converts.ToInt32(p64[i]); pr4[i]=Converts.ToInt16(p64[i]); }
        I32_I8(p32,pb1); I32_I16(p32,pb2); I64_I32(p64,pb3); I64_I16(p64,pb4);
        long e1=0,e2=0,e3=0,e4=0; for(int i=0;i<N;i++){ if(pb1[i]!=pr1[i])e1++; if(pb2[i]!=pr2[i])e2++; if(pb3[i]!=pr3[i])e3++; if(pb4[i]!=pr4[i])e4++; }
        Console.WriteLine($"correctness diffs: i32->i8={e1} i32->i16={e2} i64->i32={e3} i64->i16={e4}");
        static double Ba(delegate*<int*,sbyte*,void> f,int* s,sbyte* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        static double Bb(delegate*<int*,short*,void> f,int* s,short* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        static double Bc(delegate*<long*,int*,void> f,long* s,int* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        static double Bd(delegate*<long*,short*,void> f,long* s,short* d){double x=1e9;for(int r=0;r<7;r++){var sw=Stopwatch.StartNew();f(s,d);sw.Stop();x=Math.Min(x,sw.Elapsed.TotalMilliseconds);}return x;}
        double t1=Ba(&I32_I8,p32,pb1), t2=Bb(&I32_I16,p32,pb2), t3=Bc(&I64_I32,p64,pb3), t4=Bd(&I64_I16,p64,pb4);
        Console.WriteLine($"i32->i8   NS {t1:F3}  NumPy 1.240  NPY/NS {1.240/t1:F2}");
        Console.WriteLine($"i32->i16  NS {t2:F3}  NumPy 1.607  NPY/NS {1.607/t2:F2}");
        Console.WriteLine($"i64->i32  NS {t3:F3}  NumPy 3.465  NPY/NS {3.465/t3:F2}");
        Console.WriteLine($"i64->i16  NS {t4:F3}  NumPy 2.363  NPY/NS {2.363/t4:F2}");
    }
}
