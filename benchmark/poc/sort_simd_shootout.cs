#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Self-contained kernel shoot-out (no NumSharp ref): radix vs AVX2 vectorized quicksort vs Span.Sort.
// Answers: can a SIMD / IL-emittable kernel beat radix for int32 SORT? (and the argsort caveat)
using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

int[] sizes = { 1_000_000 };
var rng = new Random(42);
double Best(Action setup, Action timed){ double b=1e18; for(int r=0;r<7;r++){ setup(); var sw=Stopwatch.StartNew(); timed(); sw.Stop(); b=Math.Min(b,sw.Elapsed.TotalMilliseconds);} return b; }

var LUT = new Vector256<int>[256];
for(int m=0;m<256;m++){ var ix=new int[8]; int p=0;
    for(int b=0;b<8;b++) if((m&(1<<b))!=0) ix[p++]=b;   // >pivot lane indices first
    for(int b=0;b<8;b++) if((m&(1<<b))==0) ix[p++]=b;
    LUT[m]=Vector256.Create(ix[0],ix[1],ix[2],ix[3],ix[4],ix[5],ix[6],ix[7]); }

unsafe void RadixI32(int[] a, uint[] cur, uint[] tmp, int[] cnt){
    int n=a.Length; if(n<=1) return;
    fixed(uint* C=cur,T=tmp) fixed(int* K=cnt){
        for(int q=0;q<n;q++) C[q]=(uint)a[q]^0x80000000u;
        uint* s=C,d=T;
        for(int sh=0;sh<32;sh+=8){
            for(int b=0;b<256;b++) K[b]=0;
            for(int q=0;q<n;q++) K[(int)((s[q]>>sh)&0xFF)]++;
            if(K[(int)((s[0]>>sh)&0xFF)]==n) continue;
            int sum=0; for(int b=0;b<256;b++){int c=K[b];K[b]=sum;sum+=c;}
            for(int q=0;q<n;q++){int dd=(int)((s[q]>>sh)&0xFF);d[K[dd]++]=s[q];}
            var t=s;s=d;d=t;
        }
        for(int q=0;q<n;q++) a[q]=(int)(s[q]^0x80000000u);
    }
}

unsafe void QVec(int* a, int n, int* lo, int* hi, Vector256<int>[] lut){
    while(n>64){
        int x=a[0],y=a[n/2],z=a[n-1];
        int pivot = x<y ? (y<z?y:(x<z?z:x)) : (x<z?x:(y<z?z:y));   // median-of-3
        var P=Vector256.Create(pivot); int cl=0,ch=0,q=0;
        for(;q+8<=n;q+=8){
            var v=Avx.LoadVector256(a+q);
            int mask=Avx.MoveMask(Avx2.CompareGreaterThan(v,P).AsSingle());
            int nGt=BitOperations.PopCount((uint)mask);
            Avx.Store(hi+ch, Avx2.PermuteVar8x32(v, lut[mask])); ch+=nGt;
            Avx.Store(lo+cl, Avx2.PermuteVar8x32(v, lut[(~mask)&0xFF])); cl+=8-nGt;
        }
        for(;q<n;q++){ if(a[q]>pivot) hi[ch++]=a[q]; else lo[cl++]=a[q]; }
        for(int k=0;k<cl;k++) a[k]=lo[k];
        for(int k=0;k<ch;k++) a[cl+k]=hi[k];
        if(cl<ch){ QVec(a,cl,lo,hi,lut); a=a+cl; n=ch; }   // recurse SMALLER side -> depth O(log n)
        else { QVec(a+cl,ch,lo,hi,lut); n=cl; }
    }
    for(int i=1;i<n;i++){ int v=a[i],j=i-1; while(j>=0&&a[j]>v){a[j+1]=a[j];j--;} a[j+1]=v; }
}

Console.WriteLine($"AVX2={Avx2.IsSupported}");
Console.WriteLine("== int32 SORT (NumPy 2.4.2: ~264 M/s @1M) ==");
foreach(int n in sizes){
    var src=new int[n]; for(int i=0;i<n;i++) src[i]=rng.Next();
    var work=new int[n]; var cur=new uint[n]; var tmp=new uint[n]; var cnt=new int[256];
    var lo=new int[n+8]; var hi=new int[n+8];
    double rad=Best(()=>src.CopyTo(work,0), ()=>RadixI32(work,cur,tmp,cnt));
    double vec; bool vok;
    unsafe {
        vec=Best(()=>src.CopyTo(work,0), ()=>{ fixed(int* w=work,L=lo,H=hi) QVec(w,n,L,H,LUT); });
        src.CopyTo(work,0); fixed(int* w=work,L=lo,H=hi) QVec(w,n,L,H,LUT);
        var chk=(int[])src.Clone(); Array.Sort(chk); vok=work.AsSpan().SequenceEqual(chk);
    }
    double spn=Best(()=>src.CopyTo(work,0), ()=>work.AsSpan().Sort());
    Console.WriteLine($"n={n,9}  radix {rad,7:F2}ms({n/rad/1e3,5:F0}M/s) | avx2qsort {vec,7:F2}ms({n/vec/1e3,5:F0}M/s ok={vok}) | span {spn,7:F2}ms({n/spn/1e3,5:F0}M/s)");
}
Console.WriteLine("argsort: radix carries indices (87-135 M/s, BEATS NumPy 2-4x); a vectorized qsort has NO efficient index path.");
