#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;

static double Best(Action f,int reps=30){ double b=1e9; for(int k=0;k<reps;k++){ var sw=Stopwatch.StartNew(); f(); sw.Stop(); b=Math.Min(b,sw.Elapsed.TotalMilliseconds);} return b; }

foreach(int N in new[]{1_000_000,4_000_000}){
    var bc=np.broadcast_to(np.array(new byte[]{7}), new Shape(N));
    double cur=Best(()=>{ var r=bc.astype(NPTypeCode.Byte); });
    // direct broadcast-aware fill: scalar value -> InitBlock (memset)
    double fill=Best(()=>{
        unsafe {
            var ptr=(byte*)SizeBucketedBufferPool.Take(N);
            Unsafe.InitBlockUnaligned(ptr, 7, (uint)N); // memset
            // (a real impl wraps ptr in a Storage/NDArray; measure fill cost itself)
            SizeBucketedBufferPool.Return((nint)ptr, N);
        }
    });
    double npy = N==1_000_000 ? 0.012 : 0.716;
    Console.WriteLine($"bcast u8->u8 N={N}:  current(NpyIter.Copy) {cur:F4} ms   direct-fill(InitBlock) {fill:F4} ms   NumPy {npy:F3}");
    Console.WriteLine($"    speedup direct-vs-current: {cur/fill:F1}x   NPY/NS current {npy/cur:F2}  ->  NPY/NS direct {npy/fill:F2}");
}
