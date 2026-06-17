#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Numerics;
using NumSharp;

static double Time(Action a,int it){for(int i=0;i<3;i++)a();GC.Collect();GC.WaitForPendingFinalizers();GC.Collect();var sw=System.Diagnostics.Stopwatch.StartNew();for(int i=0;i<it;i++)a();sw.Stop();return sw.Elapsed.TotalMilliseconds/it;}

// The proposed UNIVERSAL reduce kernel: ONE body, generic-math constrained.
// Covers outStride==0 (reduce-axis-inner) AND outStride!=0 (slab-fold) — the two
// REUSE_REDUCE_LOOPS modes — in a single method. Works for ANY T with + identity.
static unsafe void ReduceInner<T>(byte* inp, long inS, byte* outp, long outS, long count)
    where T : unmanaged, IAdditionOperators<T,T,T>
{
    if (outS == 0) { T acc = *(T*)outp; for (long i=0;i<count;i++) acc += *(T*)(inp+i*inS); *(T*)outp = acc; }
    else           { for (long i=0;i<count;i++){ T* o=(T*)(outp+i*outS); *o += *(T*)(inp+i*inS); } }
}

unsafe void Run(int N){
    int rows=(int)Math.Sqrt(N), cols=N/rows; long total=(long)rows*cols;
    var a=(np.arange(total).astype(NPTypeCode.Double).reshape(rows,cols).astype(NPTypeCode.Complex))+new Complex(1,1);
    int it=N>=1_000_000?20:200;
    Complex* p=(Complex*)a.Address+a.Shape.offset;
    var outBuf=new Complex[cols];

    double live=Time(()=>{using var _=np.sum(a,axis:0);},it);
    // axis=0 via the universal kernel: outStride!=0 (slab-fold), one call per reduce row
    double proto=Time(()=>{
        fixed(Complex* ob=outBuf){
            for(int c=0;c<cols;c++) ob[c]=Complex.Zero;
            for(int r=0;r<rows;r++)
                ReduceInner<Complex>((byte*)(p+(long)r*cols), sizeof(Complex), (byte*)ob, sizeof(Complex), cols);
        }
    },it);
    Console.WriteLine($"  sum axis=0 {rows}x{cols}: live {live,8:F4} ms | universal-kernel {proto,7:F4} ms | {live/proto,5:F1}x");
}

Console.WriteLine("=== np.cumsum(complex) — proof the generic-math kernel path is ALREADY live for Complex ===");
var c = np.arange(6).astype(NPTypeCode.Complex) + new Complex(0,1);
Console.WriteLine($"  cumsum([0+1i..5+1i]) = {np.cumsum(c)}");

Console.WriteLine("\n=== Universal generic-math reduce kernel (Complex) vs live np.sum ===");
Console.WriteLine("    NumPy refs: sum0 100K=0.015ms 10M=7.25ms");
Run(100_000);
Run(10_000_000);
