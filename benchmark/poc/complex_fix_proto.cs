#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Numerics;
using NumSharp;

static double Time(Action a, int it) {
    for (int i=0;i<3;i++) a();
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    var sw=System.Diagnostics.Stopwatch.StartNew();
    for (int i=0;i<it;i++) a();
    sw.Stop(); return sw.Elapsed.TotalMilliseconds/it;
}

unsafe void Run(int N) {
    int rows=(int)Math.Sqrt(N), cols=N/rows; long total=(long)rows*cols;
    var a=(np.arange(total).astype(NPTypeCode.Double).reshape(rows,cols).astype(NPTypeCode.Complex))+new Complex(1,1);
    int it = N>=1_000_000?20:200;
    Complex* p=(Complex*)a.Address + a.Shape.offset;
    var outBuf=new Complex[Math.Max(rows,cols)];

    // ---- live ----
    double liveSum0=Time(()=>{using var _=np.sum(a,axis:0);},it);
    double liveSum1=Time(()=>{using var _=np.sum(a,axis:1);},it);
    double liveMean0=Time(()=>{using var _=np.mean(a,axis:0);},it);

    // ---- PROPOSED axis=0: leading-axis slab-stream (what AxisReductionSlabAccumulate does) ----
    double protoSum0=Time(()=>{
        for(int c=0;c<cols;c++) outBuf[c]=Complex.Zero;
        for(int r=0;r<rows;r++){ Complex* row=p+(long)r*cols; for(int c=0;c<cols;c++) outBuf[c]+=row[c]; }
    },it);
    double protoMean0=Time(()=>{
        for(int c=0;c<cols;c++) outBuf[c]=Complex.Zero;
        for(int r=0;r<rows;r++){ Complex* row=p+(long)r*cols; for(int c=0;c<cols;c++) outBuf[c]+=row[c]; }
        for(int c=0;c<cols;c++) outBuf[c]/=rows;
    },it);
    // ---- PROPOSED axis=1: innermost contig reduce (what DispatchInnermost does) ----
    double protoSum1=Time(()=>{
        for(int r=0;r<rows;r++){ Complex* row=p+(long)r*cols; Complex acc=Complex.Zero; for(int c=0;c<cols;c++) acc+=row[c]; outBuf[r]=acc; }
    },it);

    Console.WriteLine($"N={N,-9} ({rows}x{cols})");
    Console.WriteLine($"  sum  axis=0: live {liveSum0,8:F4} ms | proto(slab-stream) {protoSum0,7:F4} ms | {liveSum0/protoSum0,5:F1}x");
    Console.WriteLine($"  sum  axis=1: live {liveSum1,8:F4} ms | proto(innermost)   {protoSum1,7:F4} ms | {liveSum1/protoSum1,5:F1}x  (axis0/axis1 live ratio = {liveSum0/liveSum1:F1}x)");
    Console.WriteLine($"  mean axis=0: live {liveMean0,8:F4} ms | proto(slab+div)    {protoMean0,7:F4} ms | {liveMean0/protoMean0,5:F1}x");
}
Console.WriteLine("=== Proposed typed-Complex reducer (slab-stream, no boxing, no slices) vs live ===");
Console.WriteLine("    NumPy refs: sum0 100K=0.015ms 10M=7.25ms ; mean0 100K=0.017ms 10M=7.63ms");
Run(100_000);
Run(10_000_000);
