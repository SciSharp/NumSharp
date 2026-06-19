#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Diagnostics;
using NumSharp;

// amin/amax along axis at 1000x1000 — bool/char/half (the catastrophe dtypes) + double baseline.
const int N = 1000;
static double Best(Func<NDArray> f, int iters = 12)
{
    f(); f(); // warmup + JIT the kernel
    double best = double.MaxValue;
    for (int i = 0; i < iters; i++)
    {
        var sw = Stopwatch.StartNew();
        var r = f();
        sw.Stop();
        GC.KeepAlive(r);
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
    }
    return best;
}

NDArray MakeBool() { var a = new bool[N*N]; for (int i=0;i<a.Length;i++) a[i]=(i*2654435761u & 1)==0; return np.array(a).reshape(N,N); }
NDArray MakeChar() { var a = new char[N*N]; for (int i=0;i<a.Length;i++) a[i]=(char)((i*40503)&0xFFFF); return np.array(a).reshape(N,N); }
NDArray MakeHalf() { var a = new double[N*N]; for (int i=0;i<a.Length;i++) a[i]=((i*7919)%1000)-500; return np.array(a).astype(NPTypeCode.Half).reshape(N,N); }
NDArray MakeDbl()  { var a = new double[N*N]; for (int i=0;i<a.Length;i++) a[i]=((i*7919)%1000)-500; return np.array(a).reshape(N,N); }

var b = MakeBool(); var c = MakeChar(); var h = MakeHalf(); var d = MakeDbl();
Console.WriteLine($"{"dtype",-8}{"amin ax0",12}{"amax ax0",12}{"amin ax1",12}{"amax ax1",12}");
foreach (var (name, arr) in new (string,NDArray)[] { ("bool",b),("char",c),("half",h),("double",d) })
{
    double a0 = Best(() => np.amin(arr, 0));
    double x0 = Best(() => np.amax(arr, 0));
    double a1 = Best(() => np.amin(arr, 1));
    double x1 = Best(() => np.amax(arr, 1));
    Console.WriteLine($"{name,-8}{a0,12:F4}{x0,12:F4}{a1,12:F4}{x1,12:F4}");
}

// cache-bust 1781894506
