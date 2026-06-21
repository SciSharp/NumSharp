#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property TargetFramework=net10.0
// Isolate WHERE argsort hangs and HOW it scales (linear-slow vs N^2 blowup).
using System;
using System.Diagnostics;
using NumSharp;

void Line(string s){ Console.WriteLine(s); Console.Out.Flush(); }
var rng = new Random(1);

Line("== scaling probe ==");
foreach (int N in new[]{ 1000, 10_000, 100_000, 1_000_000 })
{
    var d = new int[N]; for(int i=0;i<N;i++) d[i]=rng.Next();
    var sw = Stopwatch.StartNew(); var a = np.array(d); sw.Stop();
    Line($"N={N,8}: np.array {sw.Elapsed.TotalMilliseconds,9:F2}ms");

    sw.Restart(); var s = a.copy(); s.sort(); sw.Stop();
    Line($"N={N,8}: ndarray.sort(value) {sw.Elapsed.TotalMilliseconds,9:F2}ms  ({N/sw.Elapsed.TotalMilliseconds/1e3,6:F1} M/s)");

    sw.Restart(); var g = np.argsort(a); sw.Stop();
    Line($"N={N,8}: np.argsort {sw.Elapsed.TotalMilliseconds,9:F2}ms  ({N/sw.Elapsed.TotalMilliseconds/1e3,6:F1} M/s)");
    // correctness spot-check
    bool ok = true; for(int i=1;i<Math.Min(N,1000);i++){ if(d[(int)g.GetInt64(i-1)] > d[(int)g.GetInt64(i)]){ ok=false; break; } }
    Line($"N={N,8}: argsort first-1000 sorted? {ok}");
    Line("");
}
