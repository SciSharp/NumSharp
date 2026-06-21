#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property TargetFramework=net10.0
// Validate the 1-D O(N^2) fix: scaling now O(N), correctness, in-place, vs NumPy.
using System;
using System.Diagnostics;
using NumSharp;

void Line(string s){ Console.WriteLine(s); Console.Out.Flush(); }
static double Best(int r, Action a){ double b=1e18; for(int i=0;i<r;i++){ var sw=Stopwatch.StartNew(); a(); sw.Stop(); b=Math.Min(b,sw.Elapsed.TotalMilliseconds);} return b; }
var rng = new Random(7);

// ---------- 1) correctness: 1-D sort + argsort vs Array.Sort ----------
Line("== correctness ==");
foreach (int N in new[]{ 1, 2, 17, 1000, 50000 })
{
    var d = new int[N]; for(int i=0;i<N;i++) d[i]=rng.Next(-1000,1000);
    var exp = (int[])d.Clone(); Array.Sort(exp);
    var s = np.sort(np.array((int[])d.Clone()));
    bool sortOk = true; for(int i=0;i<N;i++) if(s.GetInt32(i)!=exp[i]){ sortOk=false; break; }
    var g = np.argsort(np.array((int[])d.Clone()));
    bool argOk = true; for(int i=1;i<N;i++) if(d[(int)g.GetInt64(i-1)] > d[(int)g.GetInt64(i)]){ argOk=false; break; }
    Line($"  N={N,6}: sort={sortOk} argsort={argOk}");
}

// ---------- 2) in-place on strided 1-D view ----------
Line("== in-place strided view (a[::2].sort()) ==");
{
    var a = np.array(new[]{ 9,1,8,2,7,3,6,4,5,0 });
    var v = a["::2"]; // [9,8,7,6,5] strided
    v.sort();
    // expected: strided positions sorted -> a = [5,1,6,2,7,3,8,4,9,0]
    var got = new int[10]; for(int i=0;i<10;i++) got[i]=a.GetInt32(i);
    Line($"  a after = [{string.Join(",",got)}]  (expect 5,1,6,2,7,3,8,4,9,0)");
}

// ---------- 3) axis=None ----------
{
    var m = np.array(new[,]{ {3,1},{2,0} });
    var s = np.sort(m, (int?)null);
    Line($"== axis=None: [{s.GetInt32(0)},{s.GetInt32(1)},{s.GetInt32(2)},{s.GetInt32(3)}] (expect 0,1,2,3) ==");
}

// ---------- 4) scaling: 1-D must now be O(N) ----------
Line("== 1-D scaling (was O(N^2); expect ~linear M/s) ==");
foreach (int N in new[]{ 10_000, 100_000, 1_000_000, 4_000_000 })
{
    var d = new int[N]; for(int i=0;i<N;i++) d[i]=rng.Next();
    var a = np.array(d);
    np.argsort(a); a.copy().sort(); // warmup
    double tsort = Best(5, ()=>{ var c=a.copy(); c.sort(); });
    double targ  = Best(5, ()=>{ var g=np.argsort(a); GC.KeepAlive(g); });
    Line($"  N={N,8}: sort {tsort,7:F2}ms ({N/tsort/1e3,5:F0} M/s) | argsort {targ,7:F2}ms ({N/targ/1e3,5:F0} M/s)");
}

// ---------- 5) vs NumPy (int64/float64/float32 too) ----------
Line("== vs NumPy 2.4.2 (NPY M/s ref: int32 1M=45/4M=28, int64 1M=31/4M=20, f64 1M=40/4M=24, f32 1M=43/4M=29) ==");
int[] Ns = { 1_000_000, 4_000_000 };
foreach (int N in Ns)
{
    var di = new long[N]; for(int i=0;i<N;i++) di[i]=((long)rng.Next()<<20)^rng.Next();
    var ai = np.array(di); np.argsort(ai);
    double t64 = Best(5, ()=>{ var g=np.argsort(ai); GC.KeepAlive(g); });
    Line($"  int64 N={N,8}: argsort {N/t64/1e3,5:F0} M/s");

    var df = new double[N]; for(int i=0;i<N;i++) df[i]=rng.NextDouble()*2-1;
    var af = np.array(df); np.argsort(af);
    double tf = Best(5, ()=>{ var g=np.argsort(af); GC.KeepAlive(g); });
    Line($"  f64   N={N,8}: argsort {N/tf/1e3,5:F0} M/s");
}
