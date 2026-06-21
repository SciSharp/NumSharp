#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property TargetFramework=net10.0
// Confirm: does DriveAllButAxis invoke the kernel ONCE (count=N) or N times (count=1)?
using System;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

void Line(string s){ Console.WriteLine(s); Console.Out.Flush(); }

unsafe void DriveCount(NDArray[] ops, NpyIterPerOpFlags[] flags, int axis, string label)
{
    int ndim = ops[0].ndim;
    var kept = new int[ndim - 1];
    for (int d = 0, w = 0; d < ndim; d++) if (d != axis) kept[w++] = d;
    var opAxes = new int[ops.Length][];
    for (int i = 0; i < ops.Length; i++) opAxes[i] = kept;

    long calls = 0, totalCount = 0, firstCount = -1;
    var iter = NpyIterRef.AdvancedNew(ops.Length, ops, NpyIterGlobalFlags.None,
        NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING, flags, null, ndim - 1, opAxes);
    try
    {
        NpyInnerLoopFunc k = (p, s, c, a) => { calls++; totalCount += c; if (firstCount < 0) firstCount = c; };
        iter.ForEach(k, null);
    }
    finally { iter.Dispose(); }
    Line($"{label}: kernel CALLS={calls}  firstCount={firstCount}  sum(count)={totalCount}");
}

var rng = new Random(1);

// 1-D: N=10000, sort axis 0 (the only axis -> dropped -> oa_ndim=0)
foreach (int N in new[]{ 100, 10_000 })
{
    var d = new int[N]; for(int i=0;i<N;i++) d[i]=rng.Next();
    var a = np.array(d);
    Line($"--- 1-D N={N} (expect CALLS=1, firstCount={N}) ---");
    DriveCount(new[]{ a.copy() }, new[]{ NpyIterPerOpFlags.READWRITE }, 0, "  sort ");
    var outIdx = new NDArray(NPTypeCode.Int64, new Shape(N), false);
    DriveCount(new[]{ a, outIdx }, new[]{ NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY }, 0, "  argsort");
}

// 2-D: (M,N) sort axis=1 -> keep axis 0 (size M). expect CALLS=M, firstCount=N
{
    int M=8, N=2000; var d=new int[M*N]; for(int i=0;i<d.Length;i++) d[i]=rng.Next();
    var a = np.array(d).reshape(M,N);
    Line($"--- 2-D ({M},{N}) axis=1 (expect CALLS={M}, firstCount={N}) ---");
    DriveCount(new[]{ a.copy() }, new[]{ NpyIterPerOpFlags.READWRITE }, 1, "  sort ");
}
// 2-D: (M,N) sort axis=0 -> keep axis 1 (size N). expect CALLS=N, firstCount=M
{
    int M=2000, N=8; var d=new int[M*N]; for(int i=0;i<d.Length;i++) d[i]=rng.Next();
    var a = np.array(d).reshape(M,N);
    Line($"--- 2-D ({M},{N}) axis=0 (expect CALLS={N}, firstCount={M}) ---");
    DriveCount(new[]{ a.copy() }, new[]{ NpyIterPerOpFlags.READWRITE }, 0, "  sort ");
}
