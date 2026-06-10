#:project ../../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Optimize=true
// Constructor-cost micro-probe: times NpyIterRef.MultiNew alone (the exact
// surface Wave 1.3 touched) for the P15-class shape. Release-gate helper.
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var asm = typeof(np).Assembly;
var dbg = (System.Diagnostics.DebuggableAttribute)Attribute.GetCustomAttribute(asm, typeof(System.Diagnostics.DebuggableAttribute));
if (dbg != null && dbg.IsJITOptimizerDisabled) { Console.Error.WriteLine("DEBUG BUILD — INVALID"); return; }

var a = np.arange(1000).astype(np.float32);
var b = np.arange(1000).astype(np.float32);
var o = np.empty(new Shape(1000), np.float32);
var flags = NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP;
var opf = new[]
{
    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
    NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
};
var ops = new[] { a, b, o };

// warmup
for (int i = 0; i < 20_000; i++) { using var it = NpyIterRef.MultiNew(3, ops, flags, NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_UNSAFE_CASTING, opf); }

double best = double.MaxValue;
for (int round = 0; round < 7; round++)
{
    const int N = 200_000;
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < N; i++) { using var it = NpyIterRef.MultiNew(3, ops, flags, NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_UNSAFE_CASTING, opf); }
    sw.Stop();
    double ns = sw.Elapsed.TotalMilliseconds * 1e6 / N;
    if (ns < best) best = ns;
}
Console.WriteLine($"MultiNew(3 ops, 1K f32, ELW|CIO): best {best:F1} ns/ctor");
