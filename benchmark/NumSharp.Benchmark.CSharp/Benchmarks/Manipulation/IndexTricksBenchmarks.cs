using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Manipulation;

/// <summary>
/// Benchmarks for NumPy's index-expression DSL (journey3): np.r_, np.c_ and np.ix_.
///
/// <para>
/// Three cost classes, kept apart because a geomean across them would describe nothing:
/// </para>
/// <list type="bullet">
///   <item><b>O(N) concatenation</b> — <c>r_[a, b]</c> / <c>c_[a, b]</c>. Both sides walk an index
///         expression (directive parsing, per-entry ndmin/trans1d reshaping, a result_type pass)
///         and then land in <c>concatenate</c>, so at Medium/Large the memcpy dominates and the
///         ratio should track <c>np.concatenate</c>'s.</item>
///   <item><b>O(N) generation</b> — <c>r_["0:N"]</c> (the arange branch) and <c>r_["0:1:Nj"]</c>
///         (the imaginary-step linspace branch). No input array at all; the cost is the fill.</item>
///   <item><b>O(1) view</b> — <c>ix_(a, b)</c> inserts length-1 axes by aliasing the operand's
///         storage, so it allocates no data and its timing measures NDArray/Shape construction
///         overhead rather than throughput. A sub-1.0 cell here is dispatch cost on a no-op, not a
///         slow algorithm.</item>
/// </list>
///
/// <para>
/// The scalar-tail variants (<c>r_[a, 0, 0, b]</c>) exist to price the NEP50 weak-scalar path: each
/// literal becomes a 0-d array that must be promoted and padded to ndmin before the concatenate,
/// which is per-ENTRY work NumPy pays too.
/// </para>
///
/// NumPy twins live in numpy_benchmark.py::run_manipulation_benchmarks (float64). The labels are
/// written with PARENTHESES rather than the natural np.r_[...] brackets on purpose: merge-results'
/// normalize_op_name strips an identifier-only "(...)" arg list but leaves "[...]" alone, so
/// "np.r_(a, b)" joins onto NumPy's "np.r_" while "np.r_[a, b]" would not join at all.
/// </summary>
[BenchmarkCategory("Manipulation", "IndexTricks")]
public class IndexTricksBenchmarks : BenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private NDArray _rows = null!;
    private NDArray _cols = null!;
    private string _arangeExpr = null!;
    private string _linspaceExpr = null!;

    [Params(ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _a = np.random.rand(N) * 100;
        _b = np.random.rand(N) * 100;

        // ix_ operands: two index vectors whose cross product is the sqrt(N) x sqrt(N) mesh.
        var side = (int)Math.Sqrt(N);
        _rows = np.arange(side);
        _cols = np.arange(side);

        _arangeExpr = $"0:{N}";
        _linspaceExpr = $"0:1:{N}j";
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        _rows = null!;
        _cols = null!;
        GC.Collect();
    }

    // ========================================================================
    // O(N) concatenation
    // ========================================================================

    [Benchmark(Description = "np.r_(a, b)")]
    [BenchmarkCategory("IndexTricks")]
    public NDArray R_TwoArrays() => np.r_[_a, _b];

    [Benchmark(Description = "np.r_(a, 0, 0, b)")]
    [BenchmarkCategory("IndexTricks")]
    public NDArray R_ArraysAndScalars() => np.r_[_a, 0.0, 0.0, _b];

    [Benchmark(Description = "np.c_(a, b)")]
    [BenchmarkCategory("IndexTricks")]
    public NDArray C_TwoArrays() => np.c_[_a, _b];

    // ========================================================================
    // O(N) generation (no input array — arange / linspace branches)
    // ========================================================================

    [Benchmark(Description = "np.r_(0:N)")]
    [BenchmarkCategory("IndexTricks")]
    public NDArray R_Arange() => np.r_[_arangeExpr];

    [Benchmark(Description = "np.r_(0:1:Nj)")]
    [BenchmarkCategory("IndexTricks")]
    public NDArray R_Linspace() => np.r_[_linspaceExpr];

    // ========================================================================
    // O(1) view — measures construction overhead, not throughput
    // ========================================================================

    [Benchmark(Description = "np.ix_(rows, cols)")]
    [BenchmarkCategory("IndexTricks")]
    public NDArray[] Ix_OpenMesh() => np.ix_(_rows, _cols);
}
