#:project K:/source/NumSharp/src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property AssemblyName=NumSharp.OpenBlasBenchmark
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

#if DEBUG
Console.WriteLine("FATAL: Debug build — rerun with -c Release");
return;
#pragma warning disable CS0162
#endif

np.multithreading(false);
OpenBlasEngine.Enable(threads: 1);
Console.WriteLine($"INFO library={Path.GetFileName(OpenBlasEngine.LibraryPath)}; bundled={OpenBlasEngine.IsBundledLibrary}; core={OpenBlasEngine.CoreName}; threads=1");

double Best(Action body, int rounds)
{
    // Warm to JIT steady-state (tier-1) BEFORE timing — NumPy is native and always warm, so a single
    // body() warmup would time NumSharp's managed glue while still tier-0 and understate the small-N
    // LAPACK/product calls by ~15-30% (measured). Warm to a call-count AND wall-clock floor, capped
    // so a slow op does not warm indefinitely; NumPy's twin harness reaches steady state in one call.
    var warm = Stopwatch.StartNew();
    int warmIters = 0;
    while ((warmIters < 150 || warm.Elapsed.TotalMilliseconds < 40.0) && warm.Elapsed.TotalMilliseconds < 250.0)
    {
        body();
        warmIters++;
    }

    // best-of-21 floor (repo perf convention): a min over only 3-7 samples of a ~60 us-1.5 ms op has
    // ~15-20% jitter and swings a cell 0.7x<->1.3x run-to-run; 21 timed rounds report the steady-state
    // minimum. Applied symmetrically on the NumPy twin.
    rounds = Math.Max(rounds, 21);
    double best = double.MaxValue;
    for (int i = 0; i < rounds; i++)
    {
        var sw = Stopwatch.StartNew();
        body();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
    }
    return best;
}

void Emit(string route, string op, int n, Action body, int rounds = 5)
    => Console.WriteLine($"{route}|{op}|{n}|f64\t{Best(body, rounds):F6}");

np.random.seed(42);
const int VectorN = 100_000;
const int MatrixN = 128;
var vector = np.random.rand(VectorN);
var kernel = np.random.rand(31);
var matrixA = np.random.rand(MatrixN * MatrixN).reshape(MatrixN, MatrixN);
var matrixB = np.random.rand(MatrixN * MatrixN).reshape(MatrixN, MatrixN);
var matrixVector = np.random.rand(MatrixN);

Emit("product", "dot", VectorN, () => { using var r = np.dot(vector, vector); });
Emit("product", "inner", VectorN, () => { using var r = np.inner(vector, vector); });
Emit("product", "vdot", VectorN, () => { using var r = np.vdot(vector, vector); });
Emit("product", "vecdot", VectorN, () => { using var r = np.vecdot(vector, vector); });
Emit("product", "matmul", MatrixN, () => { using var r = np.matmul(matrixA, matrixB); });
Emit("product", "matvec", MatrixN, () => { using var r = np.matvec(matrixA, matrixVector); });
Emit("product", "vecmat", MatrixN, () => { using var r = np.vecmat(matrixVector, matrixA); });
Emit("sliding", "correlate", VectorN, () => { using var r = np.correlate(vector, kernel, "same"); });
Emit("sliding", "convolve", VectorN, () => { using var r = np.convolve(vector, kernel, "same"); });

var polyX = np.linspace(-1, 1, 1_000);
var polyY = 3.0 * polyX * polyX - 2.0 * polyX + 1.0;
var polyCoefficients = np.linspace(1, 2, 17);
Emit("composed", "polyfit", 1_000, () =>
{
    var r = np.polyfit(polyX, polyY, 3);
    foreach (var value in new[] { r.coeffs, r.residuals, r.rank, r.singular_values, r.covariance })
        if (value is not null) value.Dispose();
});
Emit("composed", "roots", 16, () => { using var r = np.roots(polyCoefficients); });

foreach (int n in new[] { 32, 96 })
{
    np.random.seed(42 + n);
    var raw = np.random.rand(n * n).reshape(n, n);
    var spd = np.matmul(raw.T, raw) + np.eye(n) * n;
    var rhs = np.random.rand(n);
    var rect = np.random.rand(2 * n * n).reshape(2 * n, n);
    var rectRhs = np.random.rand(2 * n);
    int rounds = n <= 32 ? 7 : 3;

    Emit("lapack", "cholesky", n, () => { using var r = np.linalg.cholesky(spd); }, rounds);
    Emit("lapack", "cond", n, () => { using var r = np.linalg.cond(spd); }, rounds);
    Emit("lapack", "det", n, () => { using var r = np.linalg.det(spd); }, rounds);
    Emit("lapack", "eig", n, () => { var (w, v) = np.linalg.eig(spd); w.Dispose(); v.Dispose(); }, rounds);
    Emit("lapack", "eigh", n, () => { var (w, v) = np.linalg.eigh(spd); w.Dispose(); v.Dispose(); }, rounds);
    Emit("lapack", "eigvals", n, () => { using var r = np.linalg.eigvals(spd); }, rounds);
    Emit("lapack", "eigvalsh", n, () => { using var r = np.linalg.eigvalsh(spd); }, rounds);
    Emit("lapack", "inv", n, () => { using var r = np.linalg.inv(spd); }, rounds);
    Emit("lapack", "lstsq", n, () => { var (x, residuals, rank, s) = np.linalg.lstsq(rect, rectRhs); x.Dispose(); residuals.Dispose(); rank.Dispose(); s.Dispose(); }, rounds);
    Emit("lapack", "matrix_rank", n, () => { using var r = np.linalg.matrix_rank(spd); }, rounds);
    Emit("lapack", "pinv", n, () => { using var r = np.linalg.pinv(spd); }, rounds);
    Emit("lapack", "qr", n, () => { var (q, r) = np.linalg.qr(spd); q.Dispose(); r.Dispose(); }, rounds);
    Emit("lapack", "slogdet", n, () => { var (sign, logabsdet) = np.linalg.slogdet(spd); sign.Dispose(); logabsdet.Dispose(); }, rounds);
    Emit("lapack", "solve", n, () => { using var r = np.linalg.solve(spd, rhs); }, rounds);
    Emit("lapack", "svd", n, () => { var (u, s, vh) = np.linalg.svd(spd, full_matrices: false); u.Dispose(); s.Dispose(); vh.Dispose(); }, rounds);
    Emit("lapack", "svdvals", n, () => { using var r = np.linalg.svdvals(spd); }, rounds);
    Emit("lapack", "tensorinv", n, () => { using var r = np.linalg.tensorinv(spd, ind: 1); }, rounds);
    Emit("lapack", "tensorsolve", n, () => { using var r = np.linalg.tensorsolve(spd, rhs); }, rounds);
    Emit("hybrid", "matrix_power_-1", n, () => { using var r = np.linalg.matrix_power(spd, -1); }, rounds);
    Emit("hybrid", "matrix_norm_2", n, () => { using var r = np.linalg.matrix_norm(spd, ord: 2); }, rounds);
    Emit("hybrid", "norm_2", n, () => { using var r = np.linalg.norm(spd, ord: 2); }, rounds);
}
