#:project K:/source/NumSharp/src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property AssemblyName=NumSharp.BackendProfileBenchmark
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

using System.Diagnostics;
using System.Text.Json;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

#if DEBUG
Console.Error.WriteLine("FATAL: Debug build — rerun with -c Release");
return;
#pragma warning disable CS0162
#endif

string profile = (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_PROFILE") ?? "managed")
    .Trim().ToLowerInvariant();
np.multithreading(false);
if (profile == "managed")
{
    OpenBlasEngine.Disable();
}
else if (profile == "openblas")
{
    OpenBlasEngine.Enable(threads: 1);
    if (!OpenBlasEngine.Enabled)
        throw new InvalidOperationException("The OpenBLAS profile was requested but OpenBlasEngine.Enabled is false.");
}
else
{
    throw new ArgumentException($"Unknown NUMSHARP_BENCHMARK_PROFILE '{profile}'.");
}

Console.Error.WriteLine(profile == "openblas"
    ? $"PROFILE openblas; {OpenBlasEngine.Info}"
    : "PROFILE managed; TensorEngine.Blas=null");
string backendInfo = profile == "openblas" ? OpenBlasEngine.Info : "Managed C# (TensorEngine.Blas=null)";

double Best(Action body, int rounds)
{
    body();
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

void Emit(string api, string operation, string route, int n, Action body,
    string scenario = "", string suite = "LinearAlgebra", int rounds = 5)
{
    try
    {
        double value = Best(body, rounds);
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            key = $"{api}|{scenario}|{n}|float64",
            profile,
            api,
            operation,
            suite,
            category = route,
            backend_route = route,
            dtype = "float64",
            n,
            scenario,
            availability = "available",
            numsharp_ms = value,
            actual_backend = profile,
            backend_info = backendInfo
        }));
    }
    catch (MissingBackendException ex)
    {
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            key = $"{api}|{scenario}|{n}|float64",
            profile,
            api,
            operation,
            suite,
            category = route,
            backend_route = route,
            dtype = "float64",
            n,
            scenario,
            availability = "missing_backend",
            numsharp_ms = (double?)null,
            actual_backend = (string?)null,
            backend_info = backendInfo,
            exception_type = ex.GetType().Name,
            exception_message = ex.Message
        }));
    }
    catch (NotSupportedException ex)
    {
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            key = $"{api}|{scenario}|{n}|float64",
            profile,
            api,
            operation,
            suite,
            category = route,
            backend_route = route,
            dtype = "float64",
            n,
            scenario,
            availability = "not_supported",
            numsharp_ms = (double?)null,
            actual_backend = (string?)null,
            backend_info = backendInfo,
            exception_type = ex.GetType().Name,
            exception_message = ex.Message
        }));
    }
}

np.random.seed(42);
const int VectorN = 100_000;
const int MatrixN = 128;
var vector = np.random.rand(VectorN);
var kernel = np.random.rand(31);
var matrixA = np.random.rand(MatrixN * MatrixN).reshape(MatrixN, MatrixN);
var matrixB = np.random.rand(MatrixN * MatrixN).reshape(MatrixN, MatrixN);
var matrixVector = np.random.rand(MatrixN);

Emit("np.dot", "np.dot(a, b)", "openblas_optional", VectorN, () => { using var r = np.dot(vector, vector); });
Emit("ndarray.dot", "a.dot(b)", "openblas_optional", VectorN, () => { using var r = vector.dot(vector); });
Emit("np.inner", "np.inner(a, b)", "openblas_optional", VectorN, () => { using var r = np.inner(vector, vector); });
Emit("np.vdot", "np.vdot(a, b)", "openblas_optional", VectorN, () => { using var r = np.vdot(vector, vector); });
Emit("np.vecdot", "np.vecdot(a, b)", "openblas_optional", VectorN, () => { using var r = np.vecdot(vector, vector); });
Emit("np.linalg.vecdot", "np.linalg.vecdot(a, b)", "openblas_optional", VectorN, () => { using var r = np.linalg.vecdot(vector, vector); });
Emit("np.matmul", "np.matmul(a, b)", "openblas_optional", MatrixN, () => { using var r = np.matmul(matrixA, matrixB); });
Emit("np.linalg.matmul", "np.linalg.matmul(a, b)", "openblas_optional", MatrixN, () => { using var r = np.linalg.matmul(matrixA, matrixB); });
Emit("np.matvec", "np.matvec(a, b)", "openblas_optional", MatrixN, () => { using var r = np.matvec(matrixA, matrixVector); });
Emit("np.vecmat", "np.vecmat(a, b)", "openblas_optional", MatrixN, () => { using var r = np.vecmat(matrixVector, matrixA); });
Emit("np.einsum", "np.einsum('ij,jk->ik', a, b)", "openblas_optional_composed", MatrixN, () => { using var r = np.einsum("ij,jk->ik", matrixA, matrixB); }, "ij,jk->ik");
Emit("np.tensordot", "np.tensordot(a, b, axes=1)", "openblas_optional_composed", MatrixN, () => { using var r = np.tensordot(matrixA, matrixB, 1); }, "axes=1");
Emit("np.linalg.tensordot", "np.linalg.tensordot(a, b, axes=1)", "openblas_optional_composed", MatrixN, () => { using var r = np.linalg.tensordot(matrixA, matrixB, 1); }, "axes=1");
Emit("np.linalg.multi_dot", "np.linalg.multi_dot(arrays)", "openblas_optional_composed", MatrixN, () => { using var r = np.linalg.multi_dot(new[] { matrixA, matrixB, matrixA }); });
Emit("np.correlate", "np.correlate(a, v, mode='same')", "openblas_optional_sliding_dot", VectorN, () => { using var r = np.correlate(vector, kernel, "same"); }, "mode=same,kernel=31", "Statistics");
Emit("np.convolve", "np.convolve(a, v, mode='same')", "openblas_optional_sliding_dot", VectorN, () => { using var r = np.convolve(vector, kernel, "same"); }, "mode=same,kernel=31", "Statistics");

var polyX = np.linspace(-1, 1, 1_000);
var polyY = 3.0 * polyX * polyX - 2.0 * polyX + 1.0;
var polyCoefficients = np.linspace(1, 2, 17);
Emit("np.polyfit", "np.polyfit(x, y, 3)", "openblas_required_composed", 1_000, () =>
{
    var r = np.polyfit(polyX, polyY, 3);
    foreach (var value in new[] { r.coeffs, r.residuals, r.rank, r.singular_values, r.covariance })
        value?.Dispose();
}, suite: "Api");
Emit("np.roots", "np.roots(p)", "openblas_required_composed", 16, () => { using var r = np.roots(polyCoefficients); }, suite: "Api");

foreach (int n in new[] { 32, 96 })
{
    np.random.seed(42 + n);
    var raw = np.random.rand(n * n).reshape(n, n);
    var spd = np.matmul(raw.T, raw) + np.eye(n) * n;
    var rhs = np.random.rand(n);
    var rect = np.random.rand(2 * n * n).reshape(2 * n, n);
    var rectRhs = np.random.rand(2 * n);
    int rounds = n <= 32 ? 7 : 3;

    Emit("np.linalg.cholesky", "np.linalg.cholesky(a)", "openblas_required", n, () => { using var r = np.linalg.cholesky(spd); }, rounds: rounds);
    Emit("np.linalg.cond", "np.linalg.cond(a)", "openblas_required_composed", n, () => { using var r = np.linalg.cond(spd); }, rounds: rounds);
    Emit("np.linalg.det", "np.linalg.det(a)", "openblas_required", n, () => { using var r = np.linalg.det(spd); }, rounds: rounds);
    Emit("np.linalg.eig", "np.linalg.eig(a)", "openblas_required", n, () => { var (w, v) = np.linalg.eig(spd); w.Dispose(); v.Dispose(); }, rounds: rounds);
    Emit("np.linalg.eigh", "np.linalg.eigh(a)", "openblas_required", n, () => { var (w, v) = np.linalg.eigh(spd); w.Dispose(); v.Dispose(); }, rounds: rounds);
    Emit("np.linalg.eigvals", "np.linalg.eigvals(a)", "openblas_required", n, () => { using var r = np.linalg.eigvals(spd); }, rounds: rounds);
    Emit("np.linalg.eigvalsh", "np.linalg.eigvalsh(a)", "openblas_required", n, () => { using var r = np.linalg.eigvalsh(spd); }, rounds: rounds);
    Emit("np.linalg.inv", "np.linalg.inv(a)", "openblas_required", n, () => { using var r = np.linalg.inv(spd); }, rounds: rounds);
    Emit("np.linalg.lstsq", "np.linalg.lstsq(a, b)", "openblas_required", n, () => { var (x, residuals, rank, s) = np.linalg.lstsq(rect, rectRhs); x.Dispose(); residuals.Dispose(); rank.Dispose(); s.Dispose(); }, rounds: rounds);
    Emit("np.linalg.matrix_rank", "np.linalg.matrix_rank(a)", "openblas_required_composed", n, () => { using var r = np.linalg.matrix_rank(spd); }, rounds: rounds);
    Emit("np.linalg.pinv", "np.linalg.pinv(a)", "openblas_required_composed", n, () => { using var r = np.linalg.pinv(spd); }, rounds: rounds);
    Emit("np.linalg.qr", "np.linalg.qr(a)", "openblas_required", n, () => { var (q, r) = np.linalg.qr(spd); q.Dispose(); r.Dispose(); }, rounds: rounds);
    Emit("np.linalg.slogdet", "np.linalg.slogdet(a)", "openblas_required", n, () => { var (sign, logabsdet) = np.linalg.slogdet(spd); sign.Dispose(); logabsdet.Dispose(); }, rounds: rounds);
    Emit("np.linalg.solve", "np.linalg.solve(a, b)", "openblas_required", n, () => { using var r = np.linalg.solve(spd, rhs); }, rounds: rounds);
    Emit("np.linalg.svd", "np.linalg.svd(a)", "openblas_required", n, () => { var (u, s, vh) = np.linalg.svd(spd, full_matrices: false); u.Dispose(); s.Dispose(); vh.Dispose(); }, rounds: rounds);
    Emit("np.linalg.svdvals", "np.linalg.svdvals(a)", "openblas_required", n, () => { using var r = np.linalg.svdvals(spd); }, rounds: rounds);
    Emit("np.linalg.tensorinv", "np.linalg.tensorinv(a)", "openblas_required_composed", n, () => { using var r = np.linalg.tensorinv(spd, ind: 1); }, rounds: rounds);
    Emit("np.linalg.tensorsolve", "np.linalg.tensorsolve(a, b)", "openblas_required_composed", n, () => { using var r = np.linalg.tensorsolve(spd, rhs); }, rounds: rounds);
    Emit("np.linalg.matrix_power", "np.linalg.matrix_power(a, -1)", "openblas_hybrid", n, () => { using var r = np.linalg.matrix_power(spd, -1); }, "n=-1", rounds: rounds);
    Emit("np.linalg.matrix_norm", "np.linalg.matrix_norm(a, ord=2)", "openblas_hybrid", n, () => { using var r = np.linalg.matrix_norm(spd, ord: 2); }, "ord=2", rounds: rounds);
    Emit("np.linalg.norm", "np.linalg.norm(a, ord=2)", "openblas_hybrid", n, () => { using var r = np.linalg.norm(spd, ord: 2); }, "ord=2", rounds: rounds);
}
