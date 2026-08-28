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
    // Warm to JIT steady-state (tier-1) BEFORE timing. NumPy is native and always warm, so timing
    // NumSharp's managed glue while it is still tier-0 compares cold-vs-warm and understates the
    // small-N LAPACK/product calls by ~15-30% (measured: a single body() warmup leaves e.g.
    // tensorinv/tensordot ~25% slower than their steady state). Warm until BOTH a call-count floor
    // (trips tiered compilation) and a wall-clock floor (lets the background re-JIT land) are met,
    // capped so a slow op does not warm indefinitely. NumPy reaches steady state in one call, so its
    // twin harness needs no matching change.
    var warm = Stopwatch.StartNew();
    int warmIters = 0;
    while ((warmIters < 150 || warm.Elapsed.TotalMilliseconds < 40.0) && warm.Elapsed.TotalMilliseconds < 250.0)
    {
        body();
        warmIters++;
    }

    // Min-time policy (the repo's perf convention): the small-N products/LAPACK calls are ~60 us-1.5 ms
    // with ~15-20% run-to-run jitter, so the reported number is the min over a time-budgeted sweep. A
    // call >20 ms/call runs EXACTLY 100 times; everything else runs enough single-call rounds to span
    // ~200 ms total (estimated from the warmup loop above; capped defensively). Applied symmetrically
    // on the NumPy twin.
    double perCall = warm.Elapsed.TotalMilliseconds / Math.Max(1, warmIters);
    rounds = perCall > 20.0 ? 100 : Math.Min(100_000, Math.Max(2, (int)Math.Ceiling(200.0 / Math.Max(perCall, 1e-9))));
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

// Every product below reaches an IBlasBackend member directly or reaches dot/matmul through a
// composition. Run ONE shared tier matrix in both profiles; a one-off representative case makes the
// dashboard compare unlike scenario sets and prevents its per-cell fastest-backend selection.
foreach (int n in new[] { 1_000, 100_000, 10_000_000 })
{
    np.random.seed(42 + n);
    using var vector = np.random.rand(n);

    // The official LinearAlgebra suite maps the shared workload tier N to a bounded square side.
    int side = Math.Min((int)Math.Sqrt(n), 384);
    using var matrixA = np.random.rand(side * side).reshape(side, side);
    using var matrixB = np.random.rand(side * side).reshape(side, side);
    using var matrixVector = np.random.rand(side);

    // LinalgApi's cubic compositions are capped at 128. Keep those exact operands at the shared N
    // keys so 1K/100K join the main matrix instead of cross-pairing different physical shapes.
    int linalgSide = Math.Min((int)Math.Sqrt(n), 128);
    using var linalgA = np.random.rand(linalgSide * linalgSide).reshape(linalgSide, linalgSide);
    using var linalgB = np.random.rand(linalgSide * linalgSide).reshape(linalgSide, linalgSide);

    Emit("np.dot", "np.dot(a, b)", "openblas_optional", n, () => { using var r = np.dot(vector, vector); });
    Emit("ndarray.dot", "a.dot(b) [ndarray]", "openblas_optional", n, () => { using var r = vector.dot(vector); }, suite: "NDArray");
    Emit("np.inner", "np.inner(a, b)", "openblas_optional", n, () => { using var r = np.inner(vector, vector); }, suite: "Statistics");
    Emit("np.vdot", "np.vdot(a, b)", "openblas_optional", n, () => { using var r = np.vdot(vector, vector); });
    Emit("np.vecdot", "np.vecdot(a, b)", "openblas_optional", n, () => { using var r = np.vecdot(vector, vector); });
    Emit("np.linalg.vecdot", "np.linalg.vecdot(a, b)", "openblas_optional", n, () => { using var r = np.linalg.vecdot(vector, vector); });

    Emit("np.matmul", "np.matmul(a, b)", "openblas_optional", n, () => { using var r = np.matmul(matrixA, matrixB); });
    Emit("np.linalg.matmul", "np.linalg.matmul(a, b)", "openblas_optional", n, () => { using var r = np.linalg.matmul(matrixA, matrixB); });
    Emit("np.matvec", "np.matvec(a, b)", "openblas_optional", n, () => { using var r = np.matvec(matrixA, matrixVector); });
    Emit("np.vecmat", "np.vecmat(a, b)", "openblas_optional", n, () => { using var r = np.vecmat(matrixVector, matrixA); });

    // Cover the exact op-matrix einsum/tensordot cells AND a matrix-contraction scenario. Both are
    // product compositions and therefore backend-sensitive even though they have no Try* seam of
    // their own.
    Emit("np.einsum", "np.einsum(subscripts, operands)", "openblas_optional_composed", n, () => { using var r = np.einsum("i,i->", vector, vector); });
    Emit("np.einsum", "np.einsum('ij,jk->ik', a, b)", "openblas_optional_composed", n, () => { using var r = np.einsum("ij,jk->ik", matrixA, matrixB); }, "ij,jk->ik");
    Emit("np.tensordot", "np.tensordot(a, b)", "openblas_optional_composed", n, () => { using var r = np.tensordot(vector, vector, 1); });
    Emit("np.tensordot", "np.tensordot(a, b, axes=1)", "openblas_optional_composed", n, () => { using var r = np.tensordot(matrixA, matrixB, 1); }, "axes=1,matrix");
    Emit("np.linalg.tensordot", "np.linalg.tensordot(a, b)", "openblas_optional_composed", n, () => { using var r = np.linalg.tensordot(linalgA, linalgB, 1); });
    Emit("np.linalg.multi_dot", "np.linalg.multi_dot(arrays)", "openblas_optional_composed", n, () => { using var r = np.linalg.multi_dot(new[] { linalgA, linalgB, linalgA }); });
    Emit("np.linalg.matrix_power", "np.linalg.matrix_power(a, 3)", "openblas_optional_composed", n, () => { using var r = np.linalg.matrix_power(linalgA, 3); });
}

// Sliding dot is backend-sensitive too and follows the same published workload tiers.
foreach (int n in new[] { 1_000, 100_000, 10_000_000 })
{
    np.random.seed(84 + n);
    using var signal = np.random.rand(n) * 100 - 50;
    using var kernel = np.random.rand(31) * 2 - 1;
    Emit("np.correlate", "np.correlate(a, kernel)", "openblas_optional_sliding_dot", n, () => { using var r = np.correlate(signal, kernel, "same"); }, suite: "Statistics");
    Emit("np.convolve", "np.convolve(a, kernel)", "openblas_optional_sliding_dot", n, () => { using var r = np.convolve(signal, kernel, "same"); }, suite: "Statistics");
}

// API compositions also publish only the shared workload tiers. Their physical problems grow on a
// bounded scale (samples 1K/10K/100K; polynomial degrees 16/32/64) rather than pretending a degree-
// ten-million companion matrix is a runnable benchmark.
foreach (var apiCase in new[] { (Tier: 1_000, Samples: 1_000, Coefficients: 17), (Tier: 100_000, Samples: 10_000, Coefficients: 33), (Tier: 10_000_000, Samples: 100_000, Coefficients: 65) })
{
    using var polyX = np.linspace(-1, 1, apiCase.Samples);
    using var polyY = 3.0 * polyX * polyX - 2.0 * polyX + 1.0;
    using var polyCoefficients = np.linspace(1, 2, apiCase.Coefficients);
    Emit("np.polyfit", "np.polyfit(x, y, 3)", "openblas_required_composed", apiCase.Tier, () =>
    {
        var r = np.polyfit(polyX, polyY, 3);
        foreach (var value in new[] { r.coeffs, r.residuals, r.rank, r.singular_values, r.covariance })
            value?.Dispose();
    }, suite: "Api");
    Emit("np.roots", "np.roots(p)", "openblas_required_composed", apiCase.Tier, () => { using var r = np.roots(polyCoefficients); }, suite: "Api");
}

// LAPACK's N is the SAME published workload tier as every other profile row. The physical matrix
// side is bounded per tier: 1K -> 32, 100K -> 96, 10M -> 128. This preserves the useful scaling
// sweep without exposing implementation-specific side lengths as dashboard size tiers.
foreach (var lapackCase in new[] { (Tier: 1_000, Side: 32), (Tier: 100_000, Side: 96), (Tier: 10_000_000, Side: 128) })
{
    int tierN = lapackCase.Tier;
    int matrixN = lapackCase.Side;
    np.random.seed(42 + tierN);
    using var raw = np.random.rand(matrixN * matrixN).reshape(matrixN, matrixN);
    using var spd = np.matmul(raw.T, raw) + np.eye(matrixN) * matrixN;
    using var rhs = np.random.rand(matrixN);
    using var rect = np.random.rand(2 * matrixN * matrixN).reshape(2 * matrixN, matrixN);
    using var rectRhs = np.random.rand(2 * matrixN);
    int rounds = matrixN <= 32 ? 7 : 3;

    Emit("np.linalg.cholesky", "np.linalg.cholesky(a)", "openblas_required", tierN, () => { using var r = np.linalg.cholesky(spd); }, rounds: rounds);
    Emit("np.linalg.cond", "np.linalg.cond(a)", "openblas_required_composed", tierN, () => { using var r = np.linalg.cond(spd); }, rounds: rounds);
    Emit("np.linalg.det", "np.linalg.det(a)", "openblas_required", tierN, () => { using var r = np.linalg.det(spd); }, rounds: rounds);
    Emit("np.linalg.eig", "np.linalg.eig(a)", "openblas_required", tierN, () => { var (w, v) = np.linalg.eig(spd); w.Dispose(); v.Dispose(); }, rounds: rounds);
    Emit("np.linalg.eigh", "np.linalg.eigh(a)", "openblas_required", tierN, () => { var (w, v) = np.linalg.eigh(spd); w.Dispose(); v.Dispose(); }, rounds: rounds);
    Emit("np.linalg.eigvals", "np.linalg.eigvals(a)", "openblas_required", tierN, () => { using var r = np.linalg.eigvals(spd); }, rounds: rounds);
    Emit("np.linalg.eigvalsh", "np.linalg.eigvalsh(a)", "openblas_required", tierN, () => { using var r = np.linalg.eigvalsh(spd); }, rounds: rounds);
    Emit("np.linalg.inv", "np.linalg.inv(a)", "openblas_required", tierN, () => { using var r = np.linalg.inv(spd); }, rounds: rounds);
    Emit("np.linalg.lstsq", "np.linalg.lstsq(a, b)", "openblas_required", tierN, () => { var (x, residuals, rank, s) = np.linalg.lstsq(rect, rectRhs); x.Dispose(); residuals.Dispose(); rank.Dispose(); s.Dispose(); }, rounds: rounds);
    Emit("np.linalg.matrix_rank", "np.linalg.matrix_rank(a)", "openblas_required_composed", tierN, () => { using var r = np.linalg.matrix_rank(spd); }, rounds: rounds);
    Emit("np.linalg.pinv", "np.linalg.pinv(a)", "openblas_required_composed", tierN, () => { using var r = np.linalg.pinv(spd); }, rounds: rounds);
    Emit("np.linalg.qr", "np.linalg.qr(a)", "openblas_required", tierN, () => { var (q, r) = np.linalg.qr(spd); q.Dispose(); r.Dispose(); }, rounds: rounds);
    Emit("np.linalg.slogdet", "np.linalg.slogdet(a)", "openblas_required", tierN, () => { var (sign, logabsdet) = np.linalg.slogdet(spd); sign.Dispose(); logabsdet.Dispose(); }, rounds: rounds);
    Emit("np.linalg.solve", "np.linalg.solve(a, b)", "openblas_required", tierN, () => { using var r = np.linalg.solve(spd, rhs); }, rounds: rounds);
    Emit("np.linalg.svd", "np.linalg.svd(a)", "openblas_required", tierN, () => { var (u, s, vh) = np.linalg.svd(spd, full_matrices: false); u.Dispose(); s.Dispose(); vh.Dispose(); }, rounds: rounds);
    Emit("np.linalg.svdvals", "np.linalg.svdvals(a)", "openblas_required", tierN, () => { using var r = np.linalg.svdvals(spd); }, rounds: rounds);
    Emit("np.linalg.tensorinv", "np.linalg.tensorinv(a)", "openblas_required_composed", tierN, () => { using var r = np.linalg.tensorinv(spd, ind: 1); }, rounds: rounds);
    Emit("np.linalg.tensorsolve", "np.linalg.tensorsolve(a, b)", "openblas_required_composed", tierN, () => { using var r = np.linalg.tensorsolve(spd, rhs); }, rounds: rounds);
    Emit("np.linalg.matrix_power", "np.linalg.matrix_power(a, -1)", "openblas_hybrid", tierN, () => { using var r = np.linalg.matrix_power(spd, -1); }, "n=-1", rounds: rounds);
    Emit("np.linalg.matrix_norm", "np.linalg.matrix_norm(a, ord=2)", "openblas_hybrid", tierN, () => { using var r = np.linalg.matrix_norm(spd, ord: 2); }, "ord=2", rounds: rounds);
    Emit("np.linalg.norm", "np.linalg.norm(a, ord=2)", "openblas_hybrid", tierN, () => { using var r = np.linalg.norm(spd, ord: 2); }, "ord=2", rounds: rounds);
}
