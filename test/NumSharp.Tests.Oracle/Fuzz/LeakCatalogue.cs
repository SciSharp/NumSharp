using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NumSharp.Backends;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     One catalogue entry: a direct invocation of a public NumSharp member that no corpus row can
    ///     reach, measured by the leak gate exactly like a corpus case (warm invocation, then
    ///     <see cref="ScopeAudit.MeasureConfirmedTraffic"/>, then the result disposed).
    /// </summary>
    /// <param name="Api">The inventory id the entry covers (<c>np.histogram</c>, <c>ndarray.GetData</c>,
    /// <c>NDMaskedArray.sum</c>, <c>ndarray.op_Addition</c>, …) — what
    /// <see cref="LeakSurfaceCoverageTests"/> matches against the reflected surface.</param>
    /// <param name="Label">Variant label (the overload / mode exercised), for diagnostics.</param>
    /// <param name="Run">The invocation. Receives the shared <see cref="LeakFixture"/> (build NOTHING the
    /// entry does not own inside it — fixture inputs live outside the measured region) and returns
    /// the result object, which the harness disposes through
    /// <see cref="UndisposedIntermediateTests.DisposeAny"/> (fixture arrays are never disposed). It must
    /// be re-executable: every execution must produce the same pool traffic.</param>
    /// <param name="RequiresBackend">True when the member computes only through a matrix backend
    /// (LAPACK); such entries run with OpenBLAS installed and are skipped where none loads.</param>
    /// <param name="Throws">True when the member ALWAYS raises (by design — NumPy parity such as an unhashable
    /// ndarray — or because it is unimplemented/broken): the entry is measured as an ERROR path (the throw
    /// must strand nothing), and an entry flagged so that stops throwing fails as a harness error, so the
    /// flag cannot outlive the behaviour it documents.</param>
    internal sealed record LeakCase(string Api, string Label, Func<LeakFixture, object> Run, bool RequiresBackend = false, bool Throws = false);

    /// <summary>
    ///     The shared inputs every catalogue entry and every reflective property read draws on — built
    ///     ONCE, outside every measured region, and disposed at the end. Every array it holds is in
    ///     <see cref="Keep"/>, so the result disposer can never release a fixture buffer (an entry that
    ///     returns a fixture array, or a view of one, stays safe).
    /// </summary>
    /// <remarks>
    ///     Entries MUTATE only inputs they create themselves, or the explicit per-dtype scratch arrays
    ///     (<see cref="Typed"/>), and write back values equal to what was there — so a re-executed
    ///     region sees the same state and produces the same traffic.
    /// </remarks>
    internal sealed class LeakFixture : IDisposable
    {
        /// <summary>float64 (50,) — a general 1-D operand with no special values.</summary>
        public readonly NDArray V;

        /// <summary>float64 (3, 4) C-contiguous matrix.</summary>
        public readonly NDArray M;

        /// <summary>A transposed (non-contiguous) VIEW of <see cref="M"/>, shape (4, 3).</summary>
        public readonly NDArray MT;

        /// <summary>float64 (3, 3) symmetric positive-definite matrix (linalg inputs).</summary>
        public readonly NDArray Sq;

        /// <summary>float64 (3,) vector (cross/outer/vecdot inputs).</summary>
        public readonly NDArray V3;

        /// <summary>int32 (12,) operand.</summary>
        public readonly NDArray I;

        /// <summary>int64 (3,) fancy-index vector into <see cref="M"/>'s rows / <see cref="V"/>.</summary>
        public readonly NDArray Idx;

        /// <summary>bool (50,) mask aligned with <see cref="V"/>.</summary>
        public readonly NDArray B;

        /// <summary>complex128 (6,) operand.</summary>
        public readonly NDArray C;

        /// <summary>float64 (6,) laced with NaN / ±inf.</summary>
        public readonly NDArray NanV;

        /// <summary>char (5,) — the string-conversion surface (AsString/GetString/…).</summary>
        public readonly NDArray Chars;

        /// <summary>float64 (4, 3, 2) 3-D operand (histogramdd samples, axis ops).</summary>
        public readonly NDArray Cube;

        /// <summary>0-d int32 scalar array (a HELD scalar operand — never a call-site temp).</summary>
        public readonly NDArray Zero;

        /// <summary>A masked (3, 4) float64 array with a real mask.</summary>
        public readonly NDMaskedArray MA;

        /// <summary>A second masked (3, 4) float64 array with a real mask (binary masked operands).</summary>
        public readonly NDMaskedArray MB;

        /// <summary>A nomask (3, 4) float64 masked array.</summary>
        public readonly NDMaskedArray MAn;

        /// <summary>A 1-D masked float64 array (50,) masked where <see cref="B"/> is true.</summary>
        public readonly NDMaskedArray MV;

        /// <summary>Masked VIEWS of <see cref="MV"/> built outside every region — <c>[0:4]</c>, <c>[0:5]</c>,
        /// <c>[5:10]</c>, <c>[0:10]</c>, <c>[10:13]</c>, <c>[10:20]</c> — so an entry never slices a masked
        /// operand inline (a masked slice is itself an API whose arrays the entry would then own).</summary>
        public readonly NDMaskedArray MV0_4, MV0_5, MV5_10, MV0_10, MV10_13, MV10_20;

        /// <summary>A masked transposed view of <see cref="MB"/> (the dot operand).</summary>
        public readonly NDMaskedArray MBT;

        /// <summary>A masked (3, 4) int32 array with a real mask (the bitwise/shift masked operators).</summary>
        public readonly NDMaskedArray MI;

        /// <summary>A second masked (3, 4) int32 array with a real mask.</summary>
        public readonly NDMaskedArray MI2;

        /// <summary>A typed <c>NDArray&lt;double&gt;</c> wrapper sharing <see cref="M"/>'s buffer.</summary>
        public readonly NumSharp.Generic.NDArray<double> GM;

        /// <summary>A typed <c>NDArray&lt;bool&gt;</c> wrapper sharing <see cref="B"/>'s buffer.</summary>
        public readonly NumSharp.Generic.NDArray<bool> GB;

        /// <summary>Per-dtype (2, 3) scratch arrays for the typed Get*/Set* surface (all 15 dtypes).</summary>
        public readonly Dictionary<NPTypeCode, NDArray> Typed = new();

        /// <summary>A scratch directory for file-backed IO entries (save/load/tofile).</summary>
        public readonly string Dir;

        /// <summary>A .npy file holding <see cref="M"/>.</summary>
        public readonly string NpyPath;

        /// <summary>A .npz archive holding <see cref="M"/> and <see cref="V"/> (keys <c>m</c>, <c>v</c>).</summary>
        public readonly string NpzPath;

        /// <summary>The bytes of <see cref="NpyPath"/> (in-memory load entries).</summary>
        public readonly byte[] NpyBytes;

        /// <summary>The bytes of <see cref="NpzPath"/> (in-memory load entries).</summary>
        public readonly byte[] NpzBytes;

        /// <summary>A text file of numbers for <c>loadtxt</c>/<c>fromfile</c> style entries.</summary>
        public readonly string TextPath;

        /// <summary>A poly1d of coefficients [1, -3, 2].</summary>
        public readonly poly1d P;

        /// <summary>A seeded Generator (PCG64). Draws advance its state — harmless for leak accounting.</summary>
        public readonly Generator Rng;

        /// <summary>A seeded legacy RandomState.</summary>
        public readonly NumPyRandom Rs;

        /// <summary>Every array the fixture owns — the result disposer's never-dispose set.</summary>
        public readonly HashSet<object> Keep = new(ReferenceEqualityComparer.Instance);

        /// <summary>
        ///     The base-buffer byte range of every non-empty array in <see cref="Keep"/>. A catalogue
        ///     result whose data pointer lands inside one is a VIEW of a fixture — it allocated nothing —
        ///     so the pool-bypass check must not read it as a fresh result that skipped the pool.
        /// </summary>
        public readonly List<(ulong lo, ulong hi)> Ranges = new();

        /// <summary>Builds every input and registers each array in <see cref="Keep"/>.</summary>
        /// <exception cref="IOException">The scratch directory or files cannot be created.</exception>
        public LeakFixture()
        {
            // Built under a harness scope: the throwaway parents of every view/cast chain (arange →
            // reshape → astype …) are released at scope exit while the yielded inputs survive, so the
            // fixture feeds no garbage into the finalizer backlog the measurements must drain.
            using (var scope = NDScope.Open())
            {
                V = Own(scope, (np.arange(50.0) * 0.37 - 3.0));
                M = Own(scope, np.arange(12.0).reshape(3, 4) * 1.5 + 0.25);
                MT = Own(scope, M.T);
                Sq = Own(scope, np.array(new double[,] { { 4, 1, 0.5 }, { 1, 3, 0.25 }, { 0.5, 0.25, 2 } }));
                V3 = Own(scope, np.array(new double[] { 1, 2, 3 }));
                I = Own(scope, np.arange(12).astype(np.int32));
                Idx = Own(scope, np.array(new long[] { 0, 2, 1 }));
                B = Own(scope, (np.arange(50) % 3) == 0);
                C = Own(scope, np.array(new System.Numerics.Complex[] { new(1, 2), new(-3, 0.5), new(0, 0), new(2, -1), new(0.5, 0.5), new(-1, -1) }));
                NanV = Own(scope, np.array(new double[] { 1.0, double.NaN, double.PositiveInfinity, -2.0, double.NegativeInfinity, 0.0 }));
                Chars = Own(scope, np.array("hello".ToCharArray()));
                Cube = Own(scope, np.arange(24.0).reshape(4, 3, 2));
                Zero = Own(scope, NDArray.Scalar(0));
                foreach (NPTypeCode tc in new[]
                         {
                             NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                             NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                             NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
                         })
                    Typed[tc] = Own(scope, np.arange(1, 7).reshape(2, 3).astype(tc));

                var maskM = Own(scope, np.array(new bool[,] { { false, true, false, false }, { false, false, true, false }, { true, false, false, false } }));
                var maskM2 = Own(scope, np.array(new bool[,] { { true, false, false, false }, { false, false, false, true }, { false, true, false, false } }));
                MA = np.ma.masked_array(M.copy(), maskM);
                MB = np.ma.masked_array((M * 0.5).copy(), maskM2);
                MAn = np.ma.masked_array(M.copy());
                MV = np.ma.masked_array(V.copy(), B);
                MI = np.ma.masked_array(np.arange(12).reshape(3, 4).astype(np.int32), maskM);
                MI2 = np.ma.masked_array((np.arange(12).reshape(3, 4) % 5).astype(np.int32), maskM2);
                GM = (NumSharp.Generic.NDArray<double>)Own(scope, M.MakeGeneric<double>());
                GB = (NumSharp.Generic.NDArray<bool>)Own(scope, B.MakeGeneric<bool>());
                MV0_4 = (NDMaskedArray)MV["0:4"];
                MV0_5 = (NDMaskedArray)MV["0:5"];
                MV5_10 = (NDMaskedArray)MV["5:10"];
                MV0_10 = (NDMaskedArray)MV["0:10"];
                MV10_13 = (NDMaskedArray)MV["10:13"];
                MV10_20 = (NDMaskedArray)MV["10:20"];
                MBT = MB.transpose();
                foreach (var m in new[] { MA, MB, MAn, MV, MI, MI2, MV0_4, MV0_5, MV5_10, MV0_10, MV10_13, MV10_20, MBT })
                {
                    Keep.Add(scope.Returns(m._data));
                    if (m._mask is not null)
                        Keep.Add(scope.Returns(m._mask));
                }

                P = new poly1d(Own(scope, np.array(new double[] { 1, -3, 2 })));
                Keep.Add(scope.Returns(P.coeffs));
            }

            // Process-wide np.ma singletons are never disposable results (see MaskedSingletons).
            foreach (var singleton in UndisposedIntermediateTests.MaskedSingletons())
                Keep.Add(singleton);

            Rng = np.random.default_rng(1234);
            Rs = np.random.RandomState(1234);

            Dir = Path.Combine(Path.GetTempPath(), "numsharp-leak-catalogue-" + Environment.ProcessId + "-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Dir);
            NpyPath = Path.Combine(Dir, "m.npy");
            NpzPath = Path.Combine(Dir, "mv.npz");
            TextPath = Path.Combine(Dir, "nums.txt");
            np.save(NpyPath, M);
            np.savez(NpzPath, new[] { M, V }, new Dictionary<string, NDArray> { ["m"] = M, ["v"] = V });
            File.WriteAllText(TextPath, "1 2 3\n4 5 6\n");
            NpyBytes = File.ReadAllBytes(NpyPath);
            NpzBytes = File.ReadAllBytes(NpzPath);

            // Computed LAST: every fixture array (and every np.ma singleton) is in Keep by now.
            foreach (var o in Keep)
                if (o is NDArray nd && nd.size > 0)
                    Ranges.Add(UndisposedIntermediateTests.BaseRange(nd));
        }

        /// <summary>Yields a fixture array out of the construction scope and registers it as kept.</summary>
        /// <param name="scope">The construction scope.</param>
        /// <param name="nd">The array to keep.</param>
        /// <returns><paramref name="nd"/>.</returns>
        private NDArray Own(NDScope scope, NDArray nd)
        {
            Keep.Add(scope.Returns(nd));
            return nd;
        }

        /// <summary>Disposes every fixture array (and the poly1d) and deletes the scratch directory.</summary>
        /// <remarks>The np.ma process-wide singletons ride in <see cref="Keep"/> only so no result
        /// disposer releases them; the fixture never owned them, so they are excluded here too.</remarks>
        public void Dispose()
        {
            P.Dispose();
            var seen = new HashSet<object>(UndisposedIntermediateTests.MaskedSingletons(), ReferenceEqualityComparer.Instance);
            foreach (var o in Keep)
                if (o is NDArray nd && seen.Add(nd))
                    nd.Dispose();
            try
            {
                Directory.Delete(Dir, recursive: true);
            }
            catch (IOException)
            {
                // best-effort cleanup of a temp directory: an entry's still-open handle must not fail the gate
            }
            catch (UnauthorizedAccessException)
            {
                // same: temp-file cleanup is not part of the verdict
            }
        }
    }

    /// <summary>
    ///     The leak catalogue: every public member of the NumPy-facing surface that NO corpus row
    ///     reaches (the inventory cross-reference's missing list), each as one direct invocation. Split
    ///     across partial files by owner: <c>LeakCatalogue.cs</c> (np, np.linalg, np.random and the
    ///     object surfaces), <c>LeakCatalogue.NDArray.cs</c> (ndarray, NDArray&lt;T&gt; and the operators),
    ///     <c>LeakCatalogue.Masked.cs</c> (np.ma and NDMaskedArray).
    /// </summary>
    /// <remarks>
    ///     An entry is a COVERAGE claim, not a behaviour test: the arguments only need to drive the
    ///     member through its real allocation path (values are gated elsewhere). Members whose
    ///     overloads route through materially different code (bins as count / estimator / edges; a
    ///     masked vs nomask operand) carry one entry per route.
    /// </remarks>
    internal static partial class LeakCatalogue
    {
        /// <summary>Every entry, from all partial files, in declaration order.</summary>
        public static IReadOnlyList<LeakCase> All { get; } = Build();

        /// <summary>The distinct inventory ids the catalogue covers.</summary>
        public static IReadOnlySet<string> ApiIds { get; } = new HashSet<string>(All.Select(c => c.Api), StringComparer.Ordinal);

        /// <summary>Assembles the entries of every owner group.</summary>
        /// <returns>The full entry list.</returns>
        private static List<LeakCase> Build()
        {
            var list = new List<LeakCase>();
            AddNp(list);
            AddLinalg(list);
            AddRandom(list);
            AddObjects(list);
            AddNDArray(list);
            AddOperators(list);
            AddMaskedModule(list);
            AddMaskedArray(list);
            return list;
        }

        /// <summary>Appends one entry.</summary>
        /// <param name="list">The entry list.</param>
        /// <param name="api">The covered inventory id.</param>
        /// <param name="label">Variant label.</param>
        /// <param name="run">The invocation.</param>
        /// <param name="backend">Whether the member needs a matrix backend.</param>
        private static void E(List<LeakCase> list, string api, string label, Func<LeakFixture, object> run, bool backend = false)
            => list.Add(new LeakCase(api, label, run, backend));

        /// <summary>Appends an entry for a member that ALWAYS throws: measured as an error path (see
        /// <see cref="LeakCase.Throws"/>). The label must say why it throws.</summary>
        /// <param name="list">The entry list.</param>
        /// <param name="api">The covered inventory id.</param>
        /// <param name="label">Why the member throws (by-design NumPy parity, unimplemented, broken).</param>
        /// <param name="run">The invocation that throws.</param>
        private static void T(List<LeakCase> list, string api, string label, Func<LeakFixture, object> run)
            => list.Add(new LeakCase(api, label, run, RequiresBackend: false, Throws: true));

        /// <summary>A trivially non-allocating value wrapper so scalar-returning members still yield an object.</summary>
        /// <param name="value">Any value.</param>
        /// <returns>The boxed value (the disposer ignores non-array results).</returns>
        private static object Box<T>(T value) => value;

        // ============================ np (functions no corpus row reaches) ===================

        /// <summary>np.* functions: IO, printing, dtype helpers, callables, histogram, sibling-owned predicates.</summary>
        /// <param name="l">The entry list.</param>
        private static void AddNp(List<LeakCase> l)
        {
            E(l, "np.apply_along_axis", "func1d returns a held 0-d", f => np.apply_along_axis(r => f.Zero, 1, f.M));
            E(l, "np.apply_along_axis", "func1d with args", f => np.apply_along_axis((r, a) => f.Zero, 0, f.M, 1));
            E(l, "np.apply_over_axes", "int axis", f => np.apply_over_axes((x, ax) => np.sum(x, ax, keepdims: true), f.M, 0));
            // The multi-axis form SUPERSEDES the running value; a callback returning a rank-dropping VIEW drives
            // the re-expand path and the superseded-view release without allocating per call — the arrays a
            // callback returns stay the caller's by contract (see np.apply_over_axes' remarks), so an allocating
            // callback would measure the CALLER's intermediates, not the library's.
            E(l, "np.apply_over_axes", "int[] axes, view callback (re-expand path)", f => np.apply_over_axes((x, ax) => x[ax == 0 ? "0" : ":, :, 0"], f.Cube, new[] { 0, 2 }));
            E(l, "np.are_broadcastable", "arrays", f => Box(np.are_broadcastable(f.M, f.V3)));
            E(l, "np.are_broadcastable", "shapes", f => Box(np.are_broadcastable(new long[] { 3, 4 }, new long[] { 4 })));
            E(l, "np.array2string", "float matrix", f => np.array2string(f.M));
            E(l, "np.array2string", "non-contiguous + nan", f => np.array2string(f.MT) + np.array2string(f.NanV));
            E(l, "np.array_equiv", "broadcast-consistent", f => Box(np.array_equiv(f.M, f.M)));
            E(l, "np.asscalar", "0-d", f => Box(np.asscalar<int>(f.Zero)));
            E(l, "np.base_repr", "long", f => np.base_repr(255L, 3, 2));
            E(l, "np.binary_repr", "long width", f => np.binary_repr(-5L, 8));
            E(l, "np.bmat", "NDArray[]", f => np.bmat(new[] { f.M, f.M }));
            E(l, "np.bmat", "NDArray[][]", f => np.bmat(new[] { new[] { f.Sq, f.Sq }, new[] { f.Sq, f.Sq } }));
            E(l, "np.bmat", "string + dict", f => np.bmat("A,B;B,A", new Dictionary<string, NDArray> { ["A"] = f.Sq, ["B"] = f.Sq }));
            E(l, "np.broadcast_shapes", "shapes", f => Box(np.broadcast_shapes(new Shape(3, 1), new Shape(1, 4))));
            E(l, "np.common_type_code", "arrays", f => Box(np.common_type_code(f.M, f.I)));
            E(l, "np.datetime_data", "M8[ns]", f => Box(np.datetime_data(DType.From("M8[ns]"))));
            E(l, "np.find_common_type", "DType[]", f => np.find_common_type(new DType[] { np.float32, np.int64 }, new DType[0]));
            E(l, "np.finfo", "float64", f => np.finfo(np.float64));
            E(l, "np.finfo", "array", f => np.finfo(f.M));
            E(l, "np.flat", "boxed FlatIterator over a view", f =>
            {
                var it = np.flat(f.MT);
                double s = 0;
                foreach (var v in it)
                    s += (double)v;
                return Box(s);
            });
            E(l, "np.flat", "typed by-ref FlatRefIter<T>", f =>
            {
                double s = 0;
                foreach (ref double x in np.flat<double>(f.MT))
                    s += x;
                return Box(s);
            });
            E(l, "np.format_float_positional", "double", f => np.format_float_positional(3.14159, precision: 3));
            E(l, "np.format_float_scientific", "double", f => np.format_float_scientific(314.159, precision: 3));
            E(l, "np.frompyfunc", "Call", f => np.frompyfunc((Func<double, double>)(x => x * 2), 1, 1).Call(f.V));
            E(l, "np.get_printoptions", "read", f => np.get_printoptions());
            E(l, "np.getbufsize", "read", f => Box(np.getbufsize()));
            E(l, "np.histogram", "int bins", f => (NDArray[])np.histogram(f.V, 10));
            E(l, "np.histogram", "int bins density+range", f => (NDArray[])np.histogram(f.V, 8, (-2.0, 10.0), density: true));
            E(l, "np.histogram", "int bins weights", f => (NDArray[])np.histogram(f.V, 10, weights: f.V));
            E(l, "np.histogram", "estimator bins", f => (NDArray[])np.histogram(f.V, "auto"));
            E(l, "np.histogram", "explicit edges", f => (NDArray[])np.histogram(f.V, new double[] { -3, 0, 5, 20 }));
            E(l, "np.histogram_bin_edges", "int bins", f => np.histogram_bin_edges(f.V, 10));
            E(l, "np.histogram_bin_edges", "estimator bins", f => np.histogram_bin_edges(f.V, "fd"));
            E(l, "np.histogramdd", "int bins", f =>
            {
                var (h, edges) = np.histogramdd(f.Cube.reshape(12, 2), 4);
                return new object[] { h, edges };
            });
            E(l, "np.histogram2d", "int bins", f =>
            {
                var (h, xe, ye) = np.histogram2d(f.V["0:25"], f.V["25:50"], 5);
                return new object[] { h, xe, ye };
            });
            E(l, "np.iinfo", "int32", f => np.iinfo(np.int32));
            E(l, "np.iinfo", "array", f => np.iinfo(f.I));
            E(l, "np.isneginf", "nan/inf lanes", f => np.isneginf(f.NanV));
            E(l, "np.isposinf", "nan/inf lanes", f => np.isposinf(f.NanV));
            E(l, "np.issctype", "type", f => Box(np.issctype(typeof(double))));
            E(l, "np.issubsctype", "codes", f => Box(np.issubsctype(NPTypeCode.Int32, NPTypeCode.Int64)));
            E(l, "np.load", "npy bytes", f => np.load(f.NpyBytes));
            E(l, "np.load", "npz bytes", f => np.load(f.NpzBytes));
            E(l, "np.load", "npy path", f => np.load(f.NpyPath));
            E(l, "np.load_npy", "bytes", f => np.load_npy(f.NpyBytes));
            E(l, "np.load_npy", "stream", f =>
            {
                using var s = new MemoryStream(f.NpyBytes);
                return np.load_npy(s);
            });
            E(l, "np.load_npz", "bytes + read both members", f =>
            {
                using var z = np.load_npz(f.NpzBytes);
                return new object[] { z["m"], z["v"] };
            });
            E(l, "np.maximum_sctype", "int32", f => np.maximum_sctype(NPTypeCode.Int32));
            E(l, "np.may_share_memory", "view pair", f => Box(np.may_share_memory(f.M, f.MT)));
            E(l, "np.multithreading", "idempotent off", f =>
            {
                np.multithreading(false);
                return null;
            });
            E(l, "np.nan_to_num", "copy", f => np.nan_to_num(f.NanV));
            E(l, "np.nan_to_num", "custom fills", f => np.nan_to_num(f.NanV, true, 0.0, 9.0, -9.0));
            E(l, "np.ndarray", "shape + dtype", f => np.ndarray(new Shape(2, 3), np.float64));
            E(l, "np.nditer_chunks", "typed chunks over a view", f =>
            {
                double s = 0;
                foreach (Span<double> c in np.nditer_chunks<double>(f.M))
                    foreach (var x in c)
                        s += x;
                return Box(s);
            });
            E(l, "np.polyfit", "deg 2, full outputs", f =>
            {
                var r = np.polyfit(f.V["0:10"], f.V["10:20"], 2, full: true);
                return new object[] { r.coeffs, r.residuals, r.rank, r.singular_values, r.covariance };
            }, backend: true);
            E(l, "np.printoptions", "context", f =>
            {
                using (np.printoptions(precision: 4))
                    return f.M.ToString();
            });
            E(l, "np.roots", "quadratic", f => np.roots(f.V3), backend: true);
            E(l, "np.save", "to bytes", f => np.save(f.M));
            E(l, "np.save", "to stream", f =>
            {
                using var s = new MemoryStream();
                np.save(s, f.MT);
                return Box(s.Length);
            });
            E(l, "np.save", "to path", f =>
            {
                np.save(Path.Combine(f.Dir, "save.npy"), f.M);
                return null;
            });
            E(l, "np.save_version", "v2.0 stream", f =>
            {
                using var s = new MemoryStream();
                np.save_version(s, f.M, NumSharp.IO.NpyFormat.FormatVersion.V2_0);
                return Box(s.Length);
            });
            E(l, "np.savez", "positional to bytes", f => np.savez(f.M, f.V));
            E(l, "np.savez", "keyword to stream", f =>
            {
                using var s = new MemoryStream();
                np.savez(s, new Dictionary<string, NDArray> { ["m"] = f.M });
                return Box(s.Length);
            });
            E(l, "np.savez_compressed", "positional to bytes", f => np.savez_compressed(f.M, f.V));
            E(l, "np.sctype2char", "double", f => Box(np.sctype2char(NPTypeCode.Double)));
            E(l, "np.set_printoptions", "idempotent default precision", f =>
            {
                np.set_printoptions(precision: 8);
                return null;
            });
            E(l, "np.setbufsize", "idempotent default", f => Box(np.setbufsize(np.getbufsize())));
            E(l, "np.shares_memory", "view pair", f => Box(np.shares_memory(f.M, f.MT)));
            E(l, "np.typename", "code", f => np.typename("d"));
            E(l, "np.vectorize", "element-wise Func<T,TR>", f => np.vectorize<double, double>(x => x * 2)(f.V));
            // Signature mode with a delegate returning its core VIEW: drives the broadcast, per-slice core views,
            // output allocation and slot writes without a per-slice delegate allocation (delegate results stay the
            // caller's by contract — see Vectorized's remarks).
            E(l, "np.vectorize", "Vectorized signature mode, core-view delegate", f => np.vectorize((Func<NDArray, NDArray>)(x => x), "(n)->(n)").Call(f.M));
        }

        // ============================ np.linalg (members no corpus key calls) ================

        /// <summary>The Array-API linalg forms (no corpus key calls them) and a direct entry per
        /// backend-only factorisation (also measured by the corpus backend pass).</summary>
        /// <param name="l">The entry list.</param>
        private static void AddLinalg(List<LeakCase> l)
        {
            E(l, "np.linalg.cross", "3-vectors", f => np.linalg.cross(f.V3, f.Sq["0"]));
            E(l, "np.linalg.diagonal", "last two axes", f => np.linalg.diagonal(f.Cube));
            E(l, "np.linalg.matmul", "matrix", f => np.linalg.matmul(f.M, f.MT));
            E(l, "np.linalg.matrix_transpose", "view", f => np.linalg.matrix_transpose(f.Cube));
            E(l, "np.linalg.outer", "1-D", f => np.linalg.outer(f.V3, f.V3));
            E(l, "np.linalg.tensordot", "int axes", f => np.linalg.tensordot(f.M, f.MT, 1));
            E(l, "np.linalg.tensordot", "axis lists", f => np.linalg.tensordot(f.Cube, f.Cube, new[] { 0, 1 }, new[] { 0, 1 }));
            E(l, "np.linalg.trace", "last two axes", f => np.linalg.trace(f.Cube));
            E(l, "np.linalg.vecdot", "last axis", f => np.linalg.vecdot(f.M, f.M));
            E(l, "np.linalg.cholesky", "SPD", f => np.linalg.cholesky(f.Sq), backend: true);
            E(l, "np.linalg.cond", "default", f => np.linalg.cond(f.Sq), backend: true);
            E(l, "np.linalg.eig", "general", f => np.linalg.eig(f.Sq), backend: true);
            E(l, "np.linalg.eigh", "symmetric", f => np.linalg.eigh(f.Sq), backend: true);
            E(l, "np.linalg.eigvals", "general", f => np.linalg.eigvals(f.Sq), backend: true);
            E(l, "np.linalg.eigvalsh", "symmetric", f => np.linalg.eigvalsh(f.Sq), backend: true);
            E(l, "np.linalg.lstsq", "over-determined", f => np.linalg.lstsq(f.M.T, f.V["0:4"]), backend: true);
            E(l, "np.linalg.matrix_rank", "full", f => np.linalg.matrix_rank(f.M), backend: true);
            E(l, "np.linalg.norm", "nuclear", f => np.linalg.norm(f.M, "nuc"), backend: true);
            E(l, "np.linalg.pinv", "rectangular", f => np.linalg.pinv(f.M), backend: true);
            E(l, "np.linalg.qr", "reduced", f => np.linalg.qr(f.M), backend: true);
            E(l, "np.linalg.svd", "full", f => np.linalg.svd(f.M), backend: true);
            E(l, "np.linalg.svdvals", "values", f => np.linalg.svdvals(f.M), backend: true);
        }

        // ============================ np.random (samplers outside the stream corpus) =========

        /// <summary>The samplers carved out of the byte-parity stream corpus (their VALUES are pinned
        /// under OpenBugs; their allocation paths are still leak-gated here) and the non-stream
        /// factories.</summary>
        /// <param name="l">The entry list.</param>
        private static void AddRandom(List<LeakCase> l)
        {
            E(l, "np.random.RandomState", "seeded", f => Box(np.random.RandomState(7)));
            E(l, "np.random.RandomState", "default", f => Box(np.random.RandomState()));
            E(l, "np.random.bernoulli", "sized", f => np.random.bernoulli(0.3, new Shape(20)));
            E(l, "np.random.binomial", "sized", f => np.random.binomial(10, 0.4, new Shape(20)));
            E(l, "np.random.default_rng", "seeded", f => Box(np.random.default_rng(99)));
            E(l, "np.random.f", "sized", f => np.random.f(3.0, 5.0, new Shape(20)));
            E(l, "np.random.multinomial", "sized", f => np.random.multinomial(10, new[] { 0.2, 0.3, 0.5 }, 4));
            E(l, "np.random.multivariate_normal", "NDArray mean/cov", f => np.random.multivariate_normal(f.V3, f.Sq, new Shape(5)));
            E(l, "np.random.negative_binomial", "sized", f => np.random.negative_binomial(5.0, 0.4, new Shape(20)));
            E(l, "np.random.pareto", "sized", f => np.random.pareto(2.5, new Shape(20)));
            E(l, "np.random.standard_cauchy", "sized", f => np.random.standard_cauchy(new Shape(20)));
            E(l, "np.random.bytes", "legacy", f => f.Rs.bytes(16));
            E(l, "np.random.random_integers", "legacy", f => f.Rs.random_integers(1, 6, new Shape(10)));
        }

        // ============================ object surfaces (object_surfaces.py owners) ===========

        /// <summary>
        ///     Methods of the NumPy object types the coverage artifact maps to CLR owners
        ///     (<c>coverage/object_surfaces.py</c>): Generator, SeedSequence, MT19937, the iterator
        ///     objects, DType/finfo/iinfo, NDArrayFlags, poly1d and NpzFile. (Their properties are
        ///     measured by the reflective read gate.)
        /// </summary>
        /// <param name="l">The entry list.</param>
        private static void AddObjects(List<LeakCase> l)
        {
            // ---- Generator (the stream methods ride grnd; these two do not) ----
            E(l, "Generator.ToString", "repr", f => f.Rng.ToString());
            E(l, "Generator.permuted", "axis copy", f => f.Rng.permuted(f.M, 1));

            // ---- SeedSequence / MT19937 (managed state only — must read zero pool traffic) ----
            E(l, "SeedSequence.generate_state", "uint32 words", f => new SeedSequence(5).generate_state(4));
            E(l, "MT19937.Clone", "copy", f => new MT19937(3).Clone());
            E(l, "MT19937.Next", "bounded", f => Box(new MT19937(3).Next(10) + new MT19937(3).Next(2, 9)));
            E(l, "MT19937.NextBytes", "buffer", f =>
            {
                var b = new byte[8];
                new MT19937(3).NextBytes(b);
                return b;
            });
            E(l, "MT19937.NextDouble", "draw", f => Box(new MT19937(3).NextDouble()));
            E(l, "MT19937.NextInt", "draw", f => Box(new MT19937(3).NextInt()));
            E(l, "MT19937.NextLong", "bounded", f => Box(new MT19937(3).NextLong(1L, 100L)));
            E(l, "MT19937.NextLongNumPy", "range", f => Box(new MT19937(3).NextLongNumPy(0L, 50L)));
            E(l, "MT19937.NextUInt32", "draw", f => Box(new MT19937(3).NextUInt32()));
            E(l, "MT19937.Seed", "reseed", f =>
            {
                var g = new MT19937(3);
                g.Seed(11u);
                return Box(g.NextUInt32());
            });
            E(l, "MT19937.SeedByArray", "key", f =>
            {
                var g = new MT19937(3);
                g.SeedByArray(new uint[] { 1, 2, 3 });
                return Box(g.NextUInt32());
            });
            E(l, "MT19937.SetState", "key+pos", f =>
            {
                var g = new MT19937(3);
                g.SetState(new uint[624], 624);
                return Box(g.NextUInt32());
            });

            // ---- np.nditer's NDIterator ----
            E(l, "NDIterator.iternext", "walk", f =>
            {
                using var it = np.nditer(f.M);
                long n = 0;
                while (!it.finished)
                {
                    n++;
                    it.iternext();
                }
                return Box(n);
            });
            E(l, "NDIterator.MoveNext", "manual", f =>
            {
                using var it = np.nditer(f.MT);
                long n = 0;
                while (it.MoveNext())
                    n++;
                return Box(n);
            });
            E(l, "NDIterator.GetEnumerator", "foreach", f =>
            {
                using var it = np.nditer(f.M);
                long n = 0;
                foreach (var _ in it)
                    n++;
                return Box(n);
            });
            E(l, "NDIterator.Dispose", "construct + dispose", f =>
            {
                np.nditer(f.M).Dispose();
                return null;
            });
            E(l, "NDIterator.close", "close", f =>
            {
                var it = np.nditer(f.M);
                it.close();
                return null;
            });
            E(l, "NDIterator.copy", "mid-walk copy", f =>
            {
                using var it = np.nditer(f.M);
                it.iternext();
                using var cp = it.copy();
                return Box(cp.iterindex);
            });
            E(l, "NDIterator.debug_print", "to console", f =>
            {
                using var it = np.nditer(f.I);
                var old = Console.Out;
                Console.SetOut(TextWriter.Null);
                try
                {
                    it.debug_print();
                }
                finally
                {
                    Console.SetOut(old);
                }
                return null;
            });
            E(l, "NDIterator.enable_external_loop", "delayed external loop", f =>
            {
                using var it = np.nditer(f.M, new[] { "buffered", "delay_bufalloc" });
                it.enable_external_loop();
                return Box(it.itersize);
            });
            E(l, "NDIterator.remove_axis", "multi_index then remove", f =>
            {
                using var it = np.nditer(f.M, new[] { "multi_index" });
                it.remove_axis(1);
                it.remove_multi_index();
                return Box(it.ndim);
            });
            E(l, "NDIterator.remove_multi_index", "drop tracking", f =>
            {
                using var it = np.nditer(f.M, new[] { "multi_index" });
                it.remove_multi_index();
                return Box(it.has_multi_index);
            });
            E(l, "NDIterator.reset", "walk + reset", f =>
            {
                using var it = np.nditer(f.M);
                it.iternext();
                it.reset();
                return Box(it.iterindex);
            });
            E(l, "NDIterator.Item", "0-d operand view per step of a strided walk", f =>
            {
                // it[i] builds a FRESH view object per call — the caller's to release — so the per-element
                // `it[0]` loop NumPy code ports verbatim must balance view by view, not only at the end.
                using var it = np.nditer(f.MT);
                double s = 0;
                while (!it.finished)
                {
                    using (var v = it[0])
                        s += (double)v;
                    it.iternext();
                }
                return Box(s);
            });
            E(l, "NDIterator.Item", "external-loop chunk view", f =>
            {
                // Under external_loop the view spans the inner loop (a 1-D chunk), built by the strided route.
                using var it = np.nditer(f.M, new[] { "external_loop" });
                long n = 0;
                while (!it.finished)
                {
                    using (var chunk = it[0])
                        n += chunk.size;
                    it.iternext();
                }
                return Box(n);
            });

            // ---- flatiter / ndindex / ndenumerate / broadcast ----
            E(l, "FlatIterator.AsTyped", "typed", f =>
            {
                double s = 0;
                foreach (ref double x in f.MT.flatiter.AsTyped<double>())
                    s += x;
                return Box(s);
            });
            E(l, "FlatIterator.GetEnumerator", "boxed walk", f =>
            {
                double s = 0;
                foreach (var v in f.MT.flatiter)
                    s += (double)v;
                return Box(s);
            });
            E(l, "FlatIterator.copy", "C-order copy of a view", f => f.MT.flatiter.copy());
            E(l, "FlatIterator.next", "advance", f => f.MT.flatiter.next());
            E(l, "FlatIterator.Item", "scalar get/set through a transposed owned copy", f =>
            {
                // Scalar access maps a flat index through the base's strides with no NDArray built, so the
                // round trip may cost only the owned copy and its transposed view (both released here).
                using var c = f.M.copy();
                using var t = c.T;
                var it = t.flatiter;
                object v = it[5];
                it[5] = v;     // same-dtype write-back
                it[-1] = 7;    // a weak int into float64: NumPy's cross-dtype scalar cast path
                return Box(v);
            });
            E(l, "FlatIterator.Item", "weak scalars into an int32 owned copy (bounds-check + cast route)", f =>
            {
                // A mismatched scalar into an INTEGER base takes the NumPy-exact cast route (a 1-element
                // source cast through astype) after the weak-scalar bounds check — its temps must not outlive
                // the assignment.
                using var c = f.I.copy();
                var it = c.flatiter;
                it[3] = 7L;     // int64 into int32
                it[4] = 2.9;    // a float truncates toward zero
                return Box(it[3]);
            });
            E(l, "FlatIterator.Item", "slice + int[] + long[] + NDArray gets", f =>
            {
                // Every fancy/slice get gathers into a FRESH 1-D array (returned for disposal); the slice
                // form resolves its positions through arange(size)[range] internally.
                var it = f.MT.flatiter;
                return new object[] { it["1:7:2"], it[new[] { 0, 5, 11 }], it[new long[] { 2, -1 }], it[f.Idx] };
            });
            E(l, "FlatIterator.Item", "slice + int[] + long[] + NDArray sets on an owned copy", f =>
            {
                // Every fancy/slice set casts the values to the base dtype before scattering; the array is
                // the entry's own copy so the fixture is never written.
                using var c = f.M.copy();
                var it = c.flatiter;
                it["0:3"] = f.V3;
                it[new[] { 3, 5, 7 }] = f.V3;
                it[new long[] { 8, 9, 10 }] = f.V3;
                it[f.Idx] = f.V3;
                return null;
            });
            E(l, "NDIndex.AsSpans", "span walk", f =>
            {
                long n = 0;
                foreach (var ix in np.ndindex(3, 4).AsSpans())
                    n += ix.Length;
                return Box(n);
            });
            E(l, "NDIndex.GetEnumerator", "walk", f =>
            {
                long n = 0;
                foreach (var ix in np.ndindex(3, 4))
                    n += ix.Length;
                return Box(n);
            });
            E(l, "NDIndex.MoveNext", "manual", f =>
            {
                var it = np.ndindex(2, 2);
                long n = 0;
                while (it.MoveNext())
                    n++;
                return Box(n);
            });
            E(l, "NDIndex.Dispose", "dispose", f =>
            {
                np.ndindex(2, 2).Dispose();
                return null;
            });
            E(l, "NDEnumerate.GetEnumerator", "walk a view", f =>
            {
                double s = 0;
                foreach (var (_, v) in np.ndenumerate(f.MT))
                    s += (double)v;
                return Box(s);
            });
            E(l, "NDEnumerate.MoveNext", "manual", f =>
            {
                var it = np.ndenumerate(f.M);
                long n = 0;
                while (it.MoveNext())
                    n++;
                return Box(n);
            });
            E(l, "NDEnumerate.Dispose", "dispose", f =>
            {
                np.ndenumerate(f.M).Dispose();
                return null;
            });
            E(l, "Broadcast.GetEnumerator", "walk", f =>
            {
                using var b = np.broadcast(f.M, f.V["0:4"]);
                long n = 0;
                foreach (var _ in b)
                    n++;
                return Box(n);
            });
            E(l, "Broadcast.MoveNext", "manual", f =>
            {
                using var b = np.broadcast(f.M, f.V["0:4"]);
                long n = 0;
                while (b.MoveNext())
                    n++;
                return Box(n);
            });
            E(l, "Broadcast.reset", "walk + reset", f =>
            {
                using var b = np.broadcast(f.M, f.V["0:4"]);
                b.MoveNext();
                b.reset();
                return Box(b.index);
            });
            E(l, "Broadcast.Dispose", "dispose", f =>
            {
                np.broadcast(f.M, f.M).Dispose();
                return null;
            });

            // ---- DType / finfo / iinfo / NDArrayFlags ----
            E(l, "DType.Equals", "string coercion", f => Box(np.float64.Equals("f8") && np.float64.Equals(np.float64)));
            E(l, "DType.From", "string", f => DType.From(">i4"));
            E(l, "DType.GetHashCode", "hash", f => Box(np.float64.GetHashCode()));
            E(l, "DType.GetTypeCode", "code", f => Box(np.float64.GetTypeCode()));
            E(l, "DType.ToString", "str + repr", f => np.float64.ToString() + np.float64.ToString(true));
            E(l, "DType.newbyteorder", "swap", f => np.float64.newbyteorder('>'));
            // DType's operators are pure descriptor logic (registry lookups, the safe-cast table) and must
            // read zero pool traffic — an allocation here would be one per dtype comparison, everywhere.
            E(l, "DType.op_Implicit", "Type / NPTypeCode / NPTypeCode? / string / NPY_TYPES -> DType, DType -> NPTypeCode", f =>
            {
                DType fromType = typeof(double);
                DType fromCode = NPTypeCode.Int32;
                DType fromNullable = (NPTypeCode?)NPTypeCode.Single;
                DType fromString = ">i4";
                DType fromTypeNum = NPY_TYPES.NPY_BOOL;
                NPTypeCode toCode = np.float64;
                return new object[] { fromType, fromCode, fromNullable, fromString, fromTypeNum, Box(toCode) };
            });
            E(l, "DType.op_Explicit", "DType -> Type", f => (Type)np.float64);
            E(l, "DType.op_Equality", "structural + coercing (Type converts in)", f => Box(np.float64 == np.dtype("f8") && np.int32 == typeof(int)));
            E(l, "DType.op_Inequality", "byte order differs", f => Box(np.int32 != DType.From(">i4")));
            E(l, "DType.op_LessThan", "safe-cast order", f => Box(np.int16 < np.float64));
            E(l, "DType.op_LessThanOrEqual", "safe-cast order", f => Box(np.int16 <= np.int16));
            E(l, "DType.op_GreaterThan", "safe-cast order", f => Box(np.float64 > np.int8));
            E(l, "DType.op_GreaterThanOrEqual", "safe-cast order", f => Box(np.float64 >= np.float64));
            E(l, "finfo.ToString", "repr", f => np.finfo(np.float32).ToString());
            E(l, "iinfo.ToString", "repr", f => np.iinfo(np.int16).ToString());
            E(l, "NDArrayFlags.Equals", "compare", f => Box(f.M.flags.Equals(f.M.flags)));
            E(l, "NDArrayFlags.GetHashCode", "hash", f => Box(f.M.flags.GetHashCode()));
            E(l, "NDArrayFlags.ToString", "repr", f => f.MT.flags.ToString());
            E(l, "NDArrayFlags.op_Equality", "C array vs its transpose", f => Box(f.M.flags == f.MT.flags));
            E(l, "NDArrayFlags.op_Inequality", "C array vs its transpose", f => Box(f.M.flags != f.MT.flags));
            E(l, "NDArrayFlags.Item", "key gets + writeable/aligned sets on an owned copy", f =>
            {
                // SET routes through setflags on the LIVE array; an owned buffer can always be made writeable
                // again, so the entry leaves its copy exactly as it found it.
                using var c = f.M.copy();
                var flags = c.flags;
                bool read = flags["C_CONTIGUOUS"] && flags["W"] && !flags["F"] && flags["OWNDATA"];
                flags["WRITEABLE"] = false;
                flags["W"] = true;
                flags["ALIGNED"] = true;
                return Box(read && flags["WRITEABLE"]);
            });

            // ---- poly1d ----
            E(l, "poly1d.Call", "NDArray", f => f.P.Call(f.V));
            E(l, "poly1d.Call", "scalar", f => Box(f.P.Call(2.5)));
            E(l, "poly1d.Dispose", "construct + dispose", f =>
            {
                new poly1d(f.V3).Dispose();
                return null;
            });
            E(l, "poly1d.Equals", "compare", f => Box(f.P.Equals(f.P)));
            E(l, "poly1d.GetHashCode", "hash", f => Box(f.P.GetHashCode()));
            E(l, "poly1d.ToString", "render", f => f.P.ToString());
            E(l, "poly1d.deriv", "first derivative", f => f.P.deriv());
            E(l, "poly1d.integ", "antiderivative", f => f.P.integ());
            // Every arithmetic operator returns a NEW polynomial (or a (q, r) pair of them) the caller owns —
            // the result disposer releases them through IDisposable / ITuple.
            E(l, "poly1d.op_Implicit", "poly1d -> its coefficient array", f =>
            {
                // The conversion hands out the polynomial's OWN field (NumPy's __array__), never a result to
                // release, so the entry reports the size instead of returning the array.
                NDArray c = f.P;
                return Box(c.size);
            });
            E(l, "poly1d.op_Addition", "poly+poly, poly+array", f => new object[] { f.P + f.P, f.P + f.V3 });
            E(l, "poly1d.op_Subtraction", "poly-poly (all-zero result), poly-array", f => new object[] { f.P - f.P, f.P - f.V3 });
            E(l, "poly1d.op_UnaryNegation", "negate", f => -f.P);
            // Unary plus returns the operand ITSELF — returning it would hand the fixture's polynomial to the
            // result disposer, so the entry reports the identity instead.
            E(l, "poly1d.op_UnaryPlus", "identity", f => Box(ReferenceEquals(+f.P, f.P)));
            E(l, "poly1d.op_Multiply", "poly*poly, poly*array, scalar on both sides", f => new object[] { f.P * f.P, f.P * f.V3, f.P * 2.0, 2.0 * f.P });
            E(l, "poly1d.op_Division", "by scalar, polynomial division by poly and by array", f => new object[] { f.P / 2.0, f.P / f.P, f.P / f.V3 });
            E(l, "poly1d.op_Equality", "value compare", f => Box(f.P == f.P));
            E(l, "poly1d.op_Inequality", "value compare", f => Box(f.P != f.P));
            E(l, "poly1d.Item", "coefficient views + out-of-range zero", f => new object[] { f.P[0], f.P[2], f.P[7] });
            E(l, "poly1d.Item", "set in place + grow, on an owned copy", f =>
            {
                // The copy constructor copies the coefficients, so the fixture polynomial is never written;
                // the held 0-d int32 operand (never a call-site temp) is cast into the float64 coefficients.
                using var p = new poly1d(f.P);
                p[1] = f.Zero;   // in-place write through a coefficient view
                p[4] = f.Zero;   // growing REPLACES the owned array (the old one released)
                return Box(p.order);
            });

            // ---- NpzFile ----
            E(l, "NpzFile.Close", "close", f =>
            {
                var z = np.load_npz(f.NpzBytes);
                z.Close();
                return null;
            });
            E(l, "NpzFile.ContainsKey", "probe", f =>
            {
                using var z = np.load_npz(f.NpzBytes);
                return Box(z.ContainsKey("m"));
            });
            E(l, "NpzFile.Dispose", "dispose after a read", f =>
            {
                var z = np.load_npz(f.NpzBytes);
                var m = z["m"];
                z.Dispose();
                return m;
            });
            E(l, "NpzFile.GetEnumerator", "walk members", f =>
            {
                using var z = np.load_npz(f.NpzBytes);
                var got = new List<NDArray>();
                foreach (var kv in z)
                    got.Add(kv.Value);
                return got;
            });
            E(l, "NpzFile.GetRawBytes", "member bytes", f =>
            {
                using var z = np.load_npz(f.NpzBytes);
                return z.GetRawBytes("v");
            });
            E(l, "NpzFile.IsArray", "probe", f =>
            {
                using var z = np.load_npz(f.NpzBytes);
                return Box(z.IsArray("m"));
            });
            E(l, "NpzFile.Item", "lazy load + cache, both key spellings", f =>
            {
                // The indexer loads on first access and CACHES: the ".npy" spelling must return the SAME
                // array. The cache is a BORROWED memo — whoever first reads a member owns it and the archive
                // only forgets it on Close — so the array is returned (once) for the result disposer.
                using var z = np.load_npz(f.NpzBytes);
                var m = z["m"];
                var again = z["m.npy"];
                return new object[] { m, Box(ReferenceEquals(m, again)) };
            });
            E(l, "NpzFile.ToString", "render", f =>
            {
                using var z = np.load_npz(f.NpzBytes);
                return z.ToString();
            });
            E(l, "NpzFile.TryGetValue", "read", f =>
            {
                using var z = np.load_npz(f.NpzBytes);
                z.TryGetValue("m", out var m);
                return m;
            });
        }
    }
}
