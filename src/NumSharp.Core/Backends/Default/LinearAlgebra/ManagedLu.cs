using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    ///     The pure-managed LU-based <c>np.linalg</c> factorisations — <c>det</c>, <c>slogdet</c>,
    ///     <c>solve</c> and <c>inv</c> — the fallback <see cref="TensorEngine"/> uses when no matrix
    ///     backend is installed. It lets a plain <c>NumSharp.Core</c> (no <c>NumSharp.Interop.OpenBLAS</c>
    ///     reference, no native binary) compute the LU family that used to raise
    ///     <see cref="OpenBlasMissingBackendException"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>allclose, not bit-exact — by construction.</b> NumPy computes this family through
    ///     LAPACK's BLOCKED <c>getrf</c>/<c>gesv</c>, whose Schur-complement update runs through an
    ///     arch-specific <c>gemm</c>; that accumulation order is the very thing NumSharp cannot
    ///     reproduce in managed code (the same reason the matrix PRODUCTS need the bundled binary for
    ///     byte-parity). This kernel is the classic RIGHT-LOOKING UNBLOCKED factorisation
    ///     (<c>getf2</c>) plus a triangular substitution (<c>getrs</c>), so its results agree with
    ///     NumPy to within floating-point tolerance — bit-for-bit for tiny matrices where no blocking
    ///     happens on either side, and a handful of ULP apart as the matrix grows. When a backend IS
    ///     installed it wins the seam (<c>TensorEngine.Blas</c> is consulted first), so this path only
    ///     ever runs in the no-backend configuration.
    ///     </para>
    ///     <para>
    ///     <b>The semantics are NumPy's, ported from <c>umath_linalg.cpp</c>.</b> det/slogdet fold the
    ///     LU factor's diagonal into a running sign and <c>log|det|</c> exactly as NumPy does — det is
    ///     <c>sign·exp(Σ log|Uᵢᵢ|)</c>, NOT the product of the pivots, which is why <c>det([[5.]])</c>
    ///     comes back <c>4.999999999999999</c> and not <c>5.0</c>. A singular operand (an exact zero
    ///     pivot) short-circuits to <c>(sign 0, logdet -inf)</c> / <c>det 0</c> for det/slogdet and
    ///     raises <c>LinAlgError("Singular matrix")</c> for solve/inv. An empty (0×0) matrix is the
    ///     empty product (sign 1, logdet 0, det 1). Partial pivoting is LAPACK's: the largest-magnitude
    ///     entry in the column (<c>|·|</c> for real, <c>|re|+|im|</c> — <c>dcabs1</c> — for complex,
    ///     first on ties), so the pivot SEQUENCE matches and only the eliminated values drift.
    ///     </para>
    ///     <para>
    ///     <b>Structure mirrors the OpenBLAS port</b> (<c>OpenBlasEngine.Lapack.cs</c>): the dtype is
    ///     dispatched ONCE at the entry into a generic <c>&lt;T, TOps&gt;</c> core the JIT
    ///     monomorphises, where <c>TOps</c> is a struct supplying the per-dtype primitives (the
    ///     sanctioned "Type Switch Pattern", not a per-dtype kernel). Only <c>double</c> and
    ///     <c>Complex</c> instantiate; <c>float32</c> upcasts to double, factorises, and casts back —
    ///     exactly NumPy's "lite" always-double rule. Each 2-D matrix is copied through its own strides
    ///     into a contiguous row-major buffer, so every layout (C/F/strided/transposed/reversed/
    ///     broadcast/sliced) works; the operand is never mutated. Stacks loop element by element with
    ///     the scratch buffers hoisted once per call, and results are allocated <c>fillZeros:false</c>
    ///     because every cell is written. The hot inner loop — the rank-1 elimination and the
    ///     substitution's row updates — is a contiguous <c>axpy</c>, SIMD-vectorised for the double
    ///     path via <see cref="Vector{T}"/>.
    ///     </para>
    /// </remarks>
    internal static unsafe class ManagedLu
    {
        // ----------------------------------------------------------------------------------------
        //  Per-dtype primitives (struct type argument → monomorphic JIT, like ILapackType<T>).
        // ----------------------------------------------------------------------------------------

        internal interface ILuOps<T> where T : unmanaged
        {
            T One { get; }
            T Zero { get; }
            T MinusOne { get; }

            /// <summary>An exact zero — a zero pivot means the matrix is singular.</summary>
            bool IsZero(T v);

            /// <summary>The magnitude partial pivoting selects on: <c>|v|</c> real, <c>|re|+|im|</c> complex.</summary>
            double PivotMag(T v);

            /// <summary>
            ///     <c>a / b</c>. NumSharp's managed LU divides directly (both the pivot-column
            ///     multipliers and the back-substitution scale) rather than multiplying by a hoisted
            ///     reciprocal — a deliberate departure from LAPACK's <c>DSCAL(1/pivot)</c> speed trick.
            ///     Direct division keeps the result a ULP closer to NumPy, lets a truly-singular pair
            ///     like <c>[[1,2],[2,4]]</c> cancel to an EXACT zero pivot (so it raises like NumPy
            ///     rather than returning garbage), and cannot turn a denormal pivot into a spurious
            ///     <c>inf</c> the way <c>1/denormal</c> can. It does NOT close every singular case — a
            ///     matrix whose zero pivot only appears after several eliminations (the textbook
            ///     <c>[[1,2,3],[4,5,6],[7,8,9]]</c>) still lands on ~1e-16 in unblocked arithmetic where
            ///     blocked LAPACK cancels to exact zero; that is the documented divergence. The cost —
            ///     O(m²) divisions against the O(m³) rank-1 update — is negligible.
            /// </summary>
            T Div(T a, T b);

            /// <summary><c>y[i] -= alpha·x[i]</c> for <c>i in [0,n)</c> — the rank-1 / substitution update.</summary>
            void AxpyNeg(T* y, T* x, T alpha, long n);

            /// <summary><c>y[i] /= d</c> for <c>i in [0,n)</c> — the back-substitution row scale.</summary>
            void DivRow(T* y, T d, long n);

            /// <summary>Folds one diagonal entry into the running sign and <c>log|det|</c>.</summary>
            void FoldStep(T diag, ref T sign, ref double logacc);

            /// <summary><c>sign·exp(logabsdet)</c>.</summary>
            T DetFromSign(T sign, double logabsdet);
        }

        internal readonly struct DoubleOps : ILuOps<double>
        {
            public double One => 1.0;
            public double Zero => 0.0;
            public double MinusOne => -1.0;
            public bool IsZero(double v) => v == 0.0;
            public double PivotMag(double v) => Math.Abs(v);
            public double Div(double a, double b) => a / b;

            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public void AxpyNeg(double* y, double* x, double alpha, long n)
            {
                long i = 0;
                int w = Vector<double>.Count;
                var va = new Vector<double>(alpha);
                for (; i + w <= n; i += w)
                {
                    var vy = Unsafe.ReadUnaligned<Vector<double>>(y + i);
                    var vx = Unsafe.ReadUnaligned<Vector<double>>(x + i);
                    Unsafe.WriteUnaligned(y + i, vy - va * vx);
                }

                for (; i < n; i++)
                    y[i] -= alpha * x[i];
            }

            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public void DivRow(double* y, double d, long n)
            {
                long i = 0;
                int w = Vector<double>.Count;
                var vd = new Vector<double>(d);
                for (; i + w <= n; i += w)
                {
                    var vy = Unsafe.ReadUnaligned<Vector<double>>(y + i);
                    Unsafe.WriteUnaligned(y + i, vy / vd);
                }

                for (; i < n; i++)
                    y[i] /= d;
            }

            public void FoldStep(double d, ref double sign, ref double logacc)
            {
                if (d < 0.0)
                {
                    sign = -sign;
                    d = -d;
                }

                logacc += Math.Log(d);
            }

            public double DetFromSign(double sign, double logabsdet) => sign * Math.Exp(logabsdet);
        }

        internal readonly struct ComplexOps : ILuOps<Complex>
        {
            public Complex One => Complex.One;
            public Complex Zero => Complex.Zero;
            public Complex MinusOne => new Complex(-1.0, 0.0);
            public bool IsZero(Complex v) => v.Real == 0.0 && v.Imaginary == 0.0;

            // LAPACK's izamax pivots on dcabs1 = |re| + |im|, NOT the Euclidean magnitude — reproduced
            // so the pivot SEQUENCE matches NumPy's.
            public double PivotMag(Complex v) => Math.Abs(v.Real) + Math.Abs(v.Imaginary);

            public Complex Div(Complex a, Complex b) => a / b;

            public void AxpyNeg(Complex* y, Complex* x, Complex alpha, long n)
            {
                double ar = alpha.Real, ai = alpha.Imaginary;
                for (long i = 0; i < n; i++)
                {
                    double xr = x[i].Real, xi = x[i].Imaginary;
                    // The textbook complex product, spelled out so the rounding is fixed rather than an
                    // operator's formula choice; y -= alpha·x.
                    double pr = ar * xr - ai * xi;
                    double pi = ar * xi + ai * xr;
                    y[i] = new Complex(y[i].Real - pr, y[i].Imaginary - pi);
                }
            }

            public void DivRow(Complex* y, Complex d, long n)
            {
                for (long i = 0; i < n; i++)
                    y[i] /= d;
            }

            public void FoldStep(Complex z, ref Complex sign, ref double logacc)
            {
                // NumPy: abs = npy_cabs(z); sign *= z/abs; logdet += log(abs). npy_cabs is a straight
                // hypot; the scaled form below matches it to ~1 ULP (allclose is the contract here).
                double abs = Hypot(z.Real, z.Imaginary);
                double er = z.Real / abs, ei = z.Imaginary / abs;
                double sr = sign.Real, si = sign.Imaginary;
                sign = new Complex(sr * er - si * ei, sr * ei + si * er);
                logacc += Math.Log(abs);
            }

            public Complex DetFromSign(Complex sign, double logabsdet)
            {
                double e = Math.Exp(logabsdet);
                return new Complex(sign.Real * e, sign.Imaginary * e);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double Hypot(double x, double y)
        {
            x = Math.Abs(x);
            y = Math.Abs(y);
            if (x < y)
            {
                double t = x;
                x = y;
                y = t;
            }

            if (x == 0.0)
                return 0.0;

            double r = y / x;
            return x * Math.Sqrt(1.0 + r * r);
        }

        // ----------------------------------------------------------------------------------------
        //  Entry points — dtype dispatched ONCE, then a monomorphic generic core. `a`/`b` reach here
        //  already at the common dtype (linalg's CommonType has run), so only Single/Double/Complex
        //  occur, and Single is computed in double exactly as NumPy's lite wrapper does.
        // ----------------------------------------------------------------------------------------

        internal static NDArray Det(NDArray a)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Double:
                    return DetSlogdetCore<double, DoubleOps>(a, wantDet: true).det;
                case NPTypeCode.Complex:
                    return DetSlogdetCore<Complex, ComplexOps>(a, wantDet: true).det;
                case NPTypeCode.Single:
                    return DetSlogdetCore<double, DoubleOps>(a.astype(NPTypeCode.Double), wantDet: true)
                        .det.astype(NPTypeCode.Single);
                default:
                    throw Unreachable(a);
            }
        }

        internal static (NDArray sign, NDArray logabsdet) Slogdet(NDArray a)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Double:
                {
                    var (_, sign, logdet) = DetSlogdetCore<double, DoubleOps>(a, wantDet: false);
                    return (sign, logdet);
                }
                case NPTypeCode.Complex:
                {
                    // sign is complex128, logabsdet is the real basetype (float64) — NumPy's real_t.
                    var (_, sign, logdet) = DetSlogdetCore<Complex, ComplexOps>(a, wantDet: false);
                    return (sign, logdet);
                }
                case NPTypeCode.Single:
                {
                    var (_, sign, logdet) = DetSlogdetCore<double, DoubleOps>(a.astype(NPTypeCode.Double), wantDet: false);
                    return (sign.astype(NPTypeCode.Single), logdet.astype(NPTypeCode.Single));
                }
                default:
                    throw Unreachable(a);
            }
        }

        internal static NDArray Inv(NDArray a)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Double:
                    return InvCore<double, DoubleOps>(a);
                case NPTypeCode.Complex:
                    return InvCore<Complex, ComplexOps>(a);
                case NPTypeCode.Single:
                    return InvCore<double, DoubleOps>(a.astype(NPTypeCode.Double)).astype(NPTypeCode.Single);
                default:
                    throw Unreachable(a);
            }
        }

        internal static NDArray Solve(NDArray a, NDArray b, bool oneDimensionalRhs)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Double:
                    return SolveCore<double, DoubleOps>(a, b, oneDimensionalRhs);
                case NPTypeCode.Complex:
                    return SolveCore<Complex, ComplexOps>(a, b, oneDimensionalRhs);
                case NPTypeCode.Single:
                    return SolveCore<double, DoubleOps>(
                            a.astype(NPTypeCode.Double), b.astype(NPTypeCode.Double), oneDimensionalRhs)
                        .astype(NPTypeCode.Single);
                default:
                    throw Unreachable(a);
            }
        }

        private static NotSupportedException Unreachable(NDArray a)
            => new NotSupportedException(
                $"ManagedLu received dtype {a.typecode}, but linalg's CommonType only ever hands it " +
                $"Single/Double/Complex. This is a bug in the dispatch, not a user error.");

        // ----------------------------------------------------------------------------------------
        //  Generic cores (T ∈ {double, Complex}).
        // ----------------------------------------------------------------------------------------

        private static (NDArray det, NDArray sign, NDArray logdet) DetSlogdetCore<T, TOps>(NDArray a, bool wantDet)
            where T : unmanaged
            where TOps : struct, ILuOps<T>
        {
            var ops = default(TOps);
            var shape = a.Shape;
            int nd = shape.NDim;
            long m = shape.dimensions[nd - 1];
            int nb = nd - 2;

            var batchDims = new long[nb];
            long count = 1;
            for (int i = 0; i < nb; i++)
            {
                batchDims[i] = shape.dimensions[i];
                count *= batchDims[i];
            }

            var batchShape = nb == 0 ? Shape.Scalar : new Shape(batchDims);

            NDArray det = null, sign = null, logdet = null;
            T* pdet = null, psign = null;
            double* plog = null;
            if (wantDet)
            {
                det = new NDArray(InfoOf<T>.NPTypeCode, batchShape, fillZeros: false);
                pdet = (T*)det.Address + det.Shape.offset;
            }
            else
            {
                sign = new NDArray(InfoOf<T>.NPTypeCode, batchShape, fillZeros: false);
                logdet = new NDArray(NPTypeCode.Double, batchShape, fillZeros: false);
                psign = (T*)sign.Address + sign.Shape.offset;
                plog = (double*)logdet.Address + logdet.Shape.offset;
            }

            long sr = shape.strides[nd - 2], sc = shape.strides[nd - 1];
            T* pa = (T*)a.Address;
            T* buf = Alloc<T>(m * m);
            int* pv = AllocPivots(m);
            try
            {
                var coord = new long[nb];
                for (long e = 0; e < count; e++)
                {
                    long aoff = shape.offset;
                    for (int i = 0; i < nb; i++)
                        aoff += coord[i] * shape.strides[i];

                    FactorAndFold<T, TOps>(pa + aoff, sr, sc, buf, pv, m, out T s, out double ld);
                    if (wantDet)
                        pdet[e] = ops.DetFromSign(s, ld);
                    else
                    {
                        psign[e] = s;
                        plog[e] = ld;
                    }

                    AdvanceCoord(coord, batchDims, nb);
                }
            }
            finally
            {
                Free(buf);
                NativeMemory.Free(pv);
            }

            return (det, sign, logdet);
        }

        private static NDArray InvCore<T, TOps>(NDArray a)
            where T : unmanaged
            where TOps : struct, ILuOps<T>
        {
            var shape = a.Shape;
            int nd = shape.NDim;
            long m = shape.dimensions[nd - 1];

            var result = new NDArray(InfoOf<T>.NPTypeCode, new Shape((long[])shape.dimensions.Clone()), fillZeros: false);
            if (result.size == 0)
                return result; // (…,0,0) → empty inverse, matching NumPy

            int nb = nd - 2;
            long sr = shape.strides[nd - 2], sc = shape.strides[nd - 1];
            long count = 1;
            for (int i = 0; i < nb; i++)
                count *= shape.dimensions[i];

            long block = m * m;
            long blockBytes = block * Unsafe.SizeOf<T>();
            T* pa = (T*)a.Address;
            T* po = (T*)result.Address + result.Shape.offset;
            T* bufA = Alloc<T>(m * m);
            T* rhs = Alloc<T>(m * m);
            int* pv = AllocPivots(m);
            try
            {
                var coord = new long[nb];
                for (long e = 0; e < count; e++)
                {
                    long aoff = shape.offset;
                    for (int i = 0; i < nb; i++)
                        aoff += coord[i] * shape.strides[i];

                    Linearize(pa + aoff, sr, sc, bufA, m, m);
                    int info = Factor<T, TOps>(bufA, m, pv);
                    if (info != 0)
                        throw new LinAlgError("Singular matrix");

                    IdentityRowMajor<T, TOps>(rhs, m);
                    SubstituteInPlace<T, TOps>(bufA, pv, m, m, rhs);
                    Buffer.MemoryCopy(rhs, po + e * block, blockBytes, blockBytes);
                    AdvanceCoord(coord, shape.dimensions, nb);
                }
            }
            finally
            {
                Free(bufA);
                Free(rhs);
                NativeMemory.Free(pv);
            }

            return result;
        }

        private static NDArray SolveCore<T, TOps>(NDArray a, NDArray b, bool oneDim)
            where T : unmanaged
            where TOps : struct, ILuOps<T>
        {
            var sa = a.Shape;
            var sb = b.Shape;
            int nda = sa.NDim, ndb = sb.NDim;
            long m = sa.dimensions[nda - 1];

            int aNb = nda - 2;
            long aSr = sa.strides[nda - 2], aSc = sa.strides[nda - 1];

            // RHS core: a bare vector (solve1: (m)->(m)) or a matrix column block (solve: (m,k)->(m,k)).
            long k;
            int bNb;
            long bSr, bSc;
            long[] trailing;
            if (oneDim)
            {
                k = 1;
                bNb = 0;
                bSr = sb.strides[0];
                bSc = 0;
                trailing = new[] { m };
            }
            else
            {
                k = sb.dimensions[ndb - 1];
                bNb = ndb - 2;
                bSr = sb.strides[ndb - 2];
                bSc = sb.strides[ndb - 1];
                trailing = new[] { m, k };
            }

            if (!TryBroadcastBatch(sa.dimensions, sa.strides, aNb, sb.dimensions, sb.strides, bNb,
                    out var batchShape, out var aBatchStr, out var bBatchStr))
            {
                throw new IncorrectShapeException(
                    $"np.linalg.solve: the leading dimensions of a {DimsText(sa.dimensions, aNb)} and " +
                    $"b {DimsText(sb.dimensions, bNb)} could not be broadcast together.");
            }

            int nb = batchShape.Length;
            var outDims = new long[nb + trailing.Length];
            long count = 1;
            for (int i = 0; i < nb; i++)
            {
                outDims[i] = batchShape[i];
                count *= batchShape[i];
            }

            for (int i = 0; i < trailing.Length; i++)
                outDims[nb + i] = trailing[i];

            var result = new NDArray(InfoOf<T>.NPTypeCode, new Shape(outDims), fillZeros: false);
            if (result.size == 0)
                return result;

            long block = m * k;
            long blockBytes = block * Unsafe.SizeOf<T>();
            T* pa = (T*)a.Address;
            T* pb = (T*)b.Address;
            T* po = (T*)result.Address + result.Shape.offset;
            T* bufA = Alloc<T>(m * m);
            T* rhs = Alloc<T>(m * k);
            int* pv = AllocPivots(m);
            try
            {
                var coord = new long[nb];
                for (long e = 0; e < count; e++)
                {
                    long aoff = sa.offset, boff = sb.offset;
                    for (int i = 0; i < nb; i++)
                    {
                        aoff += coord[i] * aBatchStr[i];
                        boff += coord[i] * bBatchStr[i];
                    }

                    Linearize(pa + aoff, aSr, aSc, bufA, m, m);
                    Linearize(pb + boff, bSr, bSc, rhs, m, k);
                    int info = Factor<T, TOps>(bufA, m, pv);
                    if (info != 0)
                        throw new LinAlgError("Singular matrix");

                    SubstituteInPlace<T, TOps>(bufA, pv, m, k, rhs);
                    Buffer.MemoryCopy(rhs, po + e * block, blockBytes, blockBytes);
                    AdvanceCoord(coord, batchShape, nb);
                }
            }
            finally
            {
                Free(bufA);
                Free(rhs);
                NativeMemory.Free(pv);
            }

            return result;
        }

        // ----------------------------------------------------------------------------------------
        //  The factorisation and substitution kernels.
        // ----------------------------------------------------------------------------------------

        /// <summary>
        ///     Right-looking unblocked LU with partial pivoting, in place on a row-major <c>m×m</c>
        ///     buffer (leading dimension <c>m</c>). <paramref name="pv"/>[k] is the row swapped with
        ///     row k at step k. Returns the first singular column + 1 (LAPACK's <c>info</c>), or 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static int Factor<T, TOps>(T* buf, long m, int* pv)
            where T : unmanaged
            where TOps : struct, ILuOps<T>
        {
            var ops = default(TOps);
            int info = 0;
            for (long kk = 0; kk < m; kk++)
            {
                // idamax over column kk, rows kk..m-1 (strided by m); the FIRST max wins, as LAPACK's
                // i*amax and this handles a leading NaN the LAPACK way (nothing compares > NaN).
                long p = kk;
                double best = ops.PivotMag(buf[kk * m + kk]);
                for (long i = kk + 1; i < m; i++)
                {
                    double mag = ops.PivotMag(buf[i * m + kk]);
                    if (mag > best)
                    {
                        best = mag;
                        p = i;
                    }
                }

                pv[kk] = (int)p;

                T pivot = buf[p * m + kk];
                if (ops.IsZero(pivot))
                {
                    if (info == 0)
                        info = (int)(kk + 1); // first singular column; LAPACK leaves the column as-is
                    continue;
                }

                if (p != kk)
                    SwapRows(buf, kk, p, m); // the whole row (all m columns) moves — L below moves too

                if (kk < m - 1)
                {
                    // Multipliers: column kk below the diagonal ÷= pivot, by DIRECT division (see ILuOps.Div).
                    for (long i = kk + 1; i < m; i++)
                    {
                        long idx = i * m + kk;
                        buf[idx] = ops.Div(buf[idx], pivot);
                    }

                    // Rank-1 update: each row i>kk has row_kk[kk+1:] axpy'd out of it, scaled by the
                    // multiplier just written. The row segment is contiguous → SIMD for the double path.
                    long seg = m - kk - 1;
                    T* rowk = buf + kk * m + (kk + 1);
                    for (long i = kk + 1; i < m; i++)
                        ops.AxpyNeg(buf + i * m + (kk + 1), rowk, buf[i * m + kk], seg);
                }
            }

            return info;
        }

        /// <summary>
        ///     Solves <c>A·X = B</c> in place (<paramref name="rhs"/> ← X) from the LU factor in
        ///     <paramref name="buf"/> and pivots <paramref name="pv"/>. <paramref name="rhs"/> is a
        ///     row-major <c>m×k</c> block — one column for a vector solve, <c>m</c> for an inverse.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void SubstituteInPlace<T, TOps>(T* buf, int* pv, long m, long k, T* rhs)
            where T : unmanaged
            where TOps : struct, ILuOps<T>
        {
            var ops = default(TOps);

            // Apply the row pivots forward (LAPACK laswp): rhs row i ↔ row pv[i].
            for (long i = 0; i < m; i++)
            {
                long p = pv[i];
                if (p != i)
                    SwapRows(rhs, i, p, k);
            }

            // Forward: L (unit lower) solve. Column-oriented — subtract row j's contribution from every
            // row below it, so each answer accumulates in j-order.
            for (long j = 0; j < m; j++)
            {
                T* rj = rhs + j * k;
                for (long i = j + 1; i < m; i++)
                    ops.AxpyNeg(rhs + i * k, rj, buf[i * m + j], k);
            }

            // Back: U (upper) solve. Row-oriented — subtract the already-solved rows below, then divide
            // by Uᵢᵢ.
            for (long i = m - 1; i >= 0; i--)
            {
                T* ri = rhs + i * k;
                for (long j = i + 1; j < m; j++)
                    ops.AxpyNeg(ri, rhs + j * k, buf[i * m + j], k);

                ops.DivRow(ri, buf[i * m + i], k);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void SwapRows<T>(T* buf, long r1, long r2, long width) where T : unmanaged
        {
            T* a = buf + r1 * width;
            T* b = buf + r2 * width;
            for (long j = 0; j < width; j++)
            {
                T t = a[j];
                a[j] = b[j];
                b[j] = t;
            }
        }

        /// <summary>One matrix's LU + sign/log|det| fold — NumPy's <c>slogdet_single_element</c>.</summary>
        private static void FactorAndFold<T, TOps>(T* src, long sr, long sc, T* buf, int* pv, long m,
            out T sign, out double logdet)
            where T : unmanaged
            where TOps : struct, ILuOps<T>
        {
            var ops = default(TOps);
            if (m == 0)
            {
                sign = ops.One; // empty product
                logdet = 0.0;
                return;
            }

            Linearize(src, sr, sc, buf, m, m);
            int info = Factor<T, TOps>(buf, m, pv);
            if (info != 0)
            {
                sign = ops.Zero; // singular: (sign 0, logdet -inf), which det/slogdet report, never raise
                logdet = double.NegativeInfinity;
                return;
            }

            int changes = 0;
            for (long i = 0; i < m; i++)
                if (pv[i] != i)
                    changes++;

            T s = (changes & 1) == 1 ? ops.MinusOne : ops.One;
            double acc = 0.0;
            for (long i = 0; i < m; i++)
                ops.FoldStep(buf[i * m + i], ref s, ref acc);

            sign = s;
            logdet = acc;
        }

        // ----------------------------------------------------------------------------------------
        //  Small helpers.
        // ----------------------------------------------------------------------------------------

        // AggressiveOptimization on the copy leaves: a linalg op on a small matrix is a handful of
        // calls that finish well inside tiered compilation's ~100 ms promotion delay, so without this
        // the generic instantiation runs at tier-0 for the whole call. Same rationale as the OpenBLAS
        // port's Linearize/Delinearize and NDFloatMath.Simd.
        /// <summary>Copies a strided matrix into a contiguous ROW-major buffer (leading dimension = cols).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Linearize<T>(T* src, long sr, long sc, T* dst, long rows, long cols) where T : unmanaged
        {
            for (long r = 0; r < rows; r++)
            {
                T* d = dst + r * cols;
                T* s = src + r * sr;
                // Advance the strided source pointer instead of recomputing c*sc each element.
                for (long c = 0; c < cols; c++, s += sc)
                    d[c] = *s;
            }
        }

        /// <summary>Writes a row-major identity into <paramref name="buf"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void IdentityRowMajor<T, TOps>(T* buf, long n)
            where T : unmanaged
            where TOps : struct, ILuOps<T>
        {
            long total = n * n;
            for (long i = 0; i < total; i++)
                buf[i] = default;

            T one = default(TOps).One;
            for (long i = 0; i < n; i++)
                buf[i * (n + 1)] = one;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AdvanceCoord(long[] coord, long[] dims, int nb)
        {
            for (int i = nb - 1; i >= 0; i--)
            {
                if (++coord[i] < dims[i])
                    return;
                coord[i] = 0;
            }
        }

        /// <summary>
        ///     Right-aligned broadcast of two operands' BATCH (leading) dimensions, producing the batch
        ///     shape and each operand's per-axis batch stride (0 where it broadcasts). False on an
        ///     incompatible pair.
        /// </summary>
        private static bool TryBroadcastBatch(
            long[] aDims, long[] aStr, int aNb,
            long[] bDims, long[] bStr, int bNb,
            out long[] batchShape, out long[] aBatchStr, out long[] bBatchStr)
        {
            int nb = Math.Max(aNb, bNb);
            batchShape = new long[nb];
            aBatchStr = new long[nb];
            bBatchStr = new long[nb];
            for (int i = 0; i < nb; i++)
            {
                int ai = aNb - nb + i;
                int bi = bNb - nb + i;
                long ad = ai >= 0 ? aDims[ai] : 1;
                long bd = bi >= 0 ? bDims[bi] : 1;

                long dim;
                if (ad == bd) dim = ad;
                else if (ad == 1) dim = bd;
                else if (bd == 1) dim = ad;
                else return false;

                batchShape[i] = dim;
                aBatchStr[i] = (ai >= 0 && ad != 1) ? aStr[ai] : 0;
                bBatchStr[i] = (bi >= 0 && bd != 1) ? bStr[bi] : 0;
            }

            return true;
        }

        private static string DimsText(long[] dims, int nb)
            => "(" + string.Join(",", new ArraySegment<long>(dims, 0, nb)) + ")";

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T* Alloc<T>(long count) where T : unmanaged
            => (T*)NativeMemory.Alloc((nuint)Math.Max(count, 1), (nuint)Unsafe.SizeOf<T>());

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Free<T>(T* p) where T : unmanaged
        {
            if (p != null)
                NativeMemory.Free(p);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int* AllocPivots(long m)
            => (int*)NativeMemory.Alloc((nuint)Math.Max(m, 1), sizeof(int));
    }
}
