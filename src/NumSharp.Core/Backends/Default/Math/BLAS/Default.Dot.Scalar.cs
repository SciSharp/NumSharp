using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    ///     np.dot's scalar branch: NumPy's <c>cblas_matrixproduct</c> (cblasfuncs.c) takes float32 / float64 / complex128
    ///     operands of at most two dims, and when one of them is a scalar by <c>_select_matrix_shape</c> (0-d, <c>(1,)</c> or
    ///     <c>(1, 1)</c>) it does not sum products: one output is the plain product, and more outputs are <c>?axpy</c> into a
    ///     zeroed buffer - scipy-openblas' Haswell kernels (a fused multiply-add into +0) over the first block of a
    ///     contiguous run, its plain C loop (<c>y += a*x</c>) over the rest and over a strided run.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's scalar branch of <c>cblas_matrixproduct</c> for np.dot / np.inner: when both operands have at most two
        ///     dims, their common dtype is float32, float64 or complex128, and one is a scalar by
        ///     <c>_select_matrix_shape</c>, the product computed as NumPy computes it; otherwise nothing.
        /// </summary>
        /// <param name="left">The first operand (NumPy's <c>ap1</c> before its swap).</param>
        /// <param name="right">The second operand (<c>ap2</c>).</param>
        /// <param name="result">Set to the fresh product of the common dtype when the branch applies, else null.</param>
        /// <returns>True when the branch computed the product; false to take the other routes (shapes the branch would
        ///     refuse included, so their errors stay the routes' own).</returns>
        /// <remarks>
        ///     <para>NumPy's order (PyArray_MatrixProduct2, cblas_matrixproduct): both operands converted to the common
        ///     dtype; an operand with a negative stride, or a zero one over more than one element, copied
        ///     (<c>_bad_strides</c>); the scalar moved to <c>ap2</c>; the result zeroed (<c>memset</c>). Then one output
        ///     (<c>l == 1</c>) is <c>*ap2 * *ap1</c>, the plain product (-0.0 kept: <c>[[-0.0]] . [[1.0]]</c> is
        ///     <c>[[-0.0]]</c>); more outputs are <c>?axpy(l, *ap2, ap1, ap1stride, out, 1)</c>, or for a two-dim
        ///     <c>ap1</c> one <c>?axpy</c> per line along its longer axis, so every output is <c>+0 + a*x</c>
        ///     (<c>np.dot(-0.0, [1.0, 1.0])</c> is <c>[+0.0, +0.0]</c>).</para>
        ///     <para>Which arithmetic each output gets (probed 2026-11-05 against NumPy 2.4.2 at one thread): the Haswell
        ///     kernel over the first <c>n &amp; -32</c> (saxpy) / <c>n &amp; -16</c> (daxpy, zaxpy) terms of a call whose x
        ///     and y strides are both 1 - a fused multiply-add into +0 (complex: <c>fma(-ai, xi, fma(ar, xr, 0))</c> and
        ///     <c>fma(ai, xr, fma(ar, xi, 0))</c>, 640 of 640 probe elements); the C loop otherwise - <c>y += a*x</c>,
        ///     complex <c>y += (ar*xr - ai*xi)</c> / <c>y += (ar*xi + ai*xr)</c>, uncontracted (960 of 960).</para>
        ///     <para>Two departures from NumPy's procedure, both measured (D23: a slower cell keeps its code): the outputs
        ///     are written once each in the result's memory order with the formula NumPy's calls give them, into an
        ///     unzeroed result (NumPy's zeroed buffer and per-line walk down a C-ordered matrix's columns measured
        ///     0.08-0.4x of the former multiply); and a 0-d scalar beside a matrix whose rows are not contiguous keeps the
        ///     multiply route, which returns an F-ordered result and keeps a -0.0 product where NumPy's C-ordered result
        ///     holds +0.0 (its transposing read measured 0.23-0.75x).</para>
        ///     <para>NaN: every output is NumPy's NaN too. With a scalar that is not NaN, each output's NaN can come only from
        ///     its element or an invalid product, and the vector forms keep NumPy's choice by construction (each fused
        ///     multiply-add has one multiplicand that can be NaN, and the multiplicands' NaNs come before the addend's;
        ///     <c>vaddsubpd</c> cannot be reordered); a NaN scalar, the plain product and the complex C loop's scalar tail
        ///     choose explicitly (<see cref="NDDot.ZaxpyNaN"/>, <see cref="NDDot.ScalarProductNaN(double, double)"/>).</para>
        ///     <para>The OpenBLAS backend, when installed, answered before this (its own port of the same branch).</para>
        /// </remarks>
        private bool TryDotScalarBranch(NDArray left, NDArray right, out NDArray result)
        {
            result = null;
            if (left.ndim > 2 || right.ndim > 2)
                return false;
            bool leftScalar = IsBlasScalarShape(left), rightScalar = IsBlasScalarShape(right);
            if (!leftScalar && !rightScalar)
                return false;
            // Only NumPy's numeric dtypes take part (decimal, char and text keep their routes), and only a BLAS common dtype.
            if (!IsVecdotOperand(left.typecode) || !IsVecdotOperand(right.typecode))
                return false;
            var tc = left.typecode == right.typecode
                ? left.typecode
                : DTypePromotion.PromoteTypes(left.dtype, right.dtype).GetTypeCode();
            if (tc != NPTypeCode.Single && tc != NPTypeCode.Double && tc != NPTypeCode.Complex)
                return false;

            // A 0-d scalar beside a matrix whose rows are not contiguous (F-ordered, transposed, stepped): NumPy's C-ordered
            // result needs a transposing read there, which measured 0.23-0.75x of the multiply route's contiguous pass
            // (whose result keeps the operand's K order); that cell keeps the multiply route (D23), its result F-ordered
            // where NumPy's is C, and a -0.0 product -0.0 where NumPy's C loop gives +0.0.
            {
                var m = leftScalar ? right : left;
                if (m.ndim == 2 && m.Shape.dimensions[0] > 1 && m.Shape.dimensions[1] > 1 && m.Shape.strides[1] != 1)
                    return false;
            }

            // The scalar becomes ap2 (NumPy swaps when ap1 is the scalar; a scalar pair keeps the swapped order too).
            NDArray oap1 = left, oap2 = right;
            NDArray vec = leftScalar ? right : left;   // ap1 after the swap
            NDArray sca = leftScalar ? left : right;   // ap2 after the swap
            bool vecIsScalar = leftScalar ? rightScalar : leftScalar;   // ap1's own shape class after the swap

            // The output dims and l, NumPy's statements.
            long[] dims;
            long l;
            bool vecIsRow = !vecIsScalar && vec.ndim == 2 && vec.Shape.dimensions[0] == 1;   // _row: (1, N), N > 1
            bool vecIsMatrix = !vecIsScalar && vec.ndim == 2 && vec.Shape.dimensions[0] > 1 && vec.Shape.dimensions[1] > 1;
            if (vec.ndim == 0 || sca.ndim == 0)
            {
                // A 0-d operand: the other's dims, l their product.
                var other = vec.ndim == 0 ? sca : vec;
                dims = (long[])other.Shape.dimensions.Clone();
                l = 1;
                for (int j = 0; j < dims.Length; j++)
                    l *= dims[j];
            }
            else
            {
                long contracted = oap1.Shape.dimensions[oap1.ndim - 1];
                if (oap2.Shape.dimensions[0] != contracted)
                    return false;   // dot_alignment_error: the routes below raise their misalignment
                int nd = oap1.ndim + oap2.ndim - 2;
                if (nd == 1)
                {
                    // dot((N,1), (1,)) and dot((1,), (1,N)): an (N,) result through the scalar code
                    dims = new[] { oap1.ndim == 2 ? oap1.Shape.dimensions[0] : oap2.Shape.dimensions[1] };
                    l = dims[0];
                }
                else if (nd == 2)
                {
                    dims = new[] { oap1.Shape.dimensions[0], oap2.Shape.dimensions[1] };
                    l = vecIsRow ? dims[1] : dims[0];
                }
                else
                {
                    dims = Array.Empty<long>();
                    l = 1;
                }

                // A zero-sized summation dimension sums nothing.
                if (contracted == 0)
                    l = 0;
            }

            // NumPy zeroes its buffer (memset) and adds into it; every output is written exactly once below (one term
            // each, or the plain product), so the result is taken unzeroed - a zeroed buffer comes from fresh pages -
            // except where nothing is written (an empty result, or l == 0: the zeros are the answer).
            bool nothingWritten = l == 0;
            for (int j = 0; j < dims.Length; j++)
                nothingWritten |= dims[j] == 0;
            var res = NewArray(tc, Shape.Adopt(dims), nothingWritten);
            result = res;
            if (nothingWritten)
                return true;

            // The conversions and _bad_strides copies: NumPy's PyArray_FromAny with the common descriptor casts into a
            // KEEPORDER copy (an F-ordered matrix stays F, which decides which outputs a matrix's per-line zaxpy sends
            // through the kernel), and PyArray_NewCopy(NPY_ANYORDER) copies a negative or zero stride over more than one
            // element (in C order: such a view is neither C- nor F-contiguous).
            NDArray a1 = vec.typecode == tc ? vec : vec.astype(tc, true);
            NDArray a2 = sca.typecode == tc ? sca : sca.astype(tc, true);
            if (HasBadStrides(a1))
            {
                var copy = a1.copy('C');
                if (!ReferenceEquals(a1, vec))
                    a1.Dispose();   // the conversion, superseded by its copy
                a1 = copy;
            }
            try
            {
                unsafe
                {
                    switch (tc)
                    {
                        case NPTypeCode.Single:
                            ScalarBranch<float, SaxpyModel>(*((float*)a2.Address + a2.Shape.offset), a1,
                                (float*)res.Address, l, vecIsRow, vecIsMatrix);
                            break;
                        case NPTypeCode.Double:
                            ScalarBranch<double, DaxpyModel>(*((double*)a2.Address + a2.Shape.offset), a1,
                                (double*)res.Address, l, vecIsRow, vecIsMatrix);
                            break;
                        default:
                            ScalarBranch<Complex, ZaxpyModel>(*((Complex*)a2.Address + a2.Shape.offset), a1,
                                (Complex*)res.Address, l, vecIsRow, vecIsMatrix);
                            break;
                    }
                }
            }
            finally
            {
                // Conversions and copies are this call's intermediates, released here because the engine may run outside
                // any NDScope.
                if (!ReferenceEquals(a1, vec))
                    a1.Dispose();
                if (!ReferenceEquals(a2, sca))
                    a2.Dispose();
            }

            return true;
        }

        /// <summary>
        ///     NumPy's <c>_select_matrix_shape(...) == _scalar</c> for an operand of at most two dims: 0-d, <c>(1,)</c> or
        ///     <c>(1, 1)</c>.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <returns>True for a scalar-shaped operand.</returns>
        private static bool IsBlasScalarShape(NDArray x)
            => x.ndim switch
            {
                0 => true,
                1 => x.Shape.dimensions[0] == 1,
                2 => x.Shape.dimensions[0] == 1 && x.Shape.dimensions[1] == 1,
                _ => false,
            };

        /// <summary>
        ///     NumPy's <c>_bad_strides</c> (cblasfuncs.c) for an aligned operand: a negative stride, or a zero one over more
        ///     than one element, which cblas cannot take.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <returns>True when NumPy copies the operand before the BLAS call.</returns>
        private static bool HasBadStrides(NDArray x)
        {
            for (int i = 0; i < x.ndim; i++)
            {
                long s = x.Shape.strides[i];
                if (s < 0 || (s == 0 && x.Shape.dimensions[i] > 1))
                    return true;
            }
            return false;
        }

        /// <summary>
        ///     The scalar branch's arithmetic once the operands are prepared, every output written once (the result is not
        ///     zeroed): one output is the plain product; more outputs are what NumPy's <c>?axpy</c> call(s) into the zeroed
        ///     buffer give them - the kernel's fused form for the first <c>n &amp; -block</c> terms of a call whose x and y
        ///     strides are both 1, the C loop's <c>+0 + a*x</c> for every other term - computed in the result's memory order.
        /// </summary>
        /// <typeparam name="T">float, double or Complex.</typeparam>
        /// <typeparam name="TM">The dtype's <c>?axpy</c> model.</typeparam>
        /// <param name="alpha">The scalar operand's value (<c>*ap2</c>).</param>
        /// <param name="a1">NumPy's <c>ap1</c>: the other operand, of the dtype, without bad strides.</param>
        /// <param name="o">The result's first element (a fresh C-ordered buffer, not zeroed).</param>
        /// <param name="l">NumPy's <c>l</c>: 1 for the plain product, else the outputs of the one call (non-matrix).</param>
        /// <param name="isRow">Whether <paramref name="a1"/> is NumPy's <c>_row</c> (its stride is its second axis').</param>
        /// <param name="isMatrix">Whether <paramref name="a1"/> is NumPy's <c>_matrix</c> (then a 0-d scalar beside it).</param>
        /// <remarks>
        ///     <para>Each output is independent (y starts at +0 and takes one term), so the order of computation changes no
        ///     bit; only each output's formula follows NumPy's calls. A matrix is one <c>?axpy</c> per line along its longer
        ///     axis (the first on a tie): only lines along the result's rows (more columns than rows) have an output stride
        ///     of 1, so only they reach the kernel, and only where the matrix's rows are contiguous too.</para>
        ///     <para>Why not NumPy's own traversal: its per-line calls walk a square or tall C-ordered matrix down its
        ///     columns (a cache line per element), and the zeroed buffer comes from fresh pages; measured 0.08-0.4x of the
        ///     former multiply on 10^4 - 10^6 elements before this form.</para>
        /// </remarks>
        private static unsafe void ScalarBranch<T, TM>(T alpha, NDArray a1, T* o, long l, bool isRow, bool isMatrix)
            where T : unmanaged where TM : struct, IAxpyModel<T>
        {
            T* x = (T*)a1.Address + a1.Shape.offset;
            if (l == 1)
            {
                // *ap2 * *ap1: the plain product, the scalar first, -0.0 kept.
                *o = TM.Product(alpha, *x);
                return;
            }

            if (TM.IsZero(alpha))
            {
                // OpenBLAS's cblas_?axpy returns before any arithmetic when alpha is 0 (interface/axpy.c; zaxpy.c: both
                // parts), so every output keeps the memset's +0.0 whatever x holds - a NaN or inf element included
                // (np.dot(0.0, [nan, inf]) is [0., 0.]). Every call of a matrix's per-line loop returns the same way.
                long outputs = isMatrix ? a1.Shape.dimensions[0] * a1.Shape.dimensions[1] : l;
                System.Runtime.InteropServices.NativeMemory.Clear(o, (nuint)(outputs * sizeof(T)));
                return;
            }

            if (!isMatrix)
            {
                // One ?axpy over l terms into a contiguous result: the kernel takes its block only at a unit x stride.
                long stride = isRow ? a1.Shape.strides[1] : a1.Shape.strides[0];
                if (stride == 1)
                {
                    long n1 = l & -TM.KernelBlock;
                    TM.Fused(n1, alpha, x, o);
                    TM.Plain(l - n1, alpha, x + n1, 1, o + n1);
                }
                else
                {
                    TM.Plain(l, alpha, x, stride, o);
                }

                return;
            }

            long rows = a1.Shape.dimensions[0], cols = a1.Shape.dimensions[1];
            long s0 = a1.Shape.strides[0], s1 = a1.Shape.strides[1];
            if (cols > rows && s1 == 1)
            {
                // NumPy's lines run along the result's rows (output stride 1) over contiguous rows: kernel lines.
                long n1 = cols & -TM.KernelBlock;
                for (long i = 0; i < rows; i++)
                {
                    TM.Fused(n1, alpha, x + i * s0, o + i * cols);
                    TM.Plain(cols - n1, alpha, x + i * s0 + n1, 1, o + i * cols + n1);
                }
            }
            else if (s0 == cols)
            {
                // A C-contiguous matrix whose NumPy lines are strided (down the columns): the C loop's form everywhere,
                // one flat run.
                TM.Plain(rows * cols, alpha, x, 1, o);
            }
            else
            {
                // Contiguous rows at a larger row stride (a column slice): the C loop's form everywhere, by rows.
                for (long i = 0; i < rows; i++)
                    TM.Plain(cols, alpha, x + i * s0, 1, o + i * cols);
            }
        }

        /// <summary>
        ///     The arithmetic of one dtype's <c>?axpy</c> into a zeroed y, as outputs written once: scipy-openblas' Haswell
        ///     kernel (a fused multiply-add into +0) and its C loop (<c>+0 + a*x</c>, each product rounded).
        /// </summary>
        /// <typeparam name="T">float, double or Complex.</typeparam>
        private unsafe interface IAxpyModel<T> where T : unmanaged
        {
            /// <summary>The kernel's block: it takes the first <c>n &amp; -KernelBlock</c> terms of a unit-stride call.</summary>
            static abstract long KernelBlock { get; }

            /// <summary>Whether the scalar is OpenBLAS's <c>alpha == ZERO</c> (either sign; both parts for complex), on
            ///     which cblas_?axpy returns without touching y.</summary>
            /// <param name="a">The scalar.</param>
            /// <returns>True for a zero scalar.</returns>
            static abstract bool IsZero(T a);

            /// <summary>One output, <c>a * x</c> (NumPy's <c>l == 1</c> statement; no +0, so -0.0 survives), with the NaN
            ///     NumPy's compiled product keeps when both factors hold one.</summary>
            /// <param name="a">The scalar (<c>*ap2</c>).</param>
            /// <param name="x">The other operand's element (<c>*ap1</c>).</param>
            /// <returns>The plain product.</returns>
            static abstract T Product(T a, T x);

            /// <summary>The kernel's form over <paramref name="n"/> contiguous terms (a NaN scalar: NumPy's NaN per
            ///     output, chosen explicitly).</summary>
            /// <param name="n">The terms (a multiple of the block, or 0).</param>
            /// <param name="a">The scalar.</param>
            /// <param name="x">The first term, contiguous.</param>
            /// <param name="y">The first output, contiguous.</param>
            static abstract void Fused(long n, T a, T* x, T* y);

            /// <summary>The C loop's form over <paramref name="n"/> terms into contiguous outputs (a NaN scalar: NumPy's NaN
            ///     per output, chosen explicitly).</summary>
            /// <param name="n">The terms.</param>
            /// <param name="a">The scalar.</param>
            /// <param name="x">The first term.</param>
            /// <param name="incx">The terms' element stride.</param>
            /// <param name="y">The first output, contiguous.</param>
            static abstract void Plain(long n, T a, T* x, long incx, T* y);

        }

        /// <summary>saxpy's model: the Haswell kernel's block is 32 terms (saxpy.c's <c>n &amp; -32</c>).</summary>
        private unsafe struct SaxpyModel : IAxpyModel<float>
        {
            /// <inheritdoc/>
            public static long KernelBlock => 32;

            /// <inheritdoc/>
            public static bool IsZero(float a) => a == 0f;

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static float Product(float a, float x)
            {
                float p = a * x;
                // Two NaN factors: MSVC's float32 product keeps ap1's (x's) NaN; RyuJIT's operand order is its own.
                return float.IsNaN(p) ? NDDot.ScalarProductNaN(a, x) : p;
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public static void Fused(long n, float a, float* x, float* y)
            {
                if (float.IsNaN(a))
                {
                    // The kernel's fma(a, x, 0) returns its first multiplicand's NaN whatever x holds.
                    float q = NDDot.AxpyNaNScalar(a);
                    for (long k = 0; k < n; k++)
                        y[k] = q;
                    return;
                }

                long i = 0;
                if (Fma.IsSupported && Avx.IsSupported)
                {
                    var va = Vector256.Create(a);
                    for (; i + 8 <= n; i += 8)   // vfmadd231ps (x), alpha, y with y = +0
                        Vector256.Store(Fma.MultiplyAdd(va, Vector256.Load(x + i), Vector256<float>.Zero), y + i);
                }

                for (; i < n; i++)
                    y[i] = MathF.FusedMultiplyAdd(a, x[i], 0f);
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public static void Plain(long n, float a, float* x, long incx, float* y)
            {
                if (float.IsNaN(a))
                {
                    // The C loop's `da * x[i]` keeps da's NaN first (GCC's order), whatever x holds.
                    float q = NDDot.AxpyNaNScalar(a);
                    for (long k = 0; k < n; k++)
                        y[k] = q;
                    return;
                }

                long i = 0;
                if (incx == 1 && Vector256.IsHardwareAccelerated)
                {
                    var va = Vector256.Create(a);
                    for (; i + 8 <= n; i += 8)   // y += a*x with y = +0: the product rounded, then +0 added
                        Vector256.Store(Vector256<float>.Zero + va * Vector256.Load(x + i), y + i);
                }

                for (; i < n; i++)
                {
                    float s = 0f;
                    s += a * x[i * incx];
                    y[i] = s;
                }
            }

        }

        /// <summary>daxpy's model: the Haswell kernel's block is 16 terms (daxpy.c's <c>n &amp; -16</c>).</summary>
        private unsafe struct DaxpyModel : IAxpyModel<double>
        {
            /// <inheritdoc/>
            public static long KernelBlock => 16;

            /// <inheritdoc/>
            public static bool IsZero(double a) => a == 0.0;

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static double Product(double a, double x)
            {
                double p = a * x;
                // Two NaN factors: MSVC's float64 product keeps ap2's (a's) NaN; RyuJIT's operand order is its own.
                return double.IsNaN(p) ? NDDot.ScalarProductNaN(a, x) : p;
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public static void Fused(long n, double a, double* x, double* y)
            {
                if (double.IsNaN(a))
                {
                    // The kernel's fma(a, x, 0) returns its first multiplicand's NaN whatever x holds.
                    double q = NDDot.AxpyNaNScalar(a);
                    for (long k = 0; k < n; k++)
                        y[k] = q;
                    return;
                }

                long i = 0;
                if (Fma.IsSupported && Avx.IsSupported)
                {
                    var va = Vector256.Create(a);
                    for (; i + 4 <= n; i += 4)   // vfmadd231pd (x), alpha, y with y = +0
                        Vector256.Store(Fma.MultiplyAdd(va, Vector256.Load(x + i), Vector256<double>.Zero), y + i);
                }

                for (; i < n; i++)
                    y[i] = Math.FusedMultiplyAdd(a, x[i], 0.0);
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public static void Plain(long n, double a, double* x, long incx, double* y)
            {
                if (double.IsNaN(a))
                {
                    // The C loop's `da * x[i]` keeps da's NaN first (GCC's order), whatever x holds.
                    double q = NDDot.AxpyNaNScalar(a);
                    for (long k = 0; k < n; k++)
                        y[k] = q;
                    return;
                }

                long i = 0;
                if (incx == 1 && Vector256.IsHardwareAccelerated)
                {
                    var va = Vector256.Create(a);
                    for (; i + 4 <= n; i += 4)
                        Vector256.Store(Vector256<double>.Zero + va * Vector256.Load(x + i), y + i);
                }

                for (; i < n; i++)
                {
                    double s = 0.0;
                    s += a * x[i * incx];
                    y[i] = s;
                }
            }

        }

        /// <summary>
        ///     zaxpy's model (no CONJ): the Haswell kernel's block is 16 terms - <c>fma(-ai, xi, fma(ar, xr, 0))</c>,
        ///     <c>fma(ai, xr, fma(ar, xi, 0))</c> - and the C loop's <c>+0 + (ar*xr - ai*xi)</c>, <c>+0 + (ar*xi + ai*xr)</c>.
        /// </summary>
        private unsafe struct ZaxpyModel : IAxpyModel<Complex>
        {
            /// <inheritdoc/>
            public static long KernelBlock => 16;

            /// <inheritdoc/>
            public static bool IsZero(Complex a) => a.Real == 0.0 && a.Imaginary == 0.0;

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Complex Product(Complex a, Complex x)
            {
                // npy_creal(*ptr1) * npy_creal(*ptr2) - npy_cimag(*ptr1) * npy_cimag(*ptr2), and the imaginary part
                var p = new Complex(a.Real * x.Real - a.Imaginary * x.Imaginary, a.Real * x.Imaginary + a.Imaginary * x.Real);
                // A NaN part: MSVC's operand order (the last product ptr2 first) decides which NaN, not RyuJIT's.
                return NDDot.HasNaN(p) ? NDDot.ScalarProductNaN(a, x) : p;
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public static void Fused(long n, Complex a, Complex* x, Complex* y)
            {
                if (NDDot.HasNaN(a))
                {
                    // A NaN scalar part meets the elements' NaNs in every lane: the kernel's operand order, explicitly.
                    for (long k = 0; k < n; k++)
                        y[k] = NDDot.ZaxpyNaN(a, x[k], kernel: true);
                    return;
                }

                double ar = a.Real, ai = a.Imaginary;
                double* xp = (double*)x, yp = (double*)y;
                long i = 0;
                if (Fma.IsSupported && Avx.IsSupported)
                {
                    var var_ = Vector256.Create(ar);
                    var vai = Vector256.Create(-ai, ai, -ai, ai);   // ymm1 = ai * mvec, mvec = { -1, 1, -1, 1 }
                    for (; i + 2 <= n; i += 2)
                    {
                        var vx = Avx.LoadVector256(xp + 2 * i);
                        var t = Fma.MultiplyAdd(var_, vx, Vector256<double>.Zero);               // vfmadd213pd (y = +0), ar, x
                        Avx.Store(yp + 2 * i, Fma.MultiplyAdd(vai, Avx.Permute(vx, 0b0101), t));   // vfmadd231pd ai', swap(x)
                    }
                }

                for (; i < n; i++)
                {
                    double xr = xp[2 * i], xi = xp[2 * i + 1];
                    yp[2 * i] = Math.FusedMultiplyAdd(-ai, xi, Math.FusedMultiplyAdd(ar, xr, 0.0));
                    yp[2 * i + 1] = Math.FusedMultiplyAdd(ai, xr, Math.FusedMultiplyAdd(ar, xi, 0.0));
                }
            }

            /// <inheritdoc/>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public static void Plain(long n, Complex a, Complex* x, long incx, Complex* y)
            {
                if (NDDot.HasNaN(a))
                {
                    // A NaN scalar part meets the elements' NaNs in every product: the C loop's order, explicitly.
                    for (long k = 0; k < n; k++)
                        y[k] = NDDot.ZaxpyNaN(a, x[k * incx], kernel: false);
                    return;
                }

                double ar = a.Real, ai = a.Imaginary;
                double* xp = (double*)x, yp = (double*)y;
                long i = 0;
                if (incx == 1 && Avx.IsSupported)
                {
                    var var_ = Vector256.Create(ar);
                    var vai = Vector256.Create(ai);
                    for (; i + 2 <= n; i += 2)
                    {
                        // [ar*xr, ar*xi] -/+ [ai*xi, ai*xr]: each product rounded, the lanes subtracted / added (addsub), +0
                        var vx = Avx.LoadVector256(xp + 2 * i);
                        var sum = Avx.AddSubtract(Avx.Multiply(var_, vx), Avx.Multiply(vai, Avx.Permute(vx, 0b0101)));
                        Avx.Store(yp + 2 * i, Avx.Add(Vector256<double>.Zero, sum));
                    }
                }

                long sx = 2 * incx;
                for (; i < n; i++)
                {
                    double xr = xp[i * sx], xi = xp[i * sx + 1];
                    double re = 0.0, im = 0.0;
                    re += ar * xr - ai * xi;
                    im += ar * xi + ai * xr;
                    if (double.IsNaN(re) || double.IsNaN(im))
                    {
                        // The imaginary add commutes for RyuJIT: two NaN products need NumPy's `ar*xi` first.
                        var r = NDDot.ZaxpyNaN(a, new Complex(xr, xi), kernel: false);
                        re = r.Real;
                        im = r.Imaginary;
                    }
                    yp[2 * i] = re;
                    yp[2 * i + 1] = im;
                }
            }

        }
    }
}
