using System;
using NumSharp;
using NumSharp.Backends;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>Row/column storage order — mirrors <c>enum CBLAS_ORDER</c> in npy_cblas.h.</summary>
    internal enum CBlasOrder
    {
        RowMajor = 101,
        ColMajor = 102
    }

    /// <summary>Transposition flag — mirrors <c>enum CBLAS_TRANSPOSE</c> in npy_cblas.h.</summary>
    internal enum CBlasTranspose
    {
        NoTrans = 111,
        Trans = 112,
        ConjTrans = 113
    }

    /// <summary>Triangle selector — mirrors <c>enum CBLAS_UPLO</c> in npy_cblas.h.</summary>
    internal enum CBlasUpLo
    {
        Upper = 121,
        Lower = 122
    }

    /// <summary>Unit/non-unit diagonal flag — mirrors <c>enum CBLAS_DIAG</c> in npy_cblas.h.</summary>
    internal enum CBlasDiag
    {
        NonUnit = 131,
        Unit = 132
    }

    /// <summary>Operand side for the triangular/symmetric Level-3 routines — mirrors
    /// <c>enum CBLAS_SIDE</c> in npy_cblas.h.</summary>
    internal enum CBlasSide
    {
        Left = 141,
        Right = 142
    }

    /// <summary>
    ///     Binding to an external CBLAS shared library, used exclusively by the opt-in
    ///     byte-parity matrix-product backend this package installs (<see cref="OpenBlasEngine"/> →
    ///     <see cref="OpenBlasBackend"/>).
    /// </summary>
    /// <remarks>
    ///     NumSharp's own GEMM is never routed through here — this exists so a caller can hand
    ///     NumSharp the SAME BLAS binary a given NumPy build calls, which is the only way to get
    ///     bit-identical <c>np.dot</c> / <c>np.matmul</c> results out of the two stacks (OpenBLAS'
    ///     multi-accumulator kernels are arch- and thread-count-specific and are not reproducible
    ///     by any portable algorithm — see <c>docs/GEMM_PARITY.md</c>).
    ///     <para>
    ///     Two symbol schemes are supported. NumPy's pip wheels ship an ILP64 build with the
    ///     symbols renamed <c>scipy_</c><i>name</i><c>64_</c> (e.g. <c>scipy_cblas_sgemm64_</c>),
    ///     where the BLAS integer is 64-bit; a stock LP64 OpenBLAS/reference CBLAS exports plain
    ///     <c>cblas_sgemm</c> with a 32-bit integer. Both are probed at load time and the widths
    ///     are marshalled accordingly — see <see cref="IsIlp64"/>.
    ///     </para>
    /// </remarks>
    internal static unsafe class OpenBlasNative
    {
        private static readonly object SyncRoot = new object();

        private static IntPtr _handle;

        /// <summary>Path of the currently loaded CBLAS library, or null when none is loaded.</summary>
        internal static string LibraryPath { get; private set; }

        /// <summary>True when the loaded library uses 64-bit BLAS integers (numpy's wheels do).</summary>
        internal static bool IsIlp64 { get; private set; }

        /// <summary>The symbol prefix/suffix pair that resolved, for diagnostics (e.g. "scipy_*64_").</summary>
        internal static string SymbolScheme { get; private set; }

        /// <summary>True when a library is loaded and every required symbol bound.</summary>
        internal static bool IsLoaded => _handle != IntPtr.Zero;

        #region function pointers — ILP64 (64-bit BLAS int)

        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, long, float, float*, long, float*, long, float, float*, long, void> _sgemm64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, long, double, double*, long, double*, long, double, double*, long, void> _dgemm64;
        private static delegate* unmanaged[Cdecl]<int, int, long, long, float, float*, long, float*, long, float, float*, long, void> _sgemv64;
        private static delegate* unmanaged[Cdecl]<int, int, long, long, double, double*, long, double*, long, double, double*, long, void> _dgemv64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, float, float*, long, float, float*, long, void> _ssyrk64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, double, double*, long, double, double*, long, void> _dsyrk64;
        private static delegate* unmanaged[Cdecl]<long, float*, long, float*, long, float> _sdot64;
        private static delegate* unmanaged[Cdecl]<long, double*, long, double*, long, double> _ddot64;
        private static delegate* unmanaged[Cdecl]<long, float, float*, long, float*, long, void> _saxpy64;
        private static delegate* unmanaged[Cdecl]<long, double, double*, long, double*, long, void> _daxpy64;

        // Level-3 siblings of gemm/syrk — trmm/trsm (Order,Side,Uplo,TransA,Diag,M,N,alpha,A,lda,B,ldb),
        // symm (Order,Side,Uplo,M,N,alpha,A,lda,B,ldb,beta,C,ldc) and syr2k
        // (Order,Uplo,Trans,N,K,alpha,A,lda,B,ldb,beta,C,ldc). Bound best-effort (see BindLevel3Extras):
        // nothing in the parity backend calls them, they exist as infrastructure a future consumer can
        // reach through IBlasType<T>.
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, long, long, float, float*, long, float*, long, void> _strmm64;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, long, long, double, double*, long, double*, long, void> _dtrmm64;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, long, long, float, float*, long, float*, long, void> _strsm64;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, long, long, double, double*, long, double*, long, void> _dtrsm64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, float, float*, long, float*, long, float, float*, long, void> _ssymm64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, double, double*, long, double*, long, double, double*, long, void> _dsymm64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, float, float*, long, float*, long, float, float*, long, void> _ssyr2k64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, double, double*, long, double*, long, double, double*, long, void> _dsyr2k64;

        #endregion

        #region function pointers — LP64 (32-bit BLAS int)

        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, int, float, float*, int, float*, int, float, float*, int, void> _sgemm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, int, double, double*, int, double*, int, double, double*, int, void> _dgemm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, float, float*, int, float*, int, float, float*, int, void> _sgemv32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, double, double*, int, double*, int, double, double*, int, void> _dgemv32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, float, float*, int, float, float*, int, void> _ssyrk32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, double, double*, int, double, double*, int, void> _dsyrk32;
        private static delegate* unmanaged[Cdecl]<int, float*, int, float*, int, float> _sdot32;
        private static delegate* unmanaged[Cdecl]<int, double*, int, double*, int, double> _ddot32;
        private static delegate* unmanaged[Cdecl]<int, float, float*, int, float*, int, void> _saxpy32;
        private static delegate* unmanaged[Cdecl]<int, double, double*, int, double*, int, void> _daxpy32;

        // LP64 (32-bit BLAS int) twins of the Level-3 siblings above.
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, int, int, float, float*, int, float*, int, void> _strmm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, int, int, double, double*, int, double*, int, void> _dtrmm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, int, int, float, float*, int, float*, int, void> _strsm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, int, int, double, double*, int, double*, int, void> _dtrsm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, float, float*, int, float*, int, float, float*, int, void> _ssymm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, double, double*, int, double*, int, double, double*, int, void> _dsymm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, float, float*, int, float*, int, float, float*, int, void> _ssyr2k32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, double, double*, int, double*, int, double, double*, int, void> _dsyr2k32;

        #endregion

        #region function pointers — complex128 BLAS products (zgemm / zgemv / zsyrk / zdotu_sub / zdotc_sub / zaxpy)

        // The complex CBLAS product family behind np.dot / np.matmul for complex128, and the complex
        // gufunc products (vdot/vecdot/vecmat use the CONJUGATING zdotc, matmul/dot/inner/matvec the
        // UNCONJUGATED zdotu — see cblasfuncs.c, matmul.c.src, arraytypes.c.src, vdot.c). A
        // System.Numerics.Complex is two interleaved doubles == BLAS complex*16, so a Complex* casts to
        // double* with no copy (the same trick the LAPACK z-routines above rely on).
        //
        // ABI note that differs from the real gemm/gemv/syrk/axpy above: the complex routines take
        // alpha and beta BY POINTER (const void*), not by value — so these signatures carry a `double*`
        // where the real ones carry a `float`/`double`. And the dot is the `_sub` out-pointer form
        // (cblas_zdotu_sub / cblas_zdotc_sub write the complex result through a trailing `double*`)
        // rather than a struct-return, matching what NumPy calls and avoiding the compiler-specific
        // by-value complex-struct return ABI.
        //
        // ILP64 (64-bit BLAS int): m/n/k/lda/ldb/ldc/incX/incY are 64-bit; the data and scalars stay
        // `double*` regardless of the integer width.
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, long, double*, double*, long, double*, long, double*, double*, long, void> _zgemm64;
        private static delegate* unmanaged[Cdecl]<int, int, long, long, double*, double*, long, double*, long, double*, double*, long, void> _zgemv64;
        private static delegate* unmanaged[Cdecl]<int, int, int, long, long, double*, double*, long, double*, double*, long, void> _zsyrk64;
        private static delegate* unmanaged[Cdecl]<long, double*, long, double*, long, double*, void> _zdotu_sub64;
        private static delegate* unmanaged[Cdecl]<long, double*, long, double*, long, double*, void> _zdotc_sub64;
        private static delegate* unmanaged[Cdecl]<long, double*, double*, long, double*, long, void> _zaxpy64;

        // LP64 (32-bit BLAS int): scipy-openblas32 (win-arm64, x86) and stock system OpenBLAS.
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, int, double*, double*, int, double*, int, double*, double*, int, void> _zgemm32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, double*, double*, int, double*, int, double*, double*, int, void> _zgemv32;
        private static delegate* unmanaged[Cdecl]<int, int, int, int, int, double*, double*, int, double*, double*, int, void> _zsyrk32;
        private static delegate* unmanaged[Cdecl]<int, double*, int, double*, int, double*, void> _zdotu_sub32;
        private static delegate* unmanaged[Cdecl]<int, double*, int, double*, int, double*, void> _zdotc_sub32;
        private static delegate* unmanaged[Cdecl]<int, double*, double*, int, double*, int, void> _zaxpy32;

        #endregion

        private static delegate* unmanaged[Cdecl]<int, void> _setNumThreads;
        private static delegate* unmanaged[Cdecl]<int> _getNumThreads;
        private static delegate* unmanaged[Cdecl]<IntPtr> _getConfig;
        private static delegate* unmanaged[Cdecl]<IntPtr> _getCoreName;

        #region function pointers — LAPACK LU family (getrf / gesv, for np.linalg det/slogdet/solve/inv)

        // Fortran LAPACK is column-major and passes EVERY argument by pointer, scalars included. The
        // complex routines take the matrix as `double*` — a complex*16 is two interleaved doubles,
        // which is exactly System.Numerics.Complex's [Real, Imaginary] layout, so a Complex* casts to
        // double* with no copy. Only the double/cdouble routines are bound: numpy's linalg always
        // upcasts float32→double (the 'd' gufunc signature) and casts the result back, so there is no
        // sgetrf/cgetrf path to reproduce.
        //
        // ILP64 (64-bit fortran_int): m, n, lda, ldb, ipiv entries and info are all 64-bit.
        private static delegate* unmanaged[Cdecl]<long*, long*, double*, long*, long*, long*, void> _dgetrf64;
        private static delegate* unmanaged[Cdecl]<long*, long*, double*, long*, long*, double*, long*, long*, void> _dgesv64;
        private static delegate* unmanaged[Cdecl]<long*, long*, double*, long*, long*, long*, void> _zgetrf64;
        private static delegate* unmanaged[Cdecl]<long*, long*, double*, long*, long*, double*, long*, long*, void> _zgesv64;

        // LP64 (32-bit fortran_int): the scipy-openblas32 build (win-arm64, x86) and stock system OpenBLAS.
        private static delegate* unmanaged[Cdecl]<int*, int*, double*, int*, int*, int*, void> _dgetrf32;
        private static delegate* unmanaged[Cdecl]<int*, int*, double*, int*, int*, double*, int*, int*, void> _dgesv32;
        private static delegate* unmanaged[Cdecl]<int*, int*, double*, int*, int*, int*, void> _zgetrf32;
        private static delegate* unmanaged[Cdecl]<int*, int*, double*, int*, int*, double*, int*, int*, void> _zgesv32;

        #endregion

        #region function pointers — LAPACK Cholesky/QR family (potrf / geqrf / orgqr, for np.linalg cholesky/qr)

        // Same conventions as the LU family above: column-major, every scalar by pointer, complex
        // matrices as interleaved `double*` (== System.Numerics.Complex layout), and only the double /
        // cdouble routines are bound because numpy's linalg computes float32 in double then downcasts.
        //
        // potrf's `uplo` is a Fortran CHARACTER*1; it is passed as a `byte*` to a single 'L'/'U' byte.
        // The gfortran ABI appends a hidden character-length argument, but potrf never reads it (LSAME
        // compares one char), and it is the LAST argument, so omitting it is safe under cdecl — verified
        // on the bundled scipy-openblas (info==0, bit-identical factor to numpy). geqrf/orgqr have no
        // character arguments. All three take a workspace: call with LWORK == -1 to have WORK[0] receive
        // the optimal size, then allocate and call again — exactly as numpy's init_geqrf / init_gqr do.
        private static delegate* unmanaged[Cdecl]<byte*, long*, double*, long*, long*, void> _dpotrf64;
        private static delegate* unmanaged[Cdecl]<byte*, long*, double*, long*, long*, void> _zpotrf64;
        private static delegate* unmanaged[Cdecl]<long*, long*, double*, long*, double*, double*, long*, long*, void> _dgeqrf64;
        private static delegate* unmanaged[Cdecl]<long*, long*, double*, long*, double*, double*, long*, long*, void> _zgeqrf64;
        private static delegate* unmanaged[Cdecl]<long*, long*, long*, double*, long*, double*, double*, long*, long*, void> _dorgqr64;
        private static delegate* unmanaged[Cdecl]<long*, long*, long*, double*, long*, double*, double*, long*, long*, void> _zungqr64;

        private static delegate* unmanaged[Cdecl]<byte*, int*, double*, int*, int*, void> _dpotrf32;
        private static delegate* unmanaged[Cdecl]<byte*, int*, double*, int*, int*, void> _zpotrf32;
        private static delegate* unmanaged[Cdecl]<int*, int*, double*, int*, double*, double*, int*, int*, void> _dgeqrf32;
        private static delegate* unmanaged[Cdecl]<int*, int*, double*, int*, double*, double*, int*, int*, void> _zgeqrf32;
        private static delegate* unmanaged[Cdecl]<int*, int*, int*, double*, int*, double*, double*, int*, int*, void> _dorgqr32;
        private static delegate* unmanaged[Cdecl]<int*, int*, int*, double*, int*, double*, double*, int*, int*, void> _zungqr32;

        #endregion

        #region function pointers — LAPACK SVD/least-squares family (gesdd / gelsd, for np.linalg svd/lstsq)

        // Same conventions as the LU/QR families: column-major, every scalar by pointer, complex matrices
        // as interleaved `double*` (== System.Numerics.Complex layout), and only the double / cdouble
        // routines are bound because numpy's linalg computes float32 in double then downcasts. `jobz` is a
        // Fortran CHARACTER*1 passed as a single 'N'/'S'/'A' byte (same hidden-length-arg argument the
        // potrf comment makes — it is the FIRST argument here, but LAPACK never reads the length). Both
        // take a workspace queried with LWORK == -1 (writes WORK[0]); gelsd's query ALSO writes IWORK[0]
        // with the needed integer-workspace size, exactly as numpy's init_gesdd / init_gelsd do.
        //
        // The complex routines carry an extra REAL RWORK buffer the real ones do not, and the singular
        // values S are ALWAYS the real basetype (double), so `s`/`rwork`/`rcond` are `double*` on both.
        private static delegate* unmanaged[Cdecl]<byte*, long*, long*, double*, long*, double*, double*, long*, double*, long*, double*, long*, long*, long*, void> _dgesdd64;
        private static delegate* unmanaged[Cdecl]<byte*, long*, long*, double*, long*, double*, double*, long*, double*, long*, double*, long*, double*, long*, long*, void> _zgesdd64;
        private static delegate* unmanaged[Cdecl]<long*, long*, long*, double*, long*, double*, long*, double*, double*, long*, double*, long*, long*, long*, void> _dgelsd64;
        private static delegate* unmanaged[Cdecl]<long*, long*, long*, double*, long*, double*, long*, double*, double*, long*, double*, long*, double*, long*, long*, void> _zgelsd64;

        private static delegate* unmanaged[Cdecl]<byte*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*, int*, void> _dgesdd32;
        private static delegate* unmanaged[Cdecl]<byte*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, double*, int*, int*, void> _zgesdd32;
        private static delegate* unmanaged[Cdecl]<int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*, void> _dgelsd32;
        private static delegate* unmanaged[Cdecl]<int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*, void> _zgelsd32;

        #endregion

        #region function pointers — LAPACK eigen family (syevd / heevd / geev, for np.linalg eig/eigh/eigvals/eigvalsh)

        // Same conventions as the LU/QR/SVD families: column-major, every scalar by pointer, complex
        // matrices as interleaved `double*` (== System.Numerics.Complex layout), and only the double /
        // cdouble routines are bound because numpy's linalg computes float32 in double then downcasts.
        // `jobz`/`uplo` (syevd/heevd) and `jobvl`/`jobvr` (geev) are Fortran CHARACTER*1 passed as single
        // 'N'/'V'/'L'/'U' bytes. The gfortran ABI appends one hidden character-length argument PER char
        // arg at the very END of the list (after info); LAPACK's LSAME reads one char through the pointer
        // and never the length, so — exactly as with potrf's uplo and gesdd's jobz — omitting them is safe
        // under cdecl even with TWO char args (verified: syevd/geev return info==0 and a factor
        // bit-identical to numpy). The eigenvalue vector W of syevd/heevd is ALWAYS the real basetype
        // (double), and the complex routines carry a REAL rwork buffer the real ones do not, so
        // `w`/`rwork` are `double*` on both. Workspace query: lwork==-1 writes work[0]; syevd/heevd ALSO
        // write iwork[0] (liwork==-1) and heevd rwork[0] (lrwork==-1); geev writes only work[0].
        private static delegate* unmanaged[Cdecl]<byte*, byte*, long*, double*, long*, double*, double*, long*, long*, long*, long*, void> _dsyevd64;
        private static delegate* unmanaged[Cdecl]<byte*, byte*, long*, double*, long*, double*, double*, long*, double*, long*, long*, long*, long*, void> _zheevd64;
        private static delegate* unmanaged[Cdecl]<byte*, byte*, long*, double*, long*, double*, double*, double*, long*, double*, long*, double*, long*, long*, void> _dgeev64;
        private static delegate* unmanaged[Cdecl]<byte*, byte*, long*, double*, long*, double*, double*, long*, double*, long*, double*, long*, double*, long*, void> _zgeev64;

        private static delegate* unmanaged[Cdecl]<byte*, byte*, int*, double*, int*, double*, double*, int*, int*, int*, int*, void> _dsyevd32;
        private static delegate* unmanaged[Cdecl]<byte*, byte*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*, int*, void> _zheevd32;
        private static delegate* unmanaged[Cdecl]<byte*, byte*, int*, double*, int*, double*, double*, double*, int*, double*, int*, double*, int*, int*, void> _dgeev32;
        private static delegate* unmanaged[Cdecl]<byte*, byte*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, double*, int*, void> _zgeev32;

        #endregion

        /// <summary>
        ///     True when the loaded library additionally exports the LAPACK LU routines the
        ///     factorisation backend needs (<c>getrf</c>/<c>gesv</c> for double and cdouble). A plain
        ///     reference CBLAS has no LAPACK, so this can be false while <see cref="IsLoaded"/> is true.
        /// </summary>
        internal static bool IsLapackLoaded { get; private set; }

        /// <summary>
        ///     True when the loaded library additionally exports the complex128 CBLAS products the
        ///     parity backend needs (<c>zgemm</c>/<c>zgemv</c>/<c>zsyrk</c>/<c>zdotu_sub</c>/
        ///     <c>zdotc_sub</c>/<c>zaxpy</c>). Bound best-effort and all-or-nothing: a library missing
        ///     any of them leaves this false and complex <c>np.dot</c>/<c>np.matmul</c> fall through to
        ///     NumSharp's managed complex kernel (correct, not byte-parity). Every full OpenBLAS ships
        ///     the whole set; the flag exists so a bare real-only CBLAS declines cleanly.
        /// </summary>
        internal static bool IsComplexBlasLoaded { get; private set; }

        /// <summary>The width of a LAPACK <c>fortran_int</c> in bytes — 8 for ILP64, 4 for LP64.</summary>
        internal static int FortranIntSize => IsIlp64 ? 8 : 4;

        /// <summary>
        ///     The BLAS integer ceiling — NumPy's <c>BLAS_MAXSIZE</c>, "-1 to be conservative, in case
        ///     blas internally uses a for loop with an inclusive upper bound".
        /// </summary>
        internal static long BlasMaxSize => IsIlp64 ? long.MaxValue - 1 : int.MaxValue - 1;

        /// <summary>
        ///     NumPy's <c>NPY_CBLAS_CHUNK</c> — the greatest power of two below the BLAS integer max,
        ///     used to split long <c>?dot</c> reductions when npy_intp is wider than the BLAS int.
        /// </summary>
        internal static long CBlasChunk => IsIlp64 ? long.MaxValue : int.MaxValue / 2 + 1;

        /// <summary>
        ///     Loads a CBLAS shared library and binds every symbol the parity backend needs.
        /// </summary>
        /// <param name="path">
        ///     Explicit file path, a directory to search, or null to auto-discover. Discovery runs
        ///     the delivery design's tier order (<c>docs/OPENBLAS_DELIVERY_DESIGN.md</c> §3):
        ///     override path(s) (<c>NUMSHARP_OPENBLAS_SEARCH_PATH</c>, then a build-recorded path marker) →
        ///     a build-staged version override (REQUIRED — a miss throws, never falls through) →
        ///     explicit OpenBLAS roots (<c>OPENBLAS_HOME</c>/<c>OPENBLAS_ROOT</c>) → the bundled
        ///     runtime asset → machine tooling (see <see cref="AmbientCandidates"/>).
        /// </param>
        /// <param name="coreType">
        ///     OpenBLAS <c>DYNAMIC_ARCH</c> kernel to force via <c>OPENBLAS_CORETYPE</c>, or null to
        ///     let OpenBLAS choose. Applied before the loader runs, and therefore only effective
        ///     when this call is the one that loads the library.
        /// </param>
        /// <remarks>
        ///     <b>A failed load is a no-op.</b> Nothing here touches the currently bound library
        ///     until a replacement has resolved every required symbol, so <c>Load</c> either
        ///     succeeds or leaves the previous binding exactly as it was. Tearing the working
        ///     library down first — which is what this used to do — meant a mistyped path silently
        ///     demoted every subsequent product back to NumSharp's managed kernels while
        ///     <c>OpenBlasEngine.Enabled</c> still said true: the worst outcome a parity feature has, since
        ///     the answer stays plausible.
        /// </remarks>
        /// <exception cref="DllNotFoundException">No candidate library could be loaded.</exception>
        /// <exception cref="EntryPointNotFoundException">The library loaded but is not a CBLAS provider.</exception>
        internal static void Load(string path, string coreType = null)
        {
            lock (SyncRoot)
            {
                // Must happen BEFORE the loader runs: OpenBLAS reads OPENBLAS_CORETYPE once, while
                // its DYNAMIC_ARCH initialiser is picking a micro-kernel, and never looks again.
                if (!string.IsNullOrWhiteSpace(coreType))
                {
                    EnsureCoreTypeRunnable(coreType);
                    if (!TrySetNativeEnvironment("OPENBLAS_CORETYPE", coreType))
                        throw new PlatformNotSupportedException(
                            "OpenBlasEngine.Enable: could not set OPENBLAS_CORETYPE where the native library " +
                            "can see it (the C runtime's environment is not reachable on this " +
                            "platform). Set OPENBLAS_CORETYPE in the environment before starting " +
                            "the process instead.");

                    RequestedCoreType = coreType;
                }

                // TIER 1 — an explicitly named library is BINDING: parity is a claim about one
                // specific binary, so quietly falling back to whatever else is installed would
                // produce confidently wrong bits. Only the unattended case searches.
                string requested = !string.IsNullOrWhiteSpace(path)
                    ? path
                    : EnvVars.OpenBlasLibrary; // e.g. C:/opt/openblas/scipy_openblas64.dll | /usr/lib/x86_64-linux-gnu/libopenblas.so.0 | /opt/homebrew/opt/openblas/lib/libopenblas.dylib
                bool strict = !string.IsNullOrWhiteSpace(requested);

                var tried = new List<string>();
                if (strict)
                {
                    if (TryLoadCandidates(Expand(requested), tried))
                        return;

                    throw new DllNotFoundException(
                        $"OpenBlasEngine.Enable: '{requested}' could not be loaded as a CBLAS library" +
                        (tried.Count == 0
                            ? " (no such file, and no BLAS library in it if it is a directory)."
                            : $" (tried: {string.Join(", ", tried)}).") +
                        " The named library is used as given — parity is a claim about one specific " +
                        "binary, so no other one is substituted. Clear the path (or the " +
                        "NUMSHARP_OPENBLAS_LIBRARY environment variable) to auto-discover instead.");
                }

                // Unattended discovery — the delivery design's tier order (§3). The marker is what a
                // BUILD-phase override left behind; without one this reduces to override-path env →
                // bundle → machine tooling.
                var marker = OpenBlasSourceMarker.TryFind();

                // TIER 2a — override path(s): NUMSHARP_OPENBLAS_SEARCH_PATH (env wins over metadata, so it
                // goes first), then a path-mode marker recorded by the build. Non-binding: a miss
                // falls through.
                if (TryLoadCandidates(OverridePathCandidates(marker), tried))
                    return;

                // TIER 2b — a version override staged by the build. The consumer pinned ONE specific
                // scipy-openblas build (<OpenBlasVersion> / NUMSHARP_OPENBLAS_VERSION), so this is a
                // contract: a miss THROWS rather than falling through to the bundle or machine
                // tooling — every product would still compute there, with different bits than the
                // ones the caller pinned, which is the worst outcome a pin can have.
                if (marker != null && marker.IsVersionMode)
                {
                    if (TryLoadCandidates(VersionOverrideCandidates(marker), tried))
                        return;

                    if (marker.Required)
                        throw new OpenBlasRequiredOverrideException(
                            "OpenBlasEngine.Enable: the build staged a REQUIRED OpenBLAS version override " +
                            $"({marker.Distribution ?? "scipy-openblas"} {marker.Version ?? "?"}, " +
                            $"marker '{marker.MarkerPath}') but no loadable CBLAS matched it in: " +
                            string.Join(", ", marker.ReadDirectories()) +
                            (marker.Sha256 == null
                                ? "."
                                : $" (candidates must hash to the pinned sha256 {marker.Sha256}).") +
                            " A version override is a hard requirement — discovery does not fall " +
                            "back to the bundled asset or a machine-wide OpenBLAS. Rebuild the " +
                            "project to re-stage the binary, remove the override (clear " +
                            "OpenBlasVersion / NUMSHARP_OPENBLAS_VERSION and rebuild), or bind an " +
                            "explicit library via NUMSHARP_OPENBLAS_LIBRARY.");
                }

                // TIERS 3–4 — the bundled runtime asset (the zero-config parity default), then
                // ambient machine tooling.
                if (TryLoadCandidates(AmbientCandidates(), tried))
                    return;

                throw new DllNotFoundException(
                    "OpenBlasEngine.Enable: no CBLAS library could be loaded. This package normally serves " +
                    "its bundled scipy-openblas runtime asset (runtimes/<rid>/native/); none was " +
                    "found for this platform and no other CBLAS was discovered. Point " +
                    "NUMSHARP_OPENBLAS_SEARCH_PATH at a directory holding one (non-binding, highest " +
                    "priority), or set NUMSHARP_OPENBLAS_LIBRARY to bind one specific binary. Tried: " +
                    string.Join(", ", tried));
            }
        }

        /// <summary>
        ///     Tries each candidate in order; true when one loaded AND bound completely.
        /// </summary>
        /// <remarks>
        ///     A candidate that fails to LOAD is silently skipped (that is how tiers decline); a
        ///     candidate that loads but fails to BIND throws (see <see cref="Bind"/> — the handle is
        ///     freed and the previous binding is untouched). The previous library is freed only
        ///     after its replacement has fully bound, so this either advances or changes nothing.
        /// </remarks>
        private static bool TryLoadCandidates(IEnumerable<string> candidates, List<string> tried)
        {
            foreach (var candidate in candidates)
            {
                tried.Add(candidate);
                IntPtr handle;
                if (!NativeLibrary.TryLoad(candidate, out handle))
                    continue;

                var previous = _handle;
                try
                {
                    // Commits the statics only once every required symbol resolved.
                    Bind(handle, candidate);
                }
                catch
                {
                    NativeLibrary.Free(handle);
                    throw; // previous binding untouched
                }

                if (previous != IntPtr.Zero && previous != handle)
                    NativeLibrary.Free(previous);

                return true;
            }

            return false;
        }

        /// <summary>Releases the loaded library and clears every bound symbol.</summary>
        /// <remarks>
        ///     Deliberately NOT called by <see cref="Load"/> — replacing a library must not tear the
        ///     working one down before the replacement is bound, and <c>OpenBlasEngine.Disable()</c> only
        ///     uninstalls the backend (freeing a BLAS a thread may be executing in is a fatal
        ///     access violation, not a catchable error). Kept for an explicit teardown.
        /// </remarks>
        internal static void Unload()
        {
            lock (SyncRoot)
            {
                if (_handle != IntPtr.Zero)
                    NativeLibrary.Free(_handle);

                _handle = IntPtr.Zero;
                LibraryPath = null;
                SymbolScheme = null;
                IsIlp64 = false;
                _sgemm64 = null; _dgemm64 = null; _sgemv64 = null; _dgemv64 = null;
                _ssyrk64 = null; _dsyrk64 = null; _sdot64 = null; _ddot64 = null;
                _saxpy64 = null; _daxpy64 = null;
                _sgemm32 = null; _dgemm32 = null; _sgemv32 = null; _dgemv32 = null;
                _ssyrk32 = null; _dsyrk32 = null; _sdot32 = null; _ddot32 = null;
                _saxpy32 = null; _daxpy32 = null;
                _strmm64 = null; _dtrmm64 = null; _strsm64 = null; _dtrsm64 = null;
                _ssymm64 = null; _dsymm64 = null; _ssyr2k64 = null; _dsyr2k64 = null;
                _strmm32 = null; _dtrmm32 = null; _strsm32 = null; _dtrsm32 = null;
                _ssymm32 = null; _dsymm32 = null; _ssyr2k32 = null; _dsyr2k32 = null;
                _zgemm64 = null; _zgemv64 = null; _zsyrk64 = null;
                _zdotu_sub64 = null; _zdotc_sub64 = null; _zaxpy64 = null;
                _zgemm32 = null; _zgemv32 = null; _zsyrk32 = null;
                _zdotu_sub32 = null; _zdotc_sub32 = null; _zaxpy32 = null;
                IsComplexBlasLoaded = false;
                _setNumThreads = null; _getNumThreads = null; _getConfig = null; _getCoreName = null;
                _dgetrf64 = null; _dgesv64 = null; _zgetrf64 = null; _zgesv64 = null;
                _dgetrf32 = null; _dgesv32 = null; _zgetrf32 = null; _zgesv32 = null;
                _dpotrf64 = null; _zpotrf64 = null; _dgeqrf64 = null; _zgeqrf64 = null;
                _dorgqr64 = null; _zungqr64 = null;
                _dpotrf32 = null; _zpotrf32 = null; _dgeqrf32 = null; _zgeqrf32 = null;
                _dorgqr32 = null; _zungqr32 = null;
                _dgesdd64 = null; _zgesdd64 = null; _dgelsd64 = null; _zgelsd64 = null;
                _dgesdd32 = null; _zgesdd32 = null; _dgelsd32 = null; _zgelsd32 = null;
                _dsyevd64 = null; _zheevd64 = null; _dgeev64 = null; _zgeev64 = null;
                _dsyevd32 = null; _zheevd32 = null; _dgeev32 = null; _zgeev32 = null;
                IsLapackLoaded = false;
            }
        }

        /// <summary>
        ///     Resolves every needed entry point in <paramref name="handle"/>, trying the ILP64
        ///     (numpy-wheel) scheme first and falling back to plain LP64 CBLAS names.
        /// </summary>
        /// <remarks>
        ///     Every symbol is resolved into a local BEFORE a single static is written, so this
        ///     either binds completely or not at all. A partial bind is not merely untidy: the
        ///     caller frees the handle when this throws, so a half-written <see cref="_handle"/>
        ///     would leave <see cref="IsLoaded"/> reporting true over freed memory — the next
        ///     <c>OpenBlasEngine.Enable()</c> would skip loading and call into it, and the next
        ///     <see cref="Load"/> would free it a second time.
        /// </remarks>
        private static void Bind(IntPtr handle, string path)
        {
            // (prefix, suffix, ilp64) tuples, most specific first. numpy's wheels rename symbols so
            // an ILP64 OpenBLAS can coexist with a system LP64 one in the same process.
            //
            // ("scipy_", "", false) is scipy-openblas32 — the LP64 sibling distribution, which
            // exports scipy_cblas_sgemm with a 32-bit BLAS integer. It is not an exotic case:
            // NumPy's OWN build picks it over scipy-openblas64 on Windows/ARM64 (see
            // tools/wheels/cibw_before_build.sh), so it is the library a win-arm64 parity run must
            // bind. It has to be probed BEFORE the bare ("", "", false) scheme, since a plain
            // cblas_sgemm export is what a system CBLAS provides and scipy's build has none.
            var schemes = new[]
            {
                new ValueTuple<string, string, bool>("scipy_", "64_", true),
                new ValueTuple<string, string, bool>("", "64_", true),
                new ValueTuple<string, string, bool>("", "_64", true),
                new ValueTuple<string, string, bool>("scipy_", "", false),
                new ValueTuple<string, string, bool>("", "", false)
            };

            foreach (var scheme in schemes)
            {
                string prefix = scheme.Item1, suffix = scheme.Item2;
                bool ilp64 = scheme.Item3;

                IntPtr probe;
                if (!NativeLibrary.TryGetExport(handle, prefix + "cblas_sgemm" + suffix, out probe))
                    continue;

                // Resolve first, commit second. Require() throws when a symbol is missing, and at
                // that point not one static may have been written — see the remarks above.
                IntPtr dgemm = Require(handle, prefix + "cblas_dgemm" + suffix);
                IntPtr sgemv = Require(handle, prefix + "cblas_sgemv" + suffix);
                IntPtr dgemv = Require(handle, prefix + "cblas_dgemv" + suffix);
                IntPtr ssyrk = Require(handle, prefix + "cblas_ssyrk" + suffix);
                IntPtr dsyrk = Require(handle, prefix + "cblas_dsyrk" + suffix);
                IntPtr sdot = Require(handle, prefix + "cblas_sdot" + suffix);
                IntPtr ddot = Require(handle, prefix + "cblas_ddot" + suffix);
                IntPtr saxpy = Require(handle, prefix + "cblas_saxpy" + suffix);
                IntPtr daxpy = Require(handle, prefix + "cblas_daxpy" + suffix);

                _handle = handle;
                LibraryPath = path;
                IsIlp64 = ilp64;
                SymbolScheme = prefix + "*" + suffix;

                if (ilp64)
                {
                    _sgemm64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, long, float, float*, long, float*, long, float, float*, long, void>)probe;
                    _dgemm64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, long, double, double*, long, double*, long, double, double*, long, void>)dgemm;
                    _sgemv64 = (delegate* unmanaged[Cdecl]<int, int, long, long, float, float*, long, float*, long, float, float*, long, void>)sgemv;
                    _dgemv64 = (delegate* unmanaged[Cdecl]<int, int, long, long, double, double*, long, double*, long, double, double*, long, void>)dgemv;
                    _ssyrk64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, float, float*, long, float, float*, long, void>)ssyrk;
                    _dsyrk64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, double, double*, long, double, double*, long, void>)dsyrk;
                    _sdot64 = (delegate* unmanaged[Cdecl]<long, float*, long, float*, long, float>)sdot;
                    _ddot64 = (delegate* unmanaged[Cdecl]<long, double*, long, double*, long, double>)ddot;
                    _saxpy64 = (delegate* unmanaged[Cdecl]<long, float, float*, long, float*, long, void>)saxpy;
                    _daxpy64 = (delegate* unmanaged[Cdecl]<long, double, double*, long, double*, long, void>)daxpy;
                }
                else
                {
                    _sgemm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, int, float, float*, int, float*, int, float, float*, int, void>)probe;
                    _dgemm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, int, double, double*, int, double*, int, double, double*, int, void>)dgemm;
                    _sgemv32 = (delegate* unmanaged[Cdecl]<int, int, int, int, float, float*, int, float*, int, float, float*, int, void>)sgemv;
                    _dgemv32 = (delegate* unmanaged[Cdecl]<int, int, int, int, double, double*, int, double*, int, double, double*, int, void>)dgemv;
                    _ssyrk32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, float, float*, int, float, float*, int, void>)ssyrk;
                    _dsyrk32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, double, double*, int, double, double*, int, void>)dsyrk;
                    _sdot32 = (delegate* unmanaged[Cdecl]<int, float*, int, float*, int, float>)sdot;
                    _ddot32 = (delegate* unmanaged[Cdecl]<int, double*, int, double*, int, double>)ddot;
                    _saxpy32 = (delegate* unmanaged[Cdecl]<int, float, float*, int, float*, int, void>)saxpy;
                    _daxpy32 = (delegate* unmanaged[Cdecl]<int, double, double*, int, double*, int, void>)daxpy;
                }

                // Optional OpenBLAS extensions — absent on a reference CBLAS, which is fine.
                IntPtr p;
                foreach (var name in new[] { prefix + "openblas_set_num_threads" + suffix, "openblas_set_num_threads", "goto_set_num_threads" })
                    if (NativeLibrary.TryGetExport(handle, name, out p)) { _setNumThreads = (delegate* unmanaged[Cdecl]<int, void>)p; break; }
                foreach (var name in new[] { prefix + "openblas_get_num_threads" + suffix, "openblas_get_num_threads" })
                    if (NativeLibrary.TryGetExport(handle, name, out p)) { _getNumThreads = (delegate* unmanaged[Cdecl]<int>)p; break; }
                foreach (var name in new[] { prefix + "openblas_get_config" + suffix, "openblas_get_config" })
                    if (NativeLibrary.TryGetExport(handle, name, out p)) { _getConfig = (delegate* unmanaged[Cdecl]<IntPtr>)p; break; }
                foreach (var name in new[] { prefix + "openblas_get_corename" + suffix, "openblas_get_corename" })
                    if (NativeLibrary.TryGetExport(handle, name, out p)) { _getCoreName = (delegate* unmanaged[Cdecl]<IntPtr>)p; break; }

                // Optional Level-3 BLAS siblings — trmm/trsm/symm/syr2k. Best-effort like the
                // openblas_* extensions above: they are not part of the required core (nothing in the
                // parity backend calls them), so a miss leaves a null pointer the wrapper reports, never
                // failing the gemm/gemv/syrk/dot/axpy bind.
                BindLevel3Extras(handle, prefix, suffix, ilp64);

                // Optional complex128 products — zgemm/zgemv/zsyrk/zdotu_sub/zdotc_sub/zaxpy. Bound
                // all-or-nothing (IsComplexBlasLoaded) so a bare real-only CBLAS declines complex
                // cleanly rather than dereferencing a null pointer. Not part of the required core: the
                // real s/d products are what the bind above guarantees, and a library without the
                // complex set simply leaves complex np.dot/np.matmul on the managed kernel.
                BindComplexBlas(handle, prefix, suffix, ilp64);

                // Optional LAPACK LU family — a full OpenBLAS ships it, a bare reference CBLAS does not.
                // Bound all-or-nothing so IsLapackLoaded is honest: a half-bound set would let a
                // factorisation call a null pointer.
                BindLapack(handle, prefix, suffix, ilp64);

                return;
            }

            throw new EntryPointNotFoundException(
                $"OpenBlasEngine.Enable: '{path}' loaded but exports no recognizable cblas_sgemm symbol " +
                "(tried the scipy_*64_, *64_, *_64 and plain CBLAS naming schemes). It is probably " +
                "not a CBLAS provider.");
        }

        private static IntPtr Require(IntPtr handle, string name)
        {
            IntPtr p;
            if (!NativeLibrary.TryGetExport(handle, name, out p))
                throw new EntryPointNotFoundException(
                    $"OpenBlasEngine.Enable: the loaded BLAS library is missing the required symbol '{name}'.");
            return p;
        }

        /// <summary>
        ///     Best-effort binding of the real (s/d) Level-3 BLAS siblings of gemm/syrk — trmm, trsm,
        ///     symm and syr2k. These are NOT called by the parity backend (NumPy's np.* matrix dispatch
        ///     uses only gemm/gemv/syrk/dot; trmm/trsm live in NumPy solely inside its private reference
        ///     LAPACK), so they are OPTIONAL: a library missing one leaves that pointer null and the
        ///     wrapper throws only if the routine is actually called — exactly like the openblas_*
        ///     extensions. They exist as building blocks a future consumer can reach through
        ///     <c>IBlasType&lt;T&gt;</c>. Uses the same (prefix, suffix) scheme the required core bound.
        /// </summary>
        private static void BindLevel3Extras(IntPtr handle, string prefix, string suffix, bool ilp64)
        {
            IntPtr p;
            if (ilp64)
            {
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_strmm" + suffix, out p)) _strmm64 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, long, long, float, float*, long, float*, long, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_dtrmm" + suffix, out p)) _dtrmm64 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, long, long, double, double*, long, double*, long, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_strsm" + suffix, out p)) _strsm64 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, long, long, float, float*, long, float*, long, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_dtrsm" + suffix, out p)) _dtrsm64 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, long, long, double, double*, long, double*, long, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_ssymm" + suffix, out p)) _ssymm64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, float, float*, long, float*, long, float, float*, long, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_dsymm" + suffix, out p)) _dsymm64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, double, double*, long, double*, long, double, double*, long, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_ssyr2k" + suffix, out p)) _ssyr2k64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, float, float*, long, float*, long, float, float*, long, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_dsyr2k" + suffix, out p)) _dsyr2k64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, double, double*, long, double*, long, double, double*, long, void>)p;
            }
            else
            {
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_strmm" + suffix, out p)) _strmm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, int, int, float, float*, int, float*, int, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_dtrmm" + suffix, out p)) _dtrmm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, int, int, double, double*, int, double*, int, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_strsm" + suffix, out p)) _strsm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, int, int, float, float*, int, float*, int, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_dtrsm" + suffix, out p)) _dtrsm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, int, int, double, double*, int, double*, int, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_ssymm" + suffix, out p)) _ssymm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, float, float*, int, float*, int, float, float*, int, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_dsymm" + suffix, out p)) _dsymm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, double, double*, int, double*, int, double, double*, int, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_ssyr2k" + suffix, out p)) _ssyr2k32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, float, float*, int, float*, int, float, float*, int, void>)p;
                if (NativeLibrary.TryGetExport(handle, prefix + "cblas_dsyr2k" + suffix, out p)) _dsyr2k32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, double, double*, int, double*, int, double, double*, int, void>)p;
            }
        }

        /// <summary>
        ///     Binds the complex128 CBLAS products if the library exports them. Like
        ///     <see cref="BindLapack"/> this is OPTIONAL and ALL-OR-NOTHING: a bare real-only reference
        ///     CBLAS has no complex products, so a miss simply leaves <see cref="IsComplexBlasLoaded"/>
        ///     false and complex <c>np.dot</c>/<c>np.matmul</c> fall through to NumSharp's managed
        ///     complex kernel. A partially-resolved set would let a complex product dereference a null
        ///     pointer, so every symbol is resolved into a local before a single static is written.
        /// </summary>
        private static void BindComplexBlas(IntPtr handle, string prefix, string suffix, bool ilp64)
        {
            if (!TryGet(handle, prefix + "cblas_zgemm" + suffix, out var zgemm) ||
                !TryGet(handle, prefix + "cblas_zgemv" + suffix, out var zgemv) ||
                !TryGet(handle, prefix + "cblas_zsyrk" + suffix, out var zsyrk) ||
                !TryGet(handle, prefix + "cblas_zdotu_sub" + suffix, out var zdotu) ||
                !TryGet(handle, prefix + "cblas_zdotc_sub" + suffix, out var zdotc) ||
                !TryGet(handle, prefix + "cblas_zaxpy" + suffix, out var zaxpy))
            {
                return; // no complex products — complex dot/matmul stay on the managed kernel
            }

            if (ilp64)
            {
                _zgemm64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, long, double*, double*, long, double*, long, double*, double*, long, void>)zgemm;
                _zgemv64 = (delegate* unmanaged[Cdecl]<int, int, long, long, double*, double*, long, double*, long, double*, double*, long, void>)zgemv;
                _zsyrk64 = (delegate* unmanaged[Cdecl]<int, int, int, long, long, double*, double*, long, double*, double*, long, void>)zsyrk;
                _zdotu_sub64 = (delegate* unmanaged[Cdecl]<long, double*, long, double*, long, double*, void>)zdotu;
                _zdotc_sub64 = (delegate* unmanaged[Cdecl]<long, double*, long, double*, long, double*, void>)zdotc;
                _zaxpy64 = (delegate* unmanaged[Cdecl]<long, double*, double*, long, double*, long, void>)zaxpy;
            }
            else
            {
                _zgemm32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, int, double*, double*, int, double*, int, double*, double*, int, void>)zgemm;
                _zgemv32 = (delegate* unmanaged[Cdecl]<int, int, int, int, double*, double*, int, double*, int, double*, double*, int, void>)zgemv;
                _zsyrk32 = (delegate* unmanaged[Cdecl]<int, int, int, int, int, double*, double*, int, double*, double*, int, void>)zsyrk;
                _zdotu_sub32 = (delegate* unmanaged[Cdecl]<int, double*, int, double*, int, double*, void>)zdotu;
                _zdotc_sub32 = (delegate* unmanaged[Cdecl]<int, double*, int, double*, int, double*, void>)zdotc;
                _zaxpy32 = (delegate* unmanaged[Cdecl]<int, double*, double*, int, double*, int, void>)zaxpy;
            }

            IsComplexBlasLoaded = true;
        }

        /// <summary>Thin <see cref="NativeLibrary.TryGetExport"/> wrapper for the all-or-nothing binds.</summary>
        private static bool TryGet(IntPtr handle, string name, out IntPtr p)
            => NativeLibrary.TryGetExport(handle, name, out p);

        /// <summary>
        ///     Binds the LAPACK LU routines if the library exports them. Unlike the CBLAS symbols this
        ///     is OPTIONAL: a reference CBLAS has no LAPACK, so a miss simply leaves
        ///     <see cref="IsLapackLoaded"/> false and the factorisation backend declines (the engine
        ///     then raises the same "needs a LAPACK backend" the no-backend case does). Bound
        ///     all-or-nothing — a partially-resolved set would let a call dereference a null pointer.
        /// </summary>
        private static void BindLapack(IntPtr handle, string prefix, string suffix, bool ilp64)
        {
            if (!TryResolveLapack(handle, prefix, suffix, "dgetrf", out var dgetrf) ||
                !TryResolveLapack(handle, prefix, suffix, "dgesv", out var dgesv) ||
                !TryResolveLapack(handle, prefix, suffix, "zgetrf", out var zgetrf) ||
                !TryResolveLapack(handle, prefix, suffix, "zgesv", out var zgesv) ||
                !TryResolveLapack(handle, prefix, suffix, "dpotrf", out var dpotrf) ||
                !TryResolveLapack(handle, prefix, suffix, "zpotrf", out var zpotrf) ||
                !TryResolveLapack(handle, prefix, suffix, "dgeqrf", out var dgeqrf) ||
                !TryResolveLapack(handle, prefix, suffix, "zgeqrf", out var zgeqrf) ||
                !TryResolveLapack(handle, prefix, suffix, "dorgqr", out var dorgqr) ||
                !TryResolveLapack(handle, prefix, suffix, "zungqr", out var zungqr) ||
                !TryResolveLapack(handle, prefix, suffix, "dgesdd", out var dgesdd) ||
                !TryResolveLapack(handle, prefix, suffix, "zgesdd", out var zgesdd) ||
                !TryResolveLapack(handle, prefix, suffix, "dgelsd", out var dgelsd) ||
                !TryResolveLapack(handle, prefix, suffix, "zgelsd", out var zgelsd) ||
                !TryResolveLapack(handle, prefix, suffix, "dsyevd", out var dsyevd) ||
                !TryResolveLapack(handle, prefix, suffix, "zheevd", out var zheevd) ||
                !TryResolveLapack(handle, prefix, suffix, "dgeev", out var dgeev) ||
                !TryResolveLapack(handle, prefix, suffix, "zgeev", out var zgeev))
            {
                return; // no LAPACK — factorisations decline, products still work
            }

            if (ilp64)
            {
                _dgetrf64 = (delegate* unmanaged[Cdecl]<long*, long*, double*, long*, long*, long*, void>)dgetrf;
                _dgesv64 = (delegate* unmanaged[Cdecl]<long*, long*, double*, long*, long*, double*, long*, long*, void>)dgesv;
                _zgetrf64 = (delegate* unmanaged[Cdecl]<long*, long*, double*, long*, long*, long*, void>)zgetrf;
                _zgesv64 = (delegate* unmanaged[Cdecl]<long*, long*, double*, long*, long*, double*, long*, long*, void>)zgesv;
                _dpotrf64 = (delegate* unmanaged[Cdecl]<byte*, long*, double*, long*, long*, void>)dpotrf;
                _zpotrf64 = (delegate* unmanaged[Cdecl]<byte*, long*, double*, long*, long*, void>)zpotrf;
                _dgeqrf64 = (delegate* unmanaged[Cdecl]<long*, long*, double*, long*, double*, double*, long*, long*, void>)dgeqrf;
                _zgeqrf64 = (delegate* unmanaged[Cdecl]<long*, long*, double*, long*, double*, double*, long*, long*, void>)zgeqrf;
                _dorgqr64 = (delegate* unmanaged[Cdecl]<long*, long*, long*, double*, long*, double*, double*, long*, long*, void>)dorgqr;
                _zungqr64 = (delegate* unmanaged[Cdecl]<long*, long*, long*, double*, long*, double*, double*, long*, long*, void>)zungqr;
                _dgesdd64 = (delegate* unmanaged[Cdecl]<byte*, long*, long*, double*, long*, double*, double*, long*, double*, long*, double*, long*, long*, long*, void>)dgesdd;
                _zgesdd64 = (delegate* unmanaged[Cdecl]<byte*, long*, long*, double*, long*, double*, double*, long*, double*, long*, double*, long*, double*, long*, long*, void>)zgesdd;
                _dgelsd64 = (delegate* unmanaged[Cdecl]<long*, long*, long*, double*, long*, double*, long*, double*, double*, long*, double*, long*, long*, long*, void>)dgelsd;
                _zgelsd64 = (delegate* unmanaged[Cdecl]<long*, long*, long*, double*, long*, double*, long*, double*, double*, long*, double*, long*, double*, long*, long*, void>)zgelsd;
                _dsyevd64 = (delegate* unmanaged[Cdecl]<byte*, byte*, long*, double*, long*, double*, double*, long*, long*, long*, long*, void>)dsyevd;
                _zheevd64 = (delegate* unmanaged[Cdecl]<byte*, byte*, long*, double*, long*, double*, double*, long*, double*, long*, long*, long*, long*, void>)zheevd;
                _dgeev64 = (delegate* unmanaged[Cdecl]<byte*, byte*, long*, double*, long*, double*, double*, double*, long*, double*, long*, double*, long*, long*, void>)dgeev;
                _zgeev64 = (delegate* unmanaged[Cdecl]<byte*, byte*, long*, double*, long*, double*, double*, long*, double*, long*, double*, long*, double*, long*, void>)zgeev;
            }
            else
            {
                _dgetrf32 = (delegate* unmanaged[Cdecl]<int*, int*, double*, int*, int*, int*, void>)dgetrf;
                _dgesv32 = (delegate* unmanaged[Cdecl]<int*, int*, double*, int*, int*, double*, int*, int*, void>)dgesv;
                _zgetrf32 = (delegate* unmanaged[Cdecl]<int*, int*, double*, int*, int*, int*, void>)zgetrf;
                _zgesv32 = (delegate* unmanaged[Cdecl]<int*, int*, double*, int*, int*, double*, int*, int*, void>)zgesv;
                _dpotrf32 = (delegate* unmanaged[Cdecl]<byte*, int*, double*, int*, int*, void>)dpotrf;
                _zpotrf32 = (delegate* unmanaged[Cdecl]<byte*, int*, double*, int*, int*, void>)zpotrf;
                _dgeqrf32 = (delegate* unmanaged[Cdecl]<int*, int*, double*, int*, double*, double*, int*, int*, void>)dgeqrf;
                _zgeqrf32 = (delegate* unmanaged[Cdecl]<int*, int*, double*, int*, double*, double*, int*, int*, void>)zgeqrf;
                _dorgqr32 = (delegate* unmanaged[Cdecl]<int*, int*, int*, double*, int*, double*, double*, int*, int*, void>)dorgqr;
                _zungqr32 = (delegate* unmanaged[Cdecl]<int*, int*, int*, double*, int*, double*, double*, int*, int*, void>)zungqr;
                _dgesdd32 = (delegate* unmanaged[Cdecl]<byte*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*, int*, void>)dgesdd;
                _zgesdd32 = (delegate* unmanaged[Cdecl]<byte*, int*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, double*, int*, int*, void>)zgesdd;
                _dgelsd32 = (delegate* unmanaged[Cdecl]<int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*, void>)dgelsd;
                _zgelsd32 = (delegate* unmanaged[Cdecl]<int*, int*, int*, double*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, int*, void>)zgelsd;
                _dsyevd32 = (delegate* unmanaged[Cdecl]<byte*, byte*, int*, double*, int*, double*, double*, int*, int*, int*, int*, void>)dsyevd;
                _zheevd32 = (delegate* unmanaged[Cdecl]<byte*, byte*, int*, double*, int*, double*, double*, int*, double*, int*, int*, int*, int*, void>)zheevd;
                _dgeev32 = (delegate* unmanaged[Cdecl]<byte*, byte*, int*, double*, int*, double*, double*, double*, int*, double*, int*, double*, int*, int*, void>)dgeev;
                _zgeev32 = (delegate* unmanaged[Cdecl]<byte*, byte*, int*, double*, int*, double*, double*, int*, double*, int*, double*, int*, double*, int*, void>)zgeev;
            }

            IsLapackLoaded = true;
        }

        /// <summary>
        ///     Resolves a Fortran LAPACK symbol across the spellings the supported builds use: the
        ///     scheme's own (<c>scipy_dgesv64_</c> / <c>scipy_dgesv</c>), the Fortran trailing-underscore
        ///     form a stock OpenBLAS exports (<c>dgesv_</c>), and their uppercase variants.
        /// </summary>
        private static bool TryResolveLapack(IntPtr handle, string prefix, string suffix, string baseName, out IntPtr p)
        {
            string upper = baseName.ToUpperInvariant();
            var candidates = new[]
            {
                prefix + baseName + suffix,
                prefix + baseName + "_" + suffix,
                prefix + upper + suffix,
                prefix + upper + "_" + suffix
            };

            foreach (var name in candidates)
                if (NativeLibrary.TryGetExport(handle, name, out p))
                    return true;

            p = IntPtr.Zero;
            return false;
        }

        /// <summary>
        ///     TIER 2a — the non-binding override locations: <c>NUMSHARP_OPENBLAS_SEARCH_PATH</c> (env),
        ///     then a path-mode source marker the build recorded (<c>&lt;OpenBlasPath&gt;</c> on the
        ///     <c>PackageReference</c>). Env before metadata, per the delivery design.
        /// </summary>
        private static IEnumerable<string> OverridePathCandidates(OpenBlasSourceMarker marker)
        {
            foreach (var p in UserPathCandidates())
                yield return p;

            if (marker != null && marker.IsPathMode)
                foreach (var dir in marker.ReadDirectories())
                    foreach (var p in Expand(dir))
                        yield return p;
        }

        /// <summary>
        ///     TIER 2b — the staged version override's candidates. When the marker pins a sha256,
        ///     only a file hashing to it qualifies: a same-named file with different bytes is not
        ///     the pinned binary, and loading it anyway would be a silent substitution.
        /// </summary>
        private static IEnumerable<string> VersionOverrideCandidates(OpenBlasSourceMarker marker)
        {
            foreach (var dir in marker.ReadDirectories())
                foreach (var p in Expand(dir))
                {
                    if (marker.Sha256 != null && !FileHashEquals(p, marker.Sha256))
                        continue;
                    yield return p;
                }
        }

        private static bool FileHashEquals(string file, string expectedSha256Hex)
        {
            try
            {
                using var stream = File.OpenRead(file);
                var actual = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream));
                return string.Equals(actual, expectedSha256Hex, StringComparison.OrdinalIgnoreCase);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                return false; // unreadable ⇒ unverifiable ⇒ not the pinned binary
            }
        }

        /// <summary>
        ///     TIERS 3–4: candidates for the unattended case once the overrides above have declined,
        ///     in a DELIBERATE and fixed order — first any explicit OpenBLAS root
        ///     (<c>OPENBLAS_HOME</c>/<c>OPENBLAS_ROOT</c>, which a caller sets deliberately and so
        ///     outranks the bundle), then the OpenBLAS this package bundles, then ambient machine
        ///     tooling (system install directories, bare loader names, a PATH sweep).
        /// </summary>
        /// <remarks>
        ///     <b>The bundled binary is probed ahead of all AMBIENT machine tooling, on purpose</b>
        ///     (delivery design §10.1, resolved 2026-08-13: parity-by-default preserved) — the sole
        ///     thing that outranks it is an explicitly-set <c>OPENBLAS_HOME</c>/<c>OPENBLAS_ROOT</c>
        ///     (a deliberate caller opt-in, not ambient, so setting it trades parity-by-default for
        ///     the named build). It is the
        ///     prebuilt scipy-openblas artifact NumPy itself pins (verified byte-identical to the
        ///     library in numpy 2.4.2's wheels), so referencing the package keeps giving
        ///     NumPy-identical bits by default — the answer stays a property of the PACKAGE VERSION
        ///     rather than of whatever OpenBLAS an OS package manager happened to install. A
        ///     library's numeric output should not change because an unrelated <c>apt install</c> or
        ///     <c>pip install</c> ran, and a .NET app on a bare machine should not get a different
        ///     backend from one on a developer box.
        ///     <para>
        ///     A pip-installed numpy's own <c>numpy.libs/</c> is deliberately NOT scanned (removed
        ///     2026-08-13): we never grab OpenBLAS out of a numpy installation. A conda/system
        ///     <i>OpenBLAS</i> package is still machine tooling below; a <i>numpy</i> is not. To
        ///     match a numpy whose OpenBLAS differs from the bundled one, NAME it —
        ///     <c>OpenBlasEngine.Enable(path)</c> or <c>NUMSHARP_OPENBLAS_LIBRARY</c> (binding), or point
        ///     <c>NUMSHARP_OPENBLAS_SEARCH_PATH</c> at it (non-binding priority).
        ///     <c>NUMSHARP_OPENBLAS_USE_BUNDLED=0</c> drops the bundled entry entirely, making machine
        ///     tooling the discovery default.
        ///     </para>
        /// </remarks>
        private static IEnumerable<string> AmbientCandidates()
        {
            // Explicit OpenBLAS roots (OPENBLAS_HOME / OPENBLAS_ROOT) are a deliberate "use the
            // OpenBLAS installed here" signal, so they outrank the bundled parity default — placed
            // directly below the NUMSHARP_OPENBLAS_* env-var tiers and above the bundle, the ambient
            // machine directories and the PATH sweep. (CONDA_PREFIX / VCPKG_ROOT are ambient rather
            // than this explicit, and stay in the machine-tooling tier below the bundle.)
            foreach (var dir in ExplicitOpenBlasRootDirectories())
                foreach (var p in Expand(dir))
                    yield return p;

            if (EnvVars.OpenBlasUseBundled)
                foreach (var dir in BundledDirectories())
                    foreach (var p in Expand(dir))
                        yield return p;

            // Machine-wide OpenBLAS placed by an OS package manager or a source build. These builds
            // are NOT byte-parity with NumPy, so they sit BELOW the bundled asset: a correct, fast
            // BLAS, not a bit-identical one.
            foreach (var dir in SystemBlasDirectories())
                foreach (var p in Expand(dir))
                    yield return p;

            // Bare names — resolved by the platform loader search path. Both scipy distributions
            // are covered: the ILP64 build (scipy-openblas64, "…64_") and its LP64 sibling
            // (scipy-openblas32, plain "scipy_openblas" / "libscipy_openblas" — the only build for
            // 32-bit x86, and the one NumPy itself ships on win-arm64). 64 first so a parity host
            // that has both prefers the ILP64 build NumPy uses.
            yield return "libscipy_openblas64_";
            yield return "libscipy_openblas";
            yield return "scipy_openblas";
            yield return "libopenblas64_";
            yield return "libopenblas";
            yield return "openblas";
            yield return "libblas";

            // Last resort: sweep every directory on PATH for a BLAS whose file name none of the bare
            // names above match (a renamed OpenBLAS, a vendor CBLAS). Broadest and most expensive
            // scan, and the likeliest to bind something unintended, so it runs only after every
            // more-specific source has declined — lazy iteration means it never runs when an earlier
            // tier binds.
            foreach (var dir in PathEnvDirectories())
                foreach (var p in Expand(dir))
                    yield return p;
        }

        /// <summary>
        ///     Directories that hold this package's bundled OpenBLAS runtime asset.
        /// </summary>
        /// <remarks>
        ///     NuGet lays native assets out as <c>runtimes/&lt;rid&gt;/native/</c>, which is what a
        ///     plain <c>dotnet build</c> reproduces under the output directory. A RID-specific or
        ///     self-contained publish instead FLATTENS them next to the app, so the app directory
        ///     itself is probed too. Both the running assembly's directory and
        ///     <see cref="AppContext.BaseDirectory"/> are used because they differ when NumSharp is
        ///     loaded as a plugin or from a shadow-copied location.
        /// </remarks>
        private static IEnumerable<string> BundledDirectories()
        {
            foreach (var b in ProbeBases())
            {
                foreach (var rid in RuntimeIdentifierCandidates())
                    yield return Path.Combine(b, "runtimes", rid, "native");

                // publish flattens native assets into the app directory
                yield return b;
            }
        }

        /// <summary>
        ///     Base directories the app's file assets can sit under — shared by the bundled-asset
        ///     probe and the source-marker probe (<see cref="OpenBlasSourceMarker"/>), which must
        ///     agree on where "next to the app" is.
        /// </summary>
        internal static IEnumerable<string> ProbeBases()
        {
            var bases = new List<string>();
            var appBase = AppContext.BaseDirectory;
            if (!string.IsNullOrEmpty(appBase))
                bases.Add(appBase);

            try
            {
                var loc = typeof(OpenBlasNative).Assembly.Location;
                if (!string.IsNullOrEmpty(loc))
                {
                    var dir = Path.GetDirectoryName(loc);
                    if (!string.IsNullOrEmpty(dir) && !bases.Contains(dir, StringComparer.OrdinalIgnoreCase))
                        bases.Add(dir);
                }
            }
            catch (NotSupportedException)
            {
                // single-file publish: Location is empty/unsupported, AppContext.BaseDirectory covers it
            }

            return bases;
        }

        /// <summary>RID folder names this machine could legitimately load a native asset from.</summary>
        internal static IEnumerable<string> RuntimeIdentifierCandidates()
        {
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            // The runtime's own answer first — it already knows about musl and about the exact
            // architecture, and on .NET 8+ it is the portable form ("win-x64", "linux-musl-arm64").
            string reported = null;
            try
            {
                reported = RuntimeInformation.RuntimeIdentifier;
            }
            catch (PlatformNotSupportedException)
            {
            }

            if (!string.IsNullOrEmpty(reported) && seen.Add(reported))
                yield return reported;

            string arch;
            switch (RuntimeInformation.ProcessArchitecture)
            {
                case Architecture.X64: arch = "x64"; break;
                case Architecture.Arm64: arch = "arm64"; break;
                case Architecture.X86: arch = "x86"; break;
                case Architecture.Arm: arch = "arm"; break;
                default: yield break;
            }

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (seen.Add("win-" + arch)) yield return "win-" + arch;
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                if (seen.Add("osx-" + arch)) yield return "osx-" + arch;
            }
            else
            {
                // A glibc host must not load the musl asset or vice versa, but the reported RID
                // above already distinguishes them; these are only the fallbacks for a runtime that
                // did not report one, so both spellings are offered and the loader rejects the
                // wrong one (it simply fails to load, which is a silent decline here).
                if (seen.Add("linux-" + arch)) yield return "linux-" + arch;
                if (seen.Add("linux-musl-" + arch)) yield return "linux-musl-" + arch;
            }
        }

        /// <summary>Expands a path that may be a file, or a directory holding a BLAS library.</summary>
        private static IEnumerable<string> Expand(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                yield break;

            if (File.Exists(path))
            {
                yield return path;
                yield break;
            }

            if (!Directory.Exists(path))
                yield break;

            // Newest first: a machine can hold several wheels; the freshest is the likeliest match.
            var patterns = new[] { "*scipy_openblas*", "*openblas*", "*blas*" };
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var pattern in patterns)
            {
                // Discovery runs from a [ModuleInitializer]; an unreadable directory on PATH must
                // not become a TypeInitializationException for the whole assembly.
                string[] found;
                try
                {
                    found = Directory.GetFiles(path, pattern);
                }
                catch (IOException)
                {
                    yield break;
                }
                catch (UnauthorizedAccessException)
                {
                    yield break;
                }

                var files = new List<string>();
                foreach (var f in found)
                {
                    var ext = Path.GetExtension(f);
                    if (ext.Equals(".dll", StringComparison.OrdinalIgnoreCase) ||
                        ext.Equals(".so", StringComparison.OrdinalIgnoreCase) ||
                        ext.Equals(".dylib", StringComparison.OrdinalIgnoreCase) ||
                        f.Contains(".so.", StringComparison.OrdinalIgnoreCase))
                        files.Add(f);
                }

                files.Sort(StringComparer.OrdinalIgnoreCase);
                for (int i = files.Count - 1; i >= 0; i--)
                    if (seen.Add(files[i]))
                        yield return files[i];
            }
        }

        /// <summary>
        ///     Explicit, HIGHEST-PRIORITY discovery location(s) from <c>NUMSHARP_OPENBLAS_SEARCH_PATH</c> —
        ///     one or more files or directories (separated by the platform path separator), tried
        ///     FIRST, ahead of the bundled asset and every other candidate.
        /// </summary>
        /// <remarks>
        ///     The non-binding sibling of <c>NUMSHARP_OPENBLAS_LIBRARY</c>. That one is BINDING — when set,
        ///     ONLY it is tried and a failure is fatal, because it is how a caller pins the ONE
        ///     specific binary a parity claim is about. This one is softer: it takes PRIORITY over the
        ///     bundled default and the ambient scan when it holds a loadable BLAS, but is silently
        ///     skipped (falling through to the rest) when it does not. Use it to make a local build or
        ///     an unpacked release win over the bundled binary without the all-or-nothing failure of
        ///     the binding variable.
        /// </remarks>
        private static IEnumerable<string> UserPathCandidates()
        {
            var v = EnvVars.OpenBlasSearchPath;
            if (string.IsNullOrWhiteSpace(v))
                yield break;

            foreach (var entry in v.Split(Path.PathSeparator))
                foreach (var p in Expand(entry))
                    yield return p;
        }

        /// <summary>
        ///     Explicit OpenBLAS install roots named by <c>OPENBLAS_HOME</c> / <c>OPENBLAS_ROOT</c>
        ///     (and the mixed-case <c>OpenBLAS_HOME</c> / <c>OpenBLAS_ROOT</c> some builds export).
        /// </summary>
        /// <remarks>
        ///     Setting one is a deliberate "use the OpenBLAS installed here" instruction, so — unlike
        ///     the AMBIENT <c>CONDA_PREFIX</c>/<c>VCPKG_ROOT</c> trees in
        ///     <see cref="SystemBlasDirectories"/> — it outranks the bundled parity asset: this tier
        ///     sits directly beneath the <c>NUMSHARP_OPENBLAS_*</c> env-var tiers and ABOVE the
        ///     bundle, the machine directories and the PATH sweep. A machine OpenBLAS is NOT
        ///     byte-parity with NumPy, so setting <c>OPENBLAS_HOME</c> deliberately trades
        ///     parity-by-default for the named build.
        /// </remarks>
        private static IEnumerable<string> ExplicitOpenBlasRootDirectories()
        {
            foreach (var key in new[] { "OPENBLAS_HOME", "OPENBLAS_ROOT", "OpenBLAS_HOME", "OpenBLAS_ROOT" })
            {
                var v = Environment.GetEnvironmentVariable(key);
                if (string.IsNullOrWhiteSpace(v))
                    continue;
                yield return v;
                yield return Path.Combine(v, "lib");
                yield return Path.Combine(v, "bin");
            }
        }

        /// <summary>
        ///     Conventional install directories of the OpenBLAS builds documented at
        ///     openmathlib.org/OpenBLAS/docs/install — the OS package managers (apt/dnf/pacman/zypper,
        ///     Homebrew/MacPorts, conda-forge, vcpkg, FreeBSD ports) and the source-build default
        ///     prefix.
        /// </summary>
        /// <remarks>
        ///     Scanned so a machine-wide OpenBLAS (<c>apt install libopenblas0</c>,
        ///     <c>brew install openblas</c>, …) is found even when it is not on the loader's default
        ///     search path or carries a versioned soname (<c>libopenblas.so.0</c>) the bare names
        ///     below do not match. <b>None of these is byte-parity with NumPy</b> — a different
        ///     compiler and configuration — so this tier ranks below the overrides and the bundled
        ///     asset: it yields a correct, fast BLAS, not a bit-identical one. (A conda-installed
        ///     <i>openblas</i> counts as machine tooling via <c>CONDA_PREFIX</c>; a pip-installed
        ///     <i>numpy's</i> own <c>numpy.libs/</c> is deliberately never scanned.)
        ///     The install page gives package-manager names but no literal paths,
        ///     so these are each manager's conventional prefix; the ambient env markers
        ///     (<c>VCPKG_ROOT</c>, <c>CONDA_PREFIX</c>) are honoured because that is where those
        ///     managers place their trees. (<c>OPENBLAS_HOME</c>/<c>OPENBLAS_ROOT</c> are handled one
        ///     tier up, ABOVE the bundle — see <see cref="ExplicitOpenBlasRootDirectories"/>.)
        /// </remarks>
        private static IEnumerable<string> SystemBlasDirectories()
        {
            // NOTE: the explicit OPENBLAS_HOME / OPENBLAS_ROOT roots are handled ONE tier up (see
            // ExplicitOpenBlasRootDirectories), above the bundle — a caller sets those deliberately.
            // What remains here is AMBIENT tooling: a conda/vcpkg activation or a distro package,
            // which sits BELOW the bundled parity default.
            var conda = EnvVars.CondaPrefix;
            if (!string.IsNullOrWhiteSpace(conda))
            {
                yield return Path.Combine(conda, "lib");              // posix conda
                yield return Path.Combine(conda, "Library", "bin");   // windows conda
                yield return Path.Combine(conda, "Library", "lib");
            }

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var vcpkg = EnvVars.VcpkgRoot;
                if (!string.IsNullOrWhiteSpace(vcpkg))
                    foreach (var triplet in new[] { "x64-windows", "arm64-windows", "x86-windows" })
                        yield return Path.Combine(vcpkg, "installed", triplet, "bin");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                yield return "/opt/homebrew/opt/openblas/lib";  // apple-silicon brew keg
                yield return "/usr/local/opt/openblas/lib";     // intel brew keg
                yield return "/opt/homebrew/lib";
                yield return "/usr/local/lib";
                yield return "/opt/local/lib";                  // macports
            }
            else
            {
                yield return "/usr/lib/x86_64-linux-gnu";       // debian/ubuntu multiarch
                yield return "/usr/lib/aarch64-linux-gnu";
                yield return "/usr/lib64";                      // fedora/rhel/suse
                yield return "/usr/lib";
                yield return "/usr/local/lib";                  // source install / freebsd
                yield return "/usr/local/lib64";
                yield return "/opt/OpenBLAS/lib";               // OpenBLAS `make PREFIX=/opt/OpenBLAS`
            }
        }

        /// <summary>
        ///     Every directory on the process <c>PATH</c>, swept as the LAST resort for a BLAS whose
        ///     file name is not one of the bare loader names.
        /// </summary>
        /// <remarks>
        ///     The bare names cover the common case — the OS loader already searches <c>PATH</c> on
        ///     Windows and the ld cache elsewhere — but only for a library actually NAMED
        ///     <c>libopenblas</c>/…. A CBLAS dropped on <c>PATH</c> under a different name (a vendor
        ///     MKL, a renamed OpenBLAS) is invisible to them; globbing each entry for
        ///     <c>*openblas*</c>/<c>*blas*</c> (via <see cref="Expand"/>) finds it. It is dead last
        ///     because it is the broadest, most expensive scan and the likeliest to bind something
        ///     unintended — and because iteration is lazy, it never runs when any earlier tier binds.
        /// </remarks>
        private static IEnumerable<string> PathEnvDirectories()
        {
            var pathVar = EnvVars.SystemPath;
            if (string.IsNullOrEmpty(pathVar))
                yield break;

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var entry in pathVar.Split(Path.PathSeparator))
            {
                if (string.IsNullOrWhiteSpace(entry))
                    continue;
                if (seen.Add(entry))
                    yield return entry;
            }
        }

        /// <summary>The <c>OPENBLAS_CORETYPE</c> this process asked for, or null if it never did.</summary>
        internal static string RequestedCoreType { get; private set; }

        /// <summary>
        ///     Validates a core type requested when the library is ALREADY loaded, where it can no
        ///     longer be applied.
        /// </summary>
        /// <remarks>
        ///     OpenBLAS reads <c>OPENBLAS_CORETYPE</c> once, while its DYNAMIC_ARCH initialiser
        ///     runs, so a request arriving after that is unenforceable — and silently dropping it is
        ///     the worst answer available: the caller believes the kernel is pinned, every product
        ///     still computes, and the bits are whatever the CPU happened to choose. So it either
        ///     already holds (idempotent success) or it is reported.
        /// </remarks>
        internal static void VerifyCoreTypeAlreadyInEffect(string coreType)
        {
            // An impossible request is impossible regardless of load order — report that first,
            // since it is the more specific and more actionable failure.
            EnsureCoreTypeRunnable(coreType);

            var current = GetCoreName();
            if (current == null || string.Equals(current, coreType, StringComparison.OrdinalIgnoreCase))
                return;

            throw new InvalidOperationException(
                $"OpenBlasEngine.Enable: cannot pin OPENBLAS_CORETYPE to '{coreType}' — a BLAS library is " +
                $"already loaded in this process and dispatched to '{current}'. OpenBLAS reads that " +
                "variable once, while it is selecting a micro-kernel, and never re-reads it, so the " +
                "choice is fixed for the lifetime of the process. Pin the core type on the FIRST " +
                "OpenBlasEngine.Enable call, or set OPENBLAS_CORETYPE in the environment before the process " +
                "starts.");
        }

        [DllImport("ucrtbase.dll", EntryPoint = "_putenv_s", CharSet = CharSet.Ansi)]
        private static extern int WindowsPutEnv(string name, string value);

        [DllImport("libc", EntryPoint = "setenv", CharSet = CharSet.Ansi)]
        private static extern int UnixSetEnv(string name, string value, int overwrite);

        [DllImport("ucrtbase.dll", EntryPoint = "hypot", CallingConvention = CallingConvention.Cdecl)]
        private static extern double WindowsHypot(double x, double y);

        [DllImport("libm", EntryPoint = "hypot", CallingConvention = CallingConvention.Cdecl)]
        private static extern double UnixHypot(double x, double y);

        private static readonly bool IsWindowsOS = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        private static int _hypotMode; // 0 = unprobed, 1 = system hypot, 2 = managed fallback

        /// <summary>
        ///     <c>|z|</c> via the SAME system <c>hypot</c> NumPy's <c>npy_cabs</c> links against, so the
        ///     complex determinant fold (<c>det</c>/<c>slogdet</c>) stays bit-identical to NumPy.
        /// </summary>
        /// <remarks>
        ///     .NET's <see cref="System.Numerics.Complex.Abs"/> uses its own scaled hypot, which differs
        ///     from the CRT's by ~1 ULP on ~12 % of inputs — enough to shift the last bits of a complex
        ///     determinant. NumPy's <c>npy_hypot</c> is a straight <c>return hypot(x, y)</c>, so calling
        ///     that same CRT routine reproduces it exactly. On the rare host with no callable system
        ///     <c>hypot</c> this degrades to a portable scaled hypot (correct, not bit-pinned).
        /// </remarks>
        internal static double Hypot(double x, double y)
        {
            switch (_hypotMode)
            {
                case 1:
                    return IsWindowsOS ? WindowsHypot(x, y) : UnixHypot(x, y);
                case 2:
                    break;
                default:
                    try
                    {
                        double r = IsWindowsOS ? WindowsHypot(x, y) : UnixHypot(x, y);
                        _hypotMode = 1;
                        return r;
                    }
                    catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
                    {
                        _hypotMode = 2;
                    }

                    break;
            }

            double ax = Math.Abs(x), ay = Math.Abs(y);
            if (ax < ay)
                (ax, ay) = (ay, ax);
            if (ax == 0.0)
                return 0.0;
            double t = ay / ax;
            return ax * Math.Sqrt(1.0 + t * t);
        }

        /// <summary>
        ///     Sets an environment variable somewhere a NATIVE <c>getenv()</c> can actually read it.
        /// </summary>
        /// <remarks>
        ///     <b><see cref="Environment.SetEnvironmentVariable(string,string)"/> is not enough</b>,
        ///     and fails silently, which is the worst way for this to go wrong: .NET keeps its own
        ///     environment table, so the managed getter reads back the new value while the C runtime
        ///     the native library links against never sees it. Measured directly — setting
        ///     <c>OPENBLAS_CORETYPE=Nehalem</c> through the managed API and then loading OpenBLAS
        ///     still dispatched <c>Haswell</c>, i.e. the pin appeared to be applied and was not.
        ///     Going through the CRT (<c>ucrtbase!_putenv_s</c>) dispatched <c>Nehalem</c> as asked.
        ///     Windows' own <c>SetEnvironmentVariableW</c> does NOT work either: it updates the
        ///     process environment block, not the CRT copy that <c>getenv</c> reads.
        ///     <para>The managed table is updated too, so both views agree afterwards.</para>
        /// </remarks>
        private static bool TrySetNativeEnvironment(string name, string value)
        {
            bool ok = false;
            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    ok = WindowsPutEnv(name, value) == 0;
                else
                    ok = UnixSetEnv(name, value, 1) == 0;
            }
            catch (DllNotFoundException)
            {
            }
            catch (EntryPointNotFoundException)
            {
            }

            if (ok)
                Environment.SetEnvironmentVariable(name, value);

            return ok;
        }

        /// <summary>
        ///     Refuses a core type whose instruction set this CPU cannot execute.
        /// </summary>
        /// <remarks>
        ///     <c>OPENBLAS_CORETYPE</c> BYPASSES OpenBLAS' runtime capability detection — that is
        ///     the point of it — so naming a kernel above the CPU's ISA does not degrade, it
        ///     executes an unsupported instruction. Measured: forcing <c>SkylakeX</c> on this
        ///     Haswell host killed the process with <c>Illegal instruction</c>. That is not a
        ///     catchable exception, so it has to be prevented rather than handled.
        ///     <para>
        ///     Only the families with a clear ISA floor are checked. An unrecognised name is passed
        ///     through — OpenBLAS' core list is build-specific and inventing a rejection here would
        ///     block valid kernels on architectures this table knows nothing about.
        ///     </para>
        /// </remarks>
        private static void EnsureCoreTypeRunnable(string coreType)
        {
            bool ok;
            switch (coreType.ToUpperInvariant())
            {
                case "SKYLAKEX":
                case "COOPERLAKE":
                case "SAPPHIRERAPIDS":
                    ok = Avx512F.IsSupported;
                    break;
                case "HASWELL":
                case "ZEN":
                    ok = Avx2.IsSupported && Fma.IsSupported;
                    break;
                case "SANDYBRIDGE":
                case "BULLDOZER":
                case "PILEDRIVER":
                case "STEAMROLLER":
                case "EXCAVATOR":
                    ok = Avx.IsSupported;
                    break;
                case "NEHALEM":
                    ok = Sse42.IsSupported;
                    break;
                default:
                    return; // unknown to this table — trust the caller
            }

            if (!ok)
                throw new PlatformNotSupportedException(
                    $"OpenBlasEngine.Enable: this CPU cannot execute OpenBLAS' '{coreType}' kernels. " +
                    "OPENBLAS_CORETYPE overrides OpenBLAS' own CPU detection, so loading it would " +
                    "not fall back — it would terminate the process with an illegal instruction. " +
                    "Leave the core type unset to let OpenBLAS choose the best kernel this machine " +
                    "supports (which is also what NumPy does by default).");
        }

        /// <summary>Pins the BLAS worker-thread count (OpenBLAS only); results depend on it.</summary>
        internal static bool TrySetNumThreads(int threads)
        {
            if (_setNumThreads == null)
                return false;

            _setNumThreads(threads);
            return true;
        }

        /// <summary>
        ///     Applier for <c>TensorEngine.Threading</c>'s OpenBLAS knob — pushes a thread count to the
        ///     native runtime through BOTH channels a change can need: the CRT env (so a getenv, e.g. at a
        ///     future load, sees it) and <c>openblas_set_num_threads</c> (so an already-loaded library
        ///     changes immediately). A null value (auto) is left to the library's own default and not
        ///     forced.
        /// </summary>
        internal static void ApplyThreadCount(int? threads)
        {
            if (!threads.HasValue)
                return;

            TrySetNativeEnvironment("OPENBLAS_NUM_THREADS",
                threads.Value.ToString(System.Globalization.CultureInfo.InvariantCulture));

            if (threads.Value > 0)
                TrySetNumThreads(threads.Value);
        }

        /// <summary>The BLAS worker-thread count, or -1 when the library does not report one.</summary>
        internal static int GetNumThreads()
        {
            return _getNumThreads == null ? -1 : _getNumThreads();
        }

        /// <summary>The library's self-description (OpenBLAS build string), or null.</summary>
        internal static string GetConfig()
        {
            return _getConfig == null ? null : Marshal.PtrToStringAnsi(_getConfig());
        }

        /// <summary>
        ///     The DYNAMIC_ARCH micro-kernel family this process actually dispatched to
        ///     (e.g. "Haswell"), or null when the library does not report one.
        /// </summary>
        /// <remarks>
        ///     This is a correctness-relevant fact, not a diagnostic: a DYNAMIC_ARCH OpenBLAS picks
        ///     its GEMM micro-kernel from the running CPU, and different kernels accumulate in a
        ///     different order, so the same binary gives different bits on a different machine.
        ///     Measured on one host by forcing the choice with <c>OPENBLAS_CORETYPE</c>: Haswell,
        ///     Nehalem, Sandybridge and Katmai each produced a different result for the same
        ///     float32 product.
        /// </remarks>
        internal static string GetCoreName()
        {
            return _getCoreName == null ? null : Marshal.PtrToStringAnsi(_getCoreName());
        }

        #region call wrappers (widths marshalled per IsIlp64)

        internal static void Sgemm(CBlasOrder order, CBlasTranspose transA, CBlasTranspose transB,
            long m, long n, long k, float alpha, float* a, long lda, float* b, long ldb, float beta, float* c, long ldc)
        {
            if (IsIlp64)
                _sgemm64((int)order, (int)transA, (int)transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            else
                _sgemm32((int)order, (int)transA, (int)transB, (int)m, (int)n, (int)k, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
        }

        internal static void Dgemm(CBlasOrder order, CBlasTranspose transA, CBlasTranspose transB,
            long m, long n, long k, double alpha, double* a, long lda, double* b, long ldb, double beta, double* c, long ldc)
        {
            if (IsIlp64)
                _dgemm64((int)order, (int)transA, (int)transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            else
                _dgemm32((int)order, (int)transA, (int)transB, (int)m, (int)n, (int)k, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
        }

        internal static void Sgemv(CBlasOrder order, CBlasTranspose trans, long m, long n,
            float alpha, float* a, long lda, float* x, long incX, float beta, float* y, long incY)
        {
            if (IsIlp64)
                _sgemv64((int)order, (int)trans, m, n, alpha, a, lda, x, incX, beta, y, incY);
            else
                _sgemv32((int)order, (int)trans, (int)m, (int)n, alpha, a, (int)lda, x, (int)incX, beta, y, (int)incY);
        }

        internal static void Dgemv(CBlasOrder order, CBlasTranspose trans, long m, long n,
            double alpha, double* a, long lda, double* x, long incX, double beta, double* y, long incY)
        {
            if (IsIlp64)
                _dgemv64((int)order, (int)trans, m, n, alpha, a, lda, x, incX, beta, y, incY);
            else
                _dgemv32((int)order, (int)trans, (int)m, (int)n, alpha, a, (int)lda, x, (int)incX, beta, y, (int)incY);
        }

        internal static void Ssyrk(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans,
            long n, long k, float alpha, float* a, long lda, float beta, float* c, long ldc)
        {
            if (IsIlp64)
                _ssyrk64((int)order, (int)uplo, (int)trans, n, k, alpha, a, lda, beta, c, ldc);
            else
                _ssyrk32((int)order, (int)uplo, (int)trans, (int)n, (int)k, alpha, a, (int)lda, beta, c, (int)ldc);
        }

        internal static void Dsyrk(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans,
            long n, long k, double alpha, double* a, long lda, double beta, double* c, long ldc)
        {
            if (IsIlp64)
                _dsyrk64((int)order, (int)uplo, (int)trans, n, k, alpha, a, lda, beta, c, ldc);
            else
                _dsyrk32((int)order, (int)uplo, (int)trans, (int)n, (int)k, alpha, a, (int)lda, beta, c, (int)ldc);
        }

        internal static float Sdot(long n, float* x, long incX, float* y, long incY)
        {
            return IsIlp64 ? _sdot64(n, x, incX, y, incY) : _sdot32((int)n, x, (int)incX, y, (int)incY);
        }

        internal static double Ddot(long n, double* x, long incX, double* y, long incY)
        {
            return IsIlp64 ? _ddot64(n, x, incX, y, incY) : _ddot32((int)n, x, (int)incX, y, (int)incY);
        }

        internal static void Saxpy(long n, float alpha, float* x, long incX, float* y, long incY)
        {
            if (IsIlp64)
                _saxpy64(n, alpha, x, incX, y, incY);
            else
                _saxpy32((int)n, alpha, x, (int)incX, y, (int)incY);
        }

        internal static void Daxpy(long n, double alpha, double* x, long incX, double* y, long incY)
        {
            if (IsIlp64)
                _daxpy64(n, alpha, x, incX, y, incY);
            else
                _daxpy32((int)n, alpha, x, (int)incX, y, (int)incY);
        }

        // ---- complex128 product wrappers ------------------------------------------------------------
        // A System.Numerics.Complex is two interleaved doubles == BLAS complex*16, so a Complex* casts
        // to the double* the delegates expect with no copy, and alpha/beta (a by-value Complex local)
        // are handed over as &local reinterpreted double*. cblas_z{gemm,gemv,syrk,axpy} take alpha/beta
        // BY POINTER (unlike the real routines' by-value scalars), and z{dotu,dotc}_sub write the result
        // through the trailing double* — the `_sub` out-pointer form NumPy calls. Availability is gated
        // on IsComplexBlasLoaded (bound all-or-nothing); the parity entry points check it before routing
        // here, so these never see a null pointer.

        internal static void Zgemm(CBlasOrder order, CBlasTranspose transA, CBlasTranspose transB,
            long m, long n, long k, Complex alpha, Complex* a, long lda, Complex* b, long ldb, Complex beta, Complex* c, long ldc)
        {
            if (IsIlp64)
                _zgemm64((int)order, (int)transA, (int)transB, m, n, k, (double*)&alpha, (double*)a, lda, (double*)b, ldb, (double*)&beta, (double*)c, ldc);
            else
                _zgemm32((int)order, (int)transA, (int)transB, (int)m, (int)n, (int)k, (double*)&alpha, (double*)a, (int)lda, (double*)b, (int)ldb, (double*)&beta, (double*)c, (int)ldc);
        }

        internal static void Zgemv(CBlasOrder order, CBlasTranspose trans, long m, long n,
            Complex alpha, Complex* a, long lda, Complex* x, long incX, Complex beta, Complex* y, long incY)
        {
            if (IsIlp64)
                _zgemv64((int)order, (int)trans, m, n, (double*)&alpha, (double*)a, lda, (double*)x, incX, (double*)&beta, (double*)y, incY);
            else
                _zgemv32((int)order, (int)trans, (int)m, (int)n, (double*)&alpha, (double*)a, (int)lda, (double*)x, (int)incX, (double*)&beta, (double*)y, (int)incY);
        }

        internal static void Zsyrk(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans,
            long n, long k, Complex alpha, Complex* a, long lda, Complex beta, Complex* c, long ldc)
        {
            if (IsIlp64)
                _zsyrk64((int)order, (int)uplo, (int)trans, n, k, (double*)&alpha, (double*)a, lda, (double*)&beta, (double*)c, ldc);
            else
                _zsyrk32((int)order, (int)uplo, (int)trans, (int)n, (int)k, (double*)&alpha, (double*)a, (int)lda, (double*)&beta, (double*)c, (int)ldc);
        }

        /// <summary><c>cblas_zdotu_sub</c> — the UNCONJUGATED complex dot (<c>Σ x·y</c>), for np.dot /
        /// np.matmul / np.inner / np.matvec. Result returned via the <c>_sub</c> out-pointer.</summary>
        internal static Complex Zdotu(long n, Complex* x, long incX, Complex* y, long incY)
        {
            Complex result = default;
            if (IsIlp64)
                _zdotu_sub64(n, (double*)x, incX, (double*)y, incY, (double*)&result);
            else
                _zdotu_sub32((int)n, (double*)x, (int)incX, (double*)y, (int)incY, (double*)&result);
            return result;
        }

        /// <summary><c>cblas_zdotc_sub</c> — the CONJUGATING complex dot (<c>Σ conj(x)·y</c>), for
        /// np.vdot / np.vecdot / np.vecmat. Result returned via the <c>_sub</c> out-pointer.</summary>
        internal static Complex Zdotc(long n, Complex* x, long incX, Complex* y, long incY)
        {
            Complex result = default;
            if (IsIlp64)
                _zdotc_sub64(n, (double*)x, incX, (double*)y, incY, (double*)&result);
            else
                _zdotc_sub32((int)n, (double*)x, (int)incX, (double*)y, (int)incY, (double*)&result);
            return result;
        }

        internal static void Zaxpy(long n, Complex alpha, Complex* x, long incX, Complex* y, long incY)
        {
            if (IsIlp64)
                _zaxpy64(n, (double*)&alpha, (double*)x, incX, (double*)y, incY);
            else
                _zaxpy32((int)n, (double*)&alpha, (double*)x, (int)incX, (double*)y, (int)incY);
        }

        // ---- Level-3 siblings (trmm/trsm/symm/syr2k), bound best-effort by BindLevel3Extras. Unlike
        // the gemm/gemv/syrk wrappers above (which bake numpy's alpha=1/beta=0), these expose alpha
        // (trmm/trsm) and alpha/beta (symm/syr2k) in full, since they have no fixed consumer to
        // constrain them. A routine the loaded library did not export surfaces as a clear
        // EntryPointNotFoundException rather than a null-pointer access violation.

        private static EntryPointNotFoundException NotBound(string sym)
            => new EntryPointNotFoundException(
                $"OpenBlasEngine: the loaded BLAS library does not export '{sym}'. This Level-3 routine " +
                "is bound best-effort and is unavailable on the current library.");

        /// <summary>cblas_?trmm — triangular matrix–matrix multiply <c>B := alpha·op(A)·B</c> (Side=Left)
        /// or <c>B := alpha·B·op(A)</c> (Side=Right); A is the <paramref name="uplo"/> triangle, unit or
        /// non-unit per <paramref name="diag"/>, B (m×n) is overwritten.</summary>
        internal static void Strmm(CBlasOrder order, CBlasSide side, CBlasUpLo uplo, CBlasTranspose transA,
            CBlasDiag diag, long m, long n, float alpha, float* a, long lda, float* b, long ldb)
        {
            if (IsIlp64)
            {
                if (_strmm64 == null) throw NotBound("cblas_strmm");
                _strmm64((int)order, (int)side, (int)uplo, (int)transA, (int)diag, m, n, alpha, a, lda, b, ldb);
            }
            else
            {
                if (_strmm32 == null) throw NotBound("cblas_strmm");
                _strmm32((int)order, (int)side, (int)uplo, (int)transA, (int)diag, (int)m, (int)n, alpha, a, (int)lda, b, (int)ldb);
            }
        }

        /// <summary>cblas_?trmm (double).</summary>
        internal static void Dtrmm(CBlasOrder order, CBlasSide side, CBlasUpLo uplo, CBlasTranspose transA,
            CBlasDiag diag, long m, long n, double alpha, double* a, long lda, double* b, long ldb)
        {
            if (IsIlp64)
            {
                if (_dtrmm64 == null) throw NotBound("cblas_dtrmm");
                _dtrmm64((int)order, (int)side, (int)uplo, (int)transA, (int)diag, m, n, alpha, a, lda, b, ldb);
            }
            else
            {
                if (_dtrmm32 == null) throw NotBound("cblas_dtrmm");
                _dtrmm32((int)order, (int)side, (int)uplo, (int)transA, (int)diag, (int)m, (int)n, alpha, a, (int)lda, b, (int)ldb);
            }
        }

        /// <summary>cblas_?trsm — triangular solve with multiple RHS: <c>op(A)·X = alpha·B</c> (Side=Left)
        /// or <c>X·op(A) = alpha·B</c> (Side=Right); the solution X overwrites B (m×n).</summary>
        internal static void Strsm(CBlasOrder order, CBlasSide side, CBlasUpLo uplo, CBlasTranspose transA,
            CBlasDiag diag, long m, long n, float alpha, float* a, long lda, float* b, long ldb)
        {
            if (IsIlp64)
            {
                if (_strsm64 == null) throw NotBound("cblas_strsm");
                _strsm64((int)order, (int)side, (int)uplo, (int)transA, (int)diag, m, n, alpha, a, lda, b, ldb);
            }
            else
            {
                if (_strsm32 == null) throw NotBound("cblas_strsm");
                _strsm32((int)order, (int)side, (int)uplo, (int)transA, (int)diag, (int)m, (int)n, alpha, a, (int)lda, b, (int)ldb);
            }
        }

        /// <summary>cblas_?trsm (double).</summary>
        internal static void Dtrsm(CBlasOrder order, CBlasSide side, CBlasUpLo uplo, CBlasTranspose transA,
            CBlasDiag diag, long m, long n, double alpha, double* a, long lda, double* b, long ldb)
        {
            if (IsIlp64)
            {
                if (_dtrsm64 == null) throw NotBound("cblas_dtrsm");
                _dtrsm64((int)order, (int)side, (int)uplo, (int)transA, (int)diag, m, n, alpha, a, lda, b, ldb);
            }
            else
            {
                if (_dtrsm32 == null) throw NotBound("cblas_dtrsm");
                _dtrsm32((int)order, (int)side, (int)uplo, (int)transA, (int)diag, (int)m, (int)n, alpha, a, (int)lda, b, (int)ldb);
            }
        }

        /// <summary>cblas_?symm — symmetric matrix–matrix multiply <c>C := alpha·A·B + beta·C</c>
        /// (Side=Left, A symmetric m×m) or <c>C := alpha·B·A + beta·C</c> (Side=Right, A symmetric n×n);
        /// only the <paramref name="uplo"/> triangle of A is referenced. C is m×n.</summary>
        internal static void Ssymm(CBlasOrder order, CBlasSide side, CBlasUpLo uplo, long m, long n,
            float alpha, float* a, long lda, float* b, long ldb, float beta, float* c, long ldc)
        {
            if (IsIlp64)
            {
                if (_ssymm64 == null) throw NotBound("cblas_ssymm");
                _ssymm64((int)order, (int)side, (int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            else
            {
                if (_ssymm32 == null) throw NotBound("cblas_ssymm");
                _ssymm32((int)order, (int)side, (int)uplo, (int)m, (int)n, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
            }
        }

        /// <summary>cblas_?symm (double).</summary>
        internal static void Dsymm(CBlasOrder order, CBlasSide side, CBlasUpLo uplo, long m, long n,
            double alpha, double* a, long lda, double* b, long ldb, double beta, double* c, long ldc)
        {
            if (IsIlp64)
            {
                if (_dsymm64 == null) throw NotBound("cblas_dsymm");
                _dsymm64((int)order, (int)side, (int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            else
            {
                if (_dsymm32 == null) throw NotBound("cblas_dsymm");
                _dsymm32((int)order, (int)side, (int)uplo, (int)m, (int)n, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
            }
        }

        /// <summary>cblas_?syr2k — symmetric rank-2k update <c>C := alpha·A·Bᵀ + alpha·B·Aᵀ + beta·C</c>
        /// (Trans=NoTrans, A/B n×k) or the transposed form (Trans=Trans, A/B k×n); only the
        /// <paramref name="uplo"/> triangle of the symmetric n×n C is written.</summary>
        internal static void Ssyr2k(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans, long n, long k,
            float alpha, float* a, long lda, float* b, long ldb, float beta, float* c, long ldc)
        {
            if (IsIlp64)
            {
                if (_ssyr2k64 == null) throw NotBound("cblas_ssyr2k");
                _ssyr2k64((int)order, (int)uplo, (int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            else
            {
                if (_ssyr2k32 == null) throw NotBound("cblas_ssyr2k");
                _ssyr2k32((int)order, (int)uplo, (int)trans, (int)n, (int)k, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
            }
        }

        /// <summary>cblas_?syr2k (double).</summary>
        internal static void Dsyr2k(CBlasOrder order, CBlasUpLo uplo, CBlasTranspose trans, long n, long k,
            double alpha, double* a, long lda, double* b, long ldb, double beta, double* c, long ldc)
        {
            if (IsIlp64)
            {
                if (_dsyr2k64 == null) throw NotBound("cblas_dsyr2k");
                _dsyr2k64((int)order, (int)uplo, (int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            else
            {
                if (_dsyr2k32 == null) throw NotBound("cblas_dsyr2k");
                _dsyr2k32((int)order, (int)uplo, (int)trans, (int)n, (int)k, alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc);
            }
        }

        #endregion

        #region LAPACK LU wrappers (fortran_int width marshalled per IsIlp64)

        // Each wrapper takes/returns the arguments in ELEMENTS and hands the fortran routine the
        // by-pointer scalars it expects, widened to the build's integer width. `ipiv` is caller-owned
        // storage of at least min(m,n) fortran_int entries; det/slogdet read it back (see
        // FortranIntSize), solve/inv ignore it. `info` (returned) is the LAPACK status: 0 success,
        // >0 the 1-based index of the first exactly-zero U pivot (singular).

        internal static long Dgetrf(long m, long n, double* a, long lda, void* ipiv)
        {
            if (IsIlp64)
            {
                long mm = m, nn = n, ll = lda, info = 0;
                _dgetrf64(&mm, &nn, a, &ll, (long*)ipiv, &info);
                return info;
            }
            else
            {
                int mm = (int)m, nn = (int)n, ll = (int)lda, info = 0;
                _dgetrf32(&mm, &nn, a, &ll, (int*)ipiv, &info);
                return info;
            }
        }

        internal static long Zgetrf(long m, long n, double* a, long lda, void* ipiv)
        {
            if (IsIlp64)
            {
                long mm = m, nn = n, ll = lda, info = 0;
                _zgetrf64(&mm, &nn, a, &ll, (long*)ipiv, &info);
                return info;
            }
            else
            {
                int mm = (int)m, nn = (int)n, ll = (int)lda, info = 0;
                _zgetrf32(&mm, &nn, a, &ll, (int*)ipiv, &info);
                return info;
            }
        }

        internal static long Dgesv(long n, long nrhs, double* a, long lda, void* ipiv, double* b, long ldb)
        {
            if (IsIlp64)
            {
                long nn = n, nr = nrhs, ll = lda, lb = ldb, info = 0;
                _dgesv64(&nn, &nr, a, &ll, (long*)ipiv, b, &lb, &info);
                return info;
            }
            else
            {
                int nn = (int)n, nr = (int)nrhs, ll = (int)lda, lb = (int)ldb, info = 0;
                _dgesv32(&nn, &nr, a, &ll, (int*)ipiv, b, &lb, &info);
                return info;
            }
        }

        internal static long Zgesv(long n, long nrhs, double* a, long lda, void* ipiv, double* b, long ldb)
        {
            if (IsIlp64)
            {
                long nn = n, nr = nrhs, ll = lda, lb = ldb, info = 0;
                _zgesv64(&nn, &nr, a, &ll, (long*)ipiv, b, &lb, &info);
                return info;
            }
            else
            {
                int nn = (int)n, nr = (int)nrhs, ll = (int)lda, lb = (int)ldb, info = 0;
                _zgesv32(&nn, &nr, a, &ll, (int*)ipiv, b, &lb, &info);
                return info;
            }
        }

        /// <summary>Reads pivot entry <paramref name="i"/> from a fortran_int buffer at the bound width.</summary>
        internal static long ReadPivot(void* ipiv, long i)
            => IsIlp64 ? ((long*)ipiv)[i] : ((int*)ipiv)[i];

        #endregion

        #region LAPACK Cholesky/QR wrappers (fortran_int width marshalled per IsIlp64)

        // As with the LU wrappers, arguments are in ELEMENTS and the fortran routine gets its scalars
        // by pointer widened to the build's integer width. `uplo` is 'L' or 'U' as a byte. `info` is the
        // LAPACK status: 0 success, <0 illegal argument, and for potrf >0 the leading minor order that
        // is not positive definite. geqrf/orgqr take a caller-owned workspace; call with lwork == -1 for
        // the size query (writes work[0]) then again with the allocated buffer.

        internal static long Dpotrf(byte uplo, long n, double* a, long lda)
        {
            if (IsIlp64)
            {
                byte u = uplo; long nn = n, ll = lda, info = 0;
                _dpotrf64(&u, &nn, a, &ll, &info);
                return info;
            }
            else
            {
                byte u = uplo; int nn = (int)n, ll = (int)lda, info = 0;
                _dpotrf32(&u, &nn, a, &ll, &info);
                return info;
            }
        }

        internal static long Zpotrf(byte uplo, long n, double* a, long lda)
        {
            if (IsIlp64)
            {
                byte u = uplo; long nn = n, ll = lda, info = 0;
                _zpotrf64(&u, &nn, a, &ll, &info);
                return info;
            }
            else
            {
                byte u = uplo; int nn = (int)n, ll = (int)lda, info = 0;
                _zpotrf32(&u, &nn, a, &ll, &info);
                return info;
            }
        }

        internal static long Dgeqrf(long m, long n, double* a, long lda, double* tau, double* work, long lwork)
        {
            if (IsIlp64)
            {
                long mm = m, nn = n, ll = lda, lw = lwork, info = 0;
                _dgeqrf64(&mm, &nn, a, &ll, tau, work, &lw, &info);
                return info;
            }
            else
            {
                int mm = (int)m, nn = (int)n, ll = (int)lda, lw = (int)lwork, info = 0;
                _dgeqrf32(&mm, &nn, a, &ll, tau, work, &lw, &info);
                return info;
            }
        }

        internal static long Zgeqrf(long m, long n, double* a, long lda, double* tau, double* work, long lwork)
        {
            if (IsIlp64)
            {
                long mm = m, nn = n, ll = lda, lw = lwork, info = 0;
                _zgeqrf64(&mm, &nn, a, &ll, tau, work, &lw, &info);
                return info;
            }
            else
            {
                int mm = (int)m, nn = (int)n, ll = (int)lda, lw = (int)lwork, info = 0;
                _zgeqrf32(&mm, &nn, a, &ll, tau, work, &lw, &info);
                return info;
            }
        }

        internal static long Dorgqr(long m, long n, long k, double* a, long lda, double* tau, double* work, long lwork)
        {
            if (IsIlp64)
            {
                long mm = m, nn = n, kk = k, ll = lda, lw = lwork, info = 0;
                _dorgqr64(&mm, &nn, &kk, a, &ll, tau, work, &lw, &info);
                return info;
            }
            else
            {
                int mm = (int)m, nn = (int)n, kk = (int)k, ll = (int)lda, lw = (int)lwork, info = 0;
                _dorgqr32(&mm, &nn, &kk, a, &ll, tau, work, &lw, &info);
                return info;
            }
        }

        internal static long Zungqr(long m, long n, long k, double* a, long lda, double* tau, double* work, long lwork)
        {
            if (IsIlp64)
            {
                long mm = m, nn = n, kk = k, ll = lda, lw = lwork, info = 0;
                _zungqr64(&mm, &nn, &kk, a, &ll, tau, work, &lw, &info);
                return info;
            }
            else
            {
                int mm = (int)m, nn = (int)n, kk = (int)k, ll = (int)lda, lw = (int)lwork, info = 0;
                _zungqr32(&mm, &nn, &kk, a, &ll, tau, work, &lw, &info);
                return info;
            }
        }

        #endregion

        #region LAPACK SVD/least-squares wrappers (fortran_int width marshalled per IsIlp64)

        // Arguments are in ELEMENTS; the fortran routine gets its scalars by pointer widened to the
        // build's integer width. `jobz` is 'N'/'S'/'A' as a byte. `s`/`rcond`/`rwork` are the real
        // basetype (double). `info` is the LAPACK status: 0 success, <0 illegal argument, and for
        // gesdd/gelsd >0 the algorithm failed to converge. The workspace query (lwork == -1) writes
        // work[0]; gelsd's query ALSO writes iwork[0] with the needed integer-workspace size.

        internal static long Dgesdd(byte jobz, long m, long n, double* a, long lda, double* s,
            double* u, long ldu, double* vt, long ldvt, double* work, long lwork, void* iwork)
        {
            if (IsIlp64)
            {
                byte jz = jobz; long mm = m, nn = n, la = lda, lu = ldu, lvt = ldvt, lw = lwork, info = 0;
                _dgesdd64(&jz, &mm, &nn, a, &la, s, u, &lu, vt, &lvt, work, &lw, (long*)iwork, &info);
                return info;
            }
            else
            {
                byte jz = jobz; int mm = (int)m, nn = (int)n, la = (int)lda, lu = (int)ldu, lvt = (int)ldvt, lw = (int)lwork, info = 0;
                _dgesdd32(&jz, &mm, &nn, a, &la, s, u, &lu, vt, &lvt, work, &lw, (int*)iwork, &info);
                return info;
            }
        }

        internal static long Zgesdd(byte jobz, long m, long n, double* a, long lda, double* s,
            double* u, long ldu, double* vt, long ldvt, double* work, long lwork, double* rwork, void* iwork)
        {
            if (IsIlp64)
            {
                byte jz = jobz; long mm = m, nn = n, la = lda, lu = ldu, lvt = ldvt, lw = lwork, info = 0;
                _zgesdd64(&jz, &mm, &nn, a, &la, s, u, &lu, vt, &lvt, work, &lw, rwork, (long*)iwork, &info);
                return info;
            }
            else
            {
                byte jz = jobz; int mm = (int)m, nn = (int)n, la = (int)lda, lu = (int)ldu, lvt = (int)ldvt, lw = (int)lwork, info = 0;
                _zgesdd32(&jz, &mm, &nn, a, &la, s, u, &lu, vt, &lvt, work, &lw, rwork, (int*)iwork, &info);
                return info;
            }
        }

        internal static long Dgelsd(long m, long n, long nrhs, double* a, long lda, double* b, long ldb,
            double* s, double rcond, out long rank, double* work, long lwork, void* iwork)
        {
            if (IsIlp64)
            {
                long mm = m, nn = n, nr = nrhs, la = lda, lb = ldb, rk = 0, lw = lwork, info = 0; double rc = rcond;
                _dgelsd64(&mm, &nn, &nr, a, &la, b, &lb, s, &rc, &rk, work, &lw, (long*)iwork, &info);
                rank = rk;
                return info;
            }
            else
            {
                int mm = (int)m, nn = (int)n, nr = (int)nrhs, la = (int)lda, lb = (int)ldb, rk = 0, lw = (int)lwork, info = 0; double rc = rcond;
                _dgelsd32(&mm, &nn, &nr, a, &la, b, &lb, s, &rc, &rk, work, &lw, (int*)iwork, &info);
                rank = rk;
                return info;
            }
        }

        internal static long Zgelsd(long m, long n, long nrhs, double* a, long lda, double* b, long ldb,
            double* s, double rcond, out long rank, double* work, long lwork, double* rwork, void* iwork)
        {
            if (IsIlp64)
            {
                long mm = m, nn = n, nr = nrhs, la = lda, lb = ldb, rk = 0, lw = lwork, info = 0; double rc = rcond;
                _zgelsd64(&mm, &nn, &nr, a, &la, b, &lb, s, &rc, &rk, work, &lw, rwork, (long*)iwork, &info);
                rank = rk;
                return info;
            }
            else
            {
                int mm = (int)m, nn = (int)n, nr = (int)nrhs, la = (int)lda, lb = (int)ldb, rk = 0, lw = (int)lwork, info = 0; double rc = rcond;
                _zgelsd32(&mm, &nn, &nr, a, &la, b, &lb, s, &rc, &rk, work, &lw, rwork, (int*)iwork, &info);
                rank = rk;
                return info;
            }
        }

        #endregion

        #region LAPACK eigen wrappers (fortran_int width marshalled per IsIlp64)

        // Arguments in ELEMENTS; the fortran routine gets its scalars by pointer widened to the build's
        // integer width. `jobz`/`uplo`/`jobvl`/`jobvr` are single 'N'/'V'/'L'/'U' bytes. `w` (syevd/heevd
        // eigenvalues), `rwork` and `wr`/`wi` (geev real parts) are the real basetype (double). `info` is
        // the LAPACK status: 0 success, <0 illegal argument, >0 non-convergence. The workspace query
        // (lwork==-1) writes work[0]; syevd/heevd ALSO write iwork[0] (liwork==-1) and heevd rwork[0]
        // (lrwork==-1). `iwork` is caller-owned fortran_int storage (read back at FortranIntSize).

        internal static long Dsyevd(byte jobz, byte uplo, long n, double* a, long lda, double* w,
            double* work, long lwork, void* iwork, long liwork)
        {
            if (IsIlp64)
            {
                byte jz = jobz, up = uplo; long nn = n, la = lda, lw = lwork, li = liwork, info = 0;
                _dsyevd64(&jz, &up, &nn, a, &la, w, work, &lw, (long*)iwork, &li, &info);
                return info;
            }
            else
            {
                byte jz = jobz, up = uplo; int nn = (int)n, la = (int)lda, lw = (int)lwork, li = (int)liwork, info = 0;
                _dsyevd32(&jz, &up, &nn, a, &la, w, work, &lw, (int*)iwork, &li, &info);
                return info;
            }
        }

        internal static long Zheevd(byte jobz, byte uplo, long n, double* a, long lda, double* w,
            double* work, long lwork, double* rwork, long lrwork, void* iwork, long liwork)
        {
            if (IsIlp64)
            {
                byte jz = jobz, up = uplo; long nn = n, la = lda, lw = lwork, lr = lrwork, li = liwork, info = 0;
                _zheevd64(&jz, &up, &nn, a, &la, w, work, &lw, rwork, &lr, (long*)iwork, &li, &info);
                return info;
            }
            else
            {
                byte jz = jobz, up = uplo; int nn = (int)n, la = (int)lda, lw = (int)lwork, lr = (int)lrwork, li = (int)liwork, info = 0;
                _zheevd32(&jz, &up, &nn, a, &la, w, work, &lw, rwork, &lr, (int*)iwork, &li, &info);
                return info;
            }
        }

        internal static long Dgeev(byte jobvl, byte jobvr, long n, double* a, long lda,
            double* wr, double* wi, double* vl, long ldvl, double* vr, long ldvr, double* work, long lwork)
        {
            if (IsIlp64)
            {
                byte jl = jobvl, jr = jobvr; long nn = n, la = lda, lvl = ldvl, lvr = ldvr, lw = lwork, info = 0;
                _dgeev64(&jl, &jr, &nn, a, &la, wr, wi, vl, &lvl, vr, &lvr, work, &lw, &info);
                return info;
            }
            else
            {
                byte jl = jobvl, jr = jobvr; int nn = (int)n, la = (int)lda, lvl = (int)ldvl, lvr = (int)ldvr, lw = (int)lwork, info = 0;
                _dgeev32(&jl, &jr, &nn, a, &la, wr, wi, vl, &lvl, vr, &lvr, work, &lw, &info);
                return info;
            }
        }

        internal static long Zgeev(byte jobvl, byte jobvr, long n, double* a, long lda,
            double* w, double* vl, long ldvl, double* vr, long ldvr, double* work, long lwork, double* rwork)
        {
            if (IsIlp64)
            {
                byte jl = jobvl, jr = jobvr; long nn = n, la = lda, lvl = ldvl, lvr = ldvr, lw = lwork, info = 0;
                _zgeev64(&jl, &jr, &nn, a, &la, w, vl, &lvl, vr, &lvr, work, &lw, rwork, &info);
                return info;
            }
            else
            {
                byte jl = jobvl, jr = jobvr; int nn = (int)n, la = (int)lda, lvl = (int)ldvl, lvr = (int)ldvr, lw = (int)lwork, info = 0;
                _zgeev32(&jl, &jr, &nn, a, &la, w, vl, &lvl, vr, &lvr, work, &lw, rwork, &info);
                return info;
            }
        }

        #endregion
    }
}
