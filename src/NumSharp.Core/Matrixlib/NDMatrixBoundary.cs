using System;
using System.Runtime.CompilerServices;

// =============================================================================
// NDMatrixBoundary - what a public np.* function returns for a numpy.matrix operand
// =============================================================================
//
// NumPy looks at the CLASS of an operand at run time, and each of its functions hands a matrix back by its own
// mechanism:
//
//   * a ufunc computes a plain array and matrix.__array_wrap__ returns a matrix VIEW of it (np.add, np.sin, ...),
//     and so does every function that ends in one (np.flip, np.diff, np.concatenate, np.dot ...);
//   * a fromnumeric wrapper calls the operand's own METHOD (np.sum(m, 0) is m.sum(0): a 1 x N matrix, and a
//     scalar without an axis), so `keepdims=` reaches a method that has no such keyword and raises;
//   * a function that converts with asarray returns a plain array (np.ravel, np.diagonal, np.where ...).
//
// NDMatrix binds its members and operators at COMPILE time (see NDMatrix.cs), so a matrix that reaches an np.*
// function through an NDArray parameter would be an ordinary 2-D array there. This class is the run-time half:
// the public np.* functions ask it whether an operand is a matrix and shape their result as NumPy does.
//
// THE BOUNDARY IS THE OUTERMOST PUBLIC CALL. The library calls its own public functions internally (np.std calls
// np.mean, np.cov calls np.average), and that code is written for plain arrays: a 1 x N matrix where it expects N
// elements would break it. A per-thread depth counter marks "a public function is already handling this call":
// every helper below raises it around the computation, and Sees() is false while it is raised, so nested public
// calls, NDArray members that are implemented by np.* functions, and NDMatrix members see plain-array semantics.
//
// WHAT STAYS STATIC. The operators and the indexers of an NDArray reference, and the NDArray members (a.sum(0)),
// keep the plain meaning for a matrix object: only the np.* functions follow the run-time class.
//
// A FUNCTION WITHOUT A RULE opens a region (PlainFor): the public functions it calls compute plain, so its result
// is the plain array NumPy's asarray-based functions return, and the safe answer for a function whose NumPy
// answer is not implemented (np.gradient, np.meshgrid). A function whose C# return type cannot carry the class
// (NDArray<bool>, a struct of arrays) is in that group by necessity.
//
// A CONTAINER STORES THE PLAIN VIEW (PlainOf). An object that keeps an array for later calls (a masked array's
// data and mask, a converted sampler parameter) cannot be covered by a region, which ends with the call that
// opened it; it stores a matrix as a plain array over the same buffer.
//
// THE CONTRACT IS MEASURED: no method of this library reaches the matrix rule of a public function at depth zero.
// The oracle project replays its whole corpus (every case with a two-dimensional operand: the ordinary tiers, the
// numpy.ma tiers and the index tiers) with those operands typed as matrices while AuditInternalCallers is on, and
// requires the set of library callers to stay empty (FuzzCorpusTests.MatrixOperand_LibraryNeverReachesAMatrixRule).
// A finding is closed by a region or by a rule for the calling function.
//
// NO OWNERSHIP PARITY YET. A result is always a matrix VIEW of the plain result (never an owner). NumPy's
// results own their data where the function allocated the result as the subclass (np.concatenate, np.dot); the
// matrix-operand oracle tier compares class, dtype, shape and values, not flags.owndata.
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     The run-time rule of the public <c>np.*</c> functions for a <c>numpy.matrix</c> operand: whether the
    ///     call sees a matrix (<see cref="Sees(NDArray, string)"/>) and how NumPy shapes the result (<see cref="Wrapped{TState}"/>,
    ///     <see cref="Reduced{TState}"/>, <see cref="Plain{TState}"/>, and their forms for several results). Use it
    ///     in a public function that takes an <see cref="NDArray"/>: a rule when NumPy's answer for a matrix is
    ///     known, else a region (<see cref="PlainFor(NDArray)"/>). Inside the library a matrix is a plain 2-D
    ///     array.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The pattern of a public function is one statement: when <see cref="Sees(NDArray, string)"/> is true, call a
    ///     rule with the arguments packed in a tuple and a <c>static</c> lambda that calls the function again.
    ///     The rule raises the depth, so that second call (and everything below it) computes with plain-array
    ///     semantics, and then turns the plain result into NumPy's. A <c>static</c> lambda with its state passed
    ///     by value allocates nothing; a capturing lambda would allocate a closure on EVERY call of the function,
    ///     also for plain arrays, because C# creates the closure object at method entry.
    ///     </para>
    ///     <para>
    ///     Thread safety: the depth is per thread, and a rule restores it in a <c>finally</c>, so an exception
    ///     from the computation leaves it at its entry value.
    ///     </para>
    /// </remarks>
    internal static class NDMatrixBoundary
    {
        /// <summary>
        ///     How many boundary regions are open on this thread. Zero means the next public call is the
        ///     outermost one and decides the result's class; anything else means plain-array semantics.
        /// </summary>
        [ThreadStatic] private static int t_depth;

        /// <summary>
        ///     Whether <paramref name="a"/> is a matrix OPERAND: an <see cref="NDMatrix"/> object that is not
        ///     the 0-d scalar stand-in (a C# number converted to <see cref="NDMatrix"/> is a scalar to NumPy).
        ///     Does not look at the depth; use <see cref="Sees(NDArray, string)"/> to decide a function's rule.
        /// </summary>
        /// <param name="a">The operand, or <c>null</c>.</param>
        /// <returns>True for a matrix with at least one axis.</returns>
        /// <remarks>An exact type test (one method-table compare): <see cref="NDMatrix"/> has no subclass in
        ///     the library, and <c>is NDMatrix</c> on a non-sealed class walks the type hierarchy of every
        ///     plain array that passes through a public function.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsMatrix(NDArray a)
            => a is not null && a.GetType() == typeof(NDMatrix) && a.Shape.NDim != 0;

        /// <summary>
        ///     Whether this call is the boundary for a matrix operand: <paramref name="a"/> is a matrix and no
        ///     public function above this one is already handling the call.
        /// </summary>
        /// <param name="a">The array operand, or <c>null</c>.</param>
        /// <param name="function">The name of the calling public function. Leave it to the compiler
        ///     (<see cref="CallerMemberNameAttribute"/>); only the audit reads it.</param>
        /// <returns>True when the function must apply its matrix rule; false means compute as for a plain array.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool Sees(NDArray a, [CallerMemberName] string function = null)
            => IsMatrix(a) && t_depth == 0 && (!AuditInternalCallers || Audit(function));

        /// <summary>
        ///     <see cref="Sees(NDArray, string)"/> for a function of two array operands: NumPy returns a matrix
        ///     when either operand is one (<c>matrix.__array_priority__</c> is above an array's).
        /// </summary>
        /// <param name="a">The first operand, or <c>null</c>.</param>
        /// <param name="b">The second operand, or <c>null</c>.</param>
        /// <param name="function">The calling public function's name (compiler-supplied; read by the audit).</param>
        /// <returns>True when the function must apply its matrix rule.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool Sees(NDArray a, NDArray b, [CallerMemberName] string function = null)
            => (IsMatrix(a) || IsMatrix(b)) && t_depth == 0 && (!AuditInternalCallers || Audit(function));

        /// <summary><see cref="Sees(NDArray, string)"/> for a function of three array operands.</summary>
        /// <param name="a">The first operand, or <c>null</c>.</param>
        /// <param name="b">The second operand, or <c>null</c>.</param>
        /// <param name="c">The third operand, or <c>null</c>.</param>
        /// <param name="function">The calling public function's name (compiler-supplied; read by the audit).</param>
        /// <returns>True when the function must apply its matrix rule.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool Sees(NDArray a, NDArray b, NDArray c, [CallerMemberName] string function = null)
            => (IsMatrix(a) || IsMatrix(b) || IsMatrix(c)) && t_depth == 0 && (!AuditInternalCallers || Audit(function));

        /// <summary>
        ///     <see cref="Sees(NDArray, string)"/> for a function that takes its operands as one sequence
        ///     (<c>np.concatenate</c>, <c>np.hstack</c>): true when any of them is a matrix.
        /// </summary>
        /// <param name="arrays">The operands, or <c>null</c>. A <c>null</c> element is skipped (the function
        ///     reports it itself).</param>
        /// <param name="function">The calling public function's name (compiler-supplied; read by the audit).</param>
        /// <returns>True when the function must apply its matrix rule.</returns>
        internal static bool Sees(NDArray[] arrays, [CallerMemberName] string function = null)
        {
            if (arrays is null || t_depth != 0)
                return false;
            for (int i = 0; i < arrays.Length; i++)
                if (IsMatrix(arrays[i]))
                    return !AuditInternalCallers || Audit(function);
            return false;
        }

        #region Caller audit (a test hook)

        /// <summary>
        ///     Test hook: while true, every call that <see cref="Sees(NDArray, string)"/> accepts is checked for
        ///     its CALLER, and a caller inside this library is recorded in <see cref="InternalCallers"/>. Off in
        ///     production (one static read on the matrix path only).
        /// </summary>
        /// <remarks>
        ///     The boundary design rests on one assumption: library code reaches a function with a matrix rule
        ///     only inside an open region (the depth is raised). A library method that calls such a function
        ///     with the caller's matrix at depth zero gets a matrix-shaped result where it was written for a
        ///     plain one: an <see cref="NDArray"/> member or operator that must keep the plain meaning, an engine
        ///     method, or a public function without a rule of its own. The audit turns that assumption into a
        ///     measurement: the oracle project replays its corpus with matrix operands and requires the set to
        ///     stay empty.
        /// </remarks>
        internal static bool AuditInternalCallers;

        /// <summary>
        ///     What the audit found: <c>"{caller type}.{caller method} -&gt; {function}"</c> for each library
        ///     method that reached the matrix rule of a public function at depth zero, with the number of times.
        /// </summary>
        internal static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> InternalCallers = new();

        /// <summary>
        ///     Records the caller of the public function <paramref name="function"/> when that caller belongs
        ///     to this library.
        /// </summary>
        /// <param name="function">The public function whose matrix rule is about to apply.</param>
        /// <returns>Always true, so the audit never changes what the function returns.</returns>
        /// <remarks>
        ///     Reads the stack, so it runs only under the test hook. The frame of the public function may be
        ///     missing (the JIT inlines small expression-bodied functions into their caller); the frames are
        ///     therefore matched by NAME: frames of this class and frames named like the function (the function
        ///     itself, an overload that forwards to it) are skipped, and the first frame left is the caller.
        /// </remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static bool Audit(string function)
        {
            var trace = new System.Diagnostics.StackTrace(1, false);
            for (int i = 0; i < trace.FrameCount; i++)
            {
                System.Reflection.MethodBase method = trace.GetFrame(i)?.GetMethod();
                Type owner = method?.DeclaringType;
                if (owner is null || owner == typeof(NDMatrixBoundary))
                    continue;
                // The function itself or a same-named overload that forwards (np.concatenate((a, b)) calls
                // np.concatenate(NDArray[])): the user's own call, one frame further up.
                if (method.Name == function)
                    continue;
                if (owner.Assembly == typeof(NDMatrixBoundary).Assembly)
                    InternalCallers.AddOrUpdate($"{owner.FullName}.{method.Name} -> {function}", 1, static (_, n) => n + 1);
                break;
            }
            return true;
        }

        #endregion

        /// <summary>
        ///     One slot of a multi-array result: the matrix view of a plain array when NumPy wraps that slot.
        /// </summary>
        /// <param name="slot">The plain result of the slot, or <c>null</c> (a slot the function did not compute).</param>
        /// <param name="wrap">Whether NumPy returns this slot through the operand's <c>__array_wrap__</c>.</param>
        /// <returns><paramref name="slot"/> itself when it is not wrapped, is <c>null</c> or already is a matrix;
        ///     otherwise its matrix view.</returns>
        /// <exception cref="ValueError">The slot cannot be a matrix (<c>shape too large to be a matrix.</c>).</exception>
        private static NDArray Slot(NDArray slot, bool wrap)
            => wrap && slot is not null && slot is not NDMatrix ? NDMatrix.FromWrapped(slot) : slot;

        /// <summary>
        ///     <see cref="Wrapped{TState}"/> for a function that returns two arrays of which NumPy wraps only some
        ///     (<c>np.linalg.eig</c>: plain eigenvalues, the eigenvectors as a matrix).
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments of the public function.</param>
        /// <param name="call">Calls the public function again with those arguments (a <c>static</c> lambda).</param>
        /// <param name="wrapFirst">Whether NumPy wraps the first array.</param>
        /// <param name="wrapSecond">Whether NumPy wraps the second array.</param>
        /// <returns>The pair, each wrapped slot a matrix view of its plain result.</returns>
        /// <exception cref="ValueError">A wrapped slot cannot be a matrix.</exception>
        internal static (NDArray, NDArray) WrappedPair<TState>(TState state, Func<TState, (NDArray, NDArray)> call,
            bool wrapFirst, bool wrapSecond)
        {
            NDScope scope = NDScope.Open();   // reclaims the hidden plain result of every wrapped slot (see Wrapped)
            t_depth++;
            try
            {
                (NDArray first, NDArray second) = call(state);
                return scope.Returns((Slot(first, wrapFirst), Slot(second, wrapSecond)));
            }
            finally
            {
                t_depth--;
                scope.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="Wrapped{TState}"/> for a function that returns three arrays of which NumPy wraps only some
        ///     (<c>np.linalg.svd</c>: U and Vh as matrices, the singular values plain).
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments of the public function.</param>
        /// <param name="call">Calls the public function again with those arguments (a <c>static</c> lambda).</param>
        /// <param name="wrap0">Whether NumPy wraps the first array.</param>
        /// <param name="wrap1">Whether NumPy wraps the second array.</param>
        /// <param name="wrap2">Whether NumPy wraps the third array.</param>
        /// <returns>The triple, each wrapped slot a matrix view of its plain result.</returns>
        /// <exception cref="ValueError">A wrapped slot cannot be a matrix.</exception>
        internal static (NDArray, NDArray, NDArray) WrappedTriple<TState>(TState state,
            Func<TState, (NDArray, NDArray, NDArray)> call, bool wrap0, bool wrap1, bool wrap2)
        {
            NDScope scope = NDScope.Open();   // reclaims the hidden plain result of every wrapped slot (see Wrapped)
            t_depth++;
            try
            {
                (NDArray r0, NDArray r1, NDArray r2) = call(state);
                return scope.Returns((Slot(r0, wrap0), Slot(r1, wrap1), Slot(r2, wrap2)));
            }
            finally
            {
                t_depth--;
                scope.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="Wrapped{TState}"/> for a function that returns four arrays of which NumPy wraps the first
        ///     two (<c>np.linalg.lstsq</c>: the solution and the residuals as matrices, rank and singular values
        ///     plain).
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments of the public function.</param>
        /// <param name="call">Calls the public function again with those arguments (a <c>static</c> lambda).</param>
        /// <returns>The four arrays, the first two as matrix views of their plain results.</returns>
        /// <exception cref="ValueError">A wrapped slot cannot be a matrix.</exception>
        internal static (NDArray, NDArray, NDArray, NDArray) WrappedFirstTwoOfFour<TState>(TState state,
            Func<TState, (NDArray, NDArray, NDArray, NDArray)> call)
        {
            NDScope scope = NDScope.Open();   // reclaims the hidden plain result of every wrapped slot (see Wrapped)
            t_depth++;
            try
            {
                (NDArray r0, NDArray r1, NDArray r2, NDArray r3) = call(state);
                return scope.Returns((Slot(r0, true), Slot(r1, true), r2, r3));
            }
            finally
            {
                t_depth--;
                scope.Dispose();
            }
        }

        /// <summary>
        ///     The rule of a function whose result CLASS depends on the route it took, which only the call itself
        ///     knows (<c>np.block</c>: NumPy's repeated concatenation keeps the matrix class, its single-allocation
        ///     route fills a new ndarray). Calls the function with plain-array semantics, then asks
        ///     <paramref name="keepsClass"/>: true gives <see cref="Wrapped{TState}"/>'s result, false
        ///     <see cref="Plain{TState}"/>'s.
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments of the public function.</param>
        /// <param name="call">Calls the public function again with those arguments (a <c>static</c> lambda).</param>
        /// <param name="keepsClass">Read AFTER <paramref name="call"/> returned: whether NumPy's result for the
        ///     route taken is a matrix.</param>
        /// <returns>The matrix view of the plain result, or the plain result.</returns>
        /// <exception cref="ValueError">The result cannot be a matrix (<c>shape too large to be a matrix.</c>).</exception>
        internal static NDArray WrappedIf<TState>(TState state, Func<TState, NDArray> call, Func<TState, bool> keepsClass)
        {
            // The scope reclaims the hidden plain result when a matrix view of it is returned (see Wrapped).
            NDScope scope = NDScope.Open();
            t_depth++;
            try
            {
                NDArray result = call(state);
                if (result is null)
                    return scope.Returns(result);
                if (!keepsClass(state))
                    return scope.Returns(PlainOf(result));
                return scope.Returns(result is NDMatrix ? result : NDMatrix.FromWrapped(result));
            }
            finally
            {
                t_depth--;
                scope.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's <c>matrix.__array_wrap__</c>: computes the plain result with <paramref name="call"/> and
        ///     returns it as a matrix view (the finalize rule: a 1-d result becomes a row, a 0-d result
        ///     <c>(1, 1)</c>). Use it for the ufuncs and for every function whose NumPy implementation ends in
        ///     one or allocates its result as the operand's class.
        /// </summary>
        /// <typeparam name="TState">The arguments of the function, packed by value (a tuple).</typeparam>
        /// <param name="state">The arguments, handed to <paramref name="call"/> unchanged.</param>
        /// <param name="call">Calls the public function again with the same arguments. It runs with the depth
        ///     raised, so it computes with plain-array semantics. Pass a <c>static</c> lambda.</param>
        /// <param name="out">The function's <c>out=</c> argument, or <c>null</c>. NumPy returns a given
        ///     <c>out</c> itself, whatever its class.</param>
        /// <param name="keepRank">Skip the finalize rule: the result keeps its rank under the matrix class.
        ///     Only for a function NumPy implements by INDEXING the matrix with a new axis
        ///     (<c>np.atleast_3d(m)</c> is the 3-d matrix object <c>m[:, :, newaxis]</c>), where
        ///     <c>matrix.__getitem__</c> switches the rule off.</param>
        /// <returns>The matrix; <paramref name="out"/> itself when the computation returned it; the
        ///     computation's own result when that already is a matrix (the operand handed back).</returns>
        /// <exception cref="ValueError">The result has more than two axes longer than 1
        ///     (<c>shape too large to be a matrix.</c>), as NumPy raises from <c>np.stack((m, m))</c>.</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray Wrapped<TState>(TState state, Func<TState, NDArray> call, NDArray @out = null, bool keepRank = false)
        {
            // The scope reclaims the hidden plain result: the matrix is a second array object over its buffer
            // and holds its own counted reference, so without it every call would leave one wrapper to the
            // finalizer (the buffer would stay out of the pool until then).
            NDScope scope = NDScope.Open();
            t_depth++;
            try
            {
                NDArray result = call(state);
                if (result is null || ReferenceEquals(result, @out) || result is NDMatrix)
                    return scope.Returns(result);
                return scope.Returns(keepRank ? new NDMatrix(result.Storage.Alias()) : NDMatrix.FromWrapped(result));
            }
            finally
            {
                t_depth--;
                scope.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="Wrapped{TState}"/> for a ufunc with two outputs (<c>np.modf</c>): each plain output
        ///     becomes a matrix view; a given <c>out</c> is returned itself.
        /// </summary>
        /// <typeparam name="TState">The arguments of the function, packed by value.</typeparam>
        /// <param name="state">The arguments, handed to <paramref name="call"/> unchanged.</param>
        /// <param name="call">Calls the public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <param name="out1">The function's first <c>out</c> argument, or <c>null</c>.</param>
        /// <param name="out2">The function's second <c>out</c> argument, or <c>null</c>.</param>
        /// <returns>The two results, each a matrix unless it is the caller's own <c>out</c>.</returns>
        /// <exception cref="ValueError">A result cannot be a matrix (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static (NDArray, NDArray) WrappedPair<TState>(TState state, Func<TState, (NDArray, NDArray)> call,
            NDArray out1 = null, NDArray out2 = null)
        {
            NDScope scope = NDScope.Open();
            t_depth++;
            try
            {
                (NDArray first, NDArray second) = call(state);
                if (first is not null && !ReferenceEquals(first, out1) && first is not NDMatrix)
                    first = NDMatrix.FromWrapped(first);
                if (second is not null && !ReferenceEquals(second, out2) && second is not NDMatrix)
                    second = NDMatrix.FromWrapped(second);
                return scope.Returns((first, second));
            }
            finally
            {
                t_depth--;
                scope.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="Wrapped{TState}"/> for a function that returns several arrays (<c>np.hsplit</c>): each
        ///     plain piece becomes a matrix view.
        /// </summary>
        /// <typeparam name="TState">The arguments of the function, packed by value.</typeparam>
        /// <param name="state">The arguments, handed to <paramref name="call"/> unchanged.</param>
        /// <param name="call">Calls the public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <returns>A new array of matrices in the order of the computation's results (a <c>null</c> result
        ///     stays <c>null</c>).</returns>
        /// <exception cref="ValueError">A piece cannot be a matrix (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray[] WrappedAll<TState>(TState state, Func<TState, NDArray[]> call)
        {
            NDScope scope = NDScope.Open();
            t_depth++;
            try
            {
                NDArray[] results = call(state);
                if (results is null)
                    return null;
                var wrapped = new NDArray[results.Length];
                for (int i = 0; i < results.Length; i++)
                {
                    NDArray piece = results[i];
                    wrapped[i] = piece is null || piece is NDMatrix ? piece : NDMatrix.FromWrapped(piece);
                }
                return scope.Returns(wrapped);
            }
            finally
            {
                t_depth--;
                scope.Dispose();
            }
        }

        /// <summary>
        ///     The rule of a reduction NumPy runs through the operand's METHOD (<c>np.sum(m, axis)</c> is
        ///     <c>m.sum(axis)</c>): without an axis the scalar, along axis 0 a row <c>(1, N)</c>, along axis 1 a
        ///     column <c>(N, 1)</c>. Use it for a reduction that has no <see cref="NDMatrix"/> member to delegate
        ///     to; the plain reduction supplies the values.
        /// </summary>
        /// <typeparam name="TState">The arguments of the function, packed by value.</typeparam>
        /// <param name="state">The arguments, handed to <paramref name="call"/> unchanged.</param>
        /// <param name="call">Calls the public function again (plain-array semantics, the caller's own
        ///     arguments). Pass a <c>static</c> lambda.</param>
        /// <param name="axis">The function's axis argument: it decides row or column for a 1-d plain result.</param>
        /// <param name="keepdims">The function's <c>keepdims</c> argument. NumPy forwards the keyword to the
        ///     matrix method, which does not have it.</param>
        /// <param name="method">The name of the matrix method NumPy ends up calling (<c>sum</c>, <c>mean</c>),
        ///     for the error text; <c>null</c> for a function that does not hand <c>keepdims</c> to a matrix
        ///     method (<c>np.argmax</c> falls back to the plain array when the method refuses the keyword,
        ///     <c>np.median</c> never calls one): its <c>keepdims</c> result is wrapped as it is.</param>
        /// <returns>The 0-d plain result without an axis; otherwise a matrix view of the plain result.</returns>
        /// <exception cref="TypeError"><paramref name="keepdims"/> is true and <paramref name="method"/> is not
        ///     <c>null</c>: <c>matrix.{method}() got an unexpected keyword argument 'keepdims'</c>, raised before
        ///     anything is computed, as in NumPy.</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        /// <remarks>C# cannot tell an omitted <c>keepdims</c> from <c>keepdims: false</c>, and NumPy raises for
        ///     both spellings of a PASSED keyword; only <c>keepdims: true</c> raises here.</remarks>
        internal static NDArray Reduced<TState>(TState state, Func<TState, NDArray> call, int? axis, bool keepdims, string method)
        {
            if (keepdims && method is not null)
                throw KeywordError(method, "keepdims");

            NDScope scope = NDScope.Open();
            t_depth++;
            try
            {
                NDArray result = call(state);
                if (result is null || result is NDMatrix)
                    return scope.Returns(result);
                Shape shape = result.Storage.Shape;
                if (shape.NDim == 0)
                    return scope.Returns(result);            // matrix._collapse(None): the scalar

                // matrix._collapse(axis) after keepdims=True: the reduced axis stays, with extent 1. A plain
                // reduction dropped it, so a 1-d result is put back as a column for the last axis and as a
                // row for the first.
                if (shape.NDim == 1 && axis is int along && (along == 1 || along == -1))
                    return scope.Returns(new NDMatrix(result.Storage.Alias(shape.AsMatrixColumn())));
                return scope.Returns(NDMatrix.FromWrapped(result));
            }
            finally
            {
                t_depth--;
                scope.Dispose();
            }
        }

        /// <summary>
        ///     The rule of a function NumPy converts its operand in with <c>asarray</c> (<c>np.ravel</c>,
        ///     <c>np.where</c>, <c>np.asarray</c> itself): the plain result, computed with plain-array semantics.
        /// </summary>
        /// <typeparam name="TState">The arguments of the function, packed by value.</typeparam>
        /// <param name="state">The arguments, handed to <paramref name="call"/> unchanged.</param>
        /// <param name="call">Calls the public function again. Pass a <c>static</c> lambda.</param>
        /// <returns>The computation's result; when that is a matrix object (the function handed its operand
        ///     back), a new plain <see cref="NDArray"/> view of it, as <c>np.asarray(m)</c> returns.</returns>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray Plain<TState>(TState state, Func<TState, NDArray> call)
        {
            t_depth++;
            try
            {
                NDArray result = call(state);
                return IsMatrix(result) ? ((NDMatrix)result).A : result;
            }
            finally
            {
                t_depth--;
            }
        }

        /// <summary>
        ///     Opens a region of plain-array semantics for the calling member when <paramref name="self"/> is a
        ///     matrix object: the public <c>np.*</c> functions it calls do not apply their matrix rule. Use it in
        ///     an <see cref="NDArray"/> member that is implemented by an <c>np.*</c> function (so the member
        ///     keeps the plain meaning through an <see cref="NDArray"/> reference) and in an <see cref="NDMatrix"/>
        ///     member that builds its own result from a plain computation.
        /// </summary>
        /// <param name="self">The receiver.</param>
        /// <returns>A region to dispose at the end of the member (<c>using var plain = ...</c>). For a receiver
        ///     that is not a matrix it does nothing.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static PlainRegion PlainFor(NDArray self)
        {
            if (self is null || self.GetType() != typeof(NDMatrix))
                return default;
            t_depth++;
            return new PlainRegion(true);
        }

        /// <summary><see cref="PlainFor(NDArray)"/> for a function of two array operands.</summary>
        /// <param name="a">The first operand, or <c>null</c>.</param>
        /// <param name="b">The second operand, or <c>null</c>.</param>
        /// <returns>A region to dispose at the end of the function; it does nothing when neither operand is a
        ///     matrix object.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static PlainRegion PlainFor(NDArray a, NDArray b)
        {
            if ((a is null || a.GetType() != typeof(NDMatrix)) && (b is null || b.GetType() != typeof(NDMatrix)))
                return default;
            t_depth++;
            return new PlainRegion(true);
        }

        /// <summary><see cref="PlainFor(NDArray)"/> for a function of three array operands.</summary>
        /// <param name="a">The first operand, or <c>null</c>.</param>
        /// <param name="b">The second operand, or <c>null</c>.</param>
        /// <param name="c">The third operand, or <c>null</c>.</param>
        /// <returns>A region to dispose at the end of the function; it does nothing when no operand is a matrix
        ///     object.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static PlainRegion PlainFor(NDArray a, NDArray b, NDArray c)
        {
            if ((a is null || a.GetType() != typeof(NDMatrix)) && (b is null || b.GetType() != typeof(NDMatrix))
                && (c is null || c.GetType() != typeof(NDMatrix)))
                return default;
            t_depth++;
            return new PlainRegion(true);
        }

        /// <summary><see cref="PlainFor(NDArray)"/> for operands passed as one sequence.</summary>
        /// <param name="arrays">The operands, or <c>null</c>.</param>
        /// <returns>A region to dispose at the end of the function; it does nothing when no operand is a matrix
        ///     object.</returns>
        internal static PlainRegion PlainFor(NDArray[] arrays)
        {
            if (arrays is not null)
                for (int i = 0; i < arrays.Length; i++)
                    if (arrays[i] is not null && arrays[i].GetType() == typeof(NDMatrix))
                    {
                        t_depth++;
                        return new PlainRegion(true);
                    }
            return default;
        }

        /// <summary>
        ///     <see cref="Sees(NDArray, string)"/> for a function whose operands arrive in a nested sequence
        ///     (<c>np.block</c>, <c>np.r_</c>): true when any array at any depth is a matrix.
        /// </summary>
        /// <param name="nested">An array, or a sequence (<see cref="System.Collections.IList"/>) of arrays and
        ///     sequences; anything else counts as "not a matrix".</param>
        /// <param name="function">The calling public function's name (compiler-supplied; read by the audit).</param>
        /// <returns>True when the function must apply its matrix rule.</returns>
        internal static bool SeesNested(object nested, [CallerMemberName] string function = null)
            => t_depth == 0 && HoldsMatrix(nested) && (!AuditInternalCallers || Audit(function));

        /// <summary>Whether a nested sequence holds a matrix operand at any depth.</summary>
        /// <param name="nested">An array, a sequence, or anything else.</param>
        /// <returns>True when a matrix is found.</returns>
        private static bool HoldsMatrix(object nested)
        {
            if (nested is NDArray array)
                return IsMatrix(array);
            if (nested is Array clr)
            {
                // Only a one-dimensional CLR array of references can hold an array object. An array of numbers
                // holds none, and a rectangular array is a leaf whose IList indexer refuses a single index.
                if (clr.Rank != 1 || clr.GetType().GetElementType().IsValueType)
                    return false;
            }
            else if (nested is not System.Collections.IList || nested is string)
            {
                // A scalar, a string (a sequence of chars in .NET) or any other object.
                return false;
            }
            else
            {
                // A generic list of numbers (List<double>) holds no array object: skip it without boxing its items.
                Type listType = nested.GetType();
                if (listType.IsGenericType && listType.GetGenericArguments()[0].IsValueType)
                    return false;
            }

            var list = (System.Collections.IList)nested;
            for (int i = 0; i < list.Count; i++)
                if (HoldsMatrix(list[i]))
                    return true;
            return false;
        }

        /// <summary>
        ///     The operand as a CONTAINER of the library must keep it: the array itself, or a plain view of a
        ///     matrix. Use it where an array is stored for later calls (a masked array's data and mask, a
        ///     converted sampler parameter): a region cannot cover those calls, they run after this one returns.
        /// </summary>
        /// <param name="a">The operand, or <c>null</c>.</param>
        /// <returns><paramref name="a"/> when it is not a matrix; otherwise a new plain <see cref="NDArray"/> over
        ///     the same buffer (<see cref="NDMatrix.A"/>), which the caller owns: it holds its own counted
        ///     reference, so dispose it or hand it to a scope like any other view.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NDArray PlainOf(NDArray a) => IsMatrix(a) ? ((NDMatrix)a).A : a;

        /// <summary>
        ///     Opens a region of plain-array semantics unconditionally. Use it in an <see cref="NDMatrix"/> member
        ///     or operator that calls public <c>np.*</c> functions with matrix operands and builds the matrix
        ///     result itself (<c>From</c> / <c>FromWrapped</c>): without the region those functions would hand a
        ///     finished matrix view back, and the member's own ownership rule would be lost.
        /// </summary>
        /// <returns>A region to dispose at the end of the member (<c>using var plain = ...</c>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static PlainRegion Inside()
        {
            t_depth++;
            return new PlainRegion(true);
        }

        /// <summary>
        ///     NumPy's error for a keyword a matrix method does not take (<c>np.sum(m, keepdims=True)</c>).
        /// </summary>
        /// <param name="method">The matrix method NumPy calls.</param>
        /// <param name="keyword">The keyword it was given.</param>
        /// <returns>The exception to throw, with NumPy's text.</returns>
        internal static TypeError KeywordError(string method, string keyword)
            => new TypeError($"matrix.{method}() got an unexpected keyword argument '{keyword}'");

        /// <summary>
        ///     A region opened by <see cref="PlainFor"/>: while it is alive, the public functions called on this
        ///     thread compute with plain-array semantics.
        /// </summary>
        internal readonly ref struct PlainRegion
        {
            /// <summary>Whether <see cref="PlainFor"/> raised the depth for this region.</summary>
            private readonly bool _entered;

            /// <summary>Creates a region.</summary>
            /// <param name="entered">Whether the depth was raised and must be lowered by <see cref="Dispose"/>.</param>
            internal PlainRegion(bool entered) => _entered = entered;

            /// <summary>Closes the region: lowers the depth when <see cref="PlainFor"/> raised it.</summary>
            public void Dispose()
            {
                if (_entered)
                    t_depth--;
            }
        }
    }
}
