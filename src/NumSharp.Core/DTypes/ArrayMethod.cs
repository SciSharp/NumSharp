using System;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     NEP 43's <c>ArrayMethod</c> — ONE object per (input DTypes → output DTypes) implementation of an operation:
    ///     it resolves the loop descriptors for the given operand descriptors (<see cref="resolve_descriptors"/>) and
    ///     hands out the strided inner loop that runs them (<see cref="get_strided_loop"/>). Casts are the first family
    ///     (<see cref="CastingImpl"/>); ufunc loops for a new dtype register the same way (Stage C) instead of adding a
    ///     <c>case NPTypeCode.X:</c> to the IL generators.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The loop contract is <see cref="NDInnerLoopFunc"/> — NumSharp's existing analog of NumPy's
    ///     <c>PyUFuncGenericFunction</c> / <c>PyArrayMethod_StridedLoop</c> — so <c>NDIter</c> drives a registered loop
    ///     without a new driver. <see cref="casting"/> is the method's MINIMAL cast safety (NumPy's
    ///     <c>PyArrayMethodObject.casting</c>): <c>PyArray_CheckCastSafety</c> short-circuits on it and only calls
    ///     <see cref="resolve_descriptors"/> when the requested rule is stricter.
    ///     </para>
    ///     <para>
    ///     <b>NumPy's names.</b> The members are the fields of NumPy's <c>PyArrayMethodObject</c> (<c>name</c>,
    ///     <c>nin</c>, <c>nout</c>, <c>casting</c>, <c>flags</c>, <c>resolve_descriptors</c>, <c>get_strided_loop</c>,
    ///     <c>array_method.h</c>) and of the <c>PyBoundArrayMethodObject</c> that NumPy wraps around it to hold the
    ///     operand classes (<c>dtypes</c>); NumSharp keeps both in one object, because a method here does not need to
    ///     avoid NumPy's reference cycle between a method and its classes. The same names are the fields of the public
    ///     <c>PyArrayMethod_Spec</c> a NumPy extension fills in to create a method. The test
    ///     <c>DTypeOracleTests.Members_ExistUnderNumPysNames</c> holds the C# members to the names it reads from NumPy's
    ///     headers.
    ///     </para>
    /// </remarks>
    public abstract class ArrayMethod
    {
        /// <summary>The <c>view_offset</c> value meaning "this cast is not a view" (NumPy's <c>NPY_MIN_INTP</c>).</summary>
        public const long NoView = long.MinValue;

        /// <summary>
        ///     Builds a method — the fields NumPy's <c>PyArrayMethod_FromSpec</c> copies out of a <c>PyArrayMethod_Spec</c>.
        /// </summary>
        /// <param name="name">The method name (<c>"numeric_cast"</c>, <c>"datetime_casts"</c>, …).</param>
        /// <param name="nin">The number of input operands.</param>
        /// <param name="nout">The number of output operands.</param>
        /// <param name="casting">The minimal cast safety the method guarantees for any descriptors of its classes.</param>
        /// <param name="flags">The method flags.</param>
        /// <param name="dtypes">The classes of the operands, inputs first, then outputs: <paramref name="nin"/> +
        ///     <paramref name="nout"/> entries.</param>
        /// <exception cref="ArgumentNullException"><paramref name="name"/> or <paramref name="dtypes"/> is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="nin"/> or <paramref name="nout"/> is negative.</exception>
        /// <exception cref="ArgumentException"><paramref name="dtypes"/> does not hold one class per operand.</exception>
        protected ArrayMethod(string name, int nin, int nout, NPY_CASTING casting, NDArrayMethodFlags flags, IReadOnlyList<DTypeMeta> dtypes)
        {
            // The parameters carry the fields' names (NumPy's spelling), so every field is written through `this.`.
            this.name = name ?? throw new ArgumentNullException(nameof(name));
            if (nin < 0 || nout < 0)
                throw new ArgumentOutOfRangeException(nameof(nin));
            this.dtypes = dtypes ?? throw new ArgumentNullException(nameof(dtypes));
            if (dtypes.Count != nin + nout)
                throw new ArgumentException($"ArrayMethod '{name}' declares {nin} inputs and {nout} outputs but {dtypes.Count} DTypes.", nameof(dtypes));
            this.nin = nin;
            this.nout = nout;
            this.casting = casting;
            this.flags = flags;
        }

        /// <summary>The method name (<c>"numeric_cast"</c>, <c>"datetime_casts"</c>, …) — NumPy's <c>name</c>.</summary>
        public string name { get; }

        /// <summary>The number of input operands — NumPy's <c>nin</c>.</summary>
        public int nin { get; }

        /// <summary>The number of output operands — NumPy's <c>nout</c>.</summary>
        public int nout { get; }

        /// <summary>
        ///     NumPy's <c>casting</c>: the minimal cast safety this method guarantees for ANY descriptors of its DTypes
        ///     (<c>-1</c> in NumPy means "must resolve"; here <see cref="NPY_CASTING.NPY_UNSAFE_CASTING"/> plays that role).
        ///     A per-method constant; <c>virtual</c> so an implementation whose value is derived from a table that may not be
        ///     built yet when the method is REGISTERED (the builtin numeric casts read <c>np</c>'s frozen promotion table,
        ///     and <c>np</c>'s own type initializer can be what constructs <see cref="DTypeRegistry"/>) can compute it on
        ///     first read instead.
        /// </summary>
        public virtual NPY_CASTING casting { get; }

        /// <summary>The method flags (unaligned support, floating-point-error behaviour, …) — NumPy's <c>flags</c>.</summary>
        public NDArrayMethodFlags flags { get; }

        /// <summary>
        ///     The DType classes of the operands, inputs first then outputs — NumPy's <c>PyBoundArrayMethodObject.dtypes</c>.
        /// </summary>
        public IReadOnlyList<DTypeMeta> dtypes { get; }

        /// <summary>
        ///     NumPy's <c>resolve_descriptors</c>: given the operand descriptors (an output entry may be null — "please
        ///     pick"), fills <paramref name="loopDescrs"/> with the descriptors the loop will actually run on and returns the
        ///     cast safety of doing so.
        /// </summary>
        /// <param name="givenDescrs">The operand descriptors, inputs first; an output entry may be null.</param>
        /// <param name="loopDescrs">Receives the descriptors the loop runs on, one per operand.</param>
        /// <param name="viewOffset">The byte offset at which the operation is a plain VIEW (0 for a no-op cast between
        ///     identical descriptors), or <see cref="NoView"/>.</param>
        /// <returns>The cast safety of running the loop on <paramref name="loopDescrs"/>.</returns>
        /// <exception cref="TypeError">The descriptors cannot be resolved (NumPy returns -1 with an error set).</exception>
        public abstract NPY_CASTING resolve_descriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset);

        /// <summary>
        ///     NumPy's <c>get_strided_loop</c>: the inner loop for the resolved <paramref name="context"/> and the given
        ///     fixed strides. Stage A registers no loops here — the engine still drives its own IL cast kernels directly — so
        ///     the base implementation raises <see cref="NotSupportedException"/> naming the missing piece; Stage C
        ///     overrides it for the datetime loops.
        /// </summary>
        /// <param name="context">The method and the resolved operand descriptors.</param>
        /// <param name="aligned">True when every operand is aligned.</param>
        /// <param name="strides">The fixed byte strides of the operands, or the strides of the first call.</param>
        /// <param name="flags">Receives the flags of the returned loop (they may differ from the method's
        ///     <see cref="ArrayMethod.flags"/>).</param>
        /// <returns>The inner loop.</returns>
        /// <exception cref="NotSupportedException">Always, in the base implementation: no method provides a loop yet.</exception>
        public virtual NDInnerLoopFunc get_strided_loop(ArrayMethodContext context, bool aligned, long[] strides, out NDArrayMethodFlags flags)
        {
            throw new NotSupportedException(
                $"ArrayMethod '{name}' does not provide a strided loop yet: NumSharp's engine drives its own IL kernels for the " +
                "builtin dtypes, and registered ArrayMethod loops arrive with the datetime64/timedelta64 storage (Stage C).");
        }

        /// <inheritdoc/>
        public override string ToString() => $"<ArrayMethod '{name}' ({string.Join(", ", dtypes)})>";
    }

    /// <summary>
    ///     NEP 43's <c>PyArrayMethod_Context</c>: the method being run plus the resolved descriptors — what a strided
    ///     loop receives so it can read parameters (a datetime unit) without re-resolving.
    /// </summary>
    /// <remarks>
    ///     The members are the struct's fields, <c>method</c> and <c>descriptors</c> (<c>dtype_api.h</c>). Not ported:
    ///     <c>caller</c> (the ufunc a NumPy loop is called for; NumSharp's loops run without a caller object) and the
    ///     NumPy 2.4 fields <c>_reserved</c> and <c>flags</c>.
    /// </remarks>
    public sealed class ArrayMethodContext
    {
        /// <summary>Builds the context of one call of <paramref name="method"/>.</summary>
        /// <param name="method">The method being run.</param>
        /// <param name="descriptors">The resolved operand descriptors (inputs first, then outputs).</param>
        /// <exception cref="ArgumentNullException"><paramref name="method"/> or <paramref name="descriptors"/> is null.</exception>
        public ArrayMethodContext(ArrayMethod method, IReadOnlyList<DType> descriptors)
        {
            // The parameters carry the fields' names (NumPy's spelling), so every field is written through `this.`.
            this.method = method ?? throw new ArgumentNullException(nameof(method));
            this.descriptors = descriptors ?? throw new ArgumentNullException(nameof(descriptors));
        }

        /// <summary>The method this context belongs to — NumPy's <c>method</c>.</summary>
        public ArrayMethod method { get; }

        /// <summary>The resolved operand descriptors (inputs first, then outputs) — NumPy's <c>descriptors</c>.</summary>
        public IReadOnlyList<DType> descriptors { get; }
    }

    /// <summary>
    ///     An <see cref="ArrayMethod"/> with one input and one output: the cast from <see cref="From"/> to
    ///     <see cref="To"/> (NumPy's <c>castingimpl</c>). Looked up per class pair by <see cref="DTypeMeta.GetCastingImpl"/>
    ///     and consulted by <see cref="DTypeCasting"/> (<c>can_cast</c>, cast-safety, descriptor adaptation) and by
    ///     <see cref="DTypePromotion.CastDescrToDType"/> (which is how a parametric class learns the instance an operand
    ///     of another class should become).
    /// </summary>
    /// <remarks>
    ///     <see cref="From"/>, <see cref="To"/> and <see cref="IsWithinDType"/> have no NumPy names: NumPy reads the two
    ///     classes as <c>dtypes[0]</c> and <c>dtypes[1]</c> of the bound method and compares them where it needs to.
    /// </remarks>
    public abstract class CastingImpl : ArrayMethod
    {
        /// <summary>Builds the cast from <paramref name="from"/> to <paramref name="to"/>.</summary>
        /// <param name="name">The method name.</param>
        /// <param name="casting">The minimal cast safety of the cast.</param>
        /// <param name="flags">The method flags.</param>
        /// <param name="from">The source class.</param>
        /// <param name="to">The destination class.</param>
        /// <exception cref="ArgumentNullException"><paramref name="from"/>, <paramref name="to"/> or
        ///     <paramref name="name"/> is null.</exception>
        protected CastingImpl(string name, NPY_CASTING casting, NDArrayMethodFlags flags, DTypeMeta from, DTypeMeta to)
            : base(name, 1, 1, casting, flags, new[] { from ?? throw new ArgumentNullException(nameof(from)), to ?? throw new ArgumentNullException(nameof(to)) })
        {
            From = from;
            To = to;
        }

        /// <summary>The source DType class (<c>dtypes[0]</c>).</summary>
        public DTypeMeta From { get; }

        /// <summary>The destination DType class (<c>dtypes[1]</c>).</summary>
        public DTypeMeta To { get; }

        /// <summary>True for the cast between two instances of the same class.</summary>
        public bool IsWithinDType => ReferenceEquals(From, To);
    }
}
