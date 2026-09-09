using System;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     NEP 43's <c>ArrayMethod</c> — ONE object per (input DTypes → output DTypes) implementation of an operation:
    ///     it resolves the loop descriptors for the given operand descriptors (<see cref="ResolveDescriptors"/>) and
    ///     hands out the strided inner loop that runs them (<see cref="GetStridedLoop"/>). Casts are the first family
    ///     (<see cref="CastingImpl"/>); ufunc loops for a new dtype register the same way (Stage C) instead of adding a
    ///     <c>case NPTypeCode.X:</c> to the IL generators.
    /// </summary>
    /// <remarks>
    ///     The loop contract is <see cref="NDInnerLoopFunc"/> — NumSharp's existing analog of NumPy's
    ///     <c>PyUFuncGenericFunction</c> / <c>PyArrayMethod_StridedLoop</c> — so <c>NDIter</c> drives a registered loop
    ///     without a new driver. <see cref="Casting"/> is the method's MINIMAL cast safety (NumPy's
    ///     <c>PyArrayMethodObject.casting</c>): <c>PyArray_CheckCastSafety</c> short-circuits on it and only calls
    ///     <see cref="ResolveDescriptors"/> when the requested rule is stricter.
    /// </remarks>
    public abstract class ArrayMethod
    {
        /// <summary>The <c>view_offset</c> value meaning "this cast is not a view" (NumPy's <c>NPY_MIN_INTP</c>).</summary>
        public const long NoView = long.MinValue;

        protected ArrayMethod(string name, int nin, int nout, NPY_CASTING casting, NDArrayMethodFlags flags, IReadOnlyList<DTypeMeta> dtypes)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            if (nin < 0 || nout < 0)
                throw new ArgumentOutOfRangeException(nameof(nin));
            DTypes = dtypes ?? throw new ArgumentNullException(nameof(dtypes));
            if (dtypes.Count != nin + nout)
                throw new ArgumentException($"ArrayMethod '{name}' declares {nin} inputs and {nout} outputs but {dtypes.Count} DTypes.", nameof(dtypes));
            NIn = nin;
            NOut = nout;
            Casting = casting;
            Flags = flags;
        }

        /// <summary>The method name (<c>"numeric_cast"</c>, <c>"datetime_casts"</c>, …) — NumPy's <c>name</c>.</summary>
        public string Name { get; }

        /// <summary>Number of input operands.</summary>
        public int NIn { get; }

        /// <summary>Number of output operands.</summary>
        public int NOut { get; }

        /// <summary>
        ///     The minimal cast safety this method guarantees for ANY descriptors of its DTypes (<c>-1</c> in NumPy means
        ///     "must resolve"; here <see cref="NPY_CASTING.NPY_UNSAFE_CASTING"/> plays that role). A per-method constant;
        ///     <c>virtual</c> so an implementation whose value is derived from a table that may not be built yet when the
        ///     method is REGISTERED (the builtin numeric casts read <c>np</c>'s frozen promotion table, and <c>np</c>'s own
        ///     type initializer can be what constructs <see cref="DTypeRegistry"/>) can compute it on first read instead.
        /// </summary>
        public virtual NPY_CASTING Casting { get; }

        /// <summary>The method flags (unaligned support, floating-point-error behaviour, …).</summary>
        public NDArrayMethodFlags Flags { get; }

        /// <summary>The DType classes of the operands, inputs first then outputs.</summary>
        public IReadOnlyList<DTypeMeta> DTypes { get; }

        /// <summary>
        ///     <c>resolve_descriptors</c>: given the operand descriptors (an output entry may be null — "please pick"),
        ///     fills <paramref name="loopDescrs"/> with the descriptors the loop will actually run on and returns the cast
        ///     safety of doing so. <paramref name="viewOffset"/> is the byte offset at which the operation is a plain
        ///     VIEW (0 for a no-op cast between identical descriptors), or <see cref="NoView"/>.
        /// </summary>
        /// <exception cref="TypeError">The descriptors cannot be resolved (NumPy returns -1 with an error set).</exception>
        public abstract NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset);

        /// <summary>
        ///     <c>get_strided_loop</c>: the inner loop for the resolved <paramref name="context"/> and the given fixed
        ///     strides. Stage A registers no loops here — the engine still drives its own IL cast kernels directly — so
        ///     the base implementation raises <see cref="NotSupportedException"/> naming the missing piece; Stage C
        ///     overrides it for the datetime loops.
        /// </summary>
        public virtual NDInnerLoopFunc GetStridedLoop(ArrayMethodContext context, bool aligned, long[] strides, out NDArrayMethodFlags flags)
        {
            throw new NotSupportedException(
                $"ArrayMethod '{Name}' does not provide a strided loop yet: NumSharp's engine drives its own IL kernels for the " +
                "builtin dtypes, and registered ArrayMethod loops arrive with the datetime64/timedelta64 storage (Stage C).");
        }

        /// <inheritdoc/>
        public override string ToString() => $"<ArrayMethod '{Name}' ({string.Join(", ", DTypes)})>";
    }

    /// <summary>
    ///     NEP 43's <c>PyArrayMethod_Context</c>: the method being run plus the resolved descriptors — what a strided
    ///     loop receives so it can read parameters (a datetime unit) without re-resolving.
    /// </summary>
    public sealed class ArrayMethodContext
    {
        public ArrayMethodContext(ArrayMethod method, IReadOnlyList<DType> descriptors)
        {
            Method = method ?? throw new ArgumentNullException(nameof(method));
            Descriptors = descriptors ?? throw new ArgumentNullException(nameof(descriptors));
        }

        /// <summary>The method this context belongs to.</summary>
        public ArrayMethod Method { get; }

        /// <summary>The resolved operand descriptors (inputs first, then outputs).</summary>
        public IReadOnlyList<DType> Descriptors { get; }
    }

    /// <summary>
    ///     An <see cref="ArrayMethod"/> with one input and one output: the cast from <see cref="From"/> to
    ///     <see cref="To"/> (NumPy's <c>castingimpl</c>). Looked up per class pair by <see cref="DTypeMeta.GetCastingImpl"/>
    ///     and consulted by <see cref="DTypeCasting"/> (<c>can_cast</c>, cast-safety, descriptor adaptation) and by
    ///     <see cref="DTypePromotion.CastDescrToDType"/> (which is how a parametric class learns the instance an operand
    ///     of another class should become).
    /// </summary>
    public abstract class CastingImpl : ArrayMethod
    {
        protected CastingImpl(string name, NPY_CASTING casting, NDArrayMethodFlags flags, DTypeMeta from, DTypeMeta to)
            : base(name, 1, 1, casting, flags, new[] { from ?? throw new ArgumentNullException(nameof(from)), to ?? throw new ArgumentNullException(nameof(to)) })
        {
            From = from;
            To = to;
        }

        /// <summary>The source DType class.</summary>
        public DTypeMeta From { get; }

        /// <summary>The destination DType class.</summary>
        public DTypeMeta To { get; }

        /// <summary>True for the cast between two instances of the same class.</summary>
        public bool IsWithinDType => ReferenceEquals(From, To);
    }
}
