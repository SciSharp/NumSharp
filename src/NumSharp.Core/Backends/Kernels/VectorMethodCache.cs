using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// VectorMethodCache.cs - Centralized reflection cache for Vector{128,256,512}
// =============================================================================
//
// Replaces ~10 file-private getters + ~30 inline `.GetMethods(...).Where(...).
// MakeGenericMethod(...)` patterns scattered across DirectILKernelGenerator partials.
//
// All lookups are cached as closed (already-bound-to-T) MethodInfo so callers
// don't allocate a fresh closed generic on every IL emission. Keys discriminate
// on (simdBits, name, elementType[, secondary]) so the same lookup across files
// hits the same cache entry.
//
// Naming convention for entries:
//   * Methods on the static <c>Vector{128,256,512}</c> CONTAINER type:
//     <c>Container(N).GetMethods(...)</c>
//   * Methods/properties on the typed <c>Vector{N}&lt;T&gt;</c>:
//     <c>V(N, T).GetMethod/.GetProperty</c>
//
// SCOPE — what this cache covers:
//   * Generic static helpers: Load, Store, ConditionalSelect, Equals,
//     EqualsAll, GreaterThan, ExtractMostSignificantBits, CreateScalar,
//     GetLower, GetUpper, GetElement, As{X}, OnesComplement, Multiply,
//     Divide, etc. — any generic static on the container.
//   * Non-generic statics where T is inferred from the argument:
//     WidenLower, WidenUpper, Narrow, ConvertToSingle, ConvertToDouble,
//     ConvertToInt32, the type-specific Create overloads.
//   * Typed Vector{N}&lt;T&gt;.op_X operators and the Zero getter.
//
// OUT OF SCOPE:
//   * x86 intrinsic conversions (Avx2.ConvertToVector256Int64, Sse41.…) —
//     these live on Avx2/Sse41, not on Vector*. They stay in CachedMethods.
//   * BitOperations.{PopCount,TrailingZeroCount} — same reason.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    internal static class VectorMethodCache
    {
        // Discriminator slot is freeform per lookup family (e.g. paramCount, "broadcast",
        // operator name, or 0). Lets us share one ConcurrentDictionary across all kinds.
        private readonly struct Key : IEquatable<Key>
        {
            public readonly int SimdBits;
            public readonly string Name;
            public readonly Type Elem;
            public readonly int Disc;

            public Key(int simdBits, string name, Type elem, int disc)
            {
                SimdBits = simdBits; Name = name; Elem = elem; Disc = disc;
            }

            public bool Equals(Key other)
                => SimdBits == other.SimdBits && Disc == other.Disc &&
                   ReferenceEquals(Elem, other.Elem) && Name == other.Name;

            public override bool Equals(object obj) => obj is Key k && Equals(k);
            public override int GetHashCode()
                => HashCode.Combine(SimdBits, Name, Elem, Disc);
        }

        private static readonly ConcurrentDictionary<Key, MethodInfo> _methods = new();
        private static readonly ConcurrentDictionary<Key, MethodInfo> _operators = new();
        // (simdBits, fromElem, toElem) for the V.As<from> -> V<to> family.
        private static readonly ConcurrentDictionary<(int, Type, Type), MethodInfo> _asMethods = new();
        // (simdBits, elem) -> Vector{N}<T>.Zero getter (and other properties later if needed).
        private static readonly ConcurrentDictionary<(int, Type, string), MethodInfo> _propGetters = new();
        // (simdBits, op, elem) -> x86 intrinsic MethodInfo (Avx/Avx2/Sse/Sse2). Null = no x86 path.
        private static readonly ConcurrentDictionary<(int, string, Type), MethodInfo> _x86Methods = new();

        // =================================================================
        // x86 intrinsic capability detection (cached at startup)
        // =================================================================
        // Empirical: System.Runtime.Intrinsics.Vector256.* JIT-emits ~1.8-2x slower
        // code than System.Runtime.Intrinsics.X86.Avx.* / Avx2.* on the same hardware
        // (verified .NET 10, AVX2-only host). When x86 intrinsics are present, route
        // Load / Store / Add / Sub / Mul / Div / Min / Max / Sqrt / And / Or / Xor /
        // Equals / GreaterThan / LessThan through the platform-specific MethodInfo
        // — same IL call instruction, just a faster code-gen path.

        internal static readonly bool UseX86_256 =
            System.Runtime.Intrinsics.X86.Avx.IsSupported &&
            System.Runtime.Intrinsics.X86.Avx2.IsSupported;

        internal static readonly bool UseX86_128 =
            System.Runtime.Intrinsics.X86.Sse.IsSupported &&
            System.Runtime.Intrinsics.X86.Sse2.IsSupported;

        internal static readonly bool UseX86_512 =
            System.Runtime.Intrinsics.X86.Avx512F.IsSupported;

        internal static bool UseX86For(int simdBits) => simdBits switch
        {
            512 => UseX86_512,
            256 => UseX86_256,
            128 => UseX86_128,
            _ => false
        };

        // =================================================================
        // Type resolution
        // =================================================================

        public static Type Container(int simdBits) => simdBits switch
        {
            128 => typeof(Vector128),
            256 => typeof(Vector256),
            512 => typeof(Vector512),
            _ => throw new NotSupportedException($"SIMD width {simdBits} not supported")
        };

        public static Type V(int simdBits, Type elemType) => simdBits switch
        {
            128 => typeof(Vector128<>).MakeGenericType(elemType),
            256 => typeof(Vector256<>).MakeGenericType(elemType),
            512 => typeof(Vector512<>).MakeGenericType(elemType),
            _ => throw new NotSupportedException($"SIMD width {simdBits} not supported")
        };

        // =================================================================
        // Generic container methods (Load / Store / ConditionalSelect / …)
        // =================================================================

        /// <summary>
        /// <c>Vector{N}.Load&lt;T&gt;(T*)</c> closed over <paramref name="elem"/>.
        /// </summary>
        public static MethodInfo Load(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "Load", elem, paramCount: 1, disc: 0,
                extra: m => m.GetParameters()[0].ParameterType.IsPointer);

        /// <summary>
        /// <c>Vector{N}.Store&lt;T&gt;(V&lt;T&gt;, T*)</c> closed over <paramref name="elem"/>.
        /// </summary>
        public static MethodInfo Store(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "Store", elem, paramCount: 2, disc: 0,
                extra: m => m.GetParameters()[0].ParameterType.IsGenericType);

        /// <summary>
        /// <c>Vector{N}.ConditionalSelect&lt;T&gt;(V, V, V)</c> closed over <paramref name="elem"/>.
        /// </summary>
        public static MethodInfo ConditionalSelect(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "ConditionalSelect", elem, paramCount: 3, disc: 0);

        /// <summary>
        /// Vector compare returning <c>V&lt;T&gt;</c> (lane-wise equality mask) — NOT the
        /// bool-returning <see cref="EqualsAll"/>.
        /// </summary>
        public static MethodInfo Equals(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "Equals", elem, paramCount: 2, disc: 0,
                extra: m => m.ReturnType.IsGenericType);

        /// <summary>
        /// <c>Vector{N}.EqualsAll&lt;T&gt;(V, V) -&gt; bool</c>. The bool-returning overload —
        /// disambiguated from the vector-returning <see cref="Equals"/> by return type.
        /// </summary>
        public static MethodInfo EqualsAll(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "EqualsAll", elem, paramCount: 2, disc: 0,
                extra: m => m.ReturnType == typeof(bool));

        public static MethodInfo GreaterThan(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "GreaterThan", elem, paramCount: 2, disc: 0);

        public static MethodInfo LessThan(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "LessThan", elem, paramCount: 2, disc: 0);

        public static MethodInfo ExtractMostSignificantBits(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "ExtractMostSignificantBits", elem, paramCount: 1, disc: 0);

        public static MethodInfo CreateScalar(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "CreateScalar", elem, paramCount: 1, disc: 0,
                extra: m => m.GetParameters()[0].ParameterType.IsGenericParameter);

        public static MethodInfo GetLower(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "GetLower", elem, paramCount: 1, disc: 0);

        public static MethodInfo GetUpper(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "GetUpper", elem, paramCount: 1, disc: 0);

        public static MethodInfo GetElement(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "GetElement", elem, paramCount: 2, disc: 0);

        public static MethodInfo OnesComplement(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "OnesComplement", elem, paramCount: 1, disc: 0);

        // =================================================================
        // Multiply has two generic overloads (vector-vector and vector-scalar).
        // Disambiguate via second-param type.
        // =================================================================

        public static MethodInfo MultiplyVectorVector(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "Multiply", elem, paramCount: 2, disc: /*VV*/ 1,
                extra: m =>
                {
                    var ps = m.GetParameters();
                    // Both params are the SAME generic type (V<T>) — vector-vector multiply.
                    return ps[0].ParameterType == ps[1].ParameterType;
                });

        public static MethodInfo MultiplyVectorScalar(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "Multiply", elem, paramCount: 2, disc: /*VS*/ 2,
                extra: m =>
                {
                    var ps = m.GetParameters();
                    // Second param is the generic parameter T (the scalar overload).
                    return ps[0].ParameterType != ps[1].ParameterType &&
                           ps[1].ParameterType.IsGenericParameter;
                });

        public static MethodInfo DivideVectorVector(int simdBits, Type elem)
            => GetOrAddGeneric(simdBits, "Divide", elem, paramCount: 2, disc: 1,
                extra: m =>
                {
                    var ps = m.GetParameters();
                    return ps[0].ParameterType == ps[1].ParameterType;
                });

        // =================================================================
        // Non-generic statics where T is inferred from the argument type
        // =================================================================

        public static MethodInfo WidenLower(int simdBits, Type fromElem)
            => GetOrAddNonGenericByArg(simdBits, "WidenLower", fromElem, paramCount: 1);

        public static MethodInfo WidenUpper(int simdBits, Type fromElem)
            => GetOrAddNonGenericByArg(simdBits, "WidenUpper", fromElem, paramCount: 1);

        public static MethodInfo Narrow(int simdBits, Type fromElem)
            => GetOrAddNonGenericByArg(simdBits, "Narrow", fromElem, paramCount: 2);

        public static MethodInfo ConvertToSingleFromInt32(int simdBits)
            => GetOrAddNonGenericByArg(simdBits, "ConvertToSingle", typeof(int), paramCount: 1);

        public static MethodInfo ConvertToDoubleFromInt64(int simdBits)
            => GetOrAddNonGenericByArg(simdBits, "ConvertToDouble", typeof(long), paramCount: 1);

        public static MethodInfo ConvertToInt32FromSingle(int simdBits)
            => GetOrAddNonGenericByArg(simdBits, "ConvertToInt32", typeof(float), paramCount: 1);

        /// <summary>
        /// x86 byte-lane sign-extend: <c>Avx2.ConvertToVector256Int{16,32,64}(V128&lt;byte&gt;)</c> or
        /// the SSE4.1 equivalent <c>Sse41.ConvertToVector128Int{16,32,64}(V128&lt;byte&gt;)</c>.
        /// Used by the np.where mask-expansion path to widen N bytes of bool condition into
        /// N wider lanes matching the data element size.
        /// </summary>
        /// <param name="targetSimdBits">Output vector width: 256 for Avx2, 128 for Sse41.</param>
        /// <param name="targetElemBits">Output lane bit-width: 16, 32, or 64.</param>
        public static MethodInfo ByteLaneSignExtend(int targetSimdBits, int targetElemBits)
        {
            string name = targetSimdBits switch
            {
                256 => "ConvertToVector256Int" + targetElemBits,
                128 => "ConvertToVector128Int" + targetElemBits,
                _ => throw new NotSupportedException($"SIMD width {targetSimdBits} not supported for byte-lane expand")
            };

            return _methods.GetOrAdd(new Key(targetSimdBits, name, typeof(byte), /*x86Intrinsic*/ 4000), static key =>
            {
                Type container = key.SimdBits == 256 ? typeof(Avx2) : typeof(Sse41);
                return container.GetMethod(key.Name,
                    BindingFlags.Public | BindingFlags.Static,
                    binder: null, types: new[] { typeof(Vector128<byte>) }, modifiers: null)
                    ?? throw new MissingMethodException(container.FullName, key.Name);
            });
        }

        /// <summary>
        /// <c>Vector{N}.ShiftLeft / ShiftRightLogical / ShiftRightArithmetic(V&lt;T&gt;, int)</c>
        /// — the per-type non-generic shift overload that takes a vector and a scalar shift
        /// count.
        /// </summary>
        public static MethodInfo ShiftByScalar(int simdBits, Type elem, string name)
            => _methods.GetOrAdd(new Key(simdBits, name, elem, /*shiftByInt*/ 3000), static key =>
            {
                var vT = V(key.SimdBits, key.Elem);
                return Container(key.SimdBits)
                    .GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .First(m => m.Name == key.Name && !m.IsGenericMethod &&
                                m.GetParameters().Length == 2 &&
                                m.GetParameters()[0].ParameterType == vT &&
                                m.GetParameters()[1].ParameterType == typeof(int));
            });

        // =================================================================
        // Create — broadcast and concat-from-halves
        // =================================================================

        /// <summary>
        /// Returns the static <c>Vector{N}.Create</c> overload that broadcasts a scalar to a
        /// full vector. Prefers the type-specific non-generic overload (e.g.
        /// <c>Create(double)</c>) when available — the JIT folds it to a single broadcast
        /// instruction (vbroadcastsd / vpbroadcastd) — and falls back to the generic
        /// <c>Create&lt;T&gt;(T)</c>.
        /// </summary>
        /// <remarks>
        /// On .NET 8 the generic overload routed through a runtime helper for <c>double</c>
        /// and added ~30-50% to scalar-broadcast Where kernels. The type-specific overload
        /// erases that gap.
        /// </remarks>
        public static MethodInfo CreateBroadcast(int simdBits, Type elem)
            => _methods.GetOrAdd(new Key(simdBits, "Create", elem, /*broadcast*/ 1000), static k =>
            {
                var c = Container(k.SimdBits);
                var specific = c.GetMethod("Create",
                    BindingFlags.Public | BindingFlags.Static,
                    binder: null, types: new[] { k.Elem }, modifiers: null);
                if (specific != null && specific.ReturnType.IsGenericType &&
                    specific.ReturnType.GetGenericArguments()[0] == k.Elem)
                    return specific;

                // Generic fallback Create<T>(T value) — discriminate the T-arg overload from
                // the T[], ReadOnlySpan<T>, and Vector{N/2}<T> concat overloads via
                // IsGenericParameter on the first parameter.
                return c.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .First(m => m.Name == "Create" && m.IsGenericMethod &&
                                m.GetParameters().Length == 1 &&
                                m.GetParameters()[0].ParameterType.IsGenericParameter)
                    .MakeGenericMethod(k.Elem);
            });

        /// <summary>
        /// <c>Vector{N}.Create(Vector{N/2}&lt;T&gt; lower, Vector{N/2}&lt;T&gt; upper)</c> — the
        /// concat-from-halves overload.
        /// </summary>
        public static MethodInfo CreateFromHalves(int simdBits, Type elem)
            => _methods.GetOrAdd(new Key(simdBits, "Create", elem, /*fromHalves*/ 2000), static k =>
            {
                var halfV = V(k.SimdBits / 2, k.Elem);
                return Container(k.SimdBits)
                    .GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .First(m => m.Name == "Create" && !m.IsGenericMethod &&
                                m.GetParameters().Length == 2 &&
                                m.GetParameters()[0].ParameterType == halfV);
            });

        /// <summary>
        /// The <c>Vector{N}.Create(T e0, …, T e_{lanes-1})</c> overload that packs one scalar
        /// per lane (lane count = N/8/sizeof(T) parameters, every one of type
        /// <paramref name="elem"/>). The fused strided-gather unary kernel uses this to
        /// assemble a vector directly from <em>lanes</em> strided scalar loads — no contiguous
        /// load, no scratch buffer. Discriminated from the single-arg broadcast
        /// <c>Create(T)</c> and the two-half concat <c>Create(V{N/2}, V{N/2})</c> by
        /// "non-generic, more than one parameter, all parameters of type T" — which uniquely
        /// identifies the all-lanes overload for every current Vector{128,256,512} element type.
        /// </summary>
        public static MethodInfo CreateElements(int simdBits, Type elem)
            => _methods.GetOrAdd(new Key(simdBits, "Create", elem, /*elements*/ 6000), static k =>
            {
                var c = Container(k.SimdBits);
                foreach (var m in c.GetMethods(BindingFlags.Public | BindingFlags.Static))
                {
                    if (m.Name != "Create" || m.IsGenericMethod)
                        continue;
                    var ps = m.GetParameters();
                    if (ps.Length > 1 && ps.All(p => p.ParameterType == k.Elem))
                        return m;
                }
                throw new MissingMethodException(c.FullName, $"Create({k.Elem.Name} x lanes)");
            });

        // =================================================================
        // As<from>() -> V<to>
        // =================================================================

        /// <summary>
        /// <c>Vector{N}.As&lt;from,to&gt;</c> — converts the lane interpretation without
        /// changing bits. Tries the explicit named form (<c>AsByte&lt;T&gt;</c>) first,
        /// then the two-type-parameter <c>As&lt;TFrom,TTo&gt;</c>.
        /// </summary>
        public static MethodInfo As(int simdBits, Type fromElem, Type toElem)
            => _asMethods.GetOrAdd((simdBits, fromElem, toElem), static key =>
            {
                var (bits, from, to) = key;
                string named = "As" + to.Name;
                var container = Container(bits);

                var asNamed = container
                    .GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .FirstOrDefault(m => m.Name == named && m.IsGenericMethod &&
                                         m.GetParameters().Length == 1 &&
                                         m.GetGenericArguments().Length == 1);
                if (asNamed != null)
                    return asNamed.MakeGenericMethod(from);

                var asGeneric = container
                    .GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .First(m => m.Name == "As" && m.IsGenericMethod &&
                                m.GetGenericArguments().Length == 2 &&
                                m.GetParameters().Length == 1);
                return asGeneric.MakeGenericMethod(from, to);
            });

        // =================================================================
        // Typed Vector{N}<T> operators and property getters
        // =================================================================

        /// <summary>
        /// <c>Vector{N}&lt;T&gt;.op_X(V&lt;T&gt;, V&lt;T&gt;) -&gt; V&lt;T&gt;</c> — e.g.
        /// <c>op_Addition</c>, <c>op_Multiply</c>.
        /// </summary>
        public static MethodInfo Operator(int simdBits, Type elem, string opName)
            => _operators.GetOrAdd(new Key(simdBits, opName, elem, 0), static key =>
            {
                var vT = V(key.SimdBits, key.Elem);
                return vT.GetMethod(key.Name,
                    BindingFlags.Public | BindingFlags.Static,
                    binder: null, types: new[] { vT, vT }, modifiers: null)
                    ?? throw new MissingMethodException(vT.FullName, key.Name);
            });

        /// <summary>
        /// <c>Vector{N}&lt;T&gt;.Zero</c> property getter (returned as the getter
        /// <see cref="MethodInfo"/> for direct IL emit).
        /// </summary>
        public static MethodInfo Zero(int simdBits, Type elem)
            => _propGetters.GetOrAdd((simdBits, elem, "Zero"), static key =>
            {
                var (bits, e, propName) = key;
                var vT = V(bits, e);
                var prop = vT.GetProperty(propName, BindingFlags.Public | BindingFlags.Static)
                    ?? throw new MissingMemberException(vT.FullName, propName);
                return prop.GetGetMethod()
                    ?? throw new MissingMethodException(vT.FullName, "get_" + propName);
            });

        /// <summary>
        /// <c>Vector{N}&lt;T&gt;.One</c> property getter (used by reciprocal kernels).
        /// </summary>
        public static MethodInfo One(int simdBits, Type elem)
            => _propGetters.GetOrAdd((simdBits, elem, "One"), static key =>
            {
                var (bits, e, propName) = key;
                var vT = V(bits, e);
                var prop = vT.GetProperty(propName, BindingFlags.Public | BindingFlags.Static)
                    ?? throw new MissingMemberException(vT.FullName, propName);
                return prop.GetGetMethod()
                    ?? throw new MissingMethodException(vT.FullName, "get_" + propName);
            });

        // =================================================================
        // Generic escape hatch — for less-common ops not pre-wired above
        // =================================================================

        /// <summary>
        /// Generic by name + element type. Use the named convenience methods above
        /// for the common ops; this is the fallback for one-off lookups.
        /// </summary>
        public static MethodInfo Generic(int simdBits, string name, Type elem, int? paramCount = null)
            => GetOrAddGeneric(simdBits, name, elem, paramCount: paramCount ?? -1, disc: 0);

        // =================================================================
        // x86 intrinsic routing — Avx/Avx2/Sse/Sse2/Avx512F MethodInfo lookups
        // =================================================================
        //
        // These return the platform-specific MethodInfo when the host supports the
        // matching intrinsic set. Callers in IL emission paths prefer these because
        // the JIT generates better machine code for X86.* intrinsics than for the
        // cross-platform Vector{N}.* statics (1.8-2x for hot SIMD loops).
        //
        // Return null when the requested (simdBits, op, elem) has no x86 intrinsic
        // available — caller falls back to the cross-platform Vector{N}.* path.

        /// <summary>x86 vector load (<c>Avx.LoadVector256(T*)</c>, <c>Sse.LoadVector128(T*)</c>,
        /// <c>Avx512F.LoadVector512(T*)</c>). All element types supported via Avx/Sse.</summary>
        public static MethodInfo LoadX86(int simdBits, Type elem)
        {
            if (!UseX86For(simdBits)) return null;
            return _x86Methods.GetOrAdd((simdBits, "LoadX86", elem), static k =>
            {
                Type api = X86ApiForLoadStore(k.Item1, k.Item3);
                string name = k.Item1 switch { 512 => "LoadVector512", 256 => "LoadVector256", _ => "LoadVector128" };
                return api.GetMethod(name, BindingFlags.Public | BindingFlags.Static,
                    binder: null, types: new[] { k.Item3.MakePointerType() }, modifiers: null);
            });
        }

        /// <summary>x86 vector store (<c>Avx.Store(T*, V&lt;T&gt;)</c>). NOTE: parameter
        /// order is REVERSED from <see cref="Store"/> (which is <c>(V&lt;T&gt;, T*)</c>).
        /// The IL emitter must arrange the stack accordingly.</summary>
        public static MethodInfo StoreX86(int simdBits, Type elem)
        {
            if (!UseX86For(simdBits)) return null;
            return _x86Methods.GetOrAdd((simdBits, "StoreX86", elem), static k =>
            {
                Type api = X86ApiForLoadStore(k.Item1, k.Item3);
                Type vT = V(k.Item1, k.Item3);
                return api.GetMethod("Store", BindingFlags.Public | BindingFlags.Static,
                    binder: null, types: new[] { k.Item3.MakePointerType(), vT }, modifiers: null);
            });
        }

        /// <summary>x86 vector arithmetic: <c>Avx.Add/Sub/Mul/Div/Min/Max</c> for float/double,
        /// <c>Avx2.Add/Sub/Min/Max/MultiplyLow/And/Or/Xor</c> for integer types. Returns null
        /// when the op has no x86 vector instruction (e.g. integer Divide; int64 Min/Max via Avx2).</summary>
        public static MethodInfo BinaryX86(int simdBits, string opName, Type elem)
        {
            if (!UseX86For(simdBits)) return null;
            return _x86Methods.GetOrAdd((simdBits, "Bin:" + opName, elem), static k =>
            {
                var (bits, key, e) = k;
                string op = key.Substring(4); // strip "Bin:" prefix
                Type api = ResolveX86BinaryApi(bits, op, e);
                if (api is null) return null;
                Type vT = V(bits, e);
                string methodName = TranslateBinaryOpName(op, e);
                return api.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static,
                    binder: null, types: new[] { vT, vT }, modifiers: null);
            });
        }

        /// <summary>x86 vector unary (Sqrt). Float/double only.</summary>
        public static MethodInfo UnaryX86(int simdBits, string opName, Type elem)
        {
            if (!UseX86For(simdBits)) return null;
            if (elem != typeof(float) && elem != typeof(double)) return null;
            return _x86Methods.GetOrAdd((simdBits, "Un:" + opName, elem), static k =>
            {
                var (bits, key, e) = k;
                string op = key.Substring(3);
                Type api = bits switch
                {
                    512 => typeof(System.Runtime.Intrinsics.X86.Avx512F),
                    256 => typeof(System.Runtime.Intrinsics.X86.Avx),
                    _ => typeof(System.Runtime.Intrinsics.X86.Sse) // Sqrt(V128<float>) - double on Sse2
                };
                if (bits == 128 && e == typeof(double))
                    api = typeof(System.Runtime.Intrinsics.X86.Sse2);
                Type vT = V(bits, e);
                return api.GetMethod(op, BindingFlags.Public | BindingFlags.Static,
                    binder: null, types: new[] { vT }, modifiers: null);
            });
        }

        // Resolve which X86 namespace owns the given binary op for the given element type.
        // Returns null when no x86 SIMD instruction covers (op, elem) at this width
        // (e.g. integer Divide, int64 Multiply/Min/Max on Avx2).
        private static Type ResolveX86BinaryApi(int simdBits, string op, Type elem)
        {
            bool isFp = elem == typeof(float) || elem == typeof(double);
            bool isI64 = elem == typeof(long) || elem == typeof(ulong);

            if (simdBits == 512)
            {
                // AVX-512 covers most ops including int64 min/max/mul - delegate to its container.
                return typeof(System.Runtime.Intrinsics.X86.Avx512F);
            }

            if (simdBits == 256)
            {
                if (isFp)
                {
                    if (op == "Add" || op == "Subtract" || op == "Multiply" || op == "Divide" ||
                        op == "Min" || op == "Max" || op == "And" || op == "Or" || op == "Xor")
                        return typeof(System.Runtime.Intrinsics.X86.Avx);
                    return null;
                }
                // Integer at 256-bit lives in Avx2.
                if (op == "Divide") return null;          // no integer SIMD divide
                if (isI64 && (op == "Min" || op == "Max")) return null; // Avx2 has no int64 min/max
                if (isI64 && op == "Multiply") return null;             // no int64 mul (vpmullq is Avx512DQ)
                if (op == "Add" || op == "Subtract" || op == "Multiply" || op == "Min" || op == "Max" ||
                    op == "And" || op == "Or" || op == "Xor")
                    return typeof(System.Runtime.Intrinsics.X86.Avx2);
                return null;
            }

            // 128-bit
            if (isFp)
            {
                if (elem == typeof(double))
                {
                    // double Sse2 covers Add/Sub/Mul/Div/Min/Max/And/Or/Xor
                    return typeof(System.Runtime.Intrinsics.X86.Sse2);
                }
                // float Sse covers Add/Sub/Mul/Div/Min/Max/And/Or/Xor
                return typeof(System.Runtime.Intrinsics.X86.Sse);
            }
            // Integer 128-bit lives in Sse2.
            if (op == "Divide") return null;
            if (isI64 && (op == "Min" || op == "Max")) return null;
            if (isI64 && op == "Multiply") return null;
            if (op == "Add" || op == "Subtract" || op == "Multiply" || op == "Min" || op == "Max" ||
                op == "And" || op == "Or" || op == "Xor")
                return typeof(System.Runtime.Intrinsics.X86.Sse2);
            return null;
        }

        // Map Vector256-style op names to X86 intrinsic method names.
        // "And"/"Or" on Vector{N} are spelled "BitwiseAnd"/"BitwiseOr" via Generic("BitwiseAnd") in
        // the caller; the X86 intrinsics use "And"/"Or" directly. The Multiply variant for int16/int32
        // on Avx2 is "MultiplyLow" (low 16/32 bits of product) — this matches Vector256.Multiply's
        // truncating semantics.
        private static string TranslateBinaryOpName(string op, Type elem)
        {
            if (op == "BitwiseAnd") return "And";
            if (op == "BitwiseOr") return "Or";
            bool isInt32_16 = elem == typeof(int) || elem == typeof(uint) ||
                              elem == typeof(short) || elem == typeof(ushort);
            if (op == "Multiply" && isInt32_16) return "MultiplyLow";
            return op;
        }

        // Which X86 namespace owns Load/Store for the given width + element.
        // Avx covers all element types at 256-bit via vmovups/vmovupd/vmovdqu — Avx2 not needed here.
        // 128-bit splits: Sse owns float, Sse2 owns double + all integer types.
        private static Type X86ApiForLoadStore(int simdBits, Type elem) => simdBits switch
        {
            512 => typeof(System.Runtime.Intrinsics.X86.Avx512F),
            256 => typeof(System.Runtime.Intrinsics.X86.Avx),
            _ => elem == typeof(float)
                ? typeof(System.Runtime.Intrinsics.X86.Sse)
                : typeof(System.Runtime.Intrinsics.X86.Sse2)
        };

        // =================================================================
        // Internals
        // =================================================================

        private static MethodInfo GetOrAddGeneric(
            int simdBits, string name, Type elem, int paramCount, int disc,
            Func<MethodInfo, bool> extra = null)
        {
            // The (paramCount, disc, has-extra) triple is folded into the Disc slot via
            // bit-packing so different overloads stay distinct in the cache without
            // adding a 5th dictionary key field.
            int folded = (disc << 16) | (paramCount & 0xFFFF);

            return _methods.GetOrAdd(new Key(simdBits, name, elem, folded), key =>
            {
                var container = Container(key.SimdBits);
                foreach (var m in container.GetMethods(BindingFlags.Public | BindingFlags.Static))
                {
                    if (m.Name != key.Name || !m.IsGenericMethodDefinition)
                        continue;
                    if (paramCount >= 0 && m.GetParameters().Length != paramCount)
                        continue;
                    if (extra != null && !extra(m))
                        continue;
                    return m.MakeGenericMethod(key.Elem);
                }
                throw new MissingMethodException(container.FullName, key.Name);
            });
        }

        private static MethodInfo GetOrAddNonGenericByArg(
            int simdBits, string name, Type fromElem, int paramCount)
        {
            return _methods.GetOrAdd(new Key(simdBits, name, fromElem, /*nonGeneric*/ -1 ^ paramCount), key =>
            {
                var container = Container(key.SimdBits);
                var paramV = V(key.SimdBits, key.Elem);
                foreach (var m in container.GetMethods(BindingFlags.Public | BindingFlags.Static))
                {
                    if (m.Name != key.Name || m.IsGenericMethod)
                        continue;
                    var ps = m.GetParameters();
                    if (ps.Length != paramCount)
                        continue;
                    if (ps[0].ParameterType != paramV)
                        continue;
                    // For 2-arg methods like Narrow(V<T>, V<T>) both params must match.
                    if (paramCount == 2 && ps[1].ParameterType != paramV)
                        continue;
                    return m;
                }
                throw new MissingMethodException(container.FullName, key.Name);
            });
        }
    }
}
