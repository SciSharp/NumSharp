using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Runtime.Intrinsics;

// =============================================================================
// VectorMethodCache.cs - Centralized reflection cache for Vector{128,256,512}
// =============================================================================
//
// Replaces ~10 file-private getters + ~30 inline `.GetMethods(...).Where(...).
// MakeGenericMethod(...)` patterns scattered across ILKernelGenerator partials.
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
