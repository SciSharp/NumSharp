using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    // ═══════════════════════════════════════════════════════════════════════
    //  NpFunc — Generic Type Dispatch
    // ═══════════════════════════════════════════════════════════════════════
    //
    //  Eliminates repetitive NPTypeCode switch statements by bridging a
    //  runtime type code to compile-time generic type parameters.
    //
    //  ── Usage ──────────────────────────────────────────────────────────
    //
    //  1.  Define a small generic helper method:
    //
    //        static unsafe void ClipBounds<T>(nint @out, nint min, nint max, long len)
    //            where T : unmanaged, IComparable<T>
    //            => ILKernelGenerator.ClipArrayBounds((T*)@out, (T*)min, (T*)max, len);
    //
    //  2.  Call NpFunc.Invoke — pass ANY instantiation (the <int> is a dummy;
    //      NpFunc re-instantiates for the actual type):
    //
    //        NpFunc.Invoke(typeCode, ClipBounds<int>, outAddr, minAddr, maxAddr, len);
    //
    //  3.  Returning a value:
    //
    //        static NDArray<long>[] NonZeroImpl<T>(NDArray nd) where T : unmanaged
    //            => nonzeros<T>(nd.MakeGeneric<T>());
    //
    //        var result = NpFunc.Invoke(nd.typecode, NonZeroImpl<int>, nd);
    //
    //  ── Multi-type dispatch ────────────────────────────────────────────
    //
    //  Pass multiple NPTypeCodes or Types for methods with multiple
    //  generic parameters:
    //
    //        static void Cast<TIn, TOut>(nint src, nint dst, long len) where TIn : unmanaged where TOut : unmanaged { ... }
    //
    //        NpFunc.Invoke(inputTC, outputTC, Cast<int, float>, srcAddr, dstAddr, len);
    //
    //  ── Smart matching ─────────────────────────────────────────────────
    //
    //  When the count of passed type codes ≠ count of generic parameters:
    //
    //  • 1 code, N params  →  that one type applies to ALL parameters.
    //  • M codes < N params →  positional by type identity in the dummy
    //    instantiation: the first occurrence of each distinct type binds
    //    to the next code; repeats reuse the same binding.
    //
    //    Example: Method<int, int, float> with (tcA, tcB)
    //      → int (1st distinct) → tcA, int (repeat) → tcA, float (2nd) → tcB
    //      → Method<tcA, tcA, tcB>
    //
    //  ── Performance ────────────────────────────────────────────────────
    //
    //  Hot path (cache hit):
    //    • method.Method.MethodHandle.Value  → nint (O(1))
    //    • ConcurrentDictionary<nint, Delegate[]> lookup → get per-method table
    //    • Array index by (int)NPTypeCode    → get cached delegate
    //    • Delegate invocation               → call the method
    //
    //  Cold path (first call per method+type): reflection to extract the
    //  generic definition, MakeGenericMethod, CreateDelegate. Results are
    //  cached — reflection runs at most once per (method, typeCode) pair.
    //
    //  ── API summary ────────────────────────────────────────────────────
    //
    //    Invoke(tc,  method, args...)             1 NPTypeCode,  void
    //    Invoke(tc,  method, args...)             1 NPTypeCode,  returning
    //    Invoke(tc1, tc2, method, args...)        2 NPTypeCodes, void/returning
    //    Invoke(tc1, tc2, tc3, method, args...)   3 NPTypeCodes, void/returning
    //    Invoke(type,  method, args...)           1 Type,         void/returning
    //    Invoke(t1, t2, method, args...)          2 Types,        void/returning
    //    ResolveDelegate(method, tc1..tc5)        4-5 types,      returns delegate
    //
    // ═══════════════════════════════════════════════════════════════════════

    public static class NpFunc
    {
        #region Cache — per-method Delegate[] indexed by NPTypeCode

        // Level-1 key: closed method handle → Delegate[] (one slot per NPTypeCode ordinal)
        // Hot path is: dict.TryGetValue(nint) + array[(int)tc] — no CacheKey allocation.
        private static readonly ConcurrentDictionary<nint, Delegate[]> _tables = new();
        private static readonly int _tableSize = ComputeTableSize();
        private static int ComputeTableSize()
        {
            int max = 0;
            foreach (int v in Enum.GetValues(typeof(NPTypeCode)))
                if (v > max) max = v;
            return max + 1;
        }

        // Per-arity caches for multi-type dispatch. Right-sized keys are 33% faster
        // than padding to a fixed 6-nint tuple (20ns vs 31ns per lookup).
        private static readonly ConcurrentDictionary<(nint, nint, nint), Delegate> _cache2 = new();
        private static readonly ConcurrentDictionary<(nint, nint, nint, nint), Delegate> _cache3 = new();
        private static readonly ConcurrentDictionary<(nint, nint, nint, nint, nint), Delegate> _cache4 = new();
        private static readonly ConcurrentDictionary<(nint, nint, nint, nint, nint, nint), Delegate> _cache5 = new();

        #endregion

        #region Core Resolve — single type (hot path optimized)

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TDelegate Resolve<TDelegate>(TDelegate method, NPTypeCode tc) where TDelegate : Delegate
        {
            var handle = method.Method.MethodHandle.Value;

            if (_tables.TryGetValue(handle, out var table))
            {
                var del = table[(int)tc];
                if (del != null) return (TDelegate)del;
            }

            return ResolveSlow(method, handle, tc);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static TDelegate ResolveSlow<TDelegate>(TDelegate method, nint handle, NPTypeCode tc) where TDelegate : Delegate
        {
            var table = _tables.GetOrAdd(handle, static _ => new Delegate[_tableSize]);
            var targetType = tc.AsType();
            var mi = method.Method;
            var genericDef = mi.IsGenericMethod ? mi.GetGenericMethodDefinition() : mi;
            var resolvedTypes = SmartMatchTypes(mi, new[] { targetType });
            var closed = genericDef.MakeGenericMethod(resolvedTypes);
            var del = (TDelegate)Delegate.CreateDelegate(typeof(TDelegate), method.Target, closed);
            table[(int)tc] = del;
            return del;
        }

        #endregion

        #region Core Resolve — single Type

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TDelegate Resolve<TDelegate>(TDelegate method, Type t) where TDelegate : Delegate
        {
            var tc = t.GetTypeCode();
            if (tc != NPTypeCode.Empty)
                return Resolve(method, tc);

            return ResolveByType(method, t);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static TDelegate ResolveByType<TDelegate>(TDelegate method, Type t) where TDelegate : Delegate
        {
            var key = (method.Method.MethodHandle.Value, t.TypeHandle.Value, (nint)0);
            if (_cache2.TryGetValue(key, out var cached))
                return (TDelegate)cached;

            var mi = method.Method;
            var genericDef = mi.IsGenericMethod ? mi.GetGenericMethodDefinition() : mi;
            var resolvedTypes = SmartMatchTypes(mi, new[] { t });
            var closed = genericDef.MakeGenericMethod(resolvedTypes);
            var del = (TDelegate)Delegate.CreateDelegate(typeof(TDelegate), method.Target, closed);
            _cache2[key] = del;
            return del;
        }

        #endregion

        #region Core Resolve — multiple types

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TDelegate Resolve<TDelegate>(TDelegate method, Type t1, Type t2) where TDelegate : Delegate
        {
            var key = (method.Method.MethodHandle.Value, t1.TypeHandle.Value, t2.TypeHandle.Value);
            return _cache2.TryGetValue(key, out var c) ? (TDelegate)c : ResolveSlow(method, _cache2, key, new[] { t1, t2 });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TDelegate Resolve<TDelegate>(TDelegate method, Type t1, Type t2, Type t3) where TDelegate : Delegate
        {
            var key = (method.Method.MethodHandle.Value, t1.TypeHandle.Value, t2.TypeHandle.Value, t3.TypeHandle.Value);
            return _cache3.TryGetValue(key, out var c) ? (TDelegate)c : ResolveSlow(method, _cache3, key, new[] { t1, t2, t3 });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TDelegate Resolve<TDelegate>(TDelegate method, Type t1, Type t2, Type t3, Type t4) where TDelegate : Delegate
        {
            var key = (method.Method.MethodHandle.Value, t1.TypeHandle.Value, t2.TypeHandle.Value, t3.TypeHandle.Value, t4.TypeHandle.Value);
            return _cache4.TryGetValue(key, out var c) ? (TDelegate)c : ResolveSlow(method, _cache4, key, new[] { t1, t2, t3, t4 });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TDelegate Resolve<TDelegate>(TDelegate method, Type t1, Type t2, Type t3, Type t4, Type t5) where TDelegate : Delegate
        {
            var key = (method.Method.MethodHandle.Value, t1.TypeHandle.Value, t2.TypeHandle.Value, t3.TypeHandle.Value, t4.TypeHandle.Value, t5.TypeHandle.Value);
            return _cache5.TryGetValue(key, out var c) ? (TDelegate)c : ResolveSlow(method, _cache5, key, new[] { t1, t2, t3, t4, t5 });
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static TDelegate ResolveSlow<TDelegate, TKey>(TDelegate method, ConcurrentDictionary<TKey, Delegate> cache, TKey key, Type[] targetTypes)
            where TDelegate : Delegate
            where TKey : notnull
        {
            var mi = method.Method;
            var genericDef = mi.IsGenericMethod ? mi.GetGenericMethodDefinition() : mi;
            var resolvedTypes = SmartMatchTypes(mi, targetTypes);
            var closed = genericDef.MakeGenericMethod(resolvedTypes);
            var del = (TDelegate)Delegate.CreateDelegate(typeof(TDelegate), method.Target, closed);
            cache[key] = del;
            return del;
        }

        #endregion

        #region Smart Matching

        // Maps passed target types to generic parameters using type-identity matching.
        //
        //  Count match:   [tcA, tcB] + Method<T1,T2>           → [tcA, tcB]       (positional)
        //  Single:        [tcA]      + Method<T1,T2>           → [tcA, tcA]       (broadcast)
        //  Smart:         [tcA, tcB] + Method<int,int,float>   → [tcA, tcA, tcB]  (by identity)
        //
        private static Type[] SmartMatchTypes(MethodInfo closedMethod, Type[] targetTypes)
        {
            var genericDef = closedMethod.IsGenericMethod ? closedMethod.GetGenericMethodDefinition() : closedMethod;
            var genericParams = genericDef.GetGenericArguments();
            int paramCount = genericParams.Length;

            if (targetTypes.Length == paramCount)
                return targetTypes;

            if (targetTypes.Length == 1)
            {
                var single = targetTypes[0];
                var result = new Type[paramCount];
                for (int i = 0; i < paramCount; i++) result[i] = single;
                return result;
            }

            var concreteArgs = closedMethod.GetGenericArguments();
            var typeMap = new Dictionary<Type, Type>();
            int targetIdx = 0;
            var resolved = new Type[paramCount];

            for (int i = 0; i < paramCount; i++)
            {
                if (!typeMap.TryGetValue(concreteArgs[i], out var mapped))
                {
                    if (targetIdx >= targetTypes.Length)
                        throw new ArgumentException(
                            $"Method has more distinct generic types than the {targetTypes.Length} type code(s) provided");
                    mapped = targetTypes[targetIdx++];
                    typeMap[concreteArgs[i]] = mapped;
                }
                resolved[i] = mapped;
            }

            return resolved;
        }

        #endregion

        // ═══════════════════════════════════════════════════════════════
        //  Invoke overloads — 1 NPTypeCode
        // ═══════════════════════════════════════════════════════════════

        #region 1 NPTypeCode — void

        public static void Invoke(NPTypeCode tc, Action method)
            => Resolve(method, tc)();

        public static void Invoke<T1>(NPTypeCode tc, Action<T1> method, T1 a1)
            => Resolve(method, tc)(a1);

        public static void Invoke<T1, T2>(NPTypeCode tc, Action<T1, T2> method, T1 a1, T2 a2)
            => Resolve(method, tc)(a1, a2);

        public static void Invoke<T1, T2, T3>(NPTypeCode tc, Action<T1, T2, T3> method, T1 a1, T2 a2, T3 a3)
            => Resolve(method, tc)(a1, a2, a3);

        public static void Invoke<T1, T2, T3, T4>(NPTypeCode tc, Action<T1, T2, T3, T4> method, T1 a1, T2 a2, T3 a3, T4 a4)
            => Resolve(method, tc)(a1, a2, a3, a4);

        public static void Invoke<T1, T2, T3, T4, T5>(NPTypeCode tc, Action<T1, T2, T3, T4, T5> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
            => Resolve(method, tc)(a1, a2, a3, a4, a5);

        public static void Invoke<T1, T2, T3, T4, T5, T6>(NPTypeCode tc, Action<T1, T2, T3, T4, T5, T6> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
            => Resolve(method, tc)(a1, a2, a3, a4, a5, a6);

        #endregion

        #region 1 NPTypeCode — returning

        public static TResult Invoke<TResult>(NPTypeCode tc, Func<TResult> method)
            => Resolve(method, tc)();

        public static TResult Invoke<T1, TResult>(NPTypeCode tc, Func<T1, TResult> method, T1 a1)
            => Resolve(method, tc)(a1);

        public static TResult Invoke<T1, T2, TResult>(NPTypeCode tc, Func<T1, T2, TResult> method, T1 a1, T2 a2)
            => Resolve(method, tc)(a1, a2);

        public static TResult Invoke<T1, T2, T3, TResult>(NPTypeCode tc, Func<T1, T2, T3, TResult> method, T1 a1, T2 a2, T3 a3)
            => Resolve(method, tc)(a1, a2, a3);

        public static TResult Invoke<T1, T2, T3, T4, TResult>(NPTypeCode tc, Func<T1, T2, T3, T4, TResult> method, T1 a1, T2 a2, T3 a3, T4 a4)
            => Resolve(method, tc)(a1, a2, a3, a4);

        public static TResult Invoke<T1, T2, T3, T4, T5, TResult>(NPTypeCode tc, Func<T1, T2, T3, T4, T5, TResult> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
            => Resolve(method, tc)(a1, a2, a3, a4, a5);

        public static TResult Invoke<T1, T2, T3, T4, T5, T6, TResult>(NPTypeCode tc, Func<T1, T2, T3, T4, T5, T6, TResult> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
            => Resolve(method, tc)(a1, a2, a3, a4, a5, a6);

        #endregion

        // ═══════════════════════════════════════════════════════════════
        //  Invoke overloads — 2 NPTypeCodes
        // ═══════════════════════════════════════════════════════════════

        #region 2 NPTypeCodes — void

        public static void Invoke(NPTypeCode tc1, NPTypeCode tc2, Action method)
            => Resolve(method, tc1.AsType(), tc2.AsType())();

        public static void Invoke<T1>(NPTypeCode tc1, NPTypeCode tc2, Action<T1> method, T1 a1)
            => Resolve(method, tc1.AsType(), tc2.AsType())(a1);

        public static void Invoke<T1, T2>(NPTypeCode tc1, NPTypeCode tc2, Action<T1, T2> method, T1 a1, T2 a2)
            => Resolve(method, tc1.AsType(), tc2.AsType())(a1, a2);

        public static void Invoke<T1, T2, T3>(NPTypeCode tc1, NPTypeCode tc2, Action<T1, T2, T3> method, T1 a1, T2 a2, T3 a3)
            => Resolve(method, tc1.AsType(), tc2.AsType())(a1, a2, a3);

        public static void Invoke<T1, T2, T3, T4>(NPTypeCode tc1, NPTypeCode tc2, Action<T1, T2, T3, T4> method, T1 a1, T2 a2, T3 a3, T4 a4)
            => Resolve(method, tc1.AsType(), tc2.AsType())(a1, a2, a3, a4);

        public static void Invoke<T1, T2, T3, T4, T5>(NPTypeCode tc1, NPTypeCode tc2, Action<T1, T2, T3, T4, T5> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
            => Resolve(method, tc1.AsType(), tc2.AsType())(a1, a2, a3, a4, a5);

        public static void Invoke<T1, T2, T3, T4, T5, T6>(NPTypeCode tc1, NPTypeCode tc2, Action<T1, T2, T3, T4, T5, T6> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
            => Resolve(method, tc1.AsType(), tc2.AsType())(a1, a2, a3, a4, a5, a6);

        #endregion

        #region 2 NPTypeCodes — returning

        public static TResult Invoke<TResult>(NPTypeCode tc1, NPTypeCode tc2, Func<TResult> method)
            => Resolve(method, tc1.AsType(), tc2.AsType())();

        public static TResult Invoke<T1, TResult>(NPTypeCode tc1, NPTypeCode tc2, Func<T1, TResult> method, T1 a1)
            => Resolve(method, tc1.AsType(), tc2.AsType())(a1);

        public static TResult Invoke<T1, T2, TResult>(NPTypeCode tc1, NPTypeCode tc2, Func<T1, T2, TResult> method, T1 a1, T2 a2)
            => Resolve(method, tc1.AsType(), tc2.AsType())(a1, a2);

        public static TResult Invoke<T1, T2, T3, TResult>(NPTypeCode tc1, NPTypeCode tc2, Func<T1, T2, T3, TResult> method, T1 a1, T2 a2, T3 a3)
            => Resolve(method, tc1.AsType(), tc2.AsType())(a1, a2, a3);

        #endregion

        // ═══════════════════════════════════════════════════════════════
        //  Invoke overloads — 3 NPTypeCodes
        // ═══════════════════════════════════════════════════════════════

        #region 3 NPTypeCodes — void

        public static void Invoke(NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, Action method)
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType())();

        public static void Invoke<T1>(NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, Action<T1> method, T1 a1)
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType())(a1);

        public static void Invoke<T1, T2>(NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, Action<T1, T2> method, T1 a1, T2 a2)
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType())(a1, a2);

        public static void Invoke<T1, T2, T3>(NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, Action<T1, T2, T3> method, T1 a1, T2 a2, T3 a3)
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType())(a1, a2, a3);

        public static void Invoke<T1, T2, T3, T4>(NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, Action<T1, T2, T3, T4> method, T1 a1, T2 a2, T3 a3, T4 a4)
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType())(a1, a2, a3, a4);

        public static void Invoke<T1, T2, T3, T4, T5>(NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, Action<T1, T2, T3, T4, T5> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType())(a1, a2, a3, a4, a5);

        public static void Invoke<T1, T2, T3, T4, T5, T6>(NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, Action<T1, T2, T3, T4, T5, T6> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType())(a1, a2, a3, a4, a5, a6);

        #endregion

        #region 3 NPTypeCodes — returning

        public static TResult Invoke<TResult>(NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, Func<TResult> method)
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType())();

        public static TResult Invoke<T1, TResult>(NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, Func<T1, TResult> method, T1 a1)
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType())(a1);

        #endregion

        // ═══════════════════════════════════════════════════════════════
        //  Invoke overloads — 1 Type
        // ═══════════════════════════════════════════════════════════════

        #region 1 Type — void

        public static void Invoke(Type t, Action method)
            => Resolve(method, t)();

        public static void Invoke<T1>(Type t, Action<T1> method, T1 a1)
            => Resolve(method, t)(a1);

        public static void Invoke<T1, T2>(Type t, Action<T1, T2> method, T1 a1, T2 a2)
            => Resolve(method, t)(a1, a2);

        public static void Invoke<T1, T2, T3>(Type t, Action<T1, T2, T3> method, T1 a1, T2 a2, T3 a3)
            => Resolve(method, t)(a1, a2, a3);

        public static void Invoke<T1, T2, T3, T4>(Type t, Action<T1, T2, T3, T4> method, T1 a1, T2 a2, T3 a3, T4 a4)
            => Resolve(method, t)(a1, a2, a3, a4);

        public static void Invoke<T1, T2, T3, T4, T5>(Type t, Action<T1, T2, T3, T4, T5> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
            => Resolve(method, t)(a1, a2, a3, a4, a5);

        public static void Invoke<T1, T2, T3, T4, T5, T6>(Type t, Action<T1, T2, T3, T4, T5, T6> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
            => Resolve(method, t)(a1, a2, a3, a4, a5, a6);

        #endregion

        #region 1 Type — returning

        public static TResult Invoke<TResult>(Type t, Func<TResult> method)
            => Resolve(method, t)();

        public static TResult Invoke<T1, TResult>(Type t, Func<T1, TResult> method, T1 a1)
            => Resolve(method, t)(a1);

        public static TResult Invoke<T1, T2, TResult>(Type t, Func<T1, T2, TResult> method, T1 a1, T2 a2)
            => Resolve(method, t)(a1, a2);

        public static TResult Invoke<T1, T2, T3, TResult>(Type t, Func<T1, T2, T3, TResult> method, T1 a1, T2 a2, T3 a3)
            => Resolve(method, t)(a1, a2, a3);

        public static TResult Invoke<T1, T2, T3, T4, TResult>(Type t, Func<T1, T2, T3, T4, TResult> method, T1 a1, T2 a2, T3 a3, T4 a4)
            => Resolve(method, t)(a1, a2, a3, a4);

        #endregion

        // ═══════════════════════════════════════════════════════════════
        //  Invoke overloads — 2 Types
        // ═══════════════════════════════════════════════════════════════

        #region 2 Types — void

        public static void Invoke(Type t1, Type t2, Action method)
            => Resolve(method, t1, t2)();

        public static void Invoke<T1>(Type t1, Type t2, Action<T1> method, T1 a1)
            => Resolve(method, t1, t2)(a1);

        public static void Invoke<T1, T2>(Type t1, Type t2, Action<T1, T2> method, T1 a1, T2 a2)
            => Resolve(method, t1, t2)(a1, a2);

        public static void Invoke<T1, T2, T3>(Type t1, Type t2, Action<T1, T2, T3> method, T1 a1, T2 a2, T3 a3)
            => Resolve(method, t1, t2)(a1, a2, a3);

        public static void Invoke<T1, T2, T3, T4>(Type t1, Type t2, Action<T1, T2, T3, T4> method, T1 a1, T2 a2, T3 a3, T4 a4)
            => Resolve(method, t1, t2)(a1, a2, a3, a4);

        public static void Invoke<T1, T2, T3, T4, T5>(Type t1, Type t2, Action<T1, T2, T3, T4, T5> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5)
            => Resolve(method, t1, t2)(a1, a2, a3, a4, a5);

        public static void Invoke<T1, T2, T3, T4, T5, T6>(Type t1, Type t2, Action<T1, T2, T3, T4, T5, T6> method, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6)
            => Resolve(method, t1, t2)(a1, a2, a3, a4, a5, a6);

        #endregion

        #region 2 Types — returning

        public static TResult Invoke<TResult>(Type t1, Type t2, Func<TResult> method)
            => Resolve(method, t1, t2)();

        public static TResult Invoke<T1, TResult>(Type t1, Type t2, Func<T1, TResult> method, T1 a1)
            => Resolve(method, t1, t2)(a1);

        public static TResult Invoke<T1, T2, TResult>(Type t1, Type t2, Func<T1, T2, TResult> method, T1 a1, T2 a2)
            => Resolve(method, t1, t2)(a1, a2);

        #endregion

        // ═══════════════════════════════════════════════════════════════
        //  ResolveDelegate — public, for 4-5 type codes
        // ═══════════════════════════════════════════════════════════════

        #region ResolveDelegate

        public static TDelegate ResolveDelegate<TDelegate>(TDelegate method, NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, NPTypeCode tc4) where TDelegate : Delegate
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType(), tc4.AsType());

        public static TDelegate ResolveDelegate<TDelegate>(TDelegate method, NPTypeCode tc1, NPTypeCode tc2, NPTypeCode tc3, NPTypeCode tc4, NPTypeCode tc5) where TDelegate : Delegate
            => Resolve(method, tc1.AsType(), tc2.AsType(), tc3.AsType(), tc4.AsType(), tc5.AsType());

        public static TDelegate ResolveDelegate<TDelegate>(TDelegate method, Type t1, Type t2, Type t3, Type t4) where TDelegate : Delegate
            => Resolve(method, t1, t2, t3, t4);

        public static TDelegate ResolveDelegate<TDelegate>(TDelegate method, Type t1, Type t2, Type t3, Type t4, Type t5) where TDelegate : Delegate
            => Resolve(method, t1, t2, t3, t4, t5);

        #endregion
    }
}
