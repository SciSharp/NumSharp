using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;

// =============================================================================
// ScalarMethodCache.cs - Centralized reflection cache for scalar static methods
// =============================================================================
//
// Companion to VectorMethodCache. Where that cache owns Vector{128,256,512}
// static helpers, this one owns the scalar-typed reflection that the IL kernels
// reach for: Math/MathF math functions, decimal/Half/Complex operator and
// predicate methods, BitOperations bit helpers, and the implicit/explicit
// conversion catalog on decimal.
//
// Replaces inline patterns scattered across:
//   * Comparison.cs            decimal/Half/Complex op_LessThan/op_GreaterThan etc.
//   * Unary.Decimal.cs         Half op_Multiply / IsInfinity / IsFinite, Complex op_Mul/Div
//   * Unary.Math.cs            Math/MathF unary dispatch (Sqrt, etc.)
//   * Search.cs                decimal/Half op_LessThan / op_LessThanOrEqual
//   * Scan.cs                  decimal op_Addition
//   * Reduction.cs             Math binary helpers (Max, Min)
//   * Reduction.NaN.cs         Math.Sqrt / MathF.Sqrt
//   * Clip.cs                  Math binary dispatch
//   * Binary.cs                Math.Floor
//   * NonZero.cs               BitOperations.{PopCount, TrailingZeroCount}
//   * NDExpr.cs               Math.X (expression dispatch)
//
// Naming convention follows VectorMethodCache: the API is a small set of
// strongly-typed convenience methods (BinaryOp, UnaryOp, Predicate, MathFn1/2,
// BitOp) for the recurring patterns, plus a generic `Get(owner, name, params
// Type[] argTypes)` escape hatch for one-off lookups.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    internal static class ScalarMethodCache
    {
        // (owner type, method name, list of param types) is the cache key.
        // ParamTypes is wrapped in a small class so we can implement structural
        // equality without allocating tuples on hot lookup paths.
        private sealed class Key : IEquatable<Key>
        {
            public readonly Type Owner;
            public readonly string Name;
            public readonly Type[] ParamTypes;
            private readonly int _hash;

            public Key(Type owner, string name, Type[] paramTypes)
            {
                Owner = owner;
                Name = name;
                ParamTypes = paramTypes ?? Array.Empty<Type>();

                int h = HashCode.Combine(owner, name, ParamTypes.Length);
                for (int i = 0; i < ParamTypes.Length; i++)
                    h = HashCode.Combine(h, ParamTypes[i]);
                _hash = h;
            }

            public bool Equals(Key other)
            {
                if (other is null) return false;
                if (!ReferenceEquals(Owner, other.Owner)) return false;
                if (Name != other.Name) return false;
                if (ParamTypes.Length != other.ParamTypes.Length) return false;
                for (int i = 0; i < ParamTypes.Length; i++)
                    if (!ReferenceEquals(ParamTypes[i], other.ParamTypes[i])) return false;
                return true;
            }

            public override bool Equals(object obj) => obj is Key k && Equals(k);
            public override int GetHashCode() => _hash;
        }

        private static readonly ConcurrentDictionary<Key, MethodInfo> _cache = new();

        // =================================================================
        // Base lookup — the escape hatch
        // =================================================================

        /// <summary>
        /// Static public method on <paramref name="owner"/> matching <paramref name="name"/>
        /// and the exact <paramref name="paramTypes"/> signature. Result is cached.
        /// </summary>
        public static MethodInfo Get(Type owner, string name, params Type[] paramTypes)
        {
            return _cache.GetOrAdd(new Key(owner, name, paramTypes), key =>
            {
                var m = key.Owner.GetMethod(key.Name,
                    BindingFlags.Public | BindingFlags.Static,
                    binder: null, types: key.ParamTypes, modifiers: null);
                return m ?? throw new MissingMethodException(key.Owner.FullName, key.Name);
            });
        }

        // =================================================================
        // Operator overloads on scalar types (decimal / Half / Complex)
        // =================================================================

        /// <summary>
        /// Binary static operator on a scalar type — <c>elem.op_<paramref name="opName"/>(elem, elem)</c>.
        /// Works for both element-returning operators (op_Addition, op_Multiply, op_Subtraction,
        /// op_Division) and bool-returning comparison operators (op_Equality, op_Inequality,
        /// op_LessThan, op_LessThanOrEqual, op_GreaterThan, op_GreaterThanOrEqual).
        /// </summary>
        public static MethodInfo BinaryOp(Type elem, string opName)
            => Get(elem, opName, elem, elem);

        /// <summary>
        /// Unary static operator on a scalar type — <c>elem.op_<paramref name="opName"/>(elem)</c>.
        /// Examples: <c>op_UnaryNegation</c>, <c>op_LogicalNot</c>.
        /// </summary>
        public static MethodInfo UnaryOp(Type elem, string opName)
            => Get(elem, opName, elem);

        /// <summary>
        /// Static bool predicate on a scalar type — <c>elem.<paramref name="name"/>(elem) -&gt; bool</c>.
        /// Examples: <c>Half.IsNaN</c>, <c>Half.IsInfinity</c>, <c>Half.IsFinite</c>.
        /// </summary>
        public static MethodInfo Predicate(Type elem, string name)
            => Get(elem, name, elem);

        // =================================================================
        // Math / MathF dispatch by element type
        // =================================================================

        /// <summary>
        /// Math/MathF unary function dispatch — <c>Math.<paramref name="fnName"/>(double)</c>
        /// when <paramref name="elem"/> is <c>double</c>, <c>MathF.<paramref name="fnName"/>(float)</c>
        /// when it's <c>float</c>. Use the float-elem caller side and let the cache pick MathF;
        /// double-elem callers get Math.
        /// </summary>
        public static MethodInfo MathFn1(Type elem, string fnName)
        {
            if (elem == typeof(float))  return Get(typeof(MathF), fnName, typeof(float));
            if (elem == typeof(double)) return Get(typeof(Math),  fnName, typeof(double));
            throw new NotSupportedException(
                $"MathFn1: element type {elem} not supported (only float/double)");
        }

        /// <summary>
        /// Math/MathF binary function dispatch — <c>Math.<paramref name="fnName"/>(double, double)</c>
        /// or <c>MathF.<paramref name="fnName"/>(float, float)</c>. Used for Atan2, Pow, etc.
        /// </summary>
        public static MethodInfo MathFn2(Type elem, string fnName)
        {
            if (elem == typeof(float))  return Get(typeof(MathF), fnName, typeof(float),  typeof(float));
            if (elem == typeof(double)) return Get(typeof(Math),  fnName, typeof(double), typeof(double));
            throw new NotSupportedException(
                $"MathFn2: element type {elem} not supported (only float/double)");
        }

        // =================================================================
        // BitOperations
        // =================================================================

        /// <summary>
        /// <c>BitOperations.<paramref name="name"/>(<paramref name="argType"/>)</c>.
        /// Common args: <c>uint</c>, <c>ulong</c>. Common names: <c>PopCount</c>,
        /// <c>TrailingZeroCount</c>, <c>LeadingZeroCount</c>, <c>Log2</c>.
        /// </summary>
        public static MethodInfo BitOp(string name, Type argType)
            => Get(typeof(BitOperations), name, argType);

        // =================================================================
        // Decimal conversion catalog
        // =================================================================

        /// <summary>
        /// <c>decimal.op_Implicit(<paramref name="from"/>) -&gt; decimal</c>. Used to lift
        /// integer types to decimal for high-precision math. Restricted to <c>int</c>,
        /// <c>byte</c>, <c>sbyte</c>, <c>short</c>, <c>ushort</c>, <c>uint</c>, <c>long</c>,
        /// <c>ulong</c> per <c>System.Decimal</c>'s declared overloads.
        /// </summary>
        public static MethodInfo DecimalImplicitFrom(Type from)
            => Get(typeof(decimal), "op_Implicit", from);

        /// <summary>
        /// <c>decimal.op_Explicit(<paramref name="from"/>) -&gt; decimal</c> — used for
        /// float/double sources (which can lose precision, hence explicit).
        /// </summary>
        public static MethodInfo DecimalExplicitFrom(Type from)
            => Get(typeof(decimal), "op_Explicit", from);

        /// <summary>
        /// <c>decimal.To<paramref name="targetName"/>(decimal)</c> static conversion method —
        /// e.g. <c>ToByte</c>, <c>ToInt32</c>, <c>ToSingle</c>, <c>ToDouble</c>.
        /// </summary>
        public static MethodInfo DecimalTo(string targetName)
            => Get(typeof(decimal), targetName, typeof(decimal));
    }
}
