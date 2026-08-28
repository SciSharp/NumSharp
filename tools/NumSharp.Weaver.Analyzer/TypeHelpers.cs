using Microsoft.CodeAnalysis;

namespace NumSharp.Weaver.Analyzer
{
    /// <summary>
    ///     The semantic-model mirror of the IL weaver's <c>Classify</c> / <c>IsNDArrayLike</c> /
    ///     <c>IsScalar</c> / <c>IsTaskLike</c> family (ScopeWeaver.cs). It MUST agree with the weaver:
    ///     a shape the weaver would weave must classify as supported here (or the analyzer emits a
    ///     false-positive build error), and a shape it rejects should classify as unsupported (so the
    ///     mistake is caught early). Where they cannot cheaply agree — telling a real iterator apart
    ///     from a method that merely returns a bare enumerable — the analyzer DEFERS to the weaver
    ///     rather than risk a false positive.
    /// </summary>
    internal static class TypeHelpers
    {
        /// <summary><c>NDArray</c> or any subclass (<c>NDArray&lt;T&gt;</c>); NOT an array, pointer, or type parameter (matches the weaver).</summary>
        internal static bool IsNDArrayLike(ITypeSymbol t, KnownTypes k)
        {
            if (t == null || k.NDArray == null)
                return false;
            if (t is IArrayTypeSymbol || t.TypeKind == TypeKind.Pointer || t.TypeKind == TypeKind.TypeParameter)
                return false;
            for (var cur = t; cur != null; cur = cur.BaseType)
                if (Eq(cur, k.NDArray))
                    return true;
            return false;
        }

        /// <summary>NDArray-like, or a rank-1 array of NDArray-like — the two shapes <c>Returns</c> can yield.</summary>
        internal static bool IsNDArrayCarrying(ITypeSymbol t, KnownTypes k)
            => IsNDArrayLike(t, k) || (t is IArrayTypeSymbol arr && arr.Rank == 1 && IsNDArrayLike(arr.ElementType, k));

        /// <summary>
        ///     True for the four exact task shapes (<c>Task</c>, <c>Task&lt;T&gt;</c>, <c>ValueTask</c>,
        ///     <c>ValueTask&lt;T&gt;</c>); a custom task-like is not one. <paramref name="result"/> is the
        ///     <c>T</c>, or null for the resultless forms. Matches the weaver's <c>IsTaskLike</c>.
        /// </summary>
        internal static bool IsTaskLike(ITypeSymbol t, KnownTypes k, out ITypeSymbol result)
        {
            result = null;
            if (t is INamedTypeSymbol named && named.IsGenericType && named.TypeArguments.Length == 1)
            {
                var def = named.OriginalDefinition;
                if (Eq(def, k.TaskOfT) || Eq(def, k.ValueTaskOfT))
                {
                    result = named.TypeArguments[0];
                    return true;
                }
                return false;
            }
            return Eq(t, k.Task) || Eq(t, k.ValueTask);
        }

        /// <summary>One of the four enumerable interfaces a C# iterator can return (generic or the two non-generic ones).</summary>
        internal static bool IsEnumerableInterface(ITypeSymbol t, KnownTypes k)
        {
            var def = (t as INamedTypeSymbol)?.OriginalDefinition;
            if (def != null &&
                (Eq(def, k.IEnumerableT) || Eq(def, k.IEnumeratorT) ||
                 Eq(def, k.IAsyncEnumerableT) || Eq(def, k.IAsyncEnumeratorT)))
                return true;
            return Eq(t, k.IEnumerable) || Eq(t, k.IEnumerator);
        }

        /// <summary>The weaver's <c>Classify</c> as a boolean: is this a carrier the weaver can see every NDArray through?</summary>
        internal static bool IsSupportedCarrier(ITypeSymbol t, KnownTypes k)
        {
            if (t == null)
                return true;
            if (t.SpecialType == SpecialType.System_Void)
                return true;
            if (IsNDArrayLike(t, k))
                return true;
            if (t is IArrayTypeSymbol arr && arr.Rank == 1 && IsNDArrayLike(arr.ElementType, k))
                return true;
            if (IsTaskLike(t, k, out var inner))
                return inner == null || IsSupportedCarrier(inner, k);
            if (IsScalar(t))
                return true;
            if (t.IsTupleType)          // any ValueTuple (arity/mix) — Returns(ITuple)/typed overloads
                return true;
            if (IsSystemTuple(t))       // any reference-type System.Tuple<...>
                return true;
            if (ImplementsCarrier(t, k))
                return true;
            if (Eq(t, k.IArraySlice) || Eq(t, k.UnmanagedStorage))
                return true;
            return false;
        }

        /// <summary>A value holding no NDArray — primitives, enums, string, decimal/Half/Complex, native ints.</summary>
        internal static bool IsScalar(ITypeSymbol t)
        {
            switch (t.SpecialType)
            {
                case SpecialType.System_Boolean:
                case SpecialType.System_Char:
                case SpecialType.System_SByte:
                case SpecialType.System_Byte:
                case SpecialType.System_Int16:
                case SpecialType.System_UInt16:
                case SpecialType.System_Int32:
                case SpecialType.System_UInt32:
                case SpecialType.System_Int64:
                case SpecialType.System_UInt64:
                case SpecialType.System_Single:
                case SpecialType.System_Double:
                case SpecialType.System_String:
                case SpecialType.System_IntPtr:
                case SpecialType.System_UIntPtr:
                case SpecialType.System_Decimal:
                    return true;
            }

            if (t.TypeKind == TypeKind.Enum)
                return true;

            var name = t.ToDisplayString();
            return name == "System.Half" || name == "System.Numerics.Complex";
        }

        private static bool IsSystemTuple(ITypeSymbol t)
        {
            if (t is not INamedTypeSymbol named || !named.IsGenericType || named.Name != "Tuple")
                return false;
            var ns = named.ContainingNamespace;
            return ns != null && ns.Name == "System" && (ns.ContainingNamespace?.IsGlobalNamespace ?? false);
        }

        private static bool ImplementsCarrier(ITypeSymbol t, KnownTypes k)
        {
            if (k.INDArrayCarrier == null || !t.IsValueType)
                return false;
            foreach (var i in t.AllInterfaces)
                if (Eq(i, k.INDArrayCarrier))
                    return true;
            return false;
        }

        private static bool Eq(ISymbol a, ISymbol b)
            => b != null && SymbolEqualityComparer.Default.Equals(a, b);
    }
}
