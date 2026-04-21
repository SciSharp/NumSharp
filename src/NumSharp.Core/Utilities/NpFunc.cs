using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    #region Placeholder Types for Expression-based Dispatch

    /// <summary>Placeholder type for first type argument. Replace with actual type via NPTypeCode.</summary>
    public struct TArg1 { }
    /// <summary>Placeholder type for second type argument.</summary>
    public struct TArg2 { }
    /// <summary>Placeholder type for third type argument.</summary>
    public struct TArg3 { }
    /// <summary>Placeholder type for fourth type argument.</summary>
    public struct TArg4 { }

    #endregion

    /// <summary>
    /// Generic type dispatch using Expression trees with placeholder types.
    /// </summary>
    /// <remarks>
    /// <para><b>Usage:</b></para>
    /// <code>
    /// // Single type dispatch
    /// NpFunc.Execute(
    ///     () => ILKernelGenerator.ClipArrayMin((TArg1*)outPtr, (TArg1*)minPtr, len),
    ///     typeCode
    /// );
    ///
    /// // Two type dispatch (e.g., input/output differ)
    /// NpFunc.Execute(
    ///     () => SomeKernel((TArg1*)outPtr, (TArg2*)inPtr, len),
    ///     outputTypeCode,
    ///     inputTypeCode
    /// );
    /// </code>
    ///
    /// <para>
    /// The expression is compiled once per unique type combination and cached.
    /// Subsequent calls with the same types use the cached delegate.
    /// </para>
    /// </remarks>
    public static unsafe class NpFunc
    {
        #region Expression Cache

        private static readonly ConcurrentDictionary<(int exprId, NPTypeCode t1), Action> _cache1 = new();
        private static readonly ConcurrentDictionary<(int exprId, NPTypeCode t1, NPTypeCode t2), Action> _cache2 = new();
        private static readonly ConcurrentDictionary<(int exprId, NPTypeCode t1, NPTypeCode t2, NPTypeCode t3), Action> _cache3 = new();

        private static int _nextExprId = 0;

        #endregion

        #region Execute with Single Type (TArg1)

        /// <summary>
        /// Execute an expression with TArg1 replaced by the type for typeCode1.
        /// </summary>
        /// <param name="expression">Expression using TArg1* for pointer casts</param>
        /// <param name="typeCode1">Type to substitute for TArg1</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Execute(Expression<Action> expression, NPTypeCode typeCode1)
        {
            var exprId = GetExpressionId(expression);
            var key = (exprId, typeCode1);

            if (!_cache1.TryGetValue(key, out var action))
            {
                action = CompileWithSubstitution(expression, typeCode1);
                _cache1[key] = action;
            }

            action();
        }

        /// <summary>
        /// Create a reusable dispatcher for an expression with TArg1.
        /// Call this once, then use the returned Dispatcher for fast repeated execution.
        /// </summary>
        public static Dispatcher1 Compile(Expression<Action> expression)
        {
            return new Dispatcher1(expression);
        }

        #endregion

        #region Execute with Two Types (TArg1, TArg2)

        /// <summary>
        /// Execute an expression with TArg1 and TArg2 replaced by the specified types.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Execute(Expression<Action> expression, NPTypeCode typeCode1, NPTypeCode typeCode2)
        {
            var exprId = GetExpressionId(expression);
            var key = (exprId, typeCode1, typeCode2);

            if (!_cache2.TryGetValue(key, out var action))
            {
                action = CompileWithSubstitution(expression, typeCode1, typeCode2);
                _cache2[key] = action;
            }

            action();
        }

        /// <summary>
        /// Create a reusable dispatcher for an expression with TArg1 and TArg2.
        /// </summary>
        public static Dispatcher2 Compile2(Expression<Action> expression)
        {
            return new Dispatcher2(expression);
        }

        #endregion

        #region Execute with Three Types (TArg1, TArg2, TArg3)

        /// <summary>
        /// Execute an expression with TArg1, TArg2, and TArg3 replaced.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Execute(Expression<Action> expression, NPTypeCode typeCode1, NPTypeCode typeCode2, NPTypeCode typeCode3)
        {
            var exprId = GetExpressionId(expression);
            var key = (exprId, typeCode1, typeCode2, typeCode3);

            if (!_cache3.TryGetValue(key, out var action))
            {
                action = CompileWithSubstitution(expression, typeCode1, typeCode2, typeCode3);
                _cache3[key] = action;
            }

            action();
        }

        #endregion

        #region Dispatchers (Pre-compiled, faster for repeated use)

        /// <summary>
        /// Pre-compiled dispatcher for expressions with one type parameter.
        /// </summary>
        public sealed class Dispatcher1
        {
            private readonly Expression<Action> _expression;
            private readonly Action[] _compiled = new Action[32];

            internal Dispatcher1(Expression<Action> expression) => _expression = expression;

            /// <summary>Execute with the specified type.</summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Execute(NPTypeCode typeCode)
            {
                var idx = (int)typeCode;
                var action = _compiled[idx];
                if (action == null)
                {
                    action = CompileWithSubstitution(_expression, typeCode);
                    _compiled[idx] = action;
                }
                action();
            }

            /// <summary>Indexer access for execution.</summary>
            public Action this[NPTypeCode typeCode]
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get
                {
                    var idx = (int)typeCode;
                    return _compiled[idx] ??= CompileWithSubstitution(_expression, typeCode);
                }
            }
        }

        /// <summary>
        /// Pre-compiled dispatcher for expressions with two type parameters.
        /// </summary>
        public sealed class Dispatcher2
        {
            private readonly Expression<Action> _expression;
            private readonly ConcurrentDictionary<(NPTypeCode, NPTypeCode), Action> _compiled = new();

            internal Dispatcher2(Expression<Action> expression) => _expression = expression;

            /// <summary>Execute with the specified types.</summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Execute(NPTypeCode typeCode1, NPTypeCode typeCode2)
            {
                var key = (typeCode1, typeCode2);
                if (!_compiled.TryGetValue(key, out var action))
                {
                    action = CompileWithSubstitution(_expression, typeCode1, typeCode2);
                    _compiled[key] = action;
                }
                action();
            }

            /// <summary>Indexer access for execution.</summary>
            public Action this[NPTypeCode typeCode1, NPTypeCode typeCode2]
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => _compiled.GetOrAdd((typeCode1, typeCode2),
                    _ => CompileWithSubstitution(_expression, typeCode1, typeCode2));
            }
        }

        #endregion

        #region Expression Compilation

        private static int GetExpressionId(Expression<Action> expression)
        {
            // Use expression string as identity (simple but works)
            // In production, could use a more sophisticated hash
            return expression.ToString().GetHashCode();
        }

        private static Action CompileWithSubstitution(Expression<Action> expression, NPTypeCode typeCode1)
        {
            var type1 = typeCode1.AsType();
            var visitor = new TypeSubstitutionVisitor(type1, null, null, null);
            var modified = (Expression<Action>)visitor.Visit(expression);
            return modified.Compile();
        }

        private static Action CompileWithSubstitution(Expression<Action> expression, NPTypeCode typeCode1, NPTypeCode typeCode2)
        {
            var type1 = typeCode1.AsType();
            var type2 = typeCode2.AsType();
            var visitor = new TypeSubstitutionVisitor(type1, type2, null, null);
            var modified = (Expression<Action>)visitor.Visit(expression);
            return modified.Compile();
        }

        private static Action CompileWithSubstitution(Expression<Action> expression, NPTypeCode typeCode1, NPTypeCode typeCode2, NPTypeCode typeCode3)
        {
            var type1 = typeCode1.AsType();
            var type2 = typeCode2.AsType();
            var type3 = typeCode3.AsType();
            var visitor = new TypeSubstitutionVisitor(type1, type2, type3, null);
            var modified = (Expression<Action>)visitor.Visit(expression);
            return modified.Compile();
        }

        #endregion

        #region Expression Visitor for Type Substitution

        private sealed class TypeSubstitutionVisitor : ExpressionVisitor
        {
            private readonly Type _type1;
            private readonly Type _type2;
            private readonly Type _type3;
            private readonly Type _type4;

            private static readonly Type _targ1 = typeof(TArg1);
            private static readonly Type _targ2 = typeof(TArg2);
            private static readonly Type _targ3 = typeof(TArg3);
            private static readonly Type _targ4 = typeof(TArg4);
            private static readonly Type _targ1Ptr = typeof(TArg1*);
            private static readonly Type _targ2Ptr = typeof(TArg2*);
            private static readonly Type _targ3Ptr = typeof(TArg3*);
            private static readonly Type _targ4Ptr = typeof(TArg4*);

            public TypeSubstitutionVisitor(Type type1, Type type2, Type type3, Type type4)
            {
                _type1 = type1;
                _type2 = type2;
                _type3 = type3;
                _type4 = type4;
            }

            protected override Expression VisitUnary(UnaryExpression node)
            {
                // Handle pointer casts: (TArg1*)expr -> (actualType*)expr
                if (node.NodeType == ExpressionType.Convert)
                {
                    var targetType = node.Type;
                    var newType = SubstitutePointerType(targetType);

                    if (newType != targetType)
                    {
                        var operand = Visit(node.Operand);
                        return Expression.Convert(operand, newType);
                    }
                }

                return base.VisitUnary(node);
            }

            protected override Expression VisitMethodCall(MethodCallExpression node)
            {
                // Handle generic method calls: Method<TArg1>(...) -> Method<actualType>(...)
                if (node.Method.IsGenericMethod)
                {
                    var genericDef = node.Method.GetGenericMethodDefinition();
                    var typeArgs = node.Method.GetGenericArguments();
                    var newTypeArgs = typeArgs.Select(SubstituteType).ToArray();

                    if (!typeArgs.SequenceEqual(newTypeArgs))
                    {
                        var newMethod = genericDef.MakeGenericMethod(newTypeArgs);
                        var newArgs = node.Arguments.Select(Visit).ToArray();
                        return node.Object != null
                            ? Expression.Call(Visit(node.Object), newMethod, newArgs)
                            : Expression.Call(newMethod, newArgs);
                    }
                }

                return base.VisitMethodCall(node);
            }

            private Type SubstituteType(Type type)
            {
                if (type == _targ1 && _type1 != null) return _type1;
                if (type == _targ2 && _type2 != null) return _type2;
                if (type == _targ3 && _type3 != null) return _type3;
                if (type == _targ4 && _type4 != null) return _type4;
                return type;
            }

            private Type SubstitutePointerType(Type type)
            {
                if (!type.IsPointer) return type;

                var elementType = type.GetElementType();
                if (elementType == _targ1 && _type1 != null) return _type1.MakePointerType();
                if (elementType == _targ2 && _type2 != null) return _type2.MakePointerType();
                if (elementType == _targ3 && _type3 != null) return _type3.MakePointerType();
                if (elementType == _targ4 && _type4 != null) return _type4.MakePointerType();
                return type;
            }
        }

        #endregion

        #region Legacy Table-based Dispatch (still available)

        /// <summary>Delegate for 2-pointer operations.</summary>
        public delegate void D2(nint p1, nint p2, long len);
        /// <summary>Delegate for 3-pointer operations.</summary>
        public delegate void D3(nint p1, nint p2, nint p3, long len);

        /// <summary>
        /// Create a dispatch table using switch expression factory.
        /// </summary>
        public static Table2 For2(Func<NPTypeCode, D2> factory)
        {
            var table = new D2[32];
            foreach (NPTypeCode code in Enum.GetValues(typeof(NPTypeCode)))
                if (code != NPTypeCode.Empty)
                    table[(int)code] = factory(code);
            return new Table2(table);
        }

        /// <summary>
        /// Create a dispatch table using switch expression factory.
        /// </summary>
        public static Table3 For3(Func<NPTypeCode, D3> factory)
        {
            var table = new D3[32];
            foreach (NPTypeCode code in Enum.GetValues(typeof(NPTypeCode)))
                if (code != NPTypeCode.Empty)
                    table[(int)code] = factory(code);
            return new Table3(table);
        }

        /// <summary>Dispatch table for 2-pointer operations.</summary>
        public sealed class Table2
        {
            private readonly D2[] _table;
            internal Table2(D2[] table) => _table = table;
            public D2 this[NPTypeCode code] => _table[(int)code] ?? throw new NotSupportedException($"Type {code} not supported");
        }

        /// <summary>Dispatch table for 3-pointer operations.</summary>
        public sealed class Table3
        {
            private readonly D3[] _table;
            internal Table3(D3[] table) => _table = table;
            public D3 this[NPTypeCode code] => _table[(int)code] ?? throw new NotSupportedException($"Type {code} not supported");
        }

        #endregion
    }
}
