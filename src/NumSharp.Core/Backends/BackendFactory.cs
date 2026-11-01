using System;
using System.Diagnostics;

namespace NumSharp.Backends
{
    /// <summary>
    ///     The shipped way to reach an engine: <see cref="GetEngine(BackendType)"/> is <see cref="TensorEngine.Default"/>,
    ///     and <see cref="GetEngine{T}"/> one cached instance per engine type.
    /// </summary>
    /// <remarks>
    ///     New code names <see cref="TensorEngine.Default"/> (the engine arrays bind when nothing else decides) or an
    ///     array's <see cref="NDArray.TensorEngine"/> (the engine it computes on) directly; this class stays for the callers
    ///     that already use it.
    /// </remarks>
    public class BackendFactory
    {
        /// <summary>
        ///     The engine of a backend: for <see cref="BackendType.Default"/>, the current <see cref="TensorEngine.Default"/>
        ///     (the built-in <see cref="DefaultEngine"/> until another engine is assigned there).
        /// </summary>
        /// <param name="backendType">The backend; <see cref="BackendType.Default"/> is the only one.</param>
        /// <returns><see cref="TensorEngine.Default"/> as it is at the call: an engine assigned there later is not this
        ///     one, so hold the property rather than a value read from it once.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="backendType"/> is not a
        ///     <see cref="BackendType"/> member this method knows.</exception>
        [DebuggerNonUserCode]
        public static TensorEngine GetEngine(BackendType backendType = BackendType.Default)
        {
            switch (backendType)
            {
                case BackendType.Default:
                    return TensorEngine.Default;
                default:
                    throw new ArgumentOutOfRangeException(nameof(backendType), backendType, null);
            }
        }

        /// <summary>
        ///     The one cached instance of the engine type <typeparamref name="T"/>, created on first use with its
        ///     parameterless constructor.
        /// </summary>
        /// <typeparam name="T">The engine type.</typeparam>
        /// <returns>The same instance on every call for the same <typeparamref name="T"/>. It is not
        ///     <see cref="TensorEngine.Default"/> unless assigned there: arrays bind it only when it is.</returns>
        [DebuggerNonUserCode]
        public static TensorEngine GetEngine<T>() where T : TensorEngine, new()
        {
            return EngineCache<T>.Value;
        }

        /// <summary>
        ///     The built-in <see cref="DefaultEngine"/>: the instance <see cref="TensorEngine.Default"/> returns while nothing
        ///     else is assigned, and <see cref="GetEngine{T}"/>'s for <see cref="DefaultEngine"/>.
        /// </summary>
        /// <remarks>Read on every array construction that binds the default engine: the closed generic access compiles to a
        ///     direct static load, where the shared generic <see cref="GetEngine{T}"/> may need a runtime lookup.</remarks>
        internal static DefaultEngine BuiltIn => EngineCache<DefaultEngine>.Value;

        /// <summary>The cache behind <see cref="GetEngine{T}"/>: one instance per engine type, created on first access.</summary>
        /// <typeparam name="T">The engine type.</typeparam>
        private static class EngineCache<T> where T : TensorEngine, new()
        {
            /// <summary>The cached instance (the CLR runs the initializer once, thread-safely).</summary>
            public static readonly T Value = new T();
        }
    }
}
