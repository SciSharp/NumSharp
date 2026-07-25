using System;
using System.Diagnostics;

namespace NumSharp.Backends
{
    public class BackendFactory
    {
        /// <summary>
        ///     Replaces the engine every new <see cref="NDArray"/> resolves to, or null for the
        ///     built-in pure-C# <see cref="DefaultEngine"/>.
        /// </summary>
        /// <remarks>
        ///     The install point for an out-of-box backend: an optional package (e.g.
        ///     <c>NumSharp.Interop.BLAS</c>) subclasses <see cref="DefaultEngine"/>, overrides the
        ///     operations it accelerates, and assigns the subclass here — typically from a
        ///     <c>[ModuleInitializer]</c>, so referencing the package is all the wiring there is.
        ///     NumSharp itself never sets this: with no package installed the engine is the
        ///     built-in one and every kernel is NumSharp's own managed code.
        ///     <para>
        ///     Assigning affects arrays created AFTERWARDS; arrays already alive keep the engine
        ///     they resolved at construction. Set it once at startup.
        ///     </para>
        /// </remarks>
        public static TensorEngine Default { get; set; }

        [DebuggerNonUserCode]
        public static TensorEngine GetEngine(BackendType backendType = BackendType.Default)
        {
            switch (backendType)
            {
                case BackendType.Default:
                    return Default ?? EngineCache<DefaultEngine>.Value;
                default:
                    throw new ArgumentOutOfRangeException(nameof(backendType), backendType, null);
            }
        }

        [DebuggerNonUserCode]
        public static TensorEngine GetEngine<T>() where T : TensorEngine, new()
        {
            return EngineCache<T>.Value;
        }

        private static class EngineCache<T> where T : TensorEngine, new()
        {
            public static readonly T Value = new T();
        }
    }
}
