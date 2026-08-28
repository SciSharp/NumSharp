using Microsoft.CodeAnalysis;

namespace NumSharp.Build.Analyzer
{
    /// <summary>
    ///     The symbols the analyzer keys off, resolved ONCE per compilation. If NumSharp is not
    ///     referenced (neither attribute, or no <c>NDArray</c>), <see cref="Resolve"/> returns null
    ///     and the analyzer does nothing — the same "no [NDScoped] usage, nothing to do" the weaver
    ///     reaches. The full metadata names match the IL weaver's constants exactly, so the analyzer
    ///     and the weaver classify identically.
    /// </summary>
    internal sealed class KnownTypes
    {
        public INamedTypeSymbol SyncAttr;
        public INamedTypeSymbol AsyncAttr;
        public INamedTypeSymbol CoveredAttr;
        public INamedTypeSymbol NDArray;
        public INamedTypeSymbol NDScope;
        public INamedTypeSymbol INDArrayCarrier;
        public INamedTypeSymbol IArraySlice;
        public INamedTypeSymbol UnmanagedStorage;

        /// <summary>The assembly NDArray lives in (NumSharp) — the leak analyzer's test for a NumSharp op that never disposes its NDArray inputs.</summary>
        public IAssemblySymbol NumSharpAssembly => NDArray?.ContainingAssembly;

        public INamedTypeSymbol Task;
        public INamedTypeSymbol TaskOfT;
        public INamedTypeSymbol ValueTask;
        public INamedTypeSymbol ValueTaskOfT;

        public INamedTypeSymbol IEnumerable;
        public INamedTypeSymbol IEnumerator;
        public INamedTypeSymbol IEnumerableT;
        public INamedTypeSymbol IEnumeratorT;
        public INamedTypeSymbol IAsyncEnumerableT;
        public INamedTypeSymbol IAsyncEnumeratorT;

        public static KnownTypes Resolve(Compilation c)
        {
            var sync = c.GetTypeByMetadataName("NumSharp.NDScopedAttribute");
            var asyncAttr = c.GetTypeByMetadataName("NumSharp.NDScopedAsyncAttribute");
            var ndArray = c.GetTypeByMetadataName("NumSharp.NDArray");

            // Both attributes and the NDArray anchor come from the SAME NumSharp assembly, so if the
            // attributes are present NDArray is too. Bail unless we can classify carriers (an NDArray
            // return must never be mistaken for an unsupported one).
            if ((sync == null && asyncAttr == null) || ndArray == null)
                return null;

            return new KnownTypes
            {
                SyncAttr = sync,
                AsyncAttr = asyncAttr,
                CoveredAttr = c.GetTypeByMetadataName("NumSharp.NDScopedCoveredAttribute"),
                NDArray = ndArray,
                NDScope = c.GetTypeByMetadataName("NumSharp.NDScope"),
                INDArrayCarrier = c.GetTypeByMetadataName("NumSharp.INDArrayCarrier"),
                IArraySlice = c.GetTypeByMetadataName("NumSharp.Backends.Unmanaged.IArraySlice"),
                UnmanagedStorage = c.GetTypeByMetadataName("NumSharp.Backends.UnmanagedStorage"),

                Task = c.GetTypeByMetadataName("System.Threading.Tasks.Task"),
                TaskOfT = c.GetTypeByMetadataName("System.Threading.Tasks.Task`1"),
                ValueTask = c.GetTypeByMetadataName("System.Threading.Tasks.ValueTask"),
                ValueTaskOfT = c.GetTypeByMetadataName("System.Threading.Tasks.ValueTask`1"),

                IEnumerable = c.GetTypeByMetadataName("System.Collections.IEnumerable"),
                IEnumerator = c.GetTypeByMetadataName("System.Collections.IEnumerator"),
                IEnumerableT = c.GetTypeByMetadataName("System.Collections.Generic.IEnumerable`1"),
                IEnumeratorT = c.GetTypeByMetadataName("System.Collections.Generic.IEnumerator`1"),
                IAsyncEnumerableT = c.GetTypeByMetadataName("System.Collections.Generic.IAsyncEnumerable`1"),
                IAsyncEnumeratorT = c.GetTypeByMetadataName("System.Collections.Generic.IAsyncEnumerator`1"),
            };
        }
    }
}
