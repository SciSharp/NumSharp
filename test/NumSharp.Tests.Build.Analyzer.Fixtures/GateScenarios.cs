using System.Collections.Generic;
using System.Threading.Tasks;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // The [NDScoped]/[NDScopedAsync] TARGET gate (NDW002/003/005/006/009/010/011). These are ERRORS,
    // so this project's .editorconfig silences them for the fixture's own build; NumSharp.Tests.Build.Analyzer
    // asserts them in-process. Each tag sits on the METHOD/PROPERTY declaration line — where the gate
    // analyzer reports (the symbol location), NOT on the body.
    public abstract class GateScenarios
    {
        [NDScoped]
        public static List<NDArray> UnsupportedCarrier(NDArray a)     // [NDW003]  List<NDArray> is not a woven carrier
            => new List<NDArray> { a + 1.0 };

        [NDScoped]
        public static void RefEgress(ref NDArray a)                   // [NDW002]  ref NDArray is a hidden egress
        {
            a = a + 1.0;
        }

        [NDScoped]
        public abstract NDArray NoBody();                             // [NDW005]  abstract -> nothing to weave

        [NDScoped]
        public static async Task<NDArray> AsyncUnderSync(NDArray a)   // [NDW009]  async wants [NDScopedAsync]
        {
            await Task.Yield();
            return a + 1.0;
        }

        [NDScopedAsync]
        public static NDArray SyncUnderAsync(NDArray a)               // [NDW010]  a plain sync method wants [NDScoped]
            => a + 1.0;

        [NDScoped]
        [NDScopedAsync]
        public static async Task<NDArray> BothAttributes(NDArray a)   // [NDW011]  a method has exactly one scoping model
        {
            await Task.Yield();
            return a + 1.0;
        }

        [NDScoped]
        public static NDArray WriteOnlyProperty { set { } }           // [NDW006]  setter-only property has no getter to weave
    }
}
