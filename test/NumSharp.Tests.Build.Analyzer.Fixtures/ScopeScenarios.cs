using System.Threading.Tasks;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // NDW012 scope / exemption nuances (COVERAGE_PLAN §3.4): the ways a method is exempted from the leak
    // check — the [NDScoped]/[NDScopedAsync] attributes (including on an accessor), a hand-opened
    // NDScope, and NDScope.Detach — versus a nested block (own operation block) that is analyzed anew.
    public static class ScopeScenarios
    {
        private static NDArray _a;

        // SC-1: [NDScoped] on a property getter (via AssociatedSymbol) exempts the leaking body.
        [NDScoped]
        public static NDArray ScopedGetter
        {
            get { var t = _a + 1.0; return t.copy(); }
        }

        // SC-2: a hand-opened NDScope (using STATEMENT form) exempts the whole method.
        public static NDArray HandScopeStatement(NDArray a)
        {
            using (NDScope.Open())
            {
                var t = a + 1.0;
                return t.copy();
            }
        }

        // SC-3: NDScope.Open in a NESTED block still exempts the WHOLE method (matches the weaver's
        // whole-method skip) — so the temp built OUTSIDE the nested scope also draws nothing.
        public static void HandScopeNestedBlock(NDArray a, NDArray b)
        {
            { using (NDScope.Open()) { } }
            var t = a + b;
        }

        // SC-4: a local function inside a NON-scoped method is its own operation block and IS analyzed.
        public static void LocalFunctionLeaks(NDArray a, NDArray b)
        {
            void L() { var t = a + b; }                     // [NDW012]  the local function leaks its temp
            L();
        }

        // SC-6: an [NDScopedAsync] async method is exempt.
        [NDScopedAsync]
        public static async Task<NDArray> AsyncScopedLeaks(NDArray a, NDArray b)
        {
            var t = a + b;
            await Task.Yield();
            return t.copy();
        }

        // SC-7: NDScope.Detach marks the value as intentionally un-tracked -> reclaimed, clean.
        public static void DetachedThenDropped(NDArray a, NDArray b)
        {
            var t = a + b;
            NDScope.Detach(t);
        }
    }
}
