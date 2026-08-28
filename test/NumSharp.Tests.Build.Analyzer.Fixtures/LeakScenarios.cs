namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // NDW012 (the NDArray leak warning). A line tagged with the diagnostic id in trailing brackets
    // MUST draw exactly one NDW012; every untagged line MUST stay clean. The tests in
    // NumSharp.Tests.Build.Analyzer assert this exactly (a missing warning AND an unexpected one both fail).
    public static class LeakScenarios
    {
        private static NDArray _field;
        private static void ConsumingSink(NDArray x) { }   // a foreign API assumed to take ownership

        // ----------------------------------------------------------------- MUST warn
        public static void DeadLocal(NDArray a, NDArray b)
        {
            var t = a + b;                                  // [NDW012]  created, never used again
        }

        public static double DeadLocalThenReturnScalar(NDArray a)
        {
            var t = a + 1.0;                                // [NDW012]  never used, method returns a scalar
            return 0.0;
        }

        public static NDArray ArgumentToNpOp(NDArray a, NDArray b)
        {
            return np.sum(a * b);                           // [NDW012]  the 'a * b' temp is handed to np.sum
        }

        public static void DiscardedResult(NDArray a, NDArray b)
        {
            np.add(a, b);                                   // [NDW012]  result dropped on the floor
        }

        public static NDArray LocalFedIntoNpOp(NDArray a)
        {
            var t = a + 1.0;                                // [NDW012]  t is only ever handed to np.sum
            return np.sum(t);
        }

        public static void ConstructedAndDropped(int n)
        {
            var t = new NDArray(typeof(double), new Shape(n)); // [NDW012]  constructed and dropped
        }

        // ----------------------------------------------------------------- MUST stay clean
        public static NDArray Returned(NDArray a, NDArray b)
        {
            return a + b;                                   // returned — the caller owns it
        }

        public static NDArray DisposedByUsing(NDArray a)
        {
            using var t = a + 1.0;                          // disposed at scope exit
            return t.copy();
        }

        public static NDArray DisposedExplicitly(NDArray a)
        {
            var t = a + 1.0;
            var r = t.copy();
            t.Dispose();                                    // reclaimed before it dies
            return r;
        }

        public static NDArray HandScopedAndYielded(NDArray a)
        {
            using var scope = NDScope.Open();               // opens a scope -> whole method exempt
            var t = a + 1.0;
            return scope.Returns(t.copy());
        }

        public static NDArray HandScopedAndAttached(NDArray a)
        {
            using var scope = NDScope.Open();
            var t = a + 1.0;
            NDScope.Attach(t);                              // adopted into the scope
            return a.copy();
        }

        public static void OutParameter(NDArray a, out NDArray r)
        {
            r = a + 1.0;                                    // out egress
        }

        public static void StoredInField(NDArray a)
        {
            _field = a + 1.0;                               // stored beyond the method
        }

        public static void HandedToForeignApi(NDArray a)
        {
            var t = a + 1.0;
            ConsumingSink(t);                               // handed to a non-NumSharp API (assumed consumed)
        }

        public static NDArray AliasesTheInput(NDArray a)
        {
            var t = a;                                      // aliases the input — not this method's to reclaim
            return t.copy();
        }

        public static NDArray FluentViewChain(NDArray a, NDArray b)
        {
            return (a + b).reshape(1, -1);                  // the (a+b) receiver shares the returned view's buffer
        }

        [NDScoped]
        public static NDArray ScopedAttribute(NDArray a, NDArray b)
        {
            var t = a + b;                                  // [NDScoped] -> the weaver reclaims it -> exempt
            return t.copy();
        }

        [NDScopedCovered]
        public static NDArray AmbientHelper(NDArray a, NDArray b)
        {
            var t = a + b;                                  // [NDScopedCovered] -> exempt though 't' is a dead local
            return a.copy();                                // (the assertion: this helper always runs under a caller's scope)
        }
    }
}
