namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // NDW012 owning-expression detection (COVERAGE_PLAN §3.3): which expressions are treated as a fresh
    // owned buffer (constructions, operators, calls) versus a view read (properties, indexers) that is
    // excluded to avoid flooding the fluent view chains.
    public static class OwningScenarios
    {
        // OW-1: `.T` is a property (a view), not a producer -> excluded, clean even when dropped.
        public static void PropertyViewDropped(NDArray a)
        {
            var t = a.T;
        }

        // OW-2: an indexer read is a view (and its receiver is followed) -> clean when dropped.
        public static void IndexerViewDropped(NDArray a)
        {
            var t = a["1:3"];
        }

        // OW-3: a chained operator leaks its INNER temp; the outer result is returned (clean). Reported
        // once, on the expression that produces the abandoned inner buffer.
        public static NDArray ChainedOperators(NDArray a, NDArray b, NDArray c)
        {
            return a + b + c;                               // [NDW012]  the inner (a + b) temp is abandoned
        }

        // OW-4: a constructed owned scalar dropped.
        public static void ScalarDropped()
        {
            var t = NDArray.Scalar(5.0);                    // [NDW012]  constructed, dropped
        }

        // OW-5: the generic reinterpret wrapper is itself a fresh result here (dropped), so it leaks —
        // the receiver-follow lands on the dropped MakeGeneric result.
        public static void GenericWrapperDropped(NDArray a, NDArray b)
        {
            var t = (a & b).MakeGeneric<bool>();            // [NDW012]  the MakeGeneric result is dropped
        }

        // OW-6: a compound assignment writes the new value back into the INPUT parameter — the caller
        // owns `a`, so this must not warn (R2).
        public static void CompoundOnInput(NDArray a, NDArray b)
        {
            a += b;
        }

        // OW-7: an implicit scalar conversion is not treated as a producer; pinned clean.
        public static void ImplicitScalarConversion()
        {
            NDArray t = 5;
        }
    }
}
