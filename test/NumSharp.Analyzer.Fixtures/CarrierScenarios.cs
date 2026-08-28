using NumSharp;

namespace NumSharp.Analyzer.Fixtures
{
    // NDW012 across the carrier "variations" the weaver supports: NDArray[], a ValueTuple of NDArrays,
    // and an INDArrayCarrier result struct. Tagged lines MUST warn; untagged MUST stay clean.
    public readonly struct CarrierPair : INDArrayCarrier
    {
        public readonly NDArray First;
        public readonly NDArray Second;
        public CarrierPair(NDArray first, NDArray second) { First = first; Second = second; }
        void INDArrayCarrier.YieldTo(NDScope scope) { scope.Returns(First); scope.Returns(Second); }
    }

    public static class CarrierScenarios
    {
        // Helpers: each RETURNS its temporaries (tuple / array / carrier), so each is itself clean.
        private static (NDArray, NDArray) SplitTuple(NDArray a) => (a + 1.0, a - 1.0);
        private static NDArray[] SplitArray(NDArray a) => new[] { a + 1.0, a - 1.0 };
        private static CarrierPair MakePair(NDArray a) => new CarrierPair(a + 1.0, a - 1.0);

        // ----------------------------------------------------------------- MUST warn
        public static void DroppedTuple(NDArray a)
        {
            var t = SplitTuple(a);                          // [NDW012]  a (NDArray, NDArray) result, dropped
        }

        public static NDArray DeconstructedOneSideLeaks(NDArray a)
        {
            var (q, r) = SplitTuple(a);                     // [NDW012]  r is never used (q is returned)
            return q.copy();
        }

        public static void DroppedArray(NDArray a)
        {
            var g = SplitArray(a);                          // [NDW012]  an NDArray[] result, dropped
        }

        public static void DroppedCarrier(NDArray a)
        {
            var p = MakePair(a);                            // [NDW012]  an INDArrayCarrier result, dropped
        }

        // ----------------------------------------------------------------- MUST stay clean
        public static (NDArray, NDArray) ReturnedTuple(NDArray a)
        {
            return SplitTuple(a);
        }

        public static NDArray[] ReturnedArray(NDArray a)
        {
            return SplitArray(a);
        }

        [NDScoped]
        public static CarrierPair ScopedCarrier(NDArray a)
        {
            var p = MakePair(a);                            // [NDScoped] -> exempt
            return p;
        }
    }
}
