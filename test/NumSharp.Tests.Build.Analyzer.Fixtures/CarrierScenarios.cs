namespace NumSharp.Tests.Build.Analyzer.Fixtures
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

        // A tuple LITERAL of fresh temps holds all of them at once — dropping it drops every element.
        public static void DroppedTupleLiteral(NDArray a, NDArray b)
        {
            var t = (a + 1.0, b - 1.0);                     // [NDW012]  a literal (temp, temp), dropped
        }

        // A MIXED literal still owns its produced element, so the drop is still a leak.
        public static void DroppedMixedTupleLiteral(NDArray a, NDArray b)
        {
            var t = (a + 1.0, b);                           // [NDW012]  the a+1.0 element leaks
        }

        // A NESTED literal owns through the inner tuple.
        public static void DroppedNestedTupleLiteral(NDArray a, NDArray b)
        {
            var t = ((a + 1.0, b - 1.0), 5);                // [NDW012]  both inner temps leak
        }

        public static void DroppedArrayLiteral(NDArray a)
        {
            var g = new[] { a + 1.0 };                      // [NDW012]  an array literal of a temp, dropped
        }

        // Deconstructing a tuple LITERAL tracks each side individually, like a call result.
        public static NDArray DeconstructedLiteralOneSideLeaks(NDArray a, NDArray b)
        {
            var (q, r) = (a + 1.0, b - 1.0);                // [NDW012]  r is never used (q is returned)
            return q.copy();
        }

        // Deconstruction into EXISTING locals is the same contract — the target mention is a WRITE,
        // not a use, so the unused side still leaks (reported on the declaration).
        public static NDArray DeconstructIntoExistingUnusedSide(NDArray a)
        {
            NDArray q, r;                                   // [NDW012]  r is assigned but never used
            (q, r) = SplitTuple(a);
            return q.copy();
        }

        // The C# 12 collection-expression spelling of an array literal follows the same any-element rule.
        public static void DroppedCollectionExpression(NDArray a)
        {
            NDArray[] g = [a + 1.0];                        // [NDW012]  a collection expression of a temp, dropped
        }

        // A temp INSIDE an array handed to a non-consuming NumSharp op leaks exactly like a bare temp
        // handed to one — np.concatenate reads its inputs and never disposes them (R2).
        public static NDArray TempArrayIntoNumSharpOp(NDArray a, NDArray c)
        {
            return np.concatenate(new[] { a + 1.0, c }, 0); // [NDW012]  the a+1.0 element is stranded
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

        // A tuple/array of PLAIN INPUTS owns nothing — flagging it would demand reclaiming the
        // caller's values (R2), so both drops must stay silent.
        public static void TupleOfAliasesDropped(NDArray a, NDArray b)
        {
            var t = (a, b);
        }

        public static void ArrayOfAliasesDropped(NDArray a, NDArray b)
        {
            var g = new[] { a, b };
        }

        // An EMPTY array allocation carries no buffers — nothing to reclaim.
        public static void EmptyArrayDropped()
        {
            var g = new NDArray[5];
        }

        // Escaping literal forms: returned via a local, handed to a foreign sink, stored to a field.
        private static NDArray[] _cache;
        private static void Sink(NDArray[] xs) { }

        public static (NDArray, NDArray) ReturnedTupleLiteralViaLocal(NDArray a, NDArray b)
        {
            var t = (a + 1.0, b - 1.0);
            return t;
        }

        public static void ArrayLiteralHandedToSink(NDArray a)
        {
            Sink(new[] { a + 1.0 });
        }

        public static void ArrayLiteralStoredToField(NDArray a)
        {
            var g = new[] { a + 1.0 };
            _cache = g;
        }

        // Deconstruction into existing locals with BOTH sides used stays clean.
        public static NDArray DeconstructIntoExistingBothUsed(NDArray a)
        {
            NDArray q, r;
            (q, r) = SplitTuple(a);
            r.Dispose();
            return q.copy();
        }

        // Collection-expression twins of the alias/escape rules.
        public static void CollectionExpressionOfAliases(NDArray a, NDArray b)
        {
            NDArray[] g = [a, b];
        }

        public static NDArray[] CollectionExpressionReturned(NDArray a)
        {
            NDArray[] g = [a + 1.0];
            return g;
        }

        // An alias-only array into a NumSharp op owns nothing — silent.
        public static NDArray AliasArrayIntoNumSharpOp(NDArray a, NDArray c)
        {
            return np.concatenate(new[] { a, c }, 0);
        }

        // Enumerating a produced array is CONSUMPTION — the loop owns each element's fate, which the
        // per-method analysis cannot judge, so foreach collections are conservatively clean.
        public static void ForeachOverProduced(NDArray a)
        {
            foreach (var x in SplitArray(a)) { }
        }

        [NDScoped]
        public static CarrierPair ScopedCarrier(NDArray a)
        {
            var p = MakePair(a);                            // [NDScoped] -> exempt
            return p;
        }
    }
}
