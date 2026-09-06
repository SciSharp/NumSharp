using System;
using System.Collections.Generic;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // NDW012 across the CONTAGION cases: an instance of an NDArray-owning DISPOSABLE — a consumer's
    // holder class, NumSharp's own np.nditer iterator — is an owned value the method must reclaim,
    // exactly like a bare NDArray; and a foreach over a produced NDArray (or such an instance) never
    // disposes it, because C# disposes only the enumerator it obtains. Tagged lines MUST warn;
    // untagged MUST stay clean.
    public sealed class Batch : IDisposable
    {
        private readonly NDArray _data;
        public Batch(NDArray data) { _data = data; }
        public NDArray Data => _data;
        public void Dispose() => _data?.Dispose();
    }

    [NDBorrowed] // a disposable that only BORROWS: it owns state of its own, never the array it walks
    public sealed class BorrowingCursor : IDisposable
    {
        private readonly NDArray _over;
        public BorrowingCursor(NDArray over) { _over = over; }
        public void Dispose() { }
    }

    public sealed class PlainWrapper                                    // [NDW016]  stores an array, not disposable
    {
        private readonly NDArray _data;
        public PlainWrapper(NDArray data) { _data = data; }
    }

    public static class ContagionScenarios
    {
        private static Batch _kept;
        private static void Sink(Batch b) { }
        private static NDArray[] SplitArray(NDArray a) => new[] { a + 1.0, a - 1.0 };
        private static Batch MakeBatch(NDArray a) => new Batch(a + 1.0);

        // ----------------------------------------------------------------- MUST warn
        public static void DroppedHolderInstance(NDArray a, NDArray b)
        {
            new Batch(a + b);                                           // [NDW012]  the batch (and the a+b it owns) is dropped
        }

        public static void DeadHolderLocal(NDArray a, NDArray b)
        {
            var batch = new Batch(a + b);                               // [NDW012]  never used again
        }

        public static void HolderFromFactoryDropped(NDArray a)
        {
            var batch = MakeBatch(a);                                   // [NDW012]  a factory result is owned like a constructed one
        }

        public static void ExplicitlyDiscardedHolder(NDArray a, NDArray b)
        {
            _ = new Batch(a + b);                                       // [NDW012]  an explicit discard drops it too
        }

        public static void DroppedHolderArray(NDArray a)
        {
            var batches = new[] { new Batch(a + 1.0) };                 // [NDW012]  an array literal of an owned batch, dropped
        }

        public static void NumSharpIteratorDropped(NDArray a)
        {
            var it = np.nditer(a);                                      // [NDW012]  np.nditer MUST be closed; this one never is
        }

        public static void NumSharpIteratorWalkedByForeach(NDArray a)
        {
            foreach (var x in np.nditer(a)) { }                         // [NDW012]  foreach disposes the ENUMERATOR, never the iterator
        }

        public static void ProducedNDArrayWalkedByForeach(NDArray a, NDArray b)
        {
            foreach (var row in a + b) { }                              // [NDW012]  the a+b temp is enumerated, never disposed
        }

        public static void TempHandedToIterator(NDArray a, NDArray b)
        {
            using var it = np.nditer(a + b);                            // [NDW012]  the iterator borrows a+b and never disposes it
        }

        // ----------------------------------------------------------------- MUST stay clean
        public static void HolderDisposedByUsing(NDArray a, NDArray b)
        {
            using var batch = new Batch(a + b);
        }

        public static void HolderDisposedExplicitly(NDArray a, NDArray b)
        {
            var batch = new Batch(a + b);
            batch.Dispose();
        }

        public static Batch HolderReturned(NDArray a, NDArray b)
        {
            return new Batch(a + b);
        }

        public static void HolderStored(NDArray a, NDArray b)
        {
            _kept = new Batch(a + b);
        }

        public static void HolderHandedToSink(NDArray a, NDArray b)
        {
            Sink(new Batch(a + b));
        }

        public static void HolderInUsingStatement(NDArray a, NDArray b)
        {
            using (var batch = new Batch(a + b)) { }
        }

        public static void IteratorClosed(NDArray a)
        {
            var it = np.nditer(a);
            it.close();
        }

        public static void IteratorUsingThenWalked(NDArray a)
        {
            using var it = np.nditer(a);
            foreach (var x in it) { }
        }

        public static void BorrowingDisposableDropped(NDArray a)
        {
            var c = new BorrowingCursor(a);                             // a [NDBorrowed] type is not an owned value
        }

        public static void NonDisposableWrapperDropped(NDArray a)
        {
            var w = new PlainWrapper(a);                                // not disposable: nothing here to reclaim (the type itself warns)
        }

        public static void ProducedArrayWalkedByForeach(NDArray a)
        {
            foreach (var x in SplitArray(a)) x.Dispose();               // an NDArray[] is a managed container; the loop owns its elements
        }

        public static void AliasWalkedByForeach(NDArray a)
        {
            foreach (var row in a) { }                                  // an input alias is not this method's to reclaim
        }

        [NDScoped]
        public static void ScopedHolderDropped(NDArray a, NDArray b)
        {
            var batch = new Batch(a + b);                               // the weaver's scope reclaims the a+b it holds
        }
    }
}
