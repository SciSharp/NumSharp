using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // NDW017 (a DISPOSABLE type never disposes an NDArray-holding member on any path from its
    // Dispose / Dispose(bool) / DisposeAsync / DisposeAsyncCore). The tag sits on the MEMBER
    // declaration line; every untagged member of a disposable type MUST stay clean. The clean half
    // enumerates every spelling of "disposed on the Dispose path" the analyzer recognises.

    public static class DisposeHelpers
    {
        public static void Free(NDArray x) => x?.Dispose();
    }

    public readonly struct PathCarrier : INDArrayCarrier
    {
        public readonly NDArray First;
        public readonly NDArray Second;
        public PathCarrier(NDArray first, NDArray second) { First = first; Second = second; }
        void INDArrayCarrier.YieldTo(NDScope scope) { scope.Returns(First); scope.Returns(Second); }
    }

    public abstract class DisposableBase : IDisposable
    {
        public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
        protected virtual void Dispose(bool disposing) { }
    }

    public class DisposableOwnerForPath : IDisposable
    {
        private NDArray _a;
        public void Dispose() => _a?.Dispose();
    }

    public class NonDisposableInner                                     // [NDW016]  stores an array, not disposable (the hint below points here)
    {
        private NDArray _a;
    }

    // ----------------------------------------------------------------- MUST warn
    public class ForgetsOneOfTwo : IDisposable
    {
        private NDArray _kept;
        private NDArray _forgotten;                                     // [NDW017]  never reached by Dispose
        public void Dispose() => _kept?.Dispose();
    }

    public class NullsInsteadOfDisposing : IDisposable
    {
        private NDArray _a;                                             // [NDW017]  `_a = null` drops the reference, it does not dispose
        public void Dispose() { _a = null; }
    }

    public class ClearsTheList : IDisposable
    {
        private readonly List<NDArray> _items = new List<NDArray>();    // [NDW017]  Clear() forgets the elements without disposing them
        public void Dispose() => _items.Clear();
    }

    public class ForeachWithoutDispose : IDisposable
    {
        private readonly List<NDArray> _items = new List<NDArray>();    // [NDW017]  the loop reads the elements, never disposes them
        public void Dispose() { foreach (var x in _items) Console.WriteLine(x.size); }
    }

    public class DisposesInResetOnly : IDisposable
    {
        private NDArray _a;                                             // [NDW017]  Reset() disposes it, but Dispose never calls Reset()
        public void Reset() { _a?.Dispose(); _a = null; }
        public void Dispose() { }
    }

    public class HelperNotReachable : IDisposable
    {
        private NDArray _a;                                             // [NDW017]  Release() disposes it, nothing on the Dispose path calls Release()
        private void Release() => _a?.Dispose();
        public void Dispose() { }
    }

    public class FinalizerOnly : IDisposable
    {
        private NDArray _a;                                             // [NDW017]  a finalizer is the backstop prompt disposal exists to avoid
        ~FinalizerOnly() { _a?.Dispose(); }
        public void Dispose() { }
    }

    public class LogsInsteadOfDisposing : IDisposable
    {
        private NDArray _a;                                             // [NDW017]  an argument computed FROM the member is not a hand-off of it
        public void Dispose() => Console.WriteLine(_a.ToString());
    }

    public class InheritsDisposeWithoutOverride : DisposableBase
    {
        private NDArray _own;                                           // [NDW017]  the base's Dispose cannot see this member
    }

    public class HoldsOwnerButForgets : IDisposable
    {
        private DisposableOwnerForPath _owner;                          // [NDW017]  a contagious member must be disposed too
        public void Dispose() { }
    }

    public class HoldsNonDisposableInner : IDisposable
    {
        private NonDisposableInner _inner;                              // [NDW017]  cannot be disposed until its type is made disposable
        public void Dispose() { }
    }

    public class AutoPropertyForgotten : IDisposable
    {
        public NDArray Value { get; set; }                              // [NDW017]  an auto-property holder
        public void Dispose() { }
    }

    public class AsyncForgets : IAsyncDisposable
    {
        private NDArray _a;                                             // [NDW017]  DisposeAsync never reaches it
        public ValueTask DisposeAsync() => default;
    }

    public class DisposesOnlyOneTupleSide : IDisposable
    {
        private (NDArray a, NDArray b) _pair;
        private NDArray _other;                                         // [NDW017]  the pair is handled (see below); this one is not
        public void Dispose() { _pair.a?.Dispose(); _pair.b?.Dispose(); }
    }

    public struct DisposableStructForgets : IDisposable
    {
        private NDArray _a;                                             // [NDW017]  a struct is held to the same contract
        public void Dispose() { }
    }

    // ----------------------------------------------------------------- MUST stay clean
    public class DirectDispose : IDisposable
    {
        private NDArray _a;
        public void Dispose() { _a.Dispose(); }
    }

    public class ConditionalDispose : IDisposable
    {
        private NDArray _a;
        public void Dispose() => _a?.Dispose();
    }

    public class ThisQualified : IDisposable
    {
        private NDArray _a;
        public void Dispose() { this._a?.Dispose(); }
    }

    public class CastDispose : IDisposable
    {
        private NDArray _a;
        public void Dispose() { ((IDisposable)_a).Dispose(); }
    }

    public class AsCastDispose : IDisposable
    {
        private NDArray _a;
        public void Dispose() { (_a as IDisposable)?.Dispose(); }
    }

    public class GuardedDispose : IDisposable
    {
        private NDArray _a;
        private bool _disposed;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_a is not null) { _a.Dispose(); _a = null; }
        }
    }

    public class TryFinallyDispose : IDisposable
    {
        private NDArray _a;
        public void Dispose() { try { } finally { _a?.Dispose(); } }
    }

    public class DisposeBoolPattern : IDisposable
    {
        private NDArray _a;
        public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
        protected virtual void Dispose(bool disposing) { if (disposing) _a?.Dispose(); }
        ~DisposeBoolPattern() { Dispose(false); }
    }

    public class ExplicitInterface : IDisposable
    {
        private NDArray _a;
        void IDisposable.Dispose() => _a?.Dispose();
    }

    public class ViaHelper : IDisposable
    {
        private NDArray _a;
        public void Dispose() => Release();
        private void Release() { _a?.Dispose(); _a = null; }
    }

    public class ViaHelperChain : IDisposable
    {
        private NDArray _a;
        public void Dispose() => Cleanup();
        private void Cleanup() => Free();
        private void Free() => _a?.Dispose();
    }

    public class ViaStaticHelperArgument : IDisposable
    {
        private NDArray _a;
        public void Dispose() => Free(_a);
        private static void Free(NDArray x) => x?.Dispose();
    }

    public class ViaExternalHelperArgument : IDisposable                // any method the member is handed to is assumed to dispose it
    {
        private NDArray _a;
        public void Dispose() => DisposeHelpers.Free(_a);
    }

    public class ViaPropertySetter : IDisposable                        // the setter is a same-type accessor on the path
    {
        private NDArray _buffer;
        public NDArray Buffer { get => _buffer; set { _buffer?.Dispose(); _buffer = value; } }
        public void Dispose() { Buffer = null; }
    }

    public class ViaDelegateOverHelper : IDisposable                    // a delegate over a same-type method is a callee
    {
        private NDArray _a;
        public void Dispose() { Action free = Free; free(); }
        private void Free() => _a?.Dispose();
    }

    public class ViaLocalFunction : IDisposable
    {
        private NDArray _a;
        public void Dispose() { void Free() => _a?.Dispose(); Free(); }
    }

    public class LocalCopyThenDispose : IDisposable
    {
        private NDArray _a;
        public void Dispose() { var old = _a; _a = null; old?.Dispose(); }
    }

    public class InterlockedExchange : IDisposable
    {
        private NDArray _a;
        public void Dispose() => Interlocked.Exchange(ref _a, null)?.Dispose();
    }

    public class UsingStatement : IDisposable
    {
        private NDArray _a;
        public void Dispose() { using (_a) { } }
    }

    public class UsingDeclaration : IDisposable
    {
        private NDArray _a;
        public void Dispose() { using var x = _a; }
    }

    public class OverridesBaseDispose : DisposableBase
    {
        private NDArray _own;
        protected override void Dispose(bool disposing) { _own?.Dispose(); base.Dispose(disposing); }
    }

    public class ArrayForeach : IDisposable
    {
        private NDArray[] _parts;
        public void Dispose() { foreach (var p in _parts) p?.Dispose(); }
    }

    public class ArrayForLoop : IDisposable
    {
        private NDArray[] _parts;
        public void Dispose() { for (int i = 0; i < _parts.Length; i++) _parts[i]?.Dispose(); }
    }

    public class ArrayStaticForEach : IDisposable
    {
        private NDArray[] _parts;
        public void Dispose() => Array.ForEach(_parts, p => p?.Dispose());
    }

    public class ListForeach : IDisposable
    {
        private readonly List<NDArray> _items = new List<NDArray>();
        public void Dispose() { foreach (var x in _items) x.Dispose(); _items.Clear(); }
    }

    public class ListForEachLambda : IDisposable
    {
        private readonly List<NDArray> _items = new List<NDArray>();
        public void Dispose() => _items.ForEach(x => x.Dispose());
    }

    public class ListIndexer : IDisposable
    {
        private readonly List<NDArray> _items = new List<NDArray>();
        public void Dispose() { for (int i = 0; i < _items.Count; i++) _items[i]?.Dispose(); }
    }

    public class ParallelForEach : IDisposable
    {
        private readonly List<NDArray> _items = new List<NDArray>();
        public void Dispose() => Parallel.ForEach(_items, x => x.Dispose());
    }

    public class DictionaryValues : IDisposable
    {
        private readonly Dictionary<string, NDArray> _map = new Dictionary<string, NDArray>();
        public void Dispose() { foreach (var v in _map.Values) v.Dispose(); }
    }

    public class DictionaryPairs : IDisposable
    {
        private readonly Dictionary<string, NDArray> _map = new Dictionary<string, NDArray>();
        public void Dispose() { foreach (var kv in _map) kv.Value.Dispose(); }
    }

    public class DictionaryDeconstructed : IDisposable
    {
        private readonly Dictionary<string, NDArray> _map = new Dictionary<string, NDArray>();
        public void Dispose() { foreach (var (_, v) in _map) v?.Dispose(); }
    }

    public class LinqToListForEach : IDisposable
    {
        private readonly Dictionary<string, NDArray> _map = new Dictionary<string, NDArray>();
        public void Dispose() => _map.Values.ToList().ForEach(v => v.Dispose());
    }

    public class NestedForeach : IDisposable
    {
        private NDArray[][] _rows;
        public void Dispose() { foreach (var row in _rows) foreach (var x in row) x?.Dispose(); }
    }

    public class TupleComponents : IDisposable
    {
        private (NDArray a, NDArray b) _pair;
        public void Dispose() { _pair.a?.Dispose(); _pair.b?.Dispose(); }
    }

    public class TupleDeconstructed : IDisposable
    {
        private (NDArray a, NDArray b) _pair;
        public void Dispose() { var (a, b) = _pair; a?.Dispose(); b?.Dispose(); }
    }

    public class LazyValueDisposed : IDisposable
    {
        private Lazy<NDArray> _lazy;
        public void Dispose() { if (_lazy.IsValueCreated) _lazy.Value.Dispose(); }
    }

    public class CarrierFieldDisposed : IDisposable
    {
        private PathCarrier _pair;
        public void Dispose() { _pair.First?.Dispose(); _pair.Second?.Dispose(); }
    }

    public class AutoPropertyDisposed : IDisposable
    {
        public NDArray Value { get; set; }
        public void Dispose() => Value?.Dispose();
    }

    public class ContagiousMemberDisposed : IDisposable
    {
        private DisposableOwnerForPath _owner;
        public void Dispose() => _owner?.Dispose();
    }

    public class OwnerListDisposed : IDisposable
    {
        private readonly List<DisposableOwnerForPath> _owners = new List<DisposableOwnerForPath>();
        public void Dispose() { foreach (var o in _owners) o.Dispose(); }
    }

    public class NumSharpIteratorClosed : IDisposable                   // NumSharp's NumPy-spelled close() is a dispose
    {
        private np.NDIterator _it;
        public void Dispose() => _it?.close();
    }

    public class AsyncDisposes : IAsyncDisposable
    {
        private NDArray _a;
        public async ValueTask DisposeAsync() { _a?.Dispose(); await Task.Yield(); }
    }

    public class AsyncCorePattern : IAsyncDisposable, IDisposable
    {
        private NDArray _a;
        public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
        protected virtual void Dispose(bool disposing) { if (disposing) _a?.Dispose(); }
        public async ValueTask DisposeAsync() { await DisposeAsyncCore(); Dispose(false); GC.SuppressFinalize(this); }
        protected virtual ValueTask DisposeAsyncCore() { _a?.Dispose(); return default; }
    }

    public readonly struct DisposableStruct : IDisposable
    {
        private readonly NDArray _a;
        public DisposableStruct(NDArray a) { _a = a; }
        public void Dispose() => _a?.Dispose();
    }

    public ref struct RefStructPattern
    {
        private NDArray _a;
        public RefStructPattern(NDArray a) { _a = a; }
        public void Dispose() => _a?.Dispose();
    }

    public record DisposableRecord(NDArray Data) : IDisposable
    {
        public void Dispose() => Data?.Dispose();
    }

    public partial class PartialAcrossParts : IDisposable
    {
        private NDArray _a;
    }

    public partial class PartialAcrossParts
    {
        public void Dispose() => _a?.Dispose();
    }

    public class BorrowedMemberInDisposable : IDisposable               // the borrowed member is nobody's to dispose here
    {
        [NDBorrowed] private NDArray _src;
        private NDArray _own;
        public void Dispose() => _own?.Dispose();
    }

    public class NestedTypeHasItsOwnScan : IDisposable                  // a nested type is analyzed on its own
    {
        private NDArray _a;
        public void Dispose() => _a?.Dispose();

        public class Inner : IDisposable
        {
            private NDArray _b;
            public void Dispose() => _b?.Dispose();
        }
    }
}
