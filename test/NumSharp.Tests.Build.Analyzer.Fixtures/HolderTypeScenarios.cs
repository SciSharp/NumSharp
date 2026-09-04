using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // NDW016 (a type STORES NDArrays but is not disposable). The tag sits on the line that declares
    // the TYPE — the diagnostic lands on its name — and every untagged type MUST stay clean. The
    // holder vocabulary: a bare NDArray (or subclass), an array of them, a tuple / collection /
    // generic instantiated over one, an INDArrayCarrier struct, and — ownership being contagious —
    // another type that stores them.

    /// <summary>A consumer's own generic container: holding-ness follows the type argument.</summary>
    public class Box<T>
    {
        public T Value;
    }

    // ----------------------------------------------------------------- MUST warn
    public class FieldHolder                                            // [NDW016]  a plain NDArray field
    {
        private NDArray _a;
        public FieldHolder(NDArray a) { _a = a; }
        public NDArray Peek() => _a;
    }

    public class AutoPropertyHolder                                     // [NDW016]  an auto-property is storage
    {
        public NDArray Value { get; set; }
    }

    public class InitOnlyPropertyHolder                                 // [NDW016]  an init-only auto-property too
    {
        public NDArray Value { get; init; }
    }

    public class ArrayHolder                                            // [NDW016]  NDArray[]
    {
        private NDArray[] _parts;
        public ArrayHolder(NDArray[] parts) { _parts = parts; }
    }

    public class RankTwoArrayHolder                                     // [NDW016]  an array of any rank
    {
        private NDArray[,] _grid;
    }

    public class ListHolder                                             // [NDW016]  List<NDArray>
    {
        private readonly List<NDArray> _items = new List<NDArray>();
        public void Add(NDArray x) => _items.Add(x);
    }

    public class DictionaryHolder                                       // [NDW016]  Dictionary<string, NDArray>
    {
        private readonly Dictionary<string, NDArray> _map = new Dictionary<string, NDArray>();
    }

    public class NestedGenericHolder                                    // [NDW016]  List<(string, NDArray)>
    {
        private List<(string name, NDArray value)> _named;
    }

    public class TupleHolder                                            // [NDW016]  a tuple with an NDArray component
    {
        private (NDArray a, int n) _pair;
    }

    public class CarrierFieldHolder                                     // [NDW016]  an INDArrayCarrier struct field
    {
        private ResultCarrier _pair;
    }

    public class LazyHolder                                             // [NDW016]  Lazy<NDArray>
    {
        private Lazy<NDArray> _lazy;
    }

    public class TaskHolder                                             // [NDW016]  Task<NDArray> — a pending result it will hold
    {
        private Task<NDArray> _pending;
    }

    public class TypedSubclassHolder                                    // [NDW016]  NDArray<double> (a subclass)
    {
        private NumSharp.Generic.NDArray<double> _typed;
    }

    public class ConstrainedGenericHolder<T> where T : NDArray          // [NDW016]  T : NDArray
    {
        private T _x;
    }

    public class ConsumerBoxHolder                                      // [NDW016]  Box<NDArray> — a generic over a holding type
    {
        private Box<NDArray> _box;
    }

    public record PositionalRecord(NDArray Data);                       // [NDW016]  a positional record property

    public struct PlainStruct                                           // [NDW016]  a struct stores one too (IDisposable or INDArrayCarrier)
    {
        public NDArray A;
    }

    public ref struct RefStructWithoutDispose                           // [NDW016]  a ref struct needs the Dispose() pattern
    {
        public NDArray A;
    }

    public abstract class AbstractHolder                                // [NDW016]  abstract types are analyzed like any other
    {
        protected NDArray _shared;
    }

    public class ManyHolders                                            // [NDW016]  the message names the first three and counts the rest
    {
        private NDArray _one, _two, _three, _four, _five;
    }

    public class HoldsDisposableOwner                                   // [NDW016]  stores an NDArray-owning disposable (contagious)
    {
        private DisposableOwner _owner;
    }

    public class HoldsOwnerList                                         // [NDW016]  List<DisposableOwner>
    {
        private List<DisposableOwner> _owners;
    }

    public class HoldsNonDisposableHolder                               // [NDW016]  stores a type that stores NDArrays (that type warns itself too)
    {
        private FieldHolder _inner;
    }

    public class NumSharpIteratorHolder                                 // [NDW016]  np.nditer's iterator exposes the operands it walks and must be closed
    {
        private np.NDIterator _it;
    }

    public class NpzArchiveHolder                                       // [NDW016]  an NpzFile hands out the arrays it loads
    {
        private NumSharp.IO.NpzFile _npz;
    }

    public class CycleWithArrayA                                        // [NDW016]  holds via B, which holds an array
    {
        private CycleWithArrayB _b;
    }

    public class CycleWithArrayB                                        // [NDW016]  holds '_x' (its back-reference to A adds nothing)
    {
        private CycleWithArrayA _a;
        private NDArray _x;
    }

    // ----------------------------------------------------------------- MUST stay clean
    public static class StaticOnly                                      // static storage is nobody's instance to dispose
    {
        private static NDArray _cache;
        private static List<NDArray> _pool;
    }

    public class InstanceStaticMix                                      // a static member never makes an instance an owner
    {
        private static NDArray _shared;
        private int _n;
    }

    public class ComputedPropertyOnly                                   // a computed property is a view, not storage
    {
        private int _n;
        public NDArray View => NDArray.Scalar(_n);
    }

    public class BorrowedMember                                         // [NDBorrowed] on the member: it references an array owned elsewhere
    {
        [NDBorrowed] private readonly NDArray _src;
        public BorrowedMember(NDArray src) { _src = src; }
    }

    [NDBorrowed]                                                        // [NDBorrowed] on the type: every array it references is borrowed
    public class BorrowedType
    {
        private NDArray _src;
        private List<NDArray> _more;
    }

    public class DelegateHolder                                         // a delegate that produces/consumes arrays holds none
    {
        private Func<NDArray> _make;
        private Action<NDArray> _sink;
        private Func<NDArray, NDArray> _map;
    }

    public class ComparerHolder                                         // comparers compare arrays they are handed
    {
        private IEqualityComparer<NDArray> _cmp;
        private IComparer<NDArray> _order;
    }

    public class WeakHolder                                             // a weak reference deliberately does not keep one alive
    {
        private WeakReference<NDArray> _weak;
    }

    public class ObserverHolder                                         // observers / progress sinks receive arrays, they do not store them
    {
        private IObserver<NDArray> _obs;
        private IProgress<NDArray> _progress;
    }

    public class UnconstrainedGeneric<T>                                // T could be anything — not statically an NDArray
    {
        private T _x;
        private List<T> _xs;
    }

    public class ObjectHolder                                           // object / IDisposable cannot statically name an NDArray (R2 conservatism)
    {
        private object _o;
        private IDisposable _d;
    }

    public readonly struct ResultCarrier : INDArrayCarrier              // a carrier is transient result packaging the scope yields through
    {
        public readonly NDArray A;
        public ResultCarrier(NDArray a) { A = a; }
        void INDArrayCarrier.YieldTo(NDScope scope) { scope.Returns(A); }
    }

    public class DisposableOwner : IDisposable                          // disposable AND disposes: the contract honoured
    {
        private NDArray _a;
        public DisposableOwner(NDArray a) { _a = a; }
        public void Dispose() => _a?.Dispose();
    }

    public class AsyncDisposableOwner : IAsyncDisposable                // IAsyncDisposable satisfies the contract too
    {
        private NDArray _a;
        public ValueTask DisposeAsync() { _a?.Dispose(); return default; }
    }

    public ref struct RefStructWithDispose                              // the ref-struct Dispose() pattern counts as disposable
    {
        private NDArray _a;
        public RefStructWithDispose(NDArray a) { _a = a; }
        public void Dispose() => _a?.Dispose();
    }

    [NDBorrowed]
    public sealed class BorrowedDisposable : IDisposable                // disposable but borrowing: owns unmanaged state, never its arrays
    {
        private NDArray _over;
        public void Dispose() { }
    }

    public class HoldsBorrowedDisposable                                // a [NDBorrowed] disposable is not contagious
    {
        private BorrowedDisposable _cursor;
    }

    public class HoldsFlatIterator                                      // NumSharp's own FlatIterator is [NDBorrowed] in metadata
    {
        private np.FlatIterator _flat;
    }

    public class HoldsScalarsOnly                                       // nothing NDArray-shaped at all
    {
        private double _x;
        private int[] _idx;
        private string _name;
    }

    public class DerivedNoOwnHolders : FieldHolder                      // its base is the one that stores (and warns); it adds nothing
    {
        private int _n;
        public DerivedNoOwnHolders(NDArray a) : base(a) { }
    }

    public class Node                                                   // self-referential, no arrays anywhere
    {
        private Node _next;
        private int _v;
    }

    public class CycleA                                                 // a two-type cycle with no arrays anywhere
    {
        private CycleB _b;
    }

    public class CycleB
    {
        private CycleA _a;
    }

    public interface IHasArray                                          // interfaces declare no storage
    {
        NDArray Data { get; }
    }

    public enum Kind { A, B }
    public delegate NDArray Producer(NDArray input);
}
