using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // NDW012 escape / consume nuances (COVERAGE_PLAN §3.2). Every legitimate egress — handed off,
    // stored, written back, yielded, awaited, observed — MUST stay clean. The single tagged line is a
    // deliberate leak anchor so the file is non-vacuous (a silently dead analyzer fails it).
    public static class EscapeScenarios
    {
        private static void Foo(ref NDArray x) { }
        private static void FooOut(out NDArray x) { x = null; }
        private static void FooIn(in NDArray x) { }
        private static void Bar(params NDArray[] xs) { }
        private static void Use(NDArray x) { }
        private static NDArray _refField;
        private static Task<NDArray> ComputeAsync(NDArray x) => Task.FromResult(x);

        // EC-1: a ref argument write-back is an egress the caller reads.
        public static void RefArgument(NDArray a, NDArray b)
        {
            var t = a + b;
            Foo(ref t);
        }

        // EC-1b: an out argument is likewise a write-back egress.
        public static void OutArgument(NDArray a, NDArray b)
        {
            var t = a + b;
            FooOut(out t);
        }

        // EC-2: an `in` parameter still hands the value off.
        public static void InArgument(NDArray a, NDArray b)
        {
            var t = a + b;
            FooIn(in t);
        }

        // EC-3: params-array elements are handed to a consuming API.
        public static void ParamsArray(NDArray a, NDArray b, NDArray c, NDArray d)
        {
            Bar(a + b, c - d);
        }

        // EC-4: added to a collection (stored beyond the call).
        public static void CollectionAdd(List<NDArray> list, NDArray a, NDArray b)
        {
            list.Add(a + b);
        }

        // EC-4b: a collection initializer stores the value (even if the list is later dropped, the
        // analyzer tracks the NDArray, which escaped into the collection).
        public static void CollectionInitializer(NDArray a, NDArray b)
        {
            var list = new List<NDArray> { a + b };
        }

        // EC-5: stored through an indexer.
        public static void IndexerStore(Dictionary<int, NDArray> map, int k, NDArray a, NDArray b)
        {
            map[k] = a + b;
        }

        // EC-6: captured by a lambda (used later; the analyzer cannot follow the closure, so it treats
        // the capture as an escape rather than risk a false positive).
        public static void LambdaCapture(NDArray a, NDArray b)
        {
            var t = a + b;
            Action f = () => Use(t);
            f();
        }

        // EC-7: returned from a local function — clean inside the local function's own block.
        public static NDArray LocalFunctionReturn(NDArray a, NDArray b)
        {
            NDArray L() => a + b;
            return L();
        }

        // EC-8: yielded from an iterator — the consumer owns each element.
        public static IEnumerable<NDArray> Yielded(NDArray a, NDArray b)
        {
            yield return a + b;
        }

        // EC-9: handed to a non-NumSharp async method, then awaited.
        public static async Task<NDArray> AwaitedArgument(NDArray a, NDArray b)
        {
            var v = await ComputeAsync(a + b);
            return v;
        }

        // EC-10: observed by a non-NumSharp API (returns non-NDArray -> observed, not owned).
        public static void Observed(NDArray a, NDArray b)
        {
            Console.WriteLine(a + b);
        }

        // EC-11: both temps escape through the returned tuple.
        public static (NDArray, NDArray) TupleReturn(NDArray a, NDArray b, NDArray c, NDArray d)
        {
            return (a + b, c - d);
        }

        // EC-12: stored through a ref local (an alias to a field) — the ref-local aliasing drops the
        // local from the owning set, so the stored value reads as an escape.
        public static void RefLocalStore(NDArray a, NDArray b)
        {
            ref NDArray r = ref _refField;
            r = a + b;
        }

        // The non-vacuity anchor: a dropped result really does warn.
        public static void LeakAnchor(NDArray a, NDArray b)
        {
            np.add(a, b);                                   // [NDW012]  dropped on the floor
        }
    }
}
