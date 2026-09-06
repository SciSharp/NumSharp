using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // NDW012 across every METHOD-BODY KIND the analyzer visits: async methods, iterators,
    // constructors (instance/static), finalizers, expression-bodied members, property getters,
    // user-defined operators, and lambdas. Each unscoped body kind must be analyzed like a plain
    // method; a lambda that RETURNS its temp escapes it to the delegate's consumer.
    public class BodyKindHost
    {
        private static NDArray _x;

        public BodyKindHost(NDArray a, NDArray b) { var t = a + b; }       // [NDW012]  ctor leaks

        static BodyKindHost() { var t = _x + 1.0; }                        // [NDW012]  static ctor leaks

        ~BodyKindHost() { var t = _x + 1.0; }                              // [NDW012]  finalizer leaks

        public static BodyKindHost operator +(BodyKindHost l, BodyKindHost r)
        {
            var t = _x + 1.0;                                              // [NDW012]  operator body leaks
            return null;
        }
    }

    public static class BodyKindScenarios
    {
        private static NDArray _a;

        public static async Task AsyncLeak(NDArray a, NDArray b)
        {
            var t = a + b;                                                 // [NDW012]  async body leaks
            await Task.Yield();
        }

        public static IEnumerable<NDArray> IteratorLeak(NDArray a, NDArray b)
        {
            var t = a + b;                                                 // [NDW012]  iterator body leaks
            yield return a.copy();
        }

        public static void ExprBodiedDiscard(NDArray a, NDArray b) => np.add(a, b); // [NDW012]  expression-bodied discard

        public static NDArray LeakyGetter { get { var t = _a + 1.0; return np.sum(t); } } // [NDW012]  unscoped getter leaks

        // A lambda that RETURNS the temp hands it to the delegate's consumer — clean.
        public static NDArray LambdaReturnsTemp(NDArray a, NDArray b)
        {
            Func<NDArray> f = () => a + b;
            return f();
        }
    }
}
