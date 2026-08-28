using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NDScoping.Instrumentation;

namespace NDScoping
{
    /// <summary>
    ///     LEVEL A — Manual <c>NDScope.Open()</c>. Every egress layer written by hand, so the reader
    ///     sees exactly the pattern the weaver (Level B) will later inject. Inputs are constructed by
    ///     the caller BEFORE each method's scope opens, so they are never tracked — that is what makes
    ///     "wrap every egress in <c>Returns</c>" a safe blanket rule and a passthrough a provable no-op.
    /// </summary>
    internal static class Level1_HandScope
    {
        private static NDArray A() => np.array(new double[] { 1, 2, 3 });
        private static NDArray B() => np.array(new double[] { 10, 20, 30 });
        private static NDArray C() => np.array(new double[] { 100, 200, 300 });

        public static List<Report> RunAll(bool stress)
        {
            var reports = new List<Report>();

            // L-arr
            reports.Add(ReclamationProbe.Run("Hand_Single", () =>
            {
                var temps = new List<NDArray>();
                var r = Hand_Single(A(), B(), C(), temps);
                return new DemoRun(temps, r);
            }, new double?[] { 110 }, stress));

            // L-many
            reports.Add(ReclamationProbe.Run("Hand_Array", () =>
            {
                var temps = new List<NDArray>();
                var r = Hand_Array(A(), B(), temps);
                return new DemoRun(temps, r);
            }, new double?[] { 12, -8 }, stress));

            // L-tup
            reports.Add(ReclamationProbe.Run("Hand_Tuple2", () =>
            {
                var temps = new List<NDArray>();
                var (q, r) = Hand_Tuple2(A(), B(), temps);
                return new DemoRun(temps, new[] { q, r });
            }, new double?[] { 22, 33 }, stress));

            // L-ituple
            reports.Add(ReclamationProbe.Run("Hand_ITuple", () =>
            {
                var temps = new List<NDArray>();
                var (x, n, y) = Hand_ITuple(A(), B(), temps);
                return new DemoRun(temps, new[] { x, y }, $"scalar {n} skipped");
            }, new double?[] { 11, 20 }, stress));

            // L-carrier
            reports.Add(ReclamationProbe.Run("Hand_Carrier", () =>
            {
                var temps = new List<NDArray>();
                var pair = Hand_Carrier(A(), B(), temps);
                return new DemoRun(temps, new[] { pair.First, pair.Second });
            }, new double?[] { 22, 1 }, stress));

            // L-buffer (bespoke — the result is a bare IArraySlice, not an NDArray)
            reports.Add(Hand_Buffer(stress));

            // L-out
            reports.Add(ReclamationProbe.Run("Hand_Out", () =>
            {
                var temps = new List<NDArray>();
                Hand_Out(A(), B(), C(), temps, out var r);
                return new DemoRun(temps, r);
            }, new double?[] { 110 }, stress));

            // L-scalar (bespoke — returns a double, no NDArray result)
            reports.Add(Hand_Scalar_Demo(stress));

            // L-pass
            reports.Add(ReclamationProbe.Run("Hand_Passthrough", () =>
            {
                var input = A();
                var temps = new List<NDArray>();
                var r = Hand_Passthrough(input, temps);
                return new DemoRun(temps, r, "input returned unchanged (Returns is a no-op)");
            }, new double?[] { 1 }, stress));

            // cross — the exception path still reclaims
            reports.Add(Hand_Exception_Demo(stress));

            return reports;
        }

        // ---- the workers: the hand-written pattern, one per layer -------------------------------

        // L-arr: return scope.Returns(result);
        private static NDArray Hand_Single(NDArray a, NDArray b, NDArray c, List<NDArray> temps)
        {
            using var scope = NDScope.Open();
            var ab = a * b;              // temp
            temps.Add(ab);
            return scope.Returns(ab + c); // result [110,240,390]; ab reclaimed at scope exit
        }

        // L-many: scope.Returns(new[]{ ... })
        private static NDArray[] Hand_Array(NDArray a, NDArray b, List<NDArray> temps)
        {
            using var scope = NDScope.Open();
            var scaled = a * 2.0;        // temp used by both outputs
            temps.Add(scaled);
            var sum = scaled + b;        // [12,24,36]
            var dif = scaled - b;        // [-8,-16,-24]
            return scope.Returns(new[] { sum, dif });
        }

        // L-tup: scope.Returns((q, r))
        private static (NDArray, NDArray) Hand_Tuple2(NDArray a, NDArray b, List<NDArray> temps)
        {
            using var scope = NDScope.Open();
            var s = a + b;               // temp [11,22,33]
            temps.Add(s);
            var q = s * 2.0;             // [22,44,66]
            var r = s * 3.0;             // [33,66,99]
            return scope.Returns((q, r));
        }

        // L-ituple: scope.Returns((ITuple)tuple) — NDArray members yielded, the scalar skipped
        private static (NDArray, int, NDArray) Hand_ITuple(NDArray a, NDArray b, List<NDArray> temps)
        {
            using var scope = NDScope.Open();
            var t = a * b;               // temp [10,40,90]
            temps.Add(t);
            var x = t + a;               // [11,42,93]
            var y = t + b;               // [20,60,120]
            var tuple = (x, 5, y);
            scope.Returns((ITuple)tuple);
            return tuple;
        }

        // L-carrier: ((INDArrayCarrier)result).YieldTo(scope)
        private static PairResult Hand_Carrier(NDArray a, NDArray b, List<NDArray> temps)
        {
            using var scope = NDScope.Open();
            var t = a + b;               // temp [11,22,33]
            temps.Add(t);
            var result = new PairResult(t * 2.0, t - b);   // [22,44,66], [1,2,3]
            ((INDArrayCarrier)result).YieldTo(scope);
            return result;
        }

        // L-out: r = scope.Returns(temp)
        private static bool Hand_Out(NDArray a, NDArray b, NDArray c, List<NDArray> temps, out NDArray r)
        {
            using var scope = NDScope.Open();
            var ab = a * b;              // temp
            temps.Add(ab);
            r = scope.Returns(ab + c);   // [110,240,390]
            return true;
        }

        // L-scalar: scope only — nothing to yield
        private static double Hand_Scalar(NDArray a, NDArray b, NDArray c, List<NDArray> temps)
        {
            using var scope = NDScope.Open();
            var t = a * b + c;           // temp [110,240,390]
            temps.Add(t);
            return t.GetDouble(0);       // 110; t reclaimed at scope exit
        }

        // L-pass: return scope.Returns(input) — input built BEFORE the scope, so Returns is a no-op
        private static NDArray Hand_Passthrough(NDArray input, List<NDArray> temps)
        {
            using var scope = NDScope.Open();
            var t = input * 2.0;         // a genuine temp (reclaimed)
            temps.Add(t);
            return scope.Returns(input); // untracked -> R2 for free; input untouched
        }

        // ---- the bespoke demos -----------------------------------------------------------------

        // L-buffer: a returned bare IArraySlice survives the scope's Release of the NDArray sharing
        // its buffer, because Returns(slice) takes a counted reference. The no-Returns control proves
        // the protection is load-bearing: without it the buffer is freed.
        private static Report Hand_Buffer(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                var (slice, wrapper) = BufferWithReturns(A(), B());
                bool wrapperReclaimed = wrapper.IsDisposed;   // the NDArray wrapper WAS reclaimed
                bool bufferProtected = !slice.IsReleased;     // but its buffer survived (counted ref)
                bool valueOk = bufferProtected && Math.Abs(slice.GetIndex<double>(0) - 11.0) < 1e-9;

                var (freed, _) = BufferNoReturns(A(), B());
                bool controlFreed = freed.IsReleased;         // counterfactual: no Returns -> freed

                ok &= wrapperReclaimed && bufferProtected && valueOk && controlFreed;
                detail = $"wrapper reclaimed {ReclamationProbe.Mark(wrapperReclaimed)}   buffer protected " +
                         $"{ReclamationProbe.Mark(bufferProtected)}   value {ReclamationProbe.Mark(valueOk)}   " +
                         $"(no-Returns control freed {ReclamationProbe.Mark(controlFreed)})";
                if (stress) { GC.Collect(); GC.WaitForPendingFinalizers(); }
            }

            return ReclamationProbe.Verdict("Hand_Buffer", ok, detail);
        }

        private static (IArraySlice slice, NDArray wrapper) BufferWithReturns(NDArray a, NDArray b)
        {
            using var scope = NDScope.Open();
            var t = a + b;                          // tracked; wrapper reclaimed at scope exit
            return (scope.Returns(t.GetData()), t); // buffer protected by the counted ref
        }

        private static (IArraySlice slice, NDArray wrapper) BufferNoReturns(NDArray a, NDArray b)
        {
            using var scope = NDScope.Open();
            var t = a + b;
            return (t.GetData(), t);                // NOT protected -> buffer freed at scope exit
        }

        private static Report Hand_Scalar_Demo(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                var temps = new List<NDArray>();
                double v = Hand_Scalar(A(), B(), C(), temps);
                bool reclaimed = temps.TrueForAll(t => t.IsDisposed);
                bool valueOk = Math.Abs(v - 110.0) < 1e-9;
                ok &= reclaimed && valueOk;
                detail = $"temps {temps.Count} reclaimed {ReclamationProbe.Mark(reclaimed)}   " +
                         $"scalar {ReclamationProbe.Mark(valueOk)} (= {v})";
                if (stress) { GC.Collect(); GC.WaitForPendingFinalizers(); }
            }

            return ReclamationProbe.Verdict("Hand_Scalar", ok, detail);
        }

        // cross: a throw mid-body — the using-finally still disposes the scope, so temps reclaim.
        private static Report Hand_Exception_Demo(bool stress)
        {
            var temps = new List<NDArray>();
            try
            {
                HandThrows(A(), B(), temps);
            }
            catch (InvalidOperationException)
            {
                // expected
            }

            bool reclaimed = temps.Count > 0 && temps.TrueForAll(t => t.IsDisposed);
            return ReclamationProbe.Verdict("Hand_Exception", reclaimed,
                $"temp created before the throw reclaimed on the exception path {ReclamationProbe.Mark(reclaimed)}");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray HandThrows(NDArray a, NDArray b, List<NDArray> temps)
        {
            using var scope = NDScope.Open();
            var t = a + b;
            temps.Add(t);
            throw new InvalidOperationException("boom");
        }
    }
}
