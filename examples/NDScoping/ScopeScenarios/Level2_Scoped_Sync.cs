using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NDScoping.Instrumentation;

namespace NDScoping
{
    /// <summary>
    ///     LEVEL B — <c>[NDScoped]</c>. The SAME layers as Level A, but each method keeps its 100%
    ///     original body and carries only the attribute; the build-time weaver injects the
    ///     <c>NDScope.Open()</c> prologue, the try/finally dispose, and the <c>Returns</c> egress. The
    ///     behaviour is identical — zero boilerplate. (Every method here is proven woven by the
    ///     weave-coverage line the driver prints: each carries an NDScope local.)
    /// </summary>
    internal static class Level2_Scoped_Sync
    {
        private static NDArray A() => np.array(new double[] { 1, 2, 3 });
        private static NDArray B() => np.array(new double[] { 10, 20, 30 });
        private static NDArray C() => np.array(new double[] { 100, 200, 300 });

        public static List<Report> RunAll(bool stress)
        {
            var reports = new List<Report>();

            reports.Add(ReclamationProbe.Run("Scoped_Single", () =>
            {
                var temps = new List<NDArray>();
                var r = Scoped_Single(A(), B(), C(), temps);
                return new DemoRun(temps, r);
            }, new double?[] { 110 }, stress));

            reports.Add(ReclamationProbe.Run("Scoped_Array", () =>
            {
                var temps = new List<NDArray>();
                var r = Scoped_Array(A(), B(), temps);
                return new DemoRun(temps, r);
            }, new double?[] { 12, -8 }, stress));

            reports.Add(ReclamationProbe.Run("Scoped_Tuple", () =>
            {
                var temps = new List<NDArray>();
                var (q, r) = Scoped_Tuple(A(), B(), temps);
                return new DemoRun(temps, new[] { q, r });
            }, new double?[] { 22, 33 }, stress));

            reports.Add(ReclamationProbe.Run("Scoped_Carrier", () =>
            {
                var temps = new List<NDArray>();
                var pair = Scoped_Carrier(A(), B(), temps);
                return new DemoRun(temps, new[] { pair.First, pair.Second });
            }, new double?[] { 22, 1 }, stress));

            reports.Add(Scoped_Buffer_Demo(stress));

            reports.Add(ReclamationProbe.Run("Scoped_Out", () =>
            {
                var temps = new List<NDArray>();
                Scoped_Out(A(), B(), C(), temps, out var r);
                return new DemoRun(temps, r);
            }, new double?[] { 110 }, stress));

            reports.Add(ReclamationProbe.Run("Scoped_Void", () =>
            {
                var temps = new List<NDArray>();
                Scoped_Void(A(), B(), C(), temps, out var r);
                return new DemoRun(temps, r);
            }, new double?[] { 110 }, stress));

            reports.Add(Scoped_Property_Demo(stress));
            reports.Add(Scoped_Iterator_Demo(stress));
            reports.Add(ReclamationProbe.Run("Scoped_Idempotent", () =>
            {
                var temps = new List<NDArray>();
                var r = Scoped_Idempotent(A(), B(), temps);
                return new DemoRun(temps, r, "weaver skipped (already opens a scope by hand)");
            }, new double?[] { 22 }, stress));

            return reports;
        }

        // ---- the woven workers: attribute + ORIGINAL body (no scope in source) ------------------

        [NDScoped]
        private static NDArray Scoped_Single(NDArray a, NDArray b, NDArray c, List<NDArray> temps)
        {
            var ab = a * b;
            temps.Add(ab);
            return ab + c;
        }

        [NDScoped]
        private static NDArray[] Scoped_Array(NDArray a, NDArray b, List<NDArray> temps)
        {
            var scaled = a * 2.0;
            temps.Add(scaled);
            var sum = scaled + b;
            var dif = scaled - b;
            return new[] { sum, dif };
        }

        [NDScoped]
        private static (NDArray, NDArray) Scoped_Tuple(NDArray a, NDArray b, List<NDArray> temps)
        {
            var s = a + b;
            temps.Add(s);
            return (s * 2.0, s * 3.0);
        }

        [NDScoped]
        private static PairResult Scoped_Carrier(NDArray a, NDArray b, List<NDArray> temps)
        {
            var t = a + b;
            temps.Add(t);
            return new PairResult(t * 2.0, t - b);
        }

        [NDScoped]
        private static IArraySlice Scoped_Buffer(NDArray a, NDArray b, List<NDArray> temps)
        {
            var t = a + b;
            temps.Add(t);
            return t.GetData();
        }

        [NDScoped]
        private static bool Scoped_Out(NDArray a, NDArray b, NDArray c, List<NDArray> temps, out NDArray r)
        {
            var ab = a * b;
            temps.Add(ab);
            r = ab + c;
            return true;
        }

        [NDScoped]
        private static void Scoped_Void(NDArray a, NDArray b, NDArray c, List<NDArray> temps, out NDArray r)
        {
            var ab = a * b;
            temps.Add(ab);
            r = ab + c;
        }

        // A property GETTER is a woven target too. It appends its temp to a static sink so the demo
        // can observe reclamation (a getter takes no parameters).
        private static readonly NDArray _propInput = np.array(new double[] { 1, 2, 3 });
        private static readonly List<NDArray> _propTemps = new();

        [NDScoped]
        private static NDArray ScaledProp
        {
            get
            {
                var t = _propInput * 10.0;   // temp [10,20,30]
                _propTemps.Add(t);
                return t + _propInput;       // [11,22,33]
            }
        }

        // A synchronous iterator compiles to a state machine; the weave gives it ONE invocation scope
        // (in a weaver-added field), reclaimed at the final MoveNext or the enumerator's Dispose. Each
        // yielded element is routed through Returns — the consumer owns it.
        [NDScoped]
        private static IEnumerable<NDArray> Scoped_Iterator(NDArray a, NDArray b, List<NDArray> hoisted)
        {
            var scale = a + b;              // hoisted state — reclaimed at completion / Dispose
            hoisted.Add(scale);
            for (int i = 1; i <= 3; i++)
                yield return scale * (double)i;
        }

        // [NDScoped] AND a hand-written scope: the weaver detects the Open() call and SKIPS (idempotence),
        // so the body is not double-wrapped and behaves exactly once.
        [NDScoped]
        private static NDArray Scoped_Idempotent(NDArray a, NDArray b, List<NDArray> temps)
        {
            using var scope = NDScope.Open();
            var t = a + b;
            temps.Add(t);
            return scope.Returns(t * 2.0);
        }

        // ---- the bespoke demos -----------------------------------------------------------------

        private static Report Scoped_Buffer_Demo(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                var temps = new List<NDArray>();
                var slice = Scoped_Buffer(A(), B(), temps);
                bool wrapperReclaimed = temps.TrueForAll(t => t.IsDisposed);
                bool bufferProtected = !slice.IsReleased;
                bool valueOk = bufferProtected && Math.Abs(slice.GetIndex<double>(0) - 11.0) < 1e-9;
                ok &= wrapperReclaimed && bufferProtected && valueOk;
                detail = $"wrapper reclaimed {ReclamationProbe.Mark(wrapperReclaimed)}   buffer protected " +
                         $"{ReclamationProbe.Mark(bufferProtected)}   value {ReclamationProbe.Mark(valueOk)}";
                if (stress) { GC.Collect(); GC.WaitForPendingFinalizers(); }
            }

            return ReclamationProbe.Verdict("Scoped_Buffer", ok, detail);
        }

        private static Report Scoped_Property_Demo(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                _propTemps.Clear();
                var result = ScaledProp;
                bool tempReclaimed = _propTemps.TrueForAll(t => t.IsDisposed);
                bool resultAlive = !result.IsDisposed;
                bool valueOk = Math.Abs(result.GetDouble(0) - 11.0) < 1e-9;
                ok &= tempReclaimed && resultAlive && valueOk;
                detail = $"getter woven; temp reclaimed {ReclamationProbe.Mark(tempReclaimed)}   result alive " +
                         $"{ReclamationProbe.Mark(resultAlive)}   value {ReclamationProbe.Mark(valueOk)}";
                if (stress) { result.Dispose(); GC.Collect(); GC.WaitForPendingFinalizers(); }
            }

            return ReclamationProbe.Verdict("Scoped_Property", ok, detail);
        }

        private static Report Scoped_Iterator_Demo(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                // full iteration — hoisted state reclaimed at the end, every element owned by us
                var hoisted = new List<NDArray>();
                var yielded = new List<NDArray>();
                foreach (var x in Scoped_Iterator(A(), B(), hoisted))
                    yielded.Add(x);

                bool hoistedReclaimed = hoisted.TrueForAll(t => t.IsDisposed);
                bool elementsAlive = yielded.Count == 3 && yielded.All(y => !y.IsDisposed);
                bool valueOk = elementsAlive &&
                               Math.Abs(yielded[0].GetDouble(0) - 11.0) < 1e-9 &&
                               Math.Abs(yielded[2].GetDouble(0) - 33.0) < 1e-9;

                // early break — the enumerator's Dispose reclaims the suspended invocation scope
                var hoisted2 = new List<NDArray>();
                foreach (var _ in Scoped_Iterator(A(), B(), hoisted2))
                    break;
                bool breakReclaimed = hoisted2.TrueForAll(t => t.IsDisposed);

                ok &= hoistedReclaimed && elementsAlive && valueOk && breakReclaimed;
                detail = $"elements owned {ReclamationProbe.Mark(elementsAlive)}   state reclaimed at end " +
                         $"{ReclamationProbe.Mark(hoistedReclaimed)}   value {ReclamationProbe.Mark(valueOk)}   " +
                         $"early-break reclaims {ReclamationProbe.Mark(breakReclaimed)}";
                if (stress)
                {
                    foreach (var y in yielded) y.Dispose();
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
            }

            return ReclamationProbe.Verdict("Scoped_Iterator", ok, detail);
        }
    }
}
