using System;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks.Sources;

namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Functional proof of the <see cref="NDScope"/> STATE-MACHINE seam — the members the
    ///     <c>[NDScoped]</c> weaver emits into async/iterator state machines and Task-returning
    ///     methods (<see cref="NDScope.OpenOrResume"/>, <see cref="NDScope.Suspend"/>,
    ///     <see cref="NDScope.DisposeSlot"/>, <see cref="NDScope.ExitIterator"/>,
    ///     <see cref="NDScope.CloseUnlessDeferred"/>, <c>ReturnsTask</c>/<c>ReturnsValueTask</c>).
    ///     Each test hand-drives the EXACT call protocol the woven IL performs — the same trick the
    ///     out-parameter and ITuple branches use — so the runtime semantics are pinned without
    ///     weaving this assembly; the end-to-end woven proof lives in
    ///     <c>tools/verify_build_package.sh</c>'s async consumer step.
    /// </summary>
    [TestClass]
    [DoNotParallelize]
    public class NDScopeAsyncTests
    {
        /// <summary>Runs <paramref name="action"/> on a fresh thread and joins — a REAL cross-thread resumption, not a pool hop that might land back here.</summary>
        private static void OnOtherThread(Action action)
        {
            Exception error = null;
            var thread = new Thread(() =>
            {
                try
                {
                    action();
                }
                catch (Exception e)
                {
                    error = e;
                }
            });
            thread.Start();
            thread.Join();
            if (error != null)
                throw new InvalidOperationException($"cross-thread segment failed: {error.Message}", error);
        }

        private static void PollUntil(Func<bool> condition, string what)
        {
            var deadline = DateTime.UtcNow.AddSeconds(10);
            while (!condition())
            {
                if (DateTime.UtcNow > deadline)
                    Assert.Fail($"timed out waiting for: {what}");
                Thread.Sleep(10);
            }
        }

        // ------------------------------------------------------------------ the async MoveNext protocol

        [TestMethod]
        public void OpenOrResume_SegmentsAcrossThreads_OneInvocationScope_TempsSurviveSuspension()
        {
            NDScope slot = null;

            // segment 1 (the first MoveNext, on this thread)
            var scope = NDScope.OpenOrResume(ref slot);
            Assert.IsNotNull(scope);
            Assert.AreSame(scope, slot, "the slot must hold the invocation scope");
            var seg1Temp = np.arange(4);
            var hoisted = seg1Temp + 1.0;
            NDScope.Suspend(scope);                 // the pre-schedule seam
            Assert.IsFalse(seg1Temp.IsDisposed, "suspension must NOT reclaim — an awaited callee may still use the temps");
            Assert.IsFalse(hoisted.IsDisposed);

            // segment 2 (the resumption, on another thread — the continuation)
            NDArray result = null, seg2Temp = null;
            OnOtherThread(() =>
            {
                var resumed = NDScope.OpenOrResume(ref slot);
                Assert.AreSame(scope, resumed, "resumption must re-install the SAME invocation scope");
                seg2Temp = np.ones(3);
                result = resumed.Returns(hoisted * 2.0);   // the SetResult egress
                NDScope.DisposeSlot(ref slot);             // completion
            });

            Assert.IsNull(slot, "completion must clear the slot");
            Assert.IsTrue(seg1Temp.IsDisposed, "segment-1 temp reclaimed at completion");
            Assert.IsTrue(hoisted.IsDisposed, "the hoisted (non-yielded) temp reclaimed at completion");
            Assert.IsTrue(seg2Temp.IsDisposed, "segment-2 temp reclaimed at completion");
            Assert.IsFalse(result.IsDisposed, "the yielded result survives");
            Assert.AreEqual(4.0, result.GetDouble(1));
            result.Dispose();
        }

        [TestMethod]
        public void Suspend_RestoresTheOuterChain_LaterConstructionsTrackOutward()
        {
            using var outer = NDScope.Open();
            NDScope slot = null;
            var invocation = NDScope.OpenOrResume(ref slot);
            var invocationTemp = np.arange(3);
            NDScope.Suspend(invocation);

            var afterSuspend = np.arange(2);
            Assert.AreSame(outer, afterSuspend.TrackingScope,
                "after suspension the thread's ambient scope must be the OUTER one again");
            Assert.AreSame(invocation, invocationTemp.TrackingScope, "suspension must not retrack the invocation's temps");
            Assert.IsFalse(invocationTemp.IsDisposed);

            NDScope.DisposeSlot(ref slot);
            Assert.IsTrue(invocationTemp.IsDisposed);
            Assert.IsFalse(afterSuspend.IsDisposed, "the outer scope's array is untouched by the invocation's completion");
        }

        [TestMethod]
        public void Resume_UnderAmbientScope_YieldReparentsTheResultIntoIt()
        {
            // The synchronous-completion nesting shape: an async callee resumed (or completed
            // inline) under a caller's open scope hands its result to that scope — dropping the
            // result still reclaims it eagerly, exactly like sync Returns re-parenting.
            NDScope slot = null;
            var invocation = NDScope.OpenOrResume(ref slot);
            var temp = np.arange(2);
            NDScope.Suspend(invocation);

            NDArray result;
            using (var ambient = NDScope.Open())
            {
                var resumed = NDScope.OpenOrResume(ref slot);
                Assert.AreSame(invocation, resumed);
                result = resumed.Returns(np.ones(2) + 1.0);
                NDScope.DisposeSlot(ref slot);

                Assert.IsTrue(temp.IsDisposed);
                Assert.IsFalse(result.IsDisposed, "the result must survive the invocation's completion");
                Assert.AreSame(ambient, result.TrackingScope, "the yielded result re-parents into the ambient scope");
            }

            Assert.IsTrue(result.IsDisposed, "the ambient scope reclaims the dropped inner result");
        }

        [TestMethod]
        public void DisposeSlot_IsIdempotent_AndNoOpOnEmptySlot()
        {
            NDScope slot = null;
            NDScope.DisposeSlot(ref slot);   // empty — the finally after a pre-completion disposal

            var scope = NDScope.OpenOrResume(ref slot);
            var temp = np.arange(2);
            NDScope.DisposeSlot(ref slot);
            Assert.IsTrue(temp.IsDisposed);
            Assert.IsNull(slot);
            NDScope.DisposeSlot(ref slot);   // the woven finally re-runs it — must be a no-op
            _ = scope;
        }

        [TestMethod]
        public void OpenOrResume_DisposedResidueInSlot_OpensFresh()
        {
            var stale = NDScope.Open();
            stale.Dispose();
            NDScope slot = stale;

            var fresh = NDScope.OpenOrResume(ref slot);
            var temp = np.arange(2);
            Assert.AreSame(fresh, slot);
            Assert.AreSame(fresh, temp.TrackingScope, "a disposed residue must be replaced by a live scope");
            NDScope.DisposeSlot(ref slot);
            Assert.IsTrue(temp.IsDisposed);
        }

        // ------------------------------------------------------------------ the iterator protocol

        [TestMethod]
        public void ExitIterator_HasMore_Suspends_Final_DisposesAndClears()
        {
            NDScope slot = null;
            var invocation = NDScope.OpenOrResume(ref slot);
            var hoisted = np.arange(3);
            var yielded = invocation.Returns(hoisted + 10.0);   // the <>2__current egress
            NDScope.ExitIterator(ref slot, hasMore: true);      // MoveNext returned true

            Assert.IsNotNull(slot, "a live iterator keeps its scope");
            Assert.IsFalse(hoisted.IsDisposed, "hoisted state survives between MoveNext calls");
            Assert.IsFalse(yielded.IsDisposed, "the consumer owns the yielded element");

            var resumed = NDScope.OpenOrResume(ref slot);
            Assert.AreSame(invocation, resumed);
            NDScope.ExitIterator(ref slot, hasMore: false);     // iteration finished

            Assert.IsNull(slot);
            Assert.IsTrue(hoisted.IsDisposed, "iterator state reclaimed at the end of iteration");
            Assert.IsFalse(yielded.IsDisposed);
            yielded.Dispose();
        }

        [TestMethod]
        public void DisposeSlot_FromAnotherThread_TheAbandonedEnumeratorSeam()
        {
            // foreach { break; } disposes the enumerator — possibly on a different thread than the
            // last MoveNext ran on. The calls are sequenced (the consumer owns the enumerator), so
            // DisposeSlot re-stamps the owning thread rather than asserting it.
            NDScope slot = null;
            var invocation = NDScope.OpenOrResume(ref slot);
            var hoisted = np.arange(4);
            var yielded = invocation.Returns(hoisted * 2.0);
            NDScope.ExitIterator(ref slot, hasMore: true);

            var slotBox = new NDScope[] { slot };
            OnOtherThread(() =>
            {
                NDScope.DisposeSlot(ref slotBox[0]);
            });

            Assert.IsNull(slotBox[0]);
            Assert.IsTrue(hoisted.IsDisposed, "abandonment reclaims the suspended iterator's state");
            Assert.IsFalse(yielded.IsDisposed, "already-yielded elements belong to the consumer");
            yielded.Dispose();
        }

        // ------------------------------------------------------------------ task-shaped returns (non-async methods)

        [TestMethod]
        public void ReturnsTask_Completed_YieldsResultNow_FinallyDisposesEagerly()
        {
            var scope = NDScope.Open();
            var temp = np.arange(3);
            var result = temp + 1.0;
            var task = scope.ReturnsTask(Task.FromResult(result));
            NDScope.CloseUnlessDeferred(scope);   // the woven finally

            Assert.IsTrue(temp.IsDisposed, "non-result temps reclaimed at return, like any sync method");
            Assert.IsFalse(result.IsDisposed, "a synchronously-completed task's result is yielded immediately");
            Assert.AreEqual(3.0, task.Result.GetDouble(2));
            result.Dispose();
        }

        [TestMethod]
        public void ReturnsTask_Incomplete_DefersDisposal_InFlightOperandSurvives()
        {
            var tcs = new TaskCompletionSource<NDArray>();
            var scope = NDScope.Open();
            var operand = np.arange(4) + 1.0;     // handed to the in-flight work
            var task = scope.ReturnsTask(tcs.Task);
            NDScope.CloseUnlessDeferred(scope);   // deferred — must NOT dispose

            Assert.IsFalse(operand.IsDisposed, "an in-flight callee may still be using the tracked operand");
            Assert.IsNull(NDScope.Current, "a deferred scope must be OFF the thread chain after the method returns");

            var result = np.ones(2);              // what the async work eventually produces (untracked)
            tcs.SetResult(result);

            PollUntil(() => operand.IsDisposed, "deferred disposal at task completion");
            Assert.IsFalse(result.IsDisposed);
            Assert.AreSame(result, task.Result);
            result.Dispose();
        }

        [TestMethod]
        public void ReturnsTask_Incomplete_TrackedResult_IsYieldedAtCompletion()
        {
            // The forwarding shape where the eventual result was ALREADY constructed under the
            // scope: the completion continuation must yield it before sweeping.
            var tcs = new TaskCompletionSource<NDArray>();
            var scope = NDScope.Open();
            var trackedResult = np.arange(5) * 2.0;
            var otherTemp = np.zeros(3);
            _ = scope.ReturnsTask(tcs.Task);
            NDScope.CloseUnlessDeferred(scope);

            tcs.SetResult(trackedResult);
            PollUntil(() => otherTemp.IsDisposed, "deferred sweep at completion");
            Assert.IsFalse(trackedResult.IsDisposed, "the completed task's result must be yielded before the sweep");
            trackedResult.Dispose();
        }

        [TestMethod]
        public void ReturnsTask_Faulted_DisposesAtCompletion_NothingYielded()
        {
            var tcs = new TaskCompletionSource<NDArray>();
            var scope = NDScope.Open();
            var temp = np.arange(3);
            var task = scope.ReturnsTask(tcs.Task);
            NDScope.CloseUnlessDeferred(scope);

            Assert.IsFalse(temp.IsDisposed);
            tcs.SetException(new InvalidOperationException("boom"));
            PollUntil(() => temp.IsDisposed, "deferred disposal at task fault");
            Assert.IsNotNull(task.Exception);
        }

        [TestMethod]
        public void ReturnsTask_Plain_Incomplete_DefersTheSweep()
        {
            var tcs = new TaskCompletionSource();
            var scope = NDScope.Open();
            var operand = np.arange(6);
            _ = scope.ReturnsTask((Task)tcs.Task);
            NDScope.CloseUnlessDeferred(scope);

            Assert.IsFalse(operand.IsDisposed);
            tcs.SetResult();
            PollUntil(() => operand.IsDisposed, "plain-Task deferred disposal");
        }

        [TestMethod]
        public void ReturnsTask_Plain_CarryingAnUpcastResult_SniffsAndYieldsIt()
        {
            // Task<T> : Task is an implicit conversion, so `[NDScoped] Task M() => ComputeAsync();`
            // compiles — the caller can cast the task back and read Result, which must therefore
            // never be handed back disposed. Completed path:
            var scope = NDScope.Open();
            var temp = np.arange(3);
            var result = temp + 1.0;
            var task = scope.ReturnsTask((Task)Task.FromResult(result));
            NDScope.CloseUnlessDeferred(scope);

            Assert.IsTrue(temp.IsDisposed, "the non-result temp is still reclaimed eagerly");
            Assert.IsFalse(result.IsDisposed, "the up-cast task's result must be sniffed and yielded");
            Assert.AreSame(result, ((Task<NDArray>)task).Result);
            result.Dispose();
        }

        [TestMethod]
        public void ReturnsTask_Plain_UpcastSniff_CoversGenericSubtypesAndTuples()
        {
            // The sniff recovers the result REFLECTIVELY, so it must also catch shapes no pattern
            // match could: Task<NDArray<bool>> (Task<T> is invariant — `task is Task<NDArray>` is
            // FALSE for it) and a boxed ValueTuple of arrays.
            var scope = NDScope.Open();
            NumSharp.Generic.NDArray<bool> typed = np.arange(3) > 0;
            _ = scope.ReturnsTask((Task)Task.FromResult(typed));
            NDScope.CloseUnlessDeferred(scope);
            Assert.IsFalse(typed.IsDisposed, "a generic-subtype result must survive the sniff path");
            typed.Dispose();

            var scope2 = NDScope.Open();
            var a = np.arange(2);
            var b = np.arange(3);
            _ = scope2.ReturnsTask((Task)Task.FromResult((a, b)));
            NDScope.CloseUnlessDeferred(scope2);
            Assert.IsFalse(a.IsDisposed, "tuple component must survive the boxed-ITuple sniff");
            Assert.IsFalse(b.IsDisposed);
            a.Dispose();
            b.Dispose();
        }

        [TestMethod]
        public void ReturnsTask_TupleResult_NestedAndArrayComponents_AllYielded()
        {
            // The task egress funnels through YieldBoxed → Returns(ITuple), whose recursive component
            // dispatch must see through a NESTED tuple and an NDArray[] component — a completed
            // Task<((a, b), c)> or Task<(NDArray[], n)> must never hand back disposed arrays.
            var scope = NDScope.Open();
            var a = np.arange(2);
            var b = np.arange(3);
            var c = np.arange(4);
            _ = scope.ReturnsTask(Task.FromResult(((a, b), c)));
            NDScope.CloseUnlessDeferred(scope);
            Assert.IsFalse(a.IsDisposed, "nested tuple component must survive the task egress");
            Assert.IsFalse(b.IsDisposed);
            Assert.IsFalse(c.IsDisposed);
            a.Dispose(); b.Dispose(); c.Dispose();

            var scope2 = NDScope.Open();
            var x = np.arange(2);
            var y = np.arange(3);
            _ = scope2.ReturnsTask(Task.FromResult((new[] { x, y }, 7)));
            NDScope.CloseUnlessDeferred(scope2);
            Assert.IsFalse(x.IsDisposed, "NDArray[] component's elements must survive the task egress");
            Assert.IsFalse(y.IsDisposed);
            x.Dispose(); y.Dispose();
        }

        [TestMethod]
        public void ReturnsTask_Plain_UpcastSniff_Deferred_YieldsAtCompletion()
        {
            var tcs = new TaskCompletionSource<NDArray>();
            var scope = NDScope.Open();
            var trackedResult = np.arange(4) * 2.0;
            var otherTemp = np.zeros(2);
            _ = scope.ReturnsTask((Task)tcs.Task);
            NDScope.CloseUnlessDeferred(scope);

            tcs.SetResult(trackedResult);
            PollUntil(() => otherTemp.IsDisposed, "deferred sweep at upcast completion");
            Assert.IsFalse(trackedResult.IsDisposed, "the deferred up-cast result must be yielded before the sweep");
            trackedResult.Dispose();
        }

        [TestMethod]
        public void SuspendedInvocations_InterleavedOnOneThread_StayIsolated()
        {
            // Two state machines' segments alternating on the SAME thread — the chain must
            // push/pop each invocation's scope independently.
            NDScope slotA = null, slotB = null;
            var a1 = NDScope.OpenOrResume(ref slotA);
            var tempA = np.arange(3);
            NDScope.Suspend(a1);

            var b1 = NDScope.OpenOrResume(ref slotB);
            Assert.AreNotSame(a1, b1);
            var tempB = np.arange(4);
            NDScope.Suspend(b1);

            var a2 = NDScope.OpenOrResume(ref slotA);
            Assert.AreSame(a1, a2);
            var resultA = a2.Returns(tempA + 1.0);
            NDScope.DisposeSlot(ref slotA);

            var b2 = NDScope.OpenOrResume(ref slotB);
            Assert.AreSame(b1, b2);
            var resultB = b2.Returns(tempB + 1.0);
            NDScope.DisposeSlot(ref slotB);

            Assert.IsTrue(tempA.IsDisposed, "invocation A's temp reclaimed at A's completion");
            Assert.IsTrue(tempB.IsDisposed, "invocation B's temp reclaimed at B's completion");
            Assert.IsFalse(resultA.IsDisposed);
            Assert.IsFalse(resultB.IsDisposed);
            Assert.AreEqual(1.0, resultA.GetDouble(0));
            Assert.AreEqual(1.0, resultB.GetDouble(0));
            resultA.Dispose();
            resultB.Dispose();
        }

        [TestMethod]
        public void GcPressure_WhileSuspended_TrackedArraysSurvive()
        {
            NDScope slot = null;
            var scope = NDScope.OpenOrResume(ref slot);
            var hoisted = np.arange(5) * 3.0;
            NDScope.Suspend(scope);

            for (int i = 0; i < 3; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }

            Assert.IsFalse(hoisted.IsDisposed, "a suspended scope strongly roots its tracked arrays");
            Assert.AreEqual(6.0, hoisted.GetDouble(2), "buffer must be intact after full GCs");

            var resumed = NDScope.OpenOrResume(ref slot);
            var result = resumed.Returns(hoisted + 1.0);
            NDScope.DisposeSlot(ref slot);
            Assert.IsTrue(hoisted.IsDisposed);
            Assert.IsFalse(result.IsDisposed);
            result.Dispose();
        }

        [TestMethod]
        public void ReturnsValueTask_PlainValue_YieldsNow()
        {
            var scope = NDScope.Open();
            var temp = np.arange(3);
            var result = temp * 3.0;
            var vt = scope.ReturnsValueTask(new ValueTask<NDArray>(result));
            NDScope.CloseUnlessDeferred(scope);

            Assert.IsTrue(temp.IsDisposed);
            Assert.IsFalse(result.IsDisposed);
            Assert.AreEqual(6.0, vt.Result.GetDouble(2));
            result.Dispose();
        }

        [TestMethod]
        public void ReturnsValueTask_TaskBacked_Incomplete_Defers_AndTheReturnedTaskIsAwaitable()
        {
            var tcs = new TaskCompletionSource<NDArray>();
            var scope = NDScope.Open();
            var operand = np.arange(4);
            var vt = scope.ReturnsValueTask(new ValueTask<NDArray>(tcs.Task));
            NDScope.CloseUnlessDeferred(scope);

            Assert.IsFalse(operand.IsDisposed);
            var result = np.ones(2) * 9.0;
            tcs.SetResult(result);

            PollUntil(() => operand.IsDisposed, "ValueTask deferred disposal");
            Assert.AreEqual(9.0, vt.AsTask().GetAwaiter().GetResult().GetDouble(0));
            Assert.IsFalse(result.IsDisposed);
            result.Dispose();
        }

        [TestMethod]
        public void ReturnsValueTask_SourceBacked_PreserveKeepsTheCallerConsumptionLegal()
        {
            // A ValueTask over an IValueTaskSource is single-consumption: the seam PRESERVES it
            // (consuming the original once) and hands the caller the multi-observable form — so the
            // scope's own completion observation cannot corrupt the caller's await.
            var source = new SingleUseSource();
            var scope = NDScope.Open();
            var operand = np.arange(3);
            var preserved = scope.ReturnsValueTask(new ValueTask<NDArray>(source, source.Token));
            NDScope.CloseUnlessDeferred(scope);

            Assert.IsFalse(operand.IsDisposed);
            var result = np.ones(2) * 7.0;
            source.SetResult(result);

            PollUntil(() => operand.IsDisposed, "source-backed ValueTask deferred disposal");
            Assert.AreEqual(7.0, preserved.AsTask().GetAwaiter().GetResult().GetDouble(1));
            Assert.IsFalse(result.IsDisposed);
            result.Dispose();
        }

        private sealed class SingleUseSource : IValueTaskSource<NDArray>
        {
            private ManualResetValueTaskSourceCore<NDArray> _core;

            public short Token => _core.Version;
            public void SetResult(NDArray result) => _core.SetResult(result);
            public NDArray GetResult(short token) => _core.GetResult(token);
            public ValueTaskSourceStatus GetStatus(short token) => _core.GetStatus(token);
            public void OnCompleted(Action<object> continuation, object state, short token, ValueTaskSourceOnCompletedFlags flags)
                => _core.OnCompleted(continuation, state, token, flags);
        }
    }
}
