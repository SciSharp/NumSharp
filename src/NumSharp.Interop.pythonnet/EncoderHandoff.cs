using System;
using System.Threading;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     Bounds the lifetime of the wrapper an <see cref="IPyObjectEncoder"/> hands to pythonnet.
    ///
    ///     <para>pythonnet's conversion pipeline takes its OWN reference from the <see cref="PyObject"/> an
    ///     encoder returns (<c>new NewReference(encoded)</c>) and never disposes the wrapper itself, so the
    ///     wrapper's reference lingers until the CLR finalizer runs and pythonnet's deferred-decref flush
    ///     gets to it. For an exported <see cref="NDArray"/> that reference IS the export pin: in a loop of
    ///     implicit encodes (<c>py.win = window</c>, an <see cref="NDArray"/> argument to a <c>dynamic</c>
    ///     call) every previous view stayed pinned until a garbage collection — 150 iterations, 150 live
    ///     exports, where the explicit <c>using PyObject p = nd.ToNumpy()</c> spelling holds one.</para>
    ///
    ///     <para>The fix: one slot per thread. An encode records the wrapper it returns and disposes the
    ///     wrapper the PREVIOUS encode on this thread returned — safe, because pythonnet took its reference
    ///     synchronously right after that earlier return, so by the time any later encode runs the wrapper's
    ///     reference is redundant. Encodes run under the GIL (pythonnet converts under it), so the disposal
    ///     is a legal decref; the slot is thread-static, so no encode ever disposes a wrapper another thread
    ///     just returned. The leak is thereby bounded to one wrapper per thread, released at the next
    ///     encode, the finalizer remaining the backstop for the last one.</para>
    ///
    ///     <para>A wrapper from a PREVIOUS engine session is never disposed (its pointer belongs to an
    ///     interpreter that no longer exists): the slot remembers the session it was filled in, and the
    ///     shutdown handler advances the session counter, so a stale slot is simply forgotten and left to
    ///     pythonnet's finalizer, which ignores wrappers of a finished run.</para>
    /// </summary>
    internal static class EncoderHandoff
    {
        [ThreadStatic] private static PyObject t_previous;
        [ThreadStatic] private static int t_previousSession;
        private static int s_session;

        /// <summary>
        ///     Records <paramref name="fresh"/> as this thread's outstanding handoff and disposes the wrapper the
        ///     previous encode on this thread handed over. Call under the GIL, immediately before returning
        ///     from <see cref="IPyObjectEncoder.TryEncode"/>. A <c>null</c> passes through untouched (the
        ///     encoder declined; nothing was handed over).
        /// </summary>
        internal static PyObject Hand(PyObject fresh)
        {
            if (fresh is null)
                return null;

            PyObject previous = t_previous;
            int previousSession = t_previousSession;
            int session = Volatile.Read(ref s_session);
            t_previous = fresh;
            t_previousSession = session;

            if (previous is not null && previousSession == session && PythonEngine.IsInitialized)
            {
                try { previous.Dispose(); }
                catch (Exception) { /* pythonnet already invalidated it: the finalizer backstop owns it */ }
            }

            return fresh;
        }

        /// <summary>
        ///     Disposes this thread's outstanding handoff, if any — the interop's inline housekeeping, run at the
        ///     start of every conversion verb (<see cref="PythonRuntimeInterop.DrainPending"/>), so the LAST
        ///     encode's wrapper does not stay pinned until the next encode on this thread. Takes the GIL
        ///     itself (re-entrantly) for the decref; a wrapper from a finished engine session is dropped
        ///     without a decref.
        /// </summary>
        internal static void Flush()
        {
            PyObject previous = t_previous;
            if (previous is null)
                return;

            int previousSession = t_previousSession;
            t_previous = null;
            if (previousSession != Volatile.Read(ref s_session) || !PythonEngine.IsInitialized)
                return;

            try
            {
                using (Py.GIL())
                    previous.Dispose();
            }
            catch (Exception) { /* pythonnet already invalidated it: the finalizer backstop owns it */ }
        }

        /// <summary>
        ///     The engine session ended (called from the interop's shutdown handler): every slot filled so far
        ///     holds a wrapper of the dying interpreter and must never be disposed by hand from now on.
        /// </summary>
        internal static void SessionEnded() => Interlocked.Increment(ref s_session);
    }
}
