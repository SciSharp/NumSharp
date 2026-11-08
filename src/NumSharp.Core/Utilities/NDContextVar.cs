using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     A context-local setting with the semantics of NumPy's <c>ContextVar</c>s under the oracle's Python 3.12, as close
    ///     as .NET's <see cref="ExecutionContext"/> allows: the print options and <c>np.setbufsize</c> live in two of these
    ///     (parity QA-40). A value set in a flow is seen by that flow and by the asynchronous work it starts — an
    ///     <c>await</c> continuation, <c>Task.Run</c>, the thread pool — as asyncio copies the caller's context into a task
    ///     and into <c>asyncio.to_thread</c>; a DEDICATED thread (<c>new Thread</c>, a long-running task) starts at the
    ///     defaults, as Python 3.12's <c>threading.Thread</c> starts with an empty context, unless it sets the value itself.
    /// </summary>
    /// <typeparam name="T">The setting's type (a reference type or a value type; the entry boxes it once per set).</typeparam>
    /// <remarks>
    ///     <para><b>How.</b> The value rides an <see cref="AsyncLocal{T}"/> entry that remembers the thread that set it.
    ///     .NET flows the execution context into every thread it starts, dedicated ones included, so the entry alone
    ///     would make a <c>new Thread</c> inherit; a read on a thread that is neither the setter nor a pool thread drops the
    ///     inherited entry for that thread (so the work it starts later inherits nothing either) and answers the defaults.
    ///     Reading costs one <see cref="AsyncLocal{T}"/> lookup; nothing is written on a read unless an inherited entry is
    ///     dropped, so a flow that never sets the value keeps an empty execution context.</para>
    ///     <para><b>Where .NET has no Python twin.</b> The thread pool serves both <c>asyncio.to_thread</c> (which copies the
    ///     context) and a <c>ThreadPoolExecutor</c> (which does not), so every pool work item inherits. A dedicated thread
    ///     that starts a task BEFORE its first read hands that task its parent's value (Python: its own empty context). Code
    ///     that wants a clean slate on the pool queues with <c>ThreadPool.UnsafeQueueUserWorkItem</c> or suppresses the
    ///     flow (<see cref="ExecutionContext.SuppressFlow"/>).</para>
    ///     <para>Python 3.14's free-threaded build inherits the context into new threads by default
    ///     (<c>sys.flags.thread_inherit_context</c>); the oracle's Python 3.12 does not, and this type follows 3.12.</para>
    /// </remarks>
    internal sealed class NDContextVar<T>
    {
        /// <summary>One set value and the managed id of the thread that set it.</summary>
        private sealed class Entry
        {
            /// <summary>The value.</summary>
            internal readonly T Value;

            /// <summary>The managed thread id of the setter: its own reads, and pool threads, see the value.</summary>
            internal readonly int OwnerThread;

            /// <summary>Records a value set on the current thread.</summary>
            /// <param name="value">The value.</param>
            /// <param name="ownerThread">The setter's managed thread id.</param>
            internal Entry(T value, int ownerThread)
            {
                Value = value;
                OwnerThread = ownerThread;
            }
        }

        /// <summary>The flowing entry; null while the flow has no value (the caller's defaults apply).</summary>
        private readonly AsyncLocal<Entry> _entry = new AsyncLocal<Entry>();

        /// <summary>
        ///     The value this thread and flow see, if one was set: false means the setting's default applies (nothing was set
        ///     in this flow, or the entry was inherited by a dedicated thread, which starts at the defaults).
        /// </summary>
        /// <param name="value">The value when the method returns true, else the type's default.</param>
        /// <returns>True when a value applies to the caller.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal bool TryGet(out T value)
        {
            Entry e = _entry.Value;
            if (e is not null)
            {
                if (e.OwnerThread == Environment.CurrentManagedThreadId || Thread.CurrentThread.IsThreadPoolThread)
                {
                    value = e.Value;
                    return true;
                }
                // A dedicated thread that inherited another thread's value: Python 3.12's threading.Thread starts with an
                // empty context. Drop the entry for this thread so the work it starts from here on inherits nothing.
                _entry.Value = null;
            }
            value = default;
            return false;
        }

        /// <summary>
        ///     Sets the value for the current flow: this thread, its later continuations and the asynchronous work it starts
        ///     from here on. Never visible to the caller of an asynchronous method that set it (the execution context's
        ///     copy-on-write), nor to other threads already running.
        /// </summary>
        /// <param name="value">The value.</param>
        internal void Set(T value) => _entry.Value = new Entry(value, Environment.CurrentManagedThreadId);

        /// <summary>Clears the value for the current flow: the setting's default applies again.</summary>
        internal void Reset() => _entry.Value = null;
    }
}
