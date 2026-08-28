using System.Collections.Generic;
using System.Threading.Tasks;
using NumSharp;

namespace NDScoping.CounterExamples
{
    /// <summary>
    ///     Every method here is a target the weaver CANNOT weave, so the bundled analyzer reports a
    ///     build ERROR at compile time (each expected code is in the comment). This file — and this
    ///     project — MUST NOT COMPILE; that is the whole demonstration. See README.md and run
    ///     show-analyzer.sh.
    /// </summary>
    internal static class BadShapes
    {
        // NDW002 — a 'ref NDArray' parameter is a hidden egress the weaver cannot see.
        [NDScoped]
        internal static void RefEgress(ref NDArray a)
        {
        }

        // NDW003 — List<NDArray> is an unsupported carrier: the weaver cannot see every NDArray through it.
        [NDScoped]
        internal static List<NDArray> BadCarrier(NDArray a) => new() { a };

        // NDW005 — an extern method has no body to weave.
        [NDScoped]
        internal static extern NDArray NoBody(NDArray a);

        // NDW006 — the attribute is on a setter-only property (nothing to weave; put it on the getter).
        [NDScoped]
        internal static NDArray SetterOnly
        {
            set { }
        }

        // NDW009 — an async / Task-returning method belongs on [NDScopedAsync], not [NDScoped].
        [NDScoped]
        internal static async Task<NDArray> AsyncUnderSync(NDArray a)
        {
            await Task.Yield();
            return a;
        }

        // NDW010 — a plain synchronous method belongs on [NDScoped], not [NDScopedAsync].
        [NDScopedAsync]
        internal static NDArray SyncUnderAsync(NDArray a) => a;

        // NDW011 — a method has exactly one scoping model; it cannot carry BOTH attributes.
        [NDScoped]
        [NDScopedAsync]
        internal static NDArray HasBoth(NDArray a) => a;
    }
}
