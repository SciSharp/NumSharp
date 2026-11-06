using System.Collections.Generic;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     The name → array dictionary that <c>session.Run(dict)</c> and <c>session.RunAsync</c> return: an ordinary
    ///     <see cref="Dictionary{TKey, TValue}"/> that can also hand its arrays out of an <see cref="NDScope"/>.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why it exists.</b> A scope tracks every array constructed on its thread while it is open, so the
    ///     outputs a run builds inside the caller's scope are the scope's: <c>RunAsync</c> even allocates them when the
    ///     call STARTS. A scope hands a task's result to its caller (<see cref="NDScope.ReturnsTask{TResult}"/>) by the
    ///     result's runtime type, and it knows an <see cref="NDArray"/>, an array of them, a tuple and an
    ///     <see cref="INDArrayCarrier"/> - not a dictionary. With a plain dictionary,
    ///     <c>scope.ReturnsTask(session.RunAsync(...))</c> disposed every output when the scope closed at the task's
    ///     completion, and once the run let go of its own reference the caller read freed memory (F84 of the 0.80.0-beta1
    ///     QA). As a carrier, the dictionary yields each output to the scope's parent instead.</para>
    ///     <para><b>Nothing else changes.</b> The declared return types stay
    ///     <see cref="IReadOnlyDictionary{TKey, TValue}"/>; the key comparer is the default one, as before; a caller
    ///     outside any scope never sees the difference.</para>
    /// </remarks>
    internal sealed class RunOutputs : Dictionary<string, NDArray>, INDArrayCarrier
    {
        /// <summary>An empty dictionary with room for the outputs a run fetches.</summary>
        /// <param name="capacity">The number of outputs the run fetches (a hint; the dictionary grows past it).</param>
        internal RunOutputs(int capacity) : base(capacity)
        {
        }

        /// <summary>
        ///     Hands every output to the caller of <paramref name="scope"/>: an output the scope tracks is re-tracked into
        ///     its parent scope (NDScope's rule for a yielded result), one it does not track passes through untouched.
        /// </summary>
        /// <remarks>
        ///     Called by the scope itself when the dictionary is a task's result (<c>ReturnsTask</c>) or the return value of
        ///     a woven <c>[NDScoped]</c> member - never concurrently with the run that fills it, because a scope yields a
        ///     result only once the task that produced it has completed.
        /// </remarks>
        /// <param name="scope">The scope the outputs leave.</param>
        public void YieldTo(NDScope scope)
        {
            foreach (NDArray output in Values)
                scope.Returns(output);   // null-safe and a no-op for an array the scope does not track
        }
    }
}
