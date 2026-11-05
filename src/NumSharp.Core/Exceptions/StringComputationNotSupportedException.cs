using System;

namespace NumSharp
{
    /// <summary>
    ///     A computation on text that NumSharp does not run (yet): an <c>np.evaluate</c> (and <c>NDExpr</c> evaluation) node
    ///     that takes an <c>S</c>, <c>U</c> or <c>StringDType</c> operand the current phase of np.evaluate's text support does
    ///     not compute, or a text argument to an <c>NDExpr.Call</c> delegate. <c>np.evaluate</c> is a NumSharp extension
    ///     without a NumPy function to follow; it computes text in its six comparisons, whose bool results feed the rest of the
    ///     tree, and the message names the refused node and how to compute it (the string functions first, their result
    ///     passed into the expression).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Every NumPy call over text has its answer.</b> No <c>np.*</c> function throws this type: where NumPy computes
    ///     on text (the <c>np.strings</c> functions, sorting, searching, the set routines, the
    ///     statistics and reductions over <c>StringDType</c>, the conversions between text and numbers, …) NumSharp computes
    ///     the same result, and where NumPy refuses a string operand (<c>np.subtract(u, u)</c>, <c>np.sum(u)</c>,
    ///     <c>np.dot(u, u)</c>) NumSharp raises NumPy's own exception and text. The string oracle
    ///     (<c>test/oracle/gen_strings_oracle.py</c>) holds every such case to NumPy byte for byte, with no deferral.
    ///     </para>
    ///     <para>
    ///     It is raised when the expression's program is built, before any text is read. It derives from
    ///     <see cref="NotSupportedException"/>, so existing <c>catch (NotSupportedException)</c> code sees it.
    ///     </para>
    /// </remarks>
    public sealed class StringComputationNotSupportedException : NotSupportedException, INumSharpException
    {
        /// <summary>Creates the exception with a message that names the refused node and the remedy.</summary>
        /// <param name="message">The message.</param>
        public StringComputationNotSupportedException(string message) : base(message) { }
    }
}
