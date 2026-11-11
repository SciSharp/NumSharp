using System;

namespace NumSharp
{
    /// <summary>
    ///     A computation on text that <c>np.evaluate</c> does not run: an <c>NDExpr</c> node that takes an <c>S</c>, <c>U</c> or
    ///     <c>StringDType</c> operand where NumPy's answer is not a number or a bool (a text result such as a concatenation or
    ///     a <c>Where</c> with text branches, a reduction of text other than the truth tests, an arithmetic or math node), or a
    ///     text argument to an <c>NDExpr.Call</c> delegate. <c>np.evaluate</c> is a NumSharp extension without a NumPy function
    ///     to follow; it computes text in the comparisons, the truth readers (the logical nodes, <c>Where</c>'s condition,
    ///     <c>Any</c> / <c>All</c> / <c>CountNonzero</c>), <c>IsNaN</c>, <c>Cast</c> to a number and the numpy.strings factories
    ///     (<c>StrLen</c>, the predicates, the searches), whose numeric results feed the rest of the tree, and the message names
    ///     the refused node and how to compute it (the string functions first, their result passed into the expression).
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

        /// <summary>
        ///     Creates the exception with a message and the exception that caused it, .NET's standard constructor every
        ///     NumSharp exception has.
        /// </summary>
        /// <param name="message">The message, kept verbatim (it should name the refused node and the remedy).</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public StringComputationNotSupportedException(string message, Exception innerException) : base(message, innerException) { }
    }
}
