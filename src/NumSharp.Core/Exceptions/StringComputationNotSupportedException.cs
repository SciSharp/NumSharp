using System;

namespace NumSharp
{
    /// <summary>
    ///     A computation on text that NumSharp deliberately does not run: <c>np.evaluate</c> (and <c>NDExpr</c> evaluation)
    ///     with an <c>S</c>, <c>U</c> or <c>StringDType</c> operand. <c>np.evaluate</c> is a NumSharp extension without a
    ///     NumPy function to follow; its fused kernels compute on numbers, so text is computed with the string functions
    ///     (<c>np.strings</c>, the comparisons, the truth tests) and their numeric or bool results feed the expression
    ///     (decision D20 of <c>docs/plans/strings-phase2.md</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Every NumPy call over text has its answer.</b> Since strings phase 2 (P2.7) no <c>np.*</c> function throws this
    ///     type: where NumPy computes on text (the <c>np.strings</c> functions, sorting, searching, the set routines, the
    ///     statistics and reductions over <c>StringDType</c>, the conversions between text and numbers, …) NumSharp computes
    ///     the same result, and where NumPy refuses a string operand (<c>np.subtract(u, u)</c>, <c>np.sum(u)</c>,
    ///     <c>np.dot(u, u)</c>) NumSharp raises NumPy's own exception and text. The string oracle
    ///     (<c>test/oracle/gen_strings_oracle.py</c>) holds every such case to NumPy byte for byte, with no deferral.
    ///     </para>
    ///     <para>
    ///     It derives from <see cref="NotSupportedException"/>, so existing <c>catch (NotSupportedException)</c> code sees it.
    ///     </para>
    /// </remarks>
    public sealed class StringComputationNotSupportedException : NotSupportedException, INumSharpException
    {
        /// <summary>Creates the exception with a message that names the operand's string dtype and says why it is refused.</summary>
        /// <param name="message">The message.</param>
        public StringComputationNotSupportedException(string message) : base(message) { }
    }
}
