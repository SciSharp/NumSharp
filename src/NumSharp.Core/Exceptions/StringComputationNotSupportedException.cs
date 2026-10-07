using System;

namespace NumSharp
{
    /// <summary>
    ///     A call that NumPy answers by COMPUTING on fixed-width string data (<c>S</c> / <c>U</c>), which NumSharp's string
    ///     support does not do yet. The string dtypes currently cover storage, element access, conversion to and from C#,
    ///     copying, casts between string dtypes, indexing and printing; text computation — comparisons, concatenation,
    ///     truth tests, sorting and searching, and conversion between text and numbers — is the next phase.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>When it is thrown.</b> Only where NumPy itself returns a result: <c>np.equal(u, u)</c>, <c>np.add(u, u)</c>,
    ///     <c>np.sort(u)</c>, <c>np.nonzero(u)</c>, <c>u.astype(np.int64)</c>, <c>np.array(new object[] { "a", 1 })</c> and
    ///     similar. Where NumPy raises an error for a string operand (<c>np.subtract(u, u)</c>, <c>np.sum(u)</c>,
    ///     <c>np.dot(u, u)</c>), NumSharp raises NumPy's own exception and text instead, because that answer is final.
    ///     </para>
    ///     <para>
    ///     <b>For the computational phase.</b> Every throw site is a place that phase implements: finding all references to
    ///     this type lists them. The string oracle (<c>test/oracle/gen_strings_oracle.py</c>) already holds NumPy's results
    ///     for these calls and counts them as deferred while this exception is thrown; once a site computes, the oracle
    ///     compares its result with NumPy's byte for byte.
    ///     </para>
    ///     <para>
    ///     It derives from <see cref="NotSupportedException"/>, so existing <c>catch (NotSupportedException)</c> code sees it.
    ///     </para>
    /// </remarks>
    public sealed class StringComputationNotSupportedException : NotSupportedException, INumSharpException
    {
        /// <summary>Creates the exception with a message that names the call and the string dtype involved.</summary>
        /// <param name="message">The message.</param>
        public StringComputationNotSupportedException(string message) : base(message) { }
    }
}
