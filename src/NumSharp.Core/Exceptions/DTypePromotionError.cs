using System;

namespace NumSharp
{
    /// <summary>
    ///     <c>numpy.exceptions.DTypePromotionError</c> — raised when two (or more) DTypes have no common DType,
    ///     i.e. <c>np.promote_types</c> / <c>np.result_type</c> cannot find a dtype that can hold every input
    ///     (NEP 42's <c>__common_dtype__</c> protocol returned <c>NotImplemented</c> in both directions).
    /// </summary>
    /// <remarks>
    ///     Derives from <see cref="TypeError"/> exactly as NumPy's does (<c>DTypePromotionError(TypeError)</c>), so a
    ///     <c>catch (TypeError)</c> written for NumPy 1.x code keeps working. The two message shapes are NumPy's
    ///     verbatim texts from <c>common_dtype.c</c>:
    ///     <list type="bullet">
    ///       <item><description><c>The DTypes &lt;class 'numpy.dtypes.DateTime64DType'&gt; and &lt;class 'numpy.dtypes.Float64DType'&gt;
    ///       do not have a common DType. For example they cannot be stored in a single array unless the dtype is `object`.</c>
    ///       (the pairwise <c>PyArray_CommonDType</c>)</description></item>
    ///       <item><description><c>The DType %S could not be promoted by %S. This means that no common DType exists for the
    ///       given inputs. For example they cannot be stored in a single array unless the dtype is `object`. The full list
    ///       of DTypes is: (%S, …)</c> (the sequence reduction <c>PyArray_PromoteDTypeSequence</c>)</description></item>
    ///     </list>
    /// </remarks>
    public class DTypePromotionError : TypeError
    {
        /// <summary>
        ///     Creates the error with Python's text for a bare <c>raise DTypePromotionError</c>: an empty
        ///     <see cref="Exception.Message"/>, as <c>str(np.exceptions.DTypePromotionError())</c> is <c>''</c>.
        /// </summary>
        /// <remarks>Prefer one of NumPy's two texts listed on the class: they name the DTypes that found no common
        ///     DType.</remarks>
        public DTypePromotionError() : base(string.Empty) { }

        /// <summary>Creates the error with NumPy's message.</summary>
        /// <param name="message">The text NumPy raises, verbatim (<c>The DTypes ... do not have a common DType. ...</c>).</param>
        public DTypePromotionError(string message) : base(message) { }

        /// <summary>
        ///     Creates the error with NumPy's message and the exception that caused it: Python's <c>raise
        ///     DTypePromotionError(message) from innerException</c>, the cause kept in
        ///     <see cref="Exception.InnerException"/>.
        /// </summary>
        /// <param name="message">The text NumPy raises, verbatim.</param>
        /// <param name="innerException">The exception that led to this one, or null for none.</param>
        public DTypePromotionError(string message, Exception innerException) : base(message, innerException) { }
    }
}
