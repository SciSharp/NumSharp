using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>array_richcompare</c> for <c>==</c> / <c>!=</c> (arrayobject.c): the equal / not_equal ufunc, and
        ///     where that ufunc has no loop for a string operand, <c>_failed_comparison_workaround</c>'s constant instead of
        ///     its error — NumPy's operator answers False everywhere for <c>==</c> and True for <c>!=</c> (a text array against
        ///     numbers, or two string classes that do not compare), in the broadcast shape.
        /// </summary>
        /// <param name="lhs">The left operand (not null; an uninitialized array has been answered by the caller).</param>
        /// <param name="rhs">The right operand (not null).</param>
        /// <param name="notEqual">True for <c>!=</c>, false for <c>==</c>.</param>
        /// <returns>The bool comparison as a typed array: the ufunc's result, or the failed-comparison constant.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ValueError text).</exception>
        /// <remarks>
        ///     Moved from the <c>==</c> / <c>!=</c> operators by the np-layer dtype plan's P1c, statement for statement: the
        ///     operators keep their null and uninitialized-array answers (C# protocol, no dtype involved) and call this. The
        ///     ufunc is <paramref name="lhs"/>'s engine member (<see cref="TensorEngine.Compare"/> /
        ///     <see cref="TensorEngine.NotEqual"/>), as the operators called it. Two strings of one class compare their text
        ///     there (an empty broadcast result compares nothing).
        /// </remarks>
        internal static NDArray<bool> RichCompareEquality(NDArray lhs, NDArray rhs, bool notEqual)
        {
            // A string operand: two strings of one class compare their text. For any other pair the equal / not_equal
            // ufunc has no loop, and NumPy's operator answers False (==) or True (!=) everywhere instead of raising
            // (_failed_comparison_workaround): a constant of the broadcast shape.
            if (lhs.typecode == NPTypeCode.String || rhs.typecode == NPTypeCode.String)
            {
                var constant = Strings.NDStringUfuncs.FailedComparison(lhs, rhs, notEqual);
                if (constant is not null)
                    return constant;
                // Two strings of one class: the ufunc compares their text — an empty broadcast result compares nothing
                // (the engine's string guard returns that result), any other runs the string comparison.
            }

            return notEqual
                ? lhs.TensorEngine.NotEqual(lhs, rhs).AsGeneric<bool>()
                : lhs.TensorEngine.Compare(lhs, rhs).AsGeneric<bool>();
        }
    }
}
