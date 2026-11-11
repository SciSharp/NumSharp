using System;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Bugs the API-parity batches found on the way (docs/plans/API_PARITY_DIFF.md, 2026-11-11): each is pinned here as
    ///     NumPy 2.4.2's behaviour and left for its own fix, per the owner's rule that a bug found while fixing something
    ///     else gets a reproduction and is deferred.
    /// </summary>
    public partial class OpenBugs
    {
        /// <summary>
        ///     np.concatenate's argument refusals - and np.stack's, which concatenates - raise .NET types with NumPy's texts
        ///     where NumPy raises <c>TypeError</c>: a casting refusal ("Cannot cast array data from dtype('float64') to
        ///     dtype('int32') according to the rule 'safe'") is <see cref="InvalidCastException"/>, and out with dtype
        ///     ("concatenate() only takes `out` or `dtype` as an argument, but both were provided.") is
        ///     <see cref="ArgumentException"/>. The site (<c>NDArray.concatenate</c> in Creation/NDArray.Multiarray.cs) still
        ///     follows the house convention from before NumSharp had Python's exception classes ("NumPy raises TypeError; we
        ///     raise InvalidCastException"), which the Python-exceptions work switched elsewhere. Found by API-parity batch
        ///     2's np.stack(out=, dtype=, casting=) test.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "api-parity-batch-2")]
        public void ApiParity_ConcatenateCastRefusal_IsNumPysTypeError()
        {
            var a = np.arange(6.0).reshape(2, 3);
            const string text = "Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'safe'";

            Action concatenate = () => np.concatenate(new[] { a, a }, axis: 0, dtype: np.int32, casting: "safe");
            concatenate.Should().Throw<TypeError>().WithMessage(text);

            Action stack = () => np.stack(new[] { a, a }, dtype: np.int32, casting: "safe");
            stack.Should().Throw<TypeError>().WithMessage(text);

            Action both = () => np.concatenate(new[] { a, a }, axis: 0, @out: np.zeros(new Shape(4, 3)), dtype: np.float64);
            both.Should().Throw<TypeError>()
                .WithMessage("concatenate() only takes `out` or `dtype` as an argument, but both were provided.");
        }
    }
}
