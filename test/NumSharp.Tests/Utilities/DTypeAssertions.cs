using System;
using System.Diagnostics;
using System.Linq;
using AwesomeAssertions.Execution;
using AwesomeAssertions.Primitives;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests
{
    /// <summary>
    ///     <c>x.dtype.Should()</c> for the <see cref="DType"/> descriptor (Stage B of docs/plans/dtype-system.md flipped
    ///     <c>NDArray.dtype</c> from <see cref="Type"/> to <see cref="DType"/>).
    /// </summary>
    /// <remarks>
    ///     Deliberately declared in the <c>NumSharp.Tests</c> namespace rather than <c>NumSharp.Tests.Utilities</c>: C# searches
    ///     the ENCLOSING namespaces for extension methods before it reaches the compilation unit's <c>using</c> directives, so
    ///     every test class (all live in <c>NumSharp.Tests.*</c>) binds a <see cref="DType"/> receiver here — ahead of
    ///     AwesomeAssertions' <c>Should(this object)</c> — without a per-file <c>using</c>. That keeps the ~1,000 pre-existing
    ///     <c>.dtype.Should().Be(typeof(X))</c> / <c>.Be(np.X)</c> / <c>.Be(NPTypeCode.X)</c> / <c>.Be&lt;X&gt;()</c> sites
    ///     compiling unchanged and gives them NumPy's coercing dtype equality (<c>a.dtype == np.float64</c>). Receivers of other
    ///     types are unaffected: an extension receiver never goes through a user-defined conversion, so a <see cref="Type"/>,
    ///     <see cref="NPTypeCode"/> or string subject still binds its own AwesomeAssertions <c>Should()</c>.
    /// </remarks>
    [DebuggerStepThrough]
    public static class DTypeFluentExtension
    {
        public static DTypeAssertions Should(this DType dtype)
        {
            return new DTypeAssertions(dtype);
        }
    }
}

namespace NumSharp.Tests.Utilities
{
    /// <summary>
    ///     Assertions over a <see cref="DType"/> descriptor. <c>Be</c> is NumPy's <c>dtype == other</c>: structural between
    ///     descriptors (class + byte order + parameters) and coercing for a <see cref="Type"/>, an <see cref="NPTypeCode"/> or a
    ///     dtype string — all of which convert implicitly to <see cref="DType"/>, so one overload covers every spelling.
    /// </summary>
    [DebuggerStepThrough]
    public class DTypeAssertions : ReferenceTypeAssertions<DType, DTypeAssertions>
    {
        private readonly AssertionChain _chain;

        public DTypeAssertions(DType instance)
            : this(instance, AssertionChain.GetOrCreate())
        {
        }

        public DTypeAssertions(DType instance, AssertionChain chain)
            : base(instance, chain)
        {
            _chain = chain;
        }

        protected override string Identifier => "dtype";

        /// <summary>Asserts the descriptor equals <paramref name="expected"/> (a <see cref="DType"/>, <see cref="Type"/>, <see cref="NPTypeCode"/> or dtype string).</summary>
        public AndConstraint<DTypeAssertions> Be(DType expected, string because = "", params object[] becauseArgs)
        {
            _chain
                .BecauseOf(because, becauseArgs)
                .ForCondition(Subject is null ? expected is null : Subject.Equals(expected))
                .FailWith("Expected {context:dtype} to be {0}{reason}, but found {1}.", Repr(expected), Repr(Subject));

            return new AndConstraint<DTypeAssertions>(this);
        }

        /// <summary>Asserts the descriptor is the dtype of the C# element type <typeparamref name="T"/> (<c>Be&lt;double&gt;()</c> ≙ <c>Be(typeof(double))</c>).</summary>
        public AndConstraint<DTypeAssertions> Be<T>(string because = "", params object[] becauseArgs)
        {
            return Be(typeof(T), because, becauseArgs);
        }

        /// <summary>Asserts the descriptor does not equal <paramref name="unexpected"/>.</summary>
        public AndConstraint<DTypeAssertions> NotBe(DType unexpected, string because = "", params object[] becauseArgs)
        {
            _chain
                .BecauseOf(because, becauseArgs)
                .ForCondition(!(Subject is null ? unexpected is null : Subject.Equals(unexpected)))
                .FailWith("Did not expect {context:dtype} to be {0}{reason}.", Repr(unexpected));

            return new AndConstraint<DTypeAssertions>(this);
        }

        /// <summary>Asserts the descriptor is not the dtype of the C# element type <typeparamref name="T"/>.</summary>
        public AndConstraint<DTypeAssertions> NotBe<T>(string because = "", params object[] becauseArgs)
        {
            return NotBe(typeof(T), because, becauseArgs);
        }

        /// <summary>Asserts the descriptor equals one of <paramref name="validValues"/>.</summary>
        public AndConstraint<DTypeAssertions> BeOneOf(params DType[] validValues)
        {
            if (validValues == null)
                throw new ArgumentNullException(nameof(validValues));

            _chain
                .ForCondition(validValues.Any(v => Subject is null ? v is null : Subject.Equals(v)))
                .FailWith("Expected {context:dtype} to be one of {0}, but found {1}.",
                    string.Join(", ", validValues.Select(Repr)), Repr(Subject));

            return new AndConstraint<DTypeAssertions>(this);
        }

        private static string Repr(DType dtype)
        {
            return dtype is null ? "<null>" : dtype.ToString(repr: true);
        }
    }
}
