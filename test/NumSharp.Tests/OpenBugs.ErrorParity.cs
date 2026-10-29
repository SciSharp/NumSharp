using System;
using System.Collections.Generic;
using NumSharp;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs where a function's ERROR message is not NumPy 2.4.2's — a .NET framework guard message leaks, a
    ///     bare <see cref="Exception"/> carries a typo, or an axis error uses an old parenthesized format instead of
    ///     the house <c>AxisError</c> text. Each test asserts NumPy's message, so it FAILS while NumSharp's differs and
    ///     passes once the text matches. Remove <c>[OpenBugs]</c> from a test when its path is fixed.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumSharp deliberately raises its OWN exception TYPES for NumPy's <c>ValueError</c> / <c>AxisError</c> (the
    ///     house <c>IncorrectShapeException</c> / <c>ValueError</c> / <c>AxisError</c>; <c>AxisError</c> derives from
    ///     <see cref="ArgumentOutOfRangeException"/>, so its message carries a trailing <c>(Parameter 'axis')</c>). That
    ///     type choice is a convention, not a bug. The MESSAGE TEXT, however, is meant to be NumPy's verbatim — the
    ///     library asserts verbatim texts across its surface and has an <c>errors_full</c> oracle tier for exactly this.
    ///     The cases below are where the text is still wrong, found by a 2026-10-06 error-parity sweep that probed each
    ///     call against Linux NumPy 2.4.2 (the oracle NumSharp's LP64 integer model matches).
    ///     </para>
    ///     <para>
    ///     So each pin checks the message by SUBSTRING on NumPy's core text (tolerating the house
    ///     <c>(Parameter 'axis')</c> suffix a fixed <c>AxisError</c> would add) — except the one exact-match case whose
    ///     bug is a trailing character. The expected texts are the real target: NumPy is the oracle, and sibling
    ///     functions already produce them (<c>np.sum</c> / <c>np.flip</c> give the house <c>AxisError</c> text with no
    ///     parentheses; <c>np.bmat</c> / <c>np.r_</c> / <c>np.concatenate</c> give <c>need at least one array to
    ///     concatenate</c> as a real <c>ValueError</c>), so these are inconsistencies, not open design questions.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OpenBugsErrorParityTests : TestClass
    {
        /// <summary>
        ///     The stacking family on an EMPTY operand list must raise NumPy's <c>ValueError</c> — "need at least one
        ///     array to concatenate" (<c>np.vstack</c> / <c>np.hstack</c> / <c>np.dstack</c>) or "need at least one
        ///     array to stack" (<c>np.stack</c>). NumSharp leaks .NET's generic guard-clause message
        ///     ("Value cannot be an empty collection. (Parameter '…')"), which carries none of NumPy's text. Sibling
        ///     entry points already do it right: <c>np.concatenate([])</c> and <c>np.column_stack()</c> carry NumPy's
        ///     "need at least one array to concatenate", so this is an inconsistency, not an open question.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, while the stacking family leaks the
        /// .NET guard message — the expected state until these entry points raise NumPy's text.</exception>
        [TestMethod, OpenBugs]
        public void StackingFamily_EmptyInput_LeaksDotNetMessage()
        {
            var fails = new List<string>();
            CheckContains(fails, "np.vstack()", () => np.vstack(), "need at least one array to concatenate");
            CheckContains(fails, "np.hstack()", () => np.hstack(), "need at least one array to concatenate");
            CheckContains(fails, "np.dstack()", () => np.dstack(), "need at least one array to concatenate");
            CheckContains(fails, "np.stack([])", () => np.stack(new NDArray[0]), "need at least one array to stack");
            AssertNoMismatch(fails,
                "the stacking family must raise NumPy's \"need at least one array to …\" ValueError on an empty operand " +
                "list, not .NET's generic empty-collection guard message (np.concatenate / np.column_stack already do)");
        }

        /// <summary>
        ///     The Transpose-delegating family (<c>np.moveaxis</c>, <c>np.rollaxis</c>, <c>np.swapaxes</c>) and
        ///     <c>np.squeeze</c> must report an out-of-range axis with the house <c>AxisError</c> text — NumPy's
        ///     "axis 5 is out of bounds for array of dimension 2", which NumPy prefixes with the operand's role
        ///     (<c>destination:</c> / <c>source:</c> / <c>axis2:</c>) where it normalizes more than one axis.
        ///     NumSharp instead throws a differently-formatted <c>AxisOutOfRangeException</c> ("axis (5) is out of
        ///     bounds for array of dimension (2)" — parenthesized numbers, no role prefix), and <c>np.squeeze</c> leaks
        ///     .NET's generic range message. The house format is confirmed by <c>np.sum</c> / <c>np.flip</c>, which give
        ///     "axis 5 is out of bounds for array of dimension 2" with no parentheses.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, while these axis errors use the
        /// parenthesized format or the .NET message — the expected state until they use the house AxisError text.</exception>
        [TestMethod, OpenBugs]
        public void AxisOutOfRange_TransposeFamilyAndSqueeze_NotNumPysAxisErrorText()
        {
            var fails = new List<string>();
            // The role prefix is NumPy's, emitted where it normalizes more than one axis (moveaxis/swapaxes).
            CheckContains(fails, "np.moveaxis dest OOB", () => np.moveaxis(Z(2, 3), 0, 5),
                "destination: axis 5 is out of bounds for array of dimension 2");
            CheckContains(fails, "np.moveaxis source OOB", () => np.moveaxis(Z(2, 3), 5, 0),
                "source: axis 5 is out of bounds for array of dimension 2");
            CheckContains(fails, "np.swapaxes OOB", () => np.swapaxes(Z(2, 3), 0, 5),
                "axis2: axis 5 is out of bounds for array of dimension 2");
            CheckContains(fails, "np.rollaxis OOB", () => np.rollaxis(Z(2, 3), 5),
                "axis 5 is out of bounds for array of dimension 2");
            CheckContains(fails, "np.squeeze OOB", () => np.squeeze(Z(2, 3), 5),
                "axis 5 is out of bounds for array of dimension 2");
            AssertNoMismatch(fails,
                "the Transpose family and np.squeeze must report an out-of-range axis with NumPy's AxisError text " +
                "(no parentheses around the numbers, with NumPy's role prefix), as np.sum / np.flip already do");
        }

        /// <summary>
        ///     <c>np.moveaxis</c> with <c>source</c> and <c>destination</c> of different lengths must raise NumPy's
        ///     <c>ValueError</c> — "`source` and `destination` arguments must have the same number of elements".
        ///     NumSharp throws a bare <see cref="Exception"/> whose message ends in a stray backtick
        ///     ("…same number of elements'"), so it is both the wrong (uncatchable) type and a typo'd text. Asserted by
        ///     EXACT equality because the only defect in an otherwise-correct message is that trailing character.
        /// </summary>
        /// <exception cref="Exception">The assertion framework's failure exception, while the message carries the
        /// trailing backtick — the expected state until the text matches NumPy's verbatim.</exception>
        [TestMethod, OpenBugs]
        public void Moveaxis_SourceDestinationLengthMismatch_HasTrailingBacktickTypo()
        {
            var fails = new List<string>();
            CheckExact(fails, "np.moveaxis len mismatch", () => np.moveaxis(Z(2, 3, 4), new[] { 0, 1 }, new[] { 1 }),
                "`source` and `destination` arguments must have the same number of elements");
            AssertNoMismatch(fails,
                "np.moveaxis with mismatched source/destination lengths must raise NumPy's verbatim ValueError text, " +
                "not a bare Exception whose message ends in a stray backtick");
        }

        /// <summary>Builds a fresh zero-filled array of the given shape, used only to provoke the error path.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The new array.</returns>
        private static NDArray Z(params long[] dims) => np.zeros(new Shape(dims));

        /// <summary>
        ///     Runs a call that must throw and records a line unless the thrown message CONTAINS
        ///     <paramref name="expectedSubstring"/>. A non-throwing call is a mismatch too, so a regression that stops
        ///     raising is caught. Substring is used so a fixed message may still carry the house
        ///     <c>(Parameter 'axis')</c> suffix.
        /// </summary>
        /// <param name="mismatches">Collects one line per failing case; the test asserts it is empty.</param>
        /// <param name="name">The case label used in the failure message.</param>
        /// <param name="call">The NumSharp call under test; it is expected to throw.</param>
        /// <param name="expectedSubstring">NumPy 2.4.2's message, or the core of it that a fixed message must contain.</param>
        private static void CheckContains(List<string> mismatches, string name, Func<object> call, string expectedSubstring)
        {
            string actual = Capture(call);
            if (!actual.Contains(expectedSubstring, StringComparison.Ordinal))
                mismatches.Add($"{name}: message should contain [{expectedSubstring}] but was [{actual}]");
        }

        /// <summary>
        ///     Runs a call that must throw and records a line unless the thrown message EQUALS
        ///     <paramref name="expected"/> exactly — used when the bug is a single stray character a substring check
        ///     would miss.
        /// </summary>
        /// <param name="mismatches">Collects one line per failing case; the test asserts it is empty.</param>
        /// <param name="name">The case label used in the failure message.</param>
        /// <param name="call">The NumSharp call under test; it is expected to throw.</param>
        /// <param name="expected">NumPy 2.4.2's message, verbatim.</param>
        private static void CheckExact(List<string> mismatches, string name, Func<object> call, string expected)
        {
            string actual = Capture(call);
            if (actual != expected)
                mismatches.Add($"{name}: message should be [{expected}] but was [{actual}]");
        }

        /// <summary>
        ///     Invokes <paramref name="call"/> and returns the thrown exception's message, unwrapping a reflection
        ///     wrapper, or a sentinel describing a call that did not throw.
        /// </summary>
        /// <param name="call">The NumSharp call under test.</param>
        /// <returns>The thrown message, or "NO-THROW …" when the call returned a value.</returns>
        private static string Capture(Func<object> call)
        {
            try
            {
                object r = call();
                return $"NO-THROW ({(r as NDArray)?.ToString() ?? r?.ToString() ?? "null"})";
            }
            catch (Exception e)
            {
                // Unwrap a reflection/aggregate wrapper so the leaf message (the one a user sees) is compared.
                while (e.InnerException != null && (e is System.Reflection.TargetInvocationException || e is AggregateException))
                    e = e.InnerException;
                return e.Message;
            }
        }

        /// <summary>
        ///     Fails the test when any case was recorded, listing every line so a partial fix shows which messages it
        ///     repaired.
        /// </summary>
        /// <param name="mismatches">The lines the Check helpers collected; empty means every case matched NumPy.</param>
        /// <param name="because">The parity rule the cases check, shown in the failure message.</param>
        /// <exception cref="Exception">The assertion framework's failure exception, when <paramref name="mismatches"/>
        /// is not empty.</exception>
        private static void AssertNoMismatch(List<string> mismatches, string because)
        {
            string.Join(Environment.NewLine, mismatches).Should().BeEmpty(because);
        }
    }
}
