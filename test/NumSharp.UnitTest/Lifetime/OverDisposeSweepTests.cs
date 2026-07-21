using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Lifetime
{
    /// <summary>
    ///     Sweeps the np.* surface for <b>over-disposal</b>: an operation whose <c>using</c> on an
    ///     "intermediate" frees memory that is still in use.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     This is the dangerous half of the lifetime story, and the one the <c>using</c>-on-
    ///     intermediate refactors can actually introduce. The trap is aliasing: <c>np.asanyarray</c>
    ///     hands its input straight back, broadcasting an already-correct shape can return the input
    ///     itself, and a slice is a view. Wrap any of those in a <c>using</c> and the operation frees
    ///     a caller's buffer, or its own result, while a live wrapper still points at it. Nothing
    ///     throws: the array stays reachable and correctly shaped, backed by freed pages.
    ///     </para>
    ///     <para>
    ///     Unlike deferred release, this IS exactly observable through the ARC refcount that
    ///     <see cref="LeakGuards.StillUsable"/> reads. For every operation in the catalogue the sweep
    ///     asserts that all operands survive the call, that the result survives, and — because freed
    ///     pages can still read back plausible bytes — that a second identical call over the same
    ///     operands still succeeds.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OverDisposeSweepTests
    {
        [TestMethod]
        public void NoOperationDisposesItsOperandsOrResult()
        {
            var offenders = new List<string>();

            foreach (var op in LifetimeCases.All())
            {
                var operands = op.MakeOperands();
                try
                {
                    var result = op.Run(operands);

                    for (int i = 0; i < operands.Length; i++)
                    {
                        if (operands[i].IsDisposed)
                            offenders.Add($"{op.Name}: disposed operand [{i}]");
                        else if (operands[i].Storage?.InternalArray?.IsReleased == true)
                            offenders.Add($"{op.Name}: released operand [{i}]'s buffer");
                    }

                    foreach (var produced in Produced(result))
                    {
                        if (produced.IsDisposed)
                            offenders.Add($"{op.Name}: returned a disposed array");
                        else if (produced.Storage?.InternalArray?.IsReleased == true)
                            offenders.Add($"{op.Name}: returned an array over a released buffer");
                    }

                    LifetimeCase.DisposeResult(result);

                    // A second pass over the same operands: if the first call freed something it
                    // should not have, this is where a plausible-looking read becomes a fault.
                    LifetimeCase.DisposeResult(op.Run(operands));
                }
                catch (Exception e)
                {
                    offenders.Add($"{op.Name}: threw on the lifetime pass — {e.GetType().Name}: {e.Message}");
                }
                finally
                {
                    foreach (var o in operands)
                        o.Dispose();
                }
            }

            offenders.Should().BeEmpty(
                "an operation may only release its own intermediates:\n  " + string.Join("\n  ", offenders));
        }

        /// <summary>
        ///     Proves the sweep can fail: the condition it scans for must be detectable.
        /// </summary>
        [TestMethod]
        public void Sweep_DetectsAnOverDisposedOperand()
        {
            var operand = np.arange(100).astype(NPTypeCode.Double);

            // Stand-in for an intermediate that turned out to alias the input.
            operand.Dispose();

            operand.IsDisposed.Should().BeTrue();
            operand.Storage.InternalArray.IsReleased.Should().BeTrue(
                "this is the exact condition the sweep scans for");

            Action guard = () => LeakGuards.StillUsable(operand);
            guard.Should().Throw<Exception>("StillUsable must reject an over-disposed array");
        }

        private static IEnumerable<NDArray> Produced(object result)
        {
            switch (result)
            {
                case NDArray nd:
                    yield return nd;
                    break;
                case NDArray[] many:
                    foreach (var n in many)
                    {
                        // NOT `n != null` — NDArray overloads != into an elementwise comparison.
                        if (n is not null)
                            yield return n;
                    }
                    break;
            }
        }
    }
}
