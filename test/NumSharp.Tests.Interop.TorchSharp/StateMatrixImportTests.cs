using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;
using static NumSharp.Tests.Interop.TorchSharp.StateMatrixSupport;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     torch → NumSharp for every tensor STATE of <see cref="StateCatalog.TensorStates"/>, every dtype the fixture recorded
    ///     and every mode, held to PyTorch's <c>t.numpy()</c> (and <c>t.numpy(force=True)</c> where numpy() refuses).
    /// </summary>
    /// <remarks>
    ///     <para>Per state and dtype it checks: the tensor is the fixture's (shape, strides, storage offset, contiguity,
    ///     requires_grad); View shares when <see cref="StateCatalog.ImportBlocker"/> finds nothing, else throws naming the
    ///     blocker; a view's NDArray has numpy()'s element strides and five flags; a copy's flags are a fresh C-contiguous
    ///     array's; the bits are the tensor's (or its exact widening / resolution); a NumSharp write into a writeable view shows
    ///     in the tensor.</para>
    ///     <para>Where the bridge differs from numpy() on purpose, asserted as such: an expanded (stride-0) tensor imports
    ///     READ-ONLY (NumSharp marks every repeated axis read-only, NumPy's broadcast rule; numpy() leaves such an array
    ///     writeable); a tensor that requires grad imports as a READ-ONLY view (numpy() refuses it, numpy(force=True) gives a
    ///     writeable view of its detach()); complex64 and bfloat16 widen to complex128 and float32 (numpy() returns complex64
    ///     and refuses bfloat16).</para>
    /// </remarks>
    [TestClass]
    public class StateMatrixImportTests : TorchTestBase
    {
        /// <summary>Every tensor state name of the fixture, one test case each.</summary>
        public static IEnumerable<object[]> States => StateOracle.TensorStates.Keys.OrderBy(k => k, StringComparer.Ordinal).Select(k => new object[] { k });

        /// <summary>One tensor state, every dtype the fixture recorded it for (that this libtorch has), every mode.</summary>
        /// <param name="state">The state's name.</param>
        [TestMethod]
        [DynamicData(nameof(States))]
        public void TensorState_CrossesAsTheRulesSay(string state)
        {
            var failures = new Failures();
            int ran = 0;
            foreach (var (dtype, record) in StateOracle.TensorStates[state])
            {
                if (!StateCatalog.TorchHas(dtype))
                    continue;
                ran++;
                string where = $"{state} [{dtype}]";
                try
                {
                    using var scope = Scope();
                    CheckState(where, StateCatalog.TensorStates[state](StateCatalog.ScalarTypeOf(dtype)), record, failures);
                }
                catch (Exception ex)
                {
                    failures.Unexpected(where, ex);
                }
            }
            if (ran == 0)
                Assert.Inconclusive($"{state}: no dtype the fixture recorded exists in TorchSharp {typeof(torch).Assembly.GetName().Version}.");
            failures.None.Should().BeTrue(failures.ToString());
        }

        /// <summary>Runs every check of one tensor state and dtype.</summary>
        /// <param name="where">The case's name, for messages.</param>
        /// <param name="t">The tensor, built by the state's recipe inside the caller's DisposeScope.</param>
        /// <param name="record">PyTorch's answers for the same recipe.</param>
        /// <param name="failures">Where failures are collected.</param>
        private static void CheckState(string where, Tensor t, StateOracle.TensorRecord record, Failures failures)
        {
            // The tensor is PyTorch's.
            StateOracle.TensorLayout expectedTensor = record.Tensor;
            failures.Check(t.shape.SequenceEqual(expectedTensor.Shape), $"{where}: tensor shape ({string.Join(",", t.shape)}) is not PyTorch's");
            failures.Check(t.stride().SequenceEqual(expectedTensor.Strides), $"{where}: tensor strides ({string.Join(",", t.stride())}) are not PyTorch's ({string.Join(",", expectedTensor.Strides)})");
            failures.Check(t.storage_offset() == expectedTensor.StorageOffset, $"{where}: storage offset {t.storage_offset()} is not PyTorch's {expectedTensor.StorageOffset}");
            failures.Check(t.is_contiguous() == expectedTensor.IsContiguous, $"{where}: is_contiguous {t.is_contiguous()} is not PyTorch's");
            failures.Check(t.requires_grad == expectedTensor.RequiresGrad, $"{where}: requires_grad {t.requires_grad} is not PyTorch's");

            string blocker = StateCatalog.ImportBlocker(t);
            // The values a correct import holds: the tensor's own, or its exact widening / resolution.
            Tensor reference = t.dtype == ScalarType.ComplexFloat32 ? t.to_type(ScalarType.ComplexFloat64)
                : t.dtype == ScalarType.BFloat16 ? t.to_type(ScalarType.Float32)
                : t.resolve_conj().resolve_neg();
            long numel = t.numel();
            // numpy()'s answer for a view; numpy(force=True)'s when numpy() refused (requires grad, conj, neg).
            StateOracle.ToNumpy viewAnswer = record.NumPy.Ok ? record.NumPy : record.NumPyForce;

            foreach (TorchCodecMode mode in Modes)
            {
                string at = $"{where} {mode}";
                if (mode == TorchCodecMode.View && blocker != null)
                {
                    try
                    {
                        using NDArray refused = t.AsNDArray(mode);
                        failures.Check(false, $"{at}: viewed a tensor it must not ({blocker})");
                    }
                    catch (InvalidOperationException ex)
                    {
                        failures.Check(ex.Message.Contains(blocker), $"{at}: the refusal does not name '{blocker}': {ex.Message}");
                    }
                    continue;
                }

                using (NDArray nd = t.AsNDArray(mode))
                {
                    StateOracle.Flags flags = StateOracle.Flags.Of(nd);
                    failures.Check(SameBits(nd, reference), $"{at}: the array's bits are not the tensor's");
                    // A dtype NumSharp widens always lands in an array NumSharp owns, whatever else the tensor needed first (a
                    // conjugated complex64 is resolved, then widened): the widening is the last step, and it allocates.
                    bool widened = t.dtype == ScalarType.ComplexFloat32 || t.dtype == ScalarType.BFloat16;
                    if (mode == TorchCodecMode.Copy || widened)
                    {
                        // A copy NumSharp owns: a fresh C-contiguous array, sharing nothing.
                        StateOracle.Flags expected = StateCatalog.OwningCopyFlags(nd.shape);
                        failures.Check(flags == expected, $"{at}: copy flags {flags}, expected {expected}");
                        failures.Check(!nd.MayShareMemory(t), $"{at}: a copy shares the tensor's memory");
                    }
                    else if (blocker != null)
                    {
                        // Auto over a pending conjugation / negation: a view of the private resolved copy - numpy(force=True)'s
                        // array exactly (a contiguous view NumPy does not own), sharing nothing with the tensor itself.
                        failures.Check(flags == viewAnswer.Layout.Flags, $"{at}: flags {flags}, numpy(force=True) {viewAnswer.Layout.Flags}");
                        failures.Check(!nd.MayShareMemory(t), $"{at}: the resolved view shares the unresolved tensor's memory");
                    }
                    else
                    {
                        // A view of the tensor's own memory: numpy()'s strides and flags, but read-only where NumSharp's broadcast
                        // rule (a repeated axis) or the requires-grad rule says so.
                        bool readOnly = HasRepeatedAxis(t.shape, t.stride()) || t.requires_grad;
                        StateOracle.Flags numpyFlags = viewAnswer.Layout.Flags;
                        StateOracle.Flags expected = numpyFlags with { W = numpyFlags.W && !readOnly };
                        failures.Check(flags == expected, $"{at}: flags {flags}, expected {expected} (numpy() {numpyFlags}{(readOnly ? ", read-only by the bridge's rule" : "")})");
                        failures.Check(SameStrides(nd.shape, nd.Shape.strides, viewAnswer.Layout.Strides),
                            $"{at}: element strides ({string.Join(",", nd.Shape.strides)}) are not numpy()'s ({string.Join(",", viewAnswer.Layout.Strides)})");
                        failures.Check(nd.MayShareMemory(t) == (numel > 0 && viewAnswer.Shares), $"{at}: MayShareMemory {nd.MayShareMemory(t)}, numpy() shares {viewAnswer.Shares}");
                        if (numel > 0 && nd.Shape.IsWriteable)
                        {
                            byte[] tensorBefore = BytesOf(t);
                            NumSharpWriteFirst(nd, Other(FirstValue(t)));
                            failures.Check(!tensorBefore.AsSpan().SequenceEqual(BytesOf(t)) && SameBits(nd, t), $"{at}: a NumSharp write did not reach the tensor");
                        }
                    }
                }
            }
        }
    }
}
