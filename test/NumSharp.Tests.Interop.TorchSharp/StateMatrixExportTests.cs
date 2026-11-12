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
    ///     NumSharp → torch for every array STATE of <see cref="StateCatalog.ArrayStates"/>, every shared dtype and every mode,
    ///     held to NumPy 2.4.2 and PyTorch (the fixture of <c>test/oracle/gen_torch_interop_states.py</c>).
    /// </summary>
    /// <remarks>
    ///     <para>Per state and dtype it checks, in order:</para>
    ///     <list type="number">
    ///         <item><description>the state itself: NumSharp's array has NumPy's shape, element strides and five flags (C, F,
    ///         OWNDATA, WRITEABLE, ALIGNED) for the same recipe, so the rest tests the state it names;</description></item>
    ///         <item><description>View: shares when <see cref="StateCatalog.ExportBlocker"/> finds nothing, else throws naming the
    ///         blocker; Auto: the same sharing, else a copy; Copy: never shares;</description></item>
    ///         <item><description>a shared tensor's layout is <c>torch.from_numpy</c>'s (strides on every axis longer than 1,
    ///         <c>is_contiguous()</c>) wherever from_numpy made one, and NumSharp's strides where it refused (a reversed length-1
    ///         axis); a copy is contiguous;</description></item>
    ///         <item><description>the bits: the tensor holds the array's values, compared through each side's own copy;</description></item>
    ///         <item><description>write-through: a torch write into a shared tensor shows in the array and a NumSharp write into
    ///         a writeable array shows in the tensor; a write into a copy changes nothing on the other side;</description></item>
    ///         <item><description>the array's flags are the same after the crossing, and after the tensor is disposed.</description></item>
    ///     </list>
    /// </remarks>
    [TestClass]
    public class StateMatrixExportTests : TorchTestBase
    {
        /// <summary>Every array state name of the fixture, one test case each.</summary>
        public static IEnumerable<object[]> States => StateOracle.ArrayStates.Keys.OrderBy(k => k, StringComparer.Ordinal).Select(k => new object[] { k });

        /// <summary>One array state, every dtype the fixture recorded it for, every mode.</summary>
        /// <param name="state">The state's name.</param>
        [TestMethod]
        [DynamicData(nameof(States))]
        public void ArrayState_CrossesAsTheRulesSay(string state)
        {
            var failures = new Failures();
            foreach (var (dtype, record) in StateOracle.ArrayStates[state])
            {
                string where = $"{state} [{dtype}]";
                try
                {
                    CheckState(where, StateCatalog.ArrayStates[state](dtype), record, failures);
                }
                catch (Exception ex)
                {
                    failures.Unexpected(where, ex);
                }
            }
            failures.None.Should().BeTrue(failures.ToString());
        }

        /// <summary>Runs every check of one state and dtype.</summary>
        /// <param name="where">The case's name, for messages.</param>
        /// <param name="nd">The array, built by the state's recipe.</param>
        /// <param name="record">NumPy's and PyTorch's answers for the same recipe.</param>
        /// <param name="failures">Where failures are collected.</param>
        private static void CheckState(string where, NDArray nd, StateOracle.ArrayRecord record, Failures failures)
        {
            using var scope = Scope();

            // 1. The state is NumPy's.
            StateOracle.Flags flags = StateOracle.Flags.Of(nd);
            failures.Check(nd.shape.SequenceEqual(record.NumPy.Shape), $"{where}: shape ({string.Join(",", nd.shape)}) is not NumPy's ({string.Join(",", record.NumPy.Shape)})");
            failures.Check(SameStrides(record.NumPy.Shape, nd.Shape.strides, record.NumPy.Strides),
                $"{where}: element strides ({string.Join(",", nd.Shape.strides)}) are not NumPy's ({string.Join(",", record.NumPy.Strides)})");
            failures.Check(flags == record.NumPy.Flags, $"{where}: flags {flags} are not NumPy's {record.NumPy.Flags}");

            string blocker = StateCatalog.ExportBlocker(nd);
            int flagsBefore = nd.flags.num;
            foreach (TorchCodecMode mode in Modes)
            {
                string at = $"{where} {mode}";
                bool expectView = mode != TorchCodecMode.Copy && blocker is null;
                if (mode == TorchCodecMode.View && blocker != null)
                {
                    try
                    {
                        using Tensor refused = nd.AsTensor(mode);
                        failures.Check(false, $"{at}: shared memory it must not ({blocker})");
                    }
                    catch (InvalidOperationException ex)
                    {
                        failures.Check(ex.Message.Contains(blocker), $"{at}: the refusal does not name '{blocker}': {ex.Message}");
                    }
                    continue;
                }

                using (Tensor t = nd.AsTensor(mode))
                {
                    // 2-4. Sharing, layout, bits.
                    failures.Check(t.dtype == NDArrayTorchInterop.ToScalarType(nd.dtype), $"{at}: dtype {t.dtype}");
                    failures.Check(t.shape.SequenceEqual(nd.shape), $"{at}: shape ({string.Join(",", t.shape)})");
                    failures.Check(SameBits(nd, t), $"{at}: the tensor's bits are not the array's");
                    failures.Check(nd.MayShareMemory(t) == (expectView && nd.size > 0), $"{at}: MayShareMemory {nd.MayShareMemory(t)}, expected {expectView && nd.size > 0}");
                    if (expectView && nd.size > 0)
                    {
                        long[] expected = record.FromNumpy.Ok ? record.FromNumpy.Strides : nd.Shape.strides;
                        failures.Check(SameStrides(nd.shape, t.stride(), expected),
                            $"{at}: torch strides ({string.Join(",", t.stride())}) are not {(record.FromNumpy.Ok ? "from_numpy's" : "NumSharp's")} ({string.Join(",", expected)})");
                        if (record.FromNumpy.Ok)
                            failures.Check(t.is_contiguous() == record.FromNumpy.IsContiguous, $"{at}: is_contiguous {t.is_contiguous()}, from_numpy {record.FromNumpy.IsContiguous}");
                    }
                    else
                    {
                        failures.Check(t.is_contiguous(), $"{at}: a copy is not contiguous");
                    }

                    // 5. Write-through, both ways.
                    if (nd.size > 0)
                    {
                        // Each write puts a value that differs from what the element holds, so "nothing changed" is visible.
                        double written = Other(FirstValue(t));
                        byte[] before = BytesOf(nd);
                        TorchWriteFirst(t, written);
                        bool changed = !before.AsSpan().SequenceEqual(BytesOf(nd));
                        if (expectView)
                        {
                            failures.Check(changed && SameBits(nd, t), $"{at}: a torch write did not reach the array");
                            if (nd.Shape.IsWriteable)
                            {
                                byte[] tensorBefore = BytesOf(t);
                                NumSharpWriteFirst(nd, Other(written));
                                failures.Check(!tensorBefore.AsSpan().SequenceEqual(BytesOf(t)) && SameBits(nd, t),
                                    $"{at}: a NumSharp write did not reach the tensor");
                            }
                        }
                        else
                        {
                            failures.Check(!changed, $"{at}: a write into the copy changed the array");
                        }
                    }
                    failures.Check(nd.flags.num == flagsBefore, $"{at}: the crossing changed the array's flags");
                }
                failures.Check(nd.flags.num == flagsBefore, $"{at}: disposing the tensor changed the array's flags");
            }
        }
    }
}
