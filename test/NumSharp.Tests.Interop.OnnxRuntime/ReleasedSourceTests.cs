using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     A source whose buffer has been released is refused by every verb that reads it, whatever its layout and dtype;
    ///     and an array too large for the copy's destination is refused before anything is allocated.
    ///
    ///     <para><b>The regression.</b> The copy verbs (<c>ToOrtValue</c>, <c>ToDenseTensor</c>, <c>session.Run</c>) densified a
    ///     strided source with <c>copy()</c>, and every verb converted a Decimal source with <c>astype</c>, BEFORE any check
    ///     that its buffer was still alive. That check is the ARC reference <c>Pin</c> takes (<c>TryAddRef</c> fails on a
    ///     released buffer), and it only ran later, on the fresh copy. So a released buffer was read silently: once the pool
    ///     had handed it to another array, the export carried that array's values (a probe read another array's 777s through
    ///     all three verbs, and <c>session.Run</c> fed them to the model). Only a contiguous source of a directly supported
    ///     dtype reached <c>Pin</c> first, which is all <see cref="LifetimeTests.DisposedSource_CannotBeExported"/> covered.
    ///     The same bug was fixed in the System.Numerics.Tensors bridge (<c>c8f1dffa</c>).</para>
    /// </summary>
    [TestClass]
    public class ReleasedSourceTests : OnnxTestBase
    {
        /// <summary>The layouts <see cref="Released"/> builds: every path the copy verbs take (memcpy, densify, empty).</summary>
        private static readonly string[] Layouts = { "contiguous", "contiguous slice", "transposed", "stepped", "broadcast", "empty" };

        /// <summary>
        ///     <c>ToOrtValue</c> and <c>ToDenseTensor</c> accept any layout, so a released source can only be refused for being
        ///     released. Float64 takes the memcpy and densify paths; Decimal additionally takes the float64 conversion, which
        ///     read the released buffer even when the source was contiguous.
        /// </summary>
        [TestMethod]
        public void CopyVerbs_ReleasedSource_Throw_EveryLayoutAndConversion()
        {
            foreach (NPTypeCode code in new[] { NPTypeCode.Double, NPTypeCode.Decimal })
            foreach (string layout in Layouts)
            {
                NDArray released = Released(layout, code);
                new Action(() => released.ToOrtValue()).Should().Throw<ObjectDisposedException>($"ToOrtValue of a released {code} {layout} array");
                new Action(() => released.ToDenseTensor<double>()).Should().Throw<ObjectDisposedException>($"ToDenseTensor of a released {code} {layout} array");
            }
        }

        /// <summary>
        ///     The zero-copy verbs share a Decimal array through a float64 conversion, which is always C-contiguous, so no
        ///     layout is refused and the conversion is the first read of the source. It used to read a released buffer and the
        ///     export then succeeded over the converted copy.
        /// </summary>
        [TestMethod]
        public void ZeroCopyVerbs_ReleasedDecimalSource_Throw_EveryLayout()
        {
            foreach (string layout in Layouts)
            {
                NDArray released = Released(layout, NPTypeCode.Decimal);
                new Action(() => released.AsOrtValue()).Should().Throw<ObjectDisposedException>($"AsOrtValue of a released Decimal {layout} array");
                new Action(() => released.AsDenseTensor<double>()).Should().Throw<ObjectDisposedException>($"AsDenseTensor of a released Decimal {layout} array");
            }
        }

        /// <summary>
        ///     <c>session.Run</c> prepares each input with a cast to the model's declared dtype (int32 into a float64 input), a
        ///     C-order copy of a strided view, or the Decimal conversion. Each of those read a released input before any check,
        ///     and the model then ran on whatever the buffer held.
        /// </summary>
        [TestMethod]
        public void SessionRun_ReleasedInput_Throws_EveryLayoutCastAndConversion()
        {
            using InferenceSession session = OpenSession("identity_float64");
            foreach (NPTypeCode code in new[] { NPTypeCode.Double, NPTypeCode.Int32, NPTypeCode.Decimal })
            foreach (string layout in Layouts)
            {
                NDArray released = Released(layout, code);
                // "same_kind": the rule that admits all three (the Decimal conversion loses digits, so "safe" refuses it
                // before the array is read, which is a different refusal from the one this test pins).
                new Action(() => session.Run(released, casting: "same_kind")).Should().Throw<ObjectDisposedException>($"Run on a released {code} {layout} input");
            }
        }

        /// <summary>
        ///     <c>ToDenseTensor</c> copies into a managed <c>T[]</c>, which holds at most <see cref="Array.MaxLength"/>
        ///     (2,147,483,591) elements. The guard compared against <see cref="int.MaxValue"/>, so the sizes in between reached
        ///     <c>new T[n]</c> and failed with OutOfMemoryException ("Array dimensions exceeded supported range") instead of
        ///     the refusal. Broadcast views give the element counts without the memory.
        /// </summary>
        [TestMethod]
        public void ToDenseTensor_PastTheManagedArrayLimit_IsRefused_NotOutOfMemory()
        {
            using NDArray one = NDArray.Scalar(true);
            foreach (long size in new[] { (long)Array.MaxLength + 1, int.MaxValue, (long)int.MaxValue + 1 })
            {
                using NDArray huge = np.broadcast_to(one, new Shape(size));
                new Action(() => huge.ToDenseTensor<bool>()).Should().Throw<NotSupportedException>($"{size} elements")
                    .WithMessage("*Array.MaxLength*");
            }
        }

        /// <summary>
        ///     <c>ToOrtValue</c> copies through ORT's <see cref="Span{T}"/> of bytes, which is limited to 2 GB. That limit was
        ///     checked only AFTER the source had been densified and ORT had allocated the tensor, so a large strided array cost
        ///     two full-size allocations just to be refused. With 2^40 elements the densifying copy needs a terabyte and
        ///     failed with OutOfMemoryException; the refusal must come before anything is allocated.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_PastTheSpanLimit_IsRefused_BeforeAnythingIsCopied()
        {
            using NDArray one = NDArray.Scalar(true);
            using NDArray huge = np.broadcast_to(one, new Shape(1L << 40));
            new Action(() => huge.ToOrtValue()).Should().Throw<NotSupportedException>().WithMessage("*2 GB*");
        }

        /// <summary>
        ///     The refusals decided by the dtype and the shape alone come BEFORE the liveness check, so they hold for a
        ///     released array too: an array that could never cross is refused for that, without anything being read,
        ///     converted or allocated to find out. Pins the order the fix put the checks in (dtype, size, then the pin).
        /// </summary>
        [TestMethod]
        public void DtypeAndSizeRefusals_ComeBeforeTheLivenessCheck()
        {
            NDArray complex = Released("contiguous", NPTypeCode.Complex);
            new Action(() => complex.ToOrtValue()).Should().Throw<NotSupportedException>("Complex has no ONNX element type");
            new Action(() => complex.AsOrtValue()).Should().Throw<NotSupportedException>("Complex has no ONNX element type");
            new Action(() => complex.ToDenseTensor<double>()).Should().Throw<NotSupportedException>("Complex has no ONNX element type");
            new Action(() => complex.AsDenseTensor<double>()).Should().Throw<NotSupportedException>("Complex has no ONNX element type");

            NDArray float64 = Released("transposed", NPTypeCode.Double);
            new Action(() => float64.ToDenseTensor<float>()).Should().Throw<ArgumentException>("a float64 array crosses as DenseTensor<double>");

            NDArray huge = ReleasedBroadcast(1L << 40);
            new Action(() => huge.ToOrtValue()).Should().Throw<NotSupportedException>().WithMessage("*2 GB*");
            new Action(() => huge.ToDenseTensor<bool>()).Should().Throw<NotSupportedException>().WithMessage("*Array.MaxLength*");
        }

        /// <summary>A broadcast view of <paramref name="size"/> booleans whose buffer has been released (see <see cref="Released"/>).</summary>
        /// <param name="size">The element count of the view; a broadcast needs no memory for it.</param>
        /// <returns>The disposed view.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray ReleasedBroadcast(long size)
        {
            NDArray one = NDArray.Scalar(true);
            NDArray view = np.broadcast_to(one, new Shape(size));
            view.Dispose();
            one.Dispose();
            return view;
        }

        /// <summary>
        ///     Builds an array of the requested layout and dtype, then disposes it together with every array it was derived
        ///     from, so no reference is left and its buffer goes back to the pool. A helper of its own (no inlining) keeps
        ///     the JIT from holding any of them alive in the caller's frame.
        /// </summary>
        /// <param name="layout">
        ///     One of "contiguous", "contiguous slice", "transposed", "stepped", "broadcast" or "empty" (a <c>(0, 3)</c> array).
        /// </param>
        /// <param name="code">The dtype of the array; Decimal and Int32 exercise the conversion and cast paths.</param>
        /// <returns>The disposed array, whose buffer has been released.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="layout"/> is not one of the names above.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray Released(string layout, NPTypeCode code)
        {
            var made = new List<NDArray>();
            NDArray Keep(NDArray a)
            {
                made.Add(a);
                return a;
            }

            NDArray owner = Keep(layout == "empty" ? np.zeros(new Shape(0, 3), code) : Arange(code, 6));
            NDArray view = layout switch
            {
                "contiguous" => owner,
                "contiguous slice" => Keep(owner["1:5"]),
                "transposed" => Keep(Keep(owner.reshape(2, 3)).T),
                "stepped" => Keep(owner["::2"]),
                "broadcast" => Keep(np.broadcast_to(Keep(owner["0:3"]), new Shape(2, 3))),
                "empty" => owner,
                _ => throw new ArgumentOutOfRangeException(nameof(layout), layout, "unknown layout"),
            };

            // Dispose the view and every array it was derived from: only then is the buffer's last reference gone.
            foreach (NDArray a in made)
                a.Dispose();
            return view;
        }
    }
}
