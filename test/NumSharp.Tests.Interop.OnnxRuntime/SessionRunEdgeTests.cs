using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Tier-2 <c>Run</c> corners <see cref="SessionRunTests"/> leaves: the null-argument guards on every
    ///     overload, a null array inside the input map, an explicitly supplied <see cref="RunOptions"/>, a
    ///     non-contiguous pre-allocated output, a shape ORT itself rejects, and a zero-copy empty output.
    /// </summary>
    [TestClass]
    public class SessionRunEdgeTests : OnnxTestBase
    {
        [TestMethod]
        public void NullSession_And_NullArgs_ThrowArgumentNull_OnEveryOverload()
        {
            InferenceSession none = null;
            using NDArray nd = Arange(NPTypeCode.Int32, 3);
            var inputs = new Dictionary<string, NDArray> { { "X", nd } };

            new Action(() => none.Run(nd)).Should().Throw<ArgumentNullException>();
            new Action(() => none.Run("X", nd, "Y")).Should().Throw<ArgumentNullException>();
            new Action(() => none.Run(inputs)).Should().Throw<ArgumentNullException>();
            new Action(() => none.RunInto(inputs, inputs)).Should().Throw<ArgumentNullException>();

            using InferenceSession session = OpenSession("identity_int32");
            new Action(() => session.Run(null, nd, "Y")).Should().Throw<ArgumentNullException>("inputName is null");
            new Action(() => session.Run("X", nd, null)).Should().Throw<ArgumentNullException>("outputName is null");
            new Action(() => session.Run((IReadOnlyDictionary<string, NDArray>)null)).Should().Throw<ArgumentNullException>();
            new Action(() => session.RunInto(inputs, (IReadOnlyDictionary<string, NDArray>)null)).Should().Throw<ArgumentNullException>("outputs is null");
        }

        [TestMethod]
        public void NullArrayInsideTheInputMap_NamesTheInput()
        {
            using InferenceSession session = OpenSession("identity_int32");
            new Action(() => session.Run(new Dictionary<string, NDArray> { { "X", null } }))
                .Should().Throw<ArgumentNullException>().WithMessage("*input 'X' is null*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void ExplicitRunOptions_ArePassedThrough()
        {
            using InferenceSession session = OpenSession("identity_int32");
            using NDArray nd = Arange(NPTypeCode.Int32, 5);
            using var ro = new RunOptions { LogId = "edge-run" };
            using NDArray y = session.Run(nd, ro);
            y.shape.Should().Equal(5);
            y.GetInt32(4).Should().Be(4);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void PreallocatedOutput_NonContiguous_IsRefused()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            using NDArray big = np.zeros(new Shape(3, 2), typeof(float));
            using NDArray transposed = big.T;   // (2,3) but F-contiguous / non-C-contiguous
            transposed.Shape.IsContiguous.Should().BeFalse();

            new Action(() => session.RunInto(new Dictionary<string, NDArray> { { "X", x } },
                    new Dictionary<string, NDArray> { { "Y1", transposed } }))
                .Should().Throw<InvalidOperationException>().WithMessage("*'Y1'*not C-contiguous*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the refused output pinned nothing");
        }

        [TestMethod]
        public void PreallocatedOutput_WrongShape_IsRejectedByOrt_AndHandlesAreReleased()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            using NDArray wrong = np.zeros(new Shape(2, 4), typeof(float));   // identity output is (2,3)

            new Action(() => session.RunInto(new Dictionary<string, NDArray> { { "X", x } },
                    new Dictionary<string, NDArray> { { "Y1", wrong } }))
                .Should().Throw<Exception>("ORT checks the pre-allocated output shape itself");
            // the finally in Run disposed both the input and output handles despite the throw
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void ZeroCopyOutputs_OfAnEmptyOutput_AreEmptyViews_WithNoLease()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray empty = np.zeros(new Shape(0, 3), typeof(float));
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(
                new Dictionary<string, NDArray> { { "X", empty } }, zeroCopyOutputs: true);

            using NDArray y = outputs["Y"];
            y.shape.Should().Equal(0, 3);
            y.typecode.Should().Be(NPTypeCode.Single);
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "an empty output owns its (already disposed) value but holds no lease");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     A model whose SECOND output is not a tensor: <c>Run</c> converts the first output (a tensor), then
        ///     refuses the second (a sequence). What it made for the first must not leak: the copy goes back to the
        ///     pool, and in the zero-copy form the view's lease on ONNX Runtime's buffer is released. Fetching only the
        ///     tensor output works.
        /// </summary>
        [TestMethod]
        public void Run_WithANonTensorOutputAfterATensorOutput_ReleasesWhatItMadeForTheFirst()
        {
            using InferenceSession session = OpenSession("tensor_then_seq");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            new Action(() => session.Run(inputs)).Should().Throw<NotSupportedException>().WithMessage("*dense tensor OrtValue*SEQUENCE*ToNDArrays*");
            Pool.Balance(() =>
            {
                try { session.Run(inputs); }
                catch (NotSupportedException) { }
            }).Should().Be(0, "the copy made for Y before S was refused is disposed");

            new Action(() => session.Run(inputs, zeroCopyOutputs: true)).Should().Throw<NotSupportedException>().WithMessage("*SEQUENCE*");
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "the view made for Y is disposed, and its lease with it");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);

            // the tensor output alone is fetched by both forms
            IReadOnlyDictionary<string, NDArray> onlyY = session.Run(inputs, new[] { "Y" });
            using (NDArray y = onlyY["Y"])
                y.ToArray<float>().Should().Equal(0f, 1f, 2f);
            IReadOnlyDictionary<string, NDArray> viewY = session.Run(inputs, new[] { "Y" }, zeroCopyOutputs: true);
            using (NDArray y = viewY["Y"])
                y.ToArray<float>().Should().Equal(0f, 1f, 2f);
        }

        /// <summary>
        ///     An input that needs BOTH a cast and a C-order copy: a transposed float64 array into a float32 input.
        ///     The cast keeps the transposed layout, so a second temporary puts the elements in logical order; the first
        ///     temporary is disposed at once. The model sees the logical array, and every temporary returns to the pool.
        /// </summary>
        [TestMethod]
        public void Run_CastOfATransposedInput_FeedsTheLogicalArray_AndReleasesBothTemporaries()
        {
            using InferenceSession session = OpenSession("matmul_f32");
            using NDArray wide = np.array(new double[,] { { 1, 4 }, { 2, 5 }, { 3, 6 } });   // (3, 2)
            using NDArray a64 = wide.T;                                                        // (2, 3), not C-contiguous
            a64.Shape.IsContiguous.Should().BeFalse("the test needs an input that is neither the declared dtype nor C-contiguous");
            using NDArray a32 = np.array(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });           // the same logical values
            using NDArray b = np.array(new float[,] { { 1, 0 }, { 0, 1 }, { 2, 2 } });        // (3, 2)

            using NDArray expected = session.Run(new Dictionary<string, NDArray> { { "A", a32 }, { "B", b } })["C"];
            var inputs = new Dictionary<string, NDArray> { { "A", a64 }, { "B", b } };
            using (NDArray got = session.Run(inputs, casting: "same_kind")["C"])
            {
                got.shape.Should().Equal(2, 2);
                BytesOf(got).Should().Equal(BytesOf(expected), "the cast input is fed in logical order");
                got.ToArray<float>().Should().Equal(7f, 8f, 16f, 17f);
            }

            Pool.Balance(() => session.Run(inputs, casting: "same_kind")["C"].Dispose())
                .Should().Be(0, "the cast and the C-order copy are both given back");
            new Action(() => session.Run(inputs, casting: "same_kind", mode: OnnxCodecMode.View)).Should().Throw<InvalidOperationException>()
                .WithMessage("*'A'*Double*Single*conversion is a copy*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     The guards of a pre-allocated output that had no test: a null array, an output that is not a tensor, an
        ///     array of a dtype ONNX has no element type for, and a null <c>inputs</c>. An explicit
        ///     <see cref="RunOptions"/> is passed through by <c>RunInto</c> too.
        /// </summary>
        [TestMethod]
        public void RunInto_OutputGuards_NullArray_NonTensorOutput_ComplexArray_AndExplicitRunOptions()
        {
            using NDArray x = Arange(NPTypeCode.Single, 3);
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            using (InferenceSession session = OpenSession("identity_float32"))
            {
                using NDArray y = np.zeros(new Shape(3), typeof(float));
                new Action(() => session.RunInto(null, new Dictionary<string, NDArray> { { "Y", y } })).Should().Throw<ArgumentNullException>("inputs is null");
                new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y", null } }))
                    .Should().Throw<ArgumentNullException>().WithMessage("*output 'Y' is null*");

                using NDArray complex = np.zeros(new Shape(3), typeof(System.Numerics.Complex));
                new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y", complex } }))
                    .Should().Throw<InvalidOperationException>().WithMessage("*'Y' is declared Float*Complex*");

                using var options = new RunOptions { LogId = "run-into" };
                session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y", y } }, options);
                y.ToArray<float>().Should().Equal(0f, 1f, 2f);
            }

            using (InferenceSession session = OpenSession("sequence_out"))
            {
                using NDArray target = np.zeros(new Shape(3), typeof(float));
                new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "seq", target } }))
                    .Should().Throw<NotSupportedException>().WithMessage("output 'seq' is ONNX_TYPE_SEQUENCE, not a tensor*");
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0, "every refusal releases the input handle prepared before it");
        }

        /// <summary>
        ///     A name can be null only in a caller's own read-only dictionary (a <see cref="Dictionary{TKey, TValue}"/>
        ///     refuses a null key). The preparation of an input and of an output refuses it before anything else.
        /// </summary>
        [TestMethod]
        public void NullName_IsRefused_ByTheInputAndTheOutputPreparation()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            new Action(() => InferenceSessionExtensions.PrepareInput(session, null, x, NumSharp.Backends.Iteration.NPY_CASTING.NPY_SAFE_CASTING, "safe", OnnxCodecMode.Auto))
                .Should().Throw<ArgumentNullException>();
            new Action(() => InferenceSessionExtensions.PrepareOutput(session, null, x)).Should().Throw<ArgumentNullException>();
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     A declared dimension with neither a value nor a name (legal ONNX) prints as <c>'?'</c> in a shape error,
        ///     next to the fixed extents: the message still shows the whole declaration. It is free, like a named one.
        /// </summary>
        [TestMethod]
        public void ShapeError_ForAnAnonymousDimension_PrintsTheDeclaration()
        {
            using InferenceSession session = OpenSession("anon_dim");
            using (NDArray wrongWidth = Arange(NPTypeCode.Single, 2, 4))
                new Action(() => session.Run(wrongWidth)).Should().Throw<ArgumentException>()
                    .WithMessage("input 'X' expects shape ('?', 3)*dimension 1 must be 3, not 4*");
            using (NDArray wrongRank = Arange(NPTypeCode.Single, 3))
                new Action(() => session.Run(wrongRank)).Should().Throw<ArgumentException>().WithMessage("input 'X' expects rank 2 ('?', 3)*");

            using NDArray ok = Arange(NPTypeCode.Single, 5, 3);
            using NDArray y = session.Run(ok);
            y.shape.Should().Equal(5, 3);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     BFloat16, the one tensor element type of ONNX Runtime's C# API that NumSharp has no dtype for. A model
        ///     INPUT of it is refused by name before the array is read. A model OUTPUT of it is refused by the import
        ///     after the run (both forms release what they hold), by <c>RunInto</c> as a dtype that no array has, and by
        ///     <c>RunAsync</c> before the run.
        /// </summary>
        [TestMethod]
        public void BFloat16_AsAModelInput_AndAsAModelOutput_IsRefusedByName()
        {
            using NDArray x = Arange(NPTypeCode.Single, 3);
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            using (InferenceSession session = OpenSession("bf16_in"))
            {
                new Action(() => session.Run(x)).Should().Throw<NotSupportedException>().WithMessage("input 'X' is declared BFloat16*");
                Pool.Takes(() => { try { session.Run(x); } catch (NotSupportedException) { } }).Should().Be(0, "the refusal reads the declaration only");
            }

            using (InferenceSession session = OpenSession("bf16_out"))
            {
                new Action(() => session.Run(x)).Should().Throw<NotSupportedException>().WithMessage("*BFloat16*");
                new Action(() => session.Run(inputs, zeroCopyOutputs: true)).Should().Throw<NotSupportedException>().WithMessage("*BFloat16*");
                using NDArray y = np.zeros(new Shape(3), typeof(float));
                new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y", y } }))
                    .Should().Throw<InvalidOperationException>().WithMessage("*'Y' is declared BFloat16*Single*");
                if (InferenceSessionExtensions.IsRunAsyncSupported)
                    new Action(() => session.RunAsync(inputs)).Should().Throw<NotSupportedException>().WithMessage("RunAsync: output 'Y' is declared BFloat16*");
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        /// <summary>
        ///     Two outputs of two dtypes: <c>Run</c> returns each with its own dtype, and <c>RunInto</c> checks each
        ///     pre-allocated array against ITS output's declaration, so two arrays swapped are refused with the name of
        ///     the output that got the wrong one.
        /// </summary>
        [TestMethod]
        public void TwoOutputsOfTwoDtypes_EachIsCheckedAgainstItsOwnDeclaration()
        {
            using InferenceSession session = OpenSession("two_dtypes");
            using NDArray x = np.array(new float[] { 1.5f, -2.5f, 3f });
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            IReadOnlyDictionary<string, NDArray> results = session.Run(inputs);
            using (NDArray yf = results["Yf"])
            using (NDArray yi = results["Yi"])
            {
                yf.typecode.Should().Be(NPTypeCode.Single);
                yf.ToArray<float>().Should().Equal(1.5f, -2.5f, 3f);
                yi.typecode.Should().Be(NPTypeCode.Int32);
                yi.ToArray<int>().Should().Equal(1, -2, 3);
            }

            using NDArray floats = np.zeros(new Shape(3), typeof(float));
            using NDArray ints = np.zeros(new Shape(3), typeof(int));
            new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "Yf", ints }, { "Yi", floats } }))
                .Should().Throw<InvalidOperationException>().WithMessage("*'Yf' is declared Float*Int32*");
            session.RunInto(inputs, new Dictionary<string, NDArray> { { "Yf", floats }, { "Yi", ints } });
            floats.ToArray<float>().Should().Equal(1.5f, -2.5f, 3f);
            ints.ToArray<int>().Should().Equal(1, -2, 3);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }
    }
}
