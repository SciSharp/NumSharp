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
    ///     The <c>Run</c> tier's rules that are not about one dtype or one layout: the input mode
    ///     (<see cref="OnnxCodecMode"/>) on every form, the casting rule's validation, inputs declared optional,
    ///     overridable initializers, and what <c>RunInto</c> does when an output shares memory with an input or with
    ///     another output.
    /// </summary>
    [TestClass]
    public class SessionRunModeTests : OnnxTestBase
    {
        /// <summary>
        ///     The rule name is checked on EVERY call, by every form. It was checked only when some input needed a cast,
        ///     so <c>casting: "sfae"</c> ran without complaint until the first input of another dtype. The names are
        ///     NumPy's and case-sensitive; <c>null</c> is not a rule.
        /// </summary>
        [TestMethod]
        public void CastingRule_IsValidatedOnEveryCall_EvenWhenNoInputNeedsACast()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray x = Arange(NPTypeCode.Single, 3);   // the declared dtype: no cast is needed
            using NDArray y = np.zeros(new Shape(3), typeof(float));
            var inputs = new Dictionary<string, NDArray> { { "X", x } };
            var outputs = new Dictionary<string, NDArray> { { "Y", y } };

            foreach (string bad in new[] { "bogus", "Safe", "same-kind", "", null })
            {
                string why = bad ?? "<null>";
                new Action(() => session.Run(x, casting: bad)).Should().Throw<ValueError>(why)
                    .WithMessage("casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '*')");
                new Action(() => session.Run("X", x, "Y", casting: bad)).Should().Throw<ValueError>(why);
                new Action(() => session.Run(inputs, casting: bad)).Should().Throw<ValueError>(why);
                new Action(() => session.RunInto(inputs, outputs, casting: bad)).Should().Throw<ValueError>(why);
            }

            foreach (string good in new[] { "no", "equiv", "safe", "same_kind", "unsafe" })
                using (NDArray ok = session.Run(x, casting: good))
                    ok.GetSingle(2).Should().Be(2f, good);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>A value that is not a member of <see cref="OnnxCodecMode"/> is refused by every form, before any input is read.</summary>
        [TestMethod]
        public void UndefinedMode_IsRefused_ByEveryForm()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            using NDArray y = np.zeros(new Shape(3), typeof(float));
            var inputs = new Dictionary<string, NDArray> { { "X", x } };
            var bad = (OnnxCodecMode)42;
            new Action(() => session.Run(x, mode: bad)).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => session.Run("X", x, "Y", mode: bad)).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => session.Run(inputs, mode: bad)).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y", y } }, mode: bad)).Should().Throw<ArgumentOutOfRangeException>();
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     The input mode, measured by the buffers a run takes from NumSharp's pool (a pre-allocated output, so the
        ///     output costs none). <see cref="OnnxCodecMode.View"/> and <see cref="OnnxCodecMode.Auto"/> share an input ORT
        ///     can read in place: 0 buffers. <see cref="OnnxCodecMode.Copy"/> feeds a copy of it: 1 buffer.
        ///     <see cref="OnnxCodecMode.Auto"/> copies a transposed input or casts an input of another dtype (1 buffer, or
        ///     2 for both), and <see cref="OnnxCodecMode.View"/> refuses each of those with the reason. The result is the
        ///     same in every mode that runs.
        /// </summary>
        [TestMethod]
        public void InputMode_View_ProvesNoCopy_Copy_AlwaysCopies_Auto_CopiesOnlyWhenItMust()
        {
            using InferenceSession f32 = OpenSession("identity_float32");
            using InferenceSession i64 = OpenSession("identity_int64");
            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            using NDArray transposedBase = Arange(NPTypeCode.Single, 3, 2);
            using NDArray transposed = transposedBase.T;   // (2,3), not C-contiguous
            using NDArray y = np.zeros(new Shape(2, 3), typeof(float));
            var outputs = new Dictionary<string, NDArray> { { "Y", y } };

            long Takes(NDArray input, OnnxCodecMode mode)
            {
                var inputs = new Dictionary<string, NDArray> { { "X", input } };
                f32.RunInto(inputs, outputs, mode: mode);   // unmeasured first call (static caches)
                return Pool.Takes(() => f32.RunInto(inputs, outputs, mode: mode));
            }

            Takes(x, OnnxCodecMode.View).Should().Be(0, "View shares the input");
            BytesOf(y).Should().Equal(BytesOf(x));
            Takes(x, OnnxCodecMode.Auto).Should().Be(0, "Auto shares an input ORT can read in place");
            Takes(x, OnnxCodecMode.Copy).Should().Be(1, "Copy feeds a copy of every input");
            BytesOf(y).Should().Equal(BytesOf(x), "the copy holds the same values");

            Takes(transposed, OnnxCodecMode.Auto).Should().Be(1, "Auto feeds a C-order copy of a view ORT cannot read");
            BytesOf(y).Should().Equal(BytesOf(transposed), "in logical order");
            Takes(transposed, OnnxCodecMode.Copy).Should().Be(1, "one copy serves both needs");
            new Action(() => f32.RunInto(new Dictionary<string, NDArray> { { "X", transposed } }, outputs, mode: OnnxCodecMode.View))
                .Should().Throw<InvalidOperationException>().WithMessage("*OnnxCodecMode.View*'X'*not C-contiguous*OnnxCodecMode.Auto*");

            // a cast: int32 into the int64 model
            using NDArray ints = Arange(NPTypeCode.Int32, 2, 3);
            using NDArray intsTransposedBase = Arange(NPTypeCode.Int32, 3, 2);
            using NDArray intsTransposed = intsTransposedBase.T;
            using NDArray wide = np.zeros(new Shape(2, 3), typeof(long));
            var wideOutputs = new Dictionary<string, NDArray> { { "Y", wide } };
            i64.RunInto(new Dictionary<string, NDArray> { { "X", ints } }, wideOutputs);
            Pool.Takes(() => i64.RunInto(new Dictionary<string, NDArray> { { "X", ints } }, wideOutputs)).Should().Be(1, "the cast is the one temporary");
            wide.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            Pool.Takes(() => i64.RunInto(new Dictionary<string, NDArray> { { "X", ints } }, wideOutputs, mode: OnnxCodecMode.Copy))
                .Should().Be(1, "a cast is already a copy: Copy makes no second one");
            Pool.Takes(() => i64.RunInto(new Dictionary<string, NDArray> { { "X", intsTransposed } }, wideOutputs))
                .Should().Be(2, "a cast of a transposed input: the conversion keeps the memory order, then the C-order pass");
            wide.ToArray<long>().Should().Equal(0L, 2L, 4L, 1L, 3L, 5L);
            new Action(() => i64.RunInto(new Dictionary<string, NDArray> { { "X", ints } }, wideOutputs, mode: OnnxCodecMode.View))
                .Should().Throw<InvalidOperationException>().WithMessage("*OnnxCodecMode.View*'X'*Int32*Int64*conversion is a copy*");

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     The defect behind the overlap rule: ORT writes an output while it still reads its inputs, so a MatMul
        ///     written into one of its own inputs gave wrong values (3,072 of 4,096 elements for 64×64). Under the default
        ///     mode the overlapped input is fed as a copy and the result is the one a separate output array gets; the
        ///     strict mode refuses instead of copying; the input that does not overlap is still shared.
        /// </summary>
        [TestMethod]
        public void RunInto_OutputThatIsAnInput_ComputesFromACopyUnderAuto_AndIsRefusedUnderView()
        {
            using InferenceSession session = OpenSession("matmul_f32");
            np.random.seed(11);
            using NDArray a = np.random.randn(64, 64).astype(NPTypeCode.Single);
            using NDArray b = np.random.randn(64, 64).astype(NPTypeCode.Single);
            using NDArray expected = np.zeros(new Shape(64, 64), typeof(float));
            var inputs = new Dictionary<string, NDArray> { { "A", a }, { "B", b } };
            session.RunInto(inputs, new Dictionary<string, NDArray> { { "C", expected } });

            new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "C", a } }, mode: OnnxCodecMode.View))
                .Should().Throw<InvalidOperationException>().WithMessage("*OnnxCodecMode.View*input 'A'*output 'C'*overlap*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the refused run released every handle it had made");

            long takes = Pool.Takes(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "C", a } }));
            takes.Should().Be(1, "only the overlapped input is copied; B is still shared");
            BytesOf(a).Should().Equal(BytesOf(expected), "A @ B written into A is the product of the ORIGINAL A");

            // the other operand as the output, and an explicit Copy
            using NDArray a2 = np.random.randn(64, 64).astype(NPTypeCode.Single);
            using NDArray b2 = np.random.randn(64, 64).astype(NPTypeCode.Single);
            var inputs2 = new Dictionary<string, NDArray> { { "A", a2 }, { "B", b2 } };
            session.RunInto(inputs2, new Dictionary<string, NDArray> { { "C", expected } });
            session.RunInto(inputs2, new Dictionary<string, NDArray> { { "C", b2 } }, mode: OnnxCodecMode.Copy);
            BytesOf(b2).Should().Equal(BytesOf(expected));
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     Overlap is decided on the bytes ORT really reads and writes: an input that overlaps the output only in part
        ///     is copied too, an input that is already fed as a temporary (a cast, a C-order copy) is never counted, a
        ///     neighbouring window that shares no byte is still shared, and an empty array overlaps nothing.
        /// </summary>
        [TestMethod]
        public void RunInto_OverlapIsDecidedOnTheBytesOrtReadsAndWrites()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray buffer = Arange(NPTypeCode.Single, 8);

            // partial overlap: input = buffer[0:4], output = buffer[2:6]
            using (NDArray input = buffer["0:4"])
            using (NDArray output = buffer["2:6"])
            {
                var inputs = new Dictionary<string, NDArray> { { "X", input } };
                var outputs = new Dictionary<string, NDArray> { { "Y", output } };
                new Action(() => session.RunInto(inputs, outputs, mode: OnnxCodecMode.View)).Should().Throw<InvalidOperationException>().WithMessage("*overlap*");
                session.RunInto(inputs, outputs);
                buffer.ToArray<float>().Should().Equal(new[] { 0f, 1f, 0f, 1f, 2f, 3f, 6f, 7f }, "the output window got the input's ORIGINAL values");
            }

            // neighbours that share no byte: input = buffer[0:4], output = buffer[4:8] — still zero-copy, also under View
            using (NDArray input = buffer["0:4"])
            using (NDArray output = buffer["4:8"])
            {
                var inputs = new Dictionary<string, NDArray> { { "X", input } };
                var outputs = new Dictionary<string, NDArray> { { "Y", output } };
                session.RunInto(inputs, outputs, mode: OnnxCodecMode.View);
                Pool.Takes(() => session.RunInto(inputs, outputs, mode: OnnxCodecMode.View)).Should().Be(0, "adjacent windows do not overlap");
                buffer.ToArray<float>().Should().Equal(new[] { 0f, 1f, 0f, 1f, 0f, 1f, 0f, 1f });
            }

            // an input that is fed as a temporary anyway (reversed) is not an overlap, even over the output's own bytes
            using (NDArray warm = Arange(NPTypeCode.Single, 4))
            using (NDArray warmReversed = warm["::-1"])
                session.RunInto(new Dictionary<string, NDArray> { { "X", warmReversed } }, new Dictionary<string, NDArray> { { "Y", warm } });   // unmeasured first use of this path
            using (NDArray ramp = Arange(NPTypeCode.Single, 4))
            using (NDArray reversed = ramp["::-1"])
            {
                var inputs = new Dictionary<string, NDArray> { { "X", reversed } };
                var outputs = new Dictionary<string, NDArray> { { "Y", ramp } };
                Pool.Takes(() => session.RunInto(inputs, outputs)).Should().Be(1, "one C-order copy, no second copy for the overlap");
                ramp.ToArray<float>().Should().Equal(3f, 2f, 1f, 0f);
            }

            // empty arrays occupy no byte
            using (NDArray empty = np.zeros(new Shape(0, 3), typeof(float)))
            {
                var same = new Dictionary<string, NDArray> { { "X", empty } };
                session.RunInto(same, new Dictionary<string, NDArray> { { "Y", empty } }, mode: OnnxCodecMode.View);
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     Two outputs in the same memory are refused in every mode: ORT would write both results into the same bytes,
        ///     and no copy of an input makes that well defined. Disjoint windows of one buffer are two outputs.
        /// </summary>
        [TestMethod]
        public void RunInto_TwoOutputsThatOverlap_AreRefused_InEveryMode()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            using NDArray buffer = np.zeros(new Shape(6), typeof(float));
            var inputs = new Dictionary<string, NDArray> { { "X", x } };

            using (NDArray whole = buffer["0:3"])
            using (NDArray shifted = buffer["1:4"])
            {
                foreach (OnnxCodecMode mode in new[] { OnnxCodecMode.Auto, OnnxCodecMode.View, OnnxCodecMode.Copy })
                {
                    new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y1", whole }, { "Y2", whole } }, mode: mode))
                        .Should().Throw<ArgumentException>($"{mode}: the same array twice").WithMessage("*'Y1'*'Y2'*overlap*");
                    new Action(() => session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y1", whole }, { "Y2", shifted } }, mode: mode))
                        .Should().Throw<ArgumentException>($"{mode}: windows that share 2 elements").WithMessage("*'Y1'*'Y2'*overlap*");
                }
            }
            buffer.ToArray<float>().Should().Equal(new float[6], "a refused run wrote nothing");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);

            using (NDArray first = buffer["0:3"])
            using (NDArray second = buffer["3:6"])
                session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y1", first }, { "Y2", second } });
            buffer.ToArray<float>().Should().Equal(0f, 1f, 2f, -0f, -1f, -2f);
        }

        /// <summary>
        ///     An input the model declares <c>optional(tensor(float))</c> takes a plain array, with the same dtype
        ///     coercion and shape handling as a tensor input. It was refused with "not a tensor; feed it with the
        ///     OrtValue API (OrtValue.CreateSequence / CreateMap)", and neither of those factories makes an optional.
        /// </summary>
        [TestMethod]
        public void OptionalInput_TakesAPlainArray_WithTheUsualCoercion()
        {
            using InferenceSession session = OpenSession("optional_input");
            session.InputMetadata["X"].OnnxValueType.Should().Be(OnnxValueType.ONNX_TYPE_OPTIONAL, "the model really declares an optional");

            using NDArray x = Arange(NPTypeCode.Single, 2, 3);
            using (NDArray y = session.Run(x))
            {
                y.typecode.Should().Be(NPTypeCode.Single);
                BytesOf(y).Should().Equal(BytesOf(x));
            }

            // the element declaration drives the dtype rule: int16 -> float32 is a safe cast, float64 -> float32 is not
            using NDArray shorts = Arange(NPTypeCode.Int16, 4);
            using (NDArray y = session.Run(shorts))
                y.ToArray<float>().Should().Equal(0f, 1f, 2f, 3f);
            using NDArray doubles = Arange(NPTypeCode.Double, 4);
            new Action(() => session.Run(doubles)).Should().Throw<InvalidCastException>().WithMessage("*'X'*Double*Single*");
            using (NDArray y = session.Run(doubles, casting: "same_kind"))
                y.typecode.Should().Be(NPTypeCode.Single);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     An overridable initializer (an initializer the model also lists as a graph input) can be fed like an input:
        ///     the run uses the fed value, and the model's own value when none is fed. It was refused with "the model has
        ///     no input named 'w'", because only the session's input metadata was searched. Its declared shape is checked,
        ///     and an unknown name lists both kinds of names.
        /// </summary>
        [TestMethod]
        public void OverridableInitializer_CanBeFed_AndDefaultsWhenOmitted()
        {
            using InferenceSession session = OpenSession("overridable_init");
            session.InputNames.Should().Equal("x");
            session.OverridableInitializerMetadata.Keys.Should().Equal("w");

            using NDArray x = np.array(new[] { 1f, 2f, 3f });
            using (NDArray y = session.Run(x))
                y.ToArray<float>().Should().Equal(new[] { 11f, 22f, 33f }, "the model's own w = [10, 20, 30]");

            using NDArray ones = np.ones(new Shape(3), typeof(float));
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(new Dictionary<string, NDArray> { { "x", x }, { "w", ones } });
            using (NDArray y = outputs["y"])
                y.ToArray<float>().Should().Equal(new[] { 2f, 3f, 4f }, "w replaced for this run");

            // it is validated like an input: its declared shape, and the name against both kinds of names
            using NDArray wrongShape = np.ones(new Shape(4), typeof(float));
            new Action(() => session.Run(new Dictionary<string, NDArray> { { "x", x }, { "w", wrongShape } }))
                .Should().Throw<ArgumentException>().WithMessage("*'w'*(3)*dimension 0 must be 3, not 4*");
            new Action(() => session.Run(new Dictionary<string, NDArray> { { "x", x }, { "v", ones } }))
                .Should().Throw<ArgumentException>().WithMessage("*no input named 'v'*inputs are [x]*overridable initializers are [w]*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     <c>session.Run(inputs, null)</c> compiles and means "all outputs". With the pre-allocated-output form named
        ///     <c>Run</c> too, a <c>null</c> second argument matched both dictionary forms and the call did not compile
        ///     (CS0121). The form with outputs is <c>RunInto</c> now.
        /// </summary>
        [TestMethod]
        public void Run_WithANullSecondArgument_Compiles_AndFetchesEveryOutput()
        {
            using InferenceSession session = OpenSession("two_outputs_f32");
            using NDArray x = Arange(NPTypeCode.Single, 3);
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(new Dictionary<string, NDArray> { { "X", x } }, null);
            outputs.Keys.Should().BeEquivalentTo("Y1", "Y2");
            foreach (NDArray output in outputs.Values)
                output.Dispose();
        }
    }
}
