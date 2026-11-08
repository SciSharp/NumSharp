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
    ///     A model OUTPUT that no node produces: it carries the name of a model input, or of an overridable initializer,
    ///     and its value is the value the caller FEEDS. ONNX Runtime puts the fed tensor itself into the output slot, so
    ///     a buffer the caller pre-allocates for it is never written with the fed value, and a value ORT returns for it
    ///     lies in the fed tensor's memory, which is NumSharp's.
    ///     <para>Models: <c>passthrough.onnx</c> (X is an input and an output; Y = Neg(X)) and
    ///     <c>overridable_out.onnx</c> (w is an overridable initializer and an output; y = x + w).</para>
    ///     <para>Before the fix: <c>RunInto</c> left the array for X as it was (or gave w the DEFAULT while the run used
    ///     the fed value), and <c>Run(zeroCopyOutputs: true)</c> returned a view of the fed memory that held no
    ///     reference on it: with an input fed as a temporary it read the next array that took the temporary's block.</para>
    /// </summary>
    [TestClass]
    public class PassThroughOutputTests : OnnxTestBase
    {
        private const float Sentinel = -777f;

        /// <summary>The address of an array's first element.</summary>
        /// <param name="a">Any array.</param>
        /// <returns>The address as an integer, for comparisons.</returns>
        private static unsafe long Address(NDArray a) => (long)((byte*)a.Storage.Address + a.Shape.Offset * a.dtypesize);

        /// <summary>A 1-D float32 array filled with the sentinel a run must overwrite.</summary>
        /// <param name="length">The element count.</param>
        /// <returns>A new owning array.</returns>
        private static NDArray Sentinels(int length)
        {
            var values = new float[length];
            Array.Fill(values, Sentinel);
            return np.array(values);
        }

        /// <summary>
        ///     <c>RunInto</c> writes the fed values into the array pre-allocated for an output that is a fed input, for an
        ///     input ORT reads in place, for one fed as a cast copy, for a stepped view, under every input mode that can
        ///     feed it, and for an empty tensor. The ordinary output next to it is written by ORT as before.
        /// </summary>
        [TestMethod]
        public void RunInto_OutputThatIsAFedInput_ReceivesTheFedValues()
        {
            using InferenceSession session = OpenSession("passthrough");

            using (NDArray x = np.array(new float[] { 1, 2, 3 }))
            using (NDArray outX = Sentinels(3))
            using (NDArray outY = Sentinels(3))
            {
                foreach (OnnxCodecMode mode in new[] { OnnxCodecMode.Auto, OnnxCodecMode.View, OnnxCodecMode.Copy })
                {
                    outX.fill(Sentinel);
                    outY.fill(Sentinel);
                    session.RunInto(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "X", outX }, { "Y", outY } }, mode: mode);
                    outX.ToArray<float>().Should().Equal(new float[] { 1, 2, 3 }, $"{mode}: ONNX Runtime does not write this array, so the bridge does");
                    outY.ToArray<float>().Should().Equal(new float[] { -1, -2, -3 }, $"{mode}: an ordinary output");
                    x.ToArray<float>().Should().Equal(new float[] { 1, 2, 3 }, $"{mode}: the input is only read");
                }
            }

            // a float64 array into the float32 input: the output receives the CAST values, the ones the model ran on
            using (NDArray x64 = np.array(new double[] { 1.5, 2.5, 3.5 }))
            using (NDArray outX = Sentinels(3))
            using (NDArray outY = Sentinels(3))
            {
                session.RunInto(new Dictionary<string, NDArray> { { "X", x64 } }, new Dictionary<string, NDArray> { { "X", outX }, { "Y", outY } }, casting: "same_kind");
                outX.ToArray<float>().Should().Equal(new float[] { 1.5f, 2.5f, 3.5f });
                outY.ToArray<float>().Should().Equal(new float[] { -1.5f, -2.5f, -3.5f });
            }

            // a stepped view: fed as a C-order copy, and the output receives its logical elements
            using (NDArray wide = np.array(new float[] { 9, 0, 8, 0, 7, 0 }))
            using (NDArray stepped = wide["::2"])
            using (NDArray outX = Sentinels(3))
            using (NDArray outY = Sentinels(3))
            {
                session.RunInto(new Dictionary<string, NDArray> { { "X", stepped } }, new Dictionary<string, NDArray> { { "X", outX }, { "Y", outY } });
                outX.ToArray<float>().Should().Equal(new float[] { 9, 8, 7 });
            }

            // an empty tensor: nothing to move, and no error
            using (NDArray none = np.zeros(new Shape(0), typeof(float)))
            using (NDArray outX = np.zeros(new Shape(0), typeof(float)))
            using (NDArray outY = np.zeros(new Shape(0), typeof(float)))
                session.RunInto(new Dictionary<string, NDArray> { { "X", none } }, new Dictionary<string, NDArray> { { "X", outX }, { "Y", outY } });

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     ONNX Runtime checks no shape for the array of an output that is a fed input (it never touches it), so the
        ///     bridge does, before the run: an array of another shape is refused with both shapes, and nothing ran.
        /// </summary>
        [TestMethod]
        public void RunInto_PassThroughOutput_OfAnotherShape_IsRefusedBeforeTheRun()
        {
            using InferenceSession session = OpenSession("passthrough");
            using NDArray x = np.array(new float[] { 1, 2, 3 });
            using NDArray outY = Sentinels(3);
            using NDArray tooLong = Sentinels(4);
            new Action(() => session.RunInto(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "X", tooLong }, { "Y", outY } }))
                .Should().Throw<ArgumentException>().WithMessage("*'X'*passed through*(3)*(4)*");
            outY.ToArray<float>().Should().Equal(new[] { Sentinel, Sentinel, Sentinel }, "the refusal comes before the run");
            tooLong.ToArray<float>().Should().Equal(new[] { Sentinel, Sentinel, Sentinel, Sentinel });
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "a refused run releases every handle it prepared");
        }

        /// <summary>
        ///     The same array as the input and as its own pass-through output: under the default mode the overlap rule
        ///     feeds a copy and the array ends with its own values; the strict mode refuses the overlap, as for any
        ///     input that shares memory with an output.
        /// </summary>
        [TestMethod]
        public void RunInto_TheSameArrayAsInputAndPassThroughOutput_KeepsItsValues()
        {
            using InferenceSession session = OpenSession("passthrough");
            using NDArray x = np.array(new float[] { 1, 2, 3 });
            using NDArray outY = Sentinels(3);
            session.RunInto(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "X", x }, { "Y", outY } });
            x.ToArray<float>().Should().Equal(new float[] { 1, 2, 3 });
            outY.ToArray<float>().Should().Equal(new float[] { -1, -2, -3 });

            new Action(() => session.RunInto(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "X", x }, { "Y", outY } },
                    mode: OnnxCodecMode.View))
                .Should().Throw<InvalidOperationException>().WithMessage("*'X'*overlap*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>The copy form returns an independent, owning copy of the fed values for a pass-through output, by every form of <c>Run</c>.</summary>
        [TestMethod]
        public void Run_CopyOutputs_PassThroughOutput_IsAnIndependentCopy()
        {
            using InferenceSession session = OpenSession("passthrough");
            using NDArray x = np.array(new float[] { 1, 2, 3 });
            IReadOnlyDictionary<string, NDArray> results = session.Run(new Dictionary<string, NDArray> { { "X", x } });
            using (NDArray copyX = results["X"])
            using (NDArray y = results["Y"])
            {
                copyX.flags.owndata.Should().BeTrue();
                copyX.flags.writeable.Should().BeTrue();
                Address(copyX).Should().NotBe(Address(x));
                copyX.ToArray<float>().Should().Equal(new float[] { 1, 2, 3 });
                y.ToArray<float>().Should().Equal(new float[] { -1, -2, -3 });
                x.SetAtIndex<float>(99f, 0);
                copyX.GetSingle(0).Should().Be(1f, "a copy does not follow the input");
                x.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("a copy holds no reference on the input's buffer");
            }

            using (NDArray named = session.Run("X", x, "X"))
                named.ToArray<float>().Should().Equal(new float[] { 99, 2, 3 }, "the named form fetches the input passed through");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        /// <summary>
        ///     With zero-copy outputs, a pass-through output of an input ORT read in place is a NumSharp view of the
        ///     caller's input array that holds its own reference: the two show each other's writes, and the view stays
        ///     valid after the input array is disposed. It is no import lease (ORT owns nothing here).
        /// </summary>
        [TestMethod]
        public void Run_ZeroCopyOutputs_PassThroughOfASharedInput_IsACountedViewOfTheInput()
        {
            using InferenceSession session = OpenSession("passthrough");
            NDArray x = np.array(new float[] { 1, 2, 3 });
            IReadOnlyDictionary<string, NDArray> results = session.Run(new Dictionary<string, NDArray> { { "X", x } }, zeroCopyOutputs: true);
            NDArray view = results["X"];
            try
            {
                using (NDArray y = results["Y"])
                {
                    y.ToArray<float>().Should().Equal(new float[] { -1, -2, -3 });
                    NDArrayOnnxInterop.LiveImports.Should().Be(1, "only Y is a lease over a buffer ORT owns");
                }

                NDArrayOnnxInterop.LiveImports.Should().Be(0, "the pass-through view is ordinary NumSharp memory, not a lease");
                NDArrayOnnxInterop.LiveExports.Should().Be(0, "the run's input handle is disposed");
                Address(view).Should().Be(Address(x), "zero-copy: the output IS the fed memory");
                view.flags.owndata.Should().BeFalse();
                view.flags.writeable.Should().BeTrue();
                x.Storage.InternalArray.IsUniquelyReferenced.Should().BeFalse("the view holds a reference on the input's buffer");

                x.SetAtIndex<float>(99f, 0);
                view.GetSingle(0).Should().Be(99f, "the view reads the input's memory");
                view.SetAtIndex<float>(5f, 1);
                x.GetSingle(1).Should().Be(5f, "and writes it");

                // The input array goes first. The view's reference keeps the buffer out of the pool.
                x.Dispose();
                var churn = new List<NDArray>();
                for (int i = 0; i < 256; i++)
                {
                    NDArray fresh = np.array(new float[] { 555, 555, 555 });
                    Address(fresh).Should().NotBe(Address(view), "the pool must not hand out a buffer a view still references");
                    churn.Add(fresh);
                }

                view.ToArray<float>().Should().Equal(new float[] { 99, 5, 3 });
                foreach (NDArray fresh in churn)
                    fresh.Dispose();
            }
            finally
            {
                view.Dispose();
                x.Dispose();
            }
        }

        /// <summary>
        ///     With zero-copy outputs, a pass-through output of an input that needed a TEMPORARY (a cast) keeps the
        ///     temporary alive: the run disposes its handle, the view's own reference holds the buffer. The view used to
        ///     read whatever array took the temporary's block next.
        /// </summary>
        [TestMethod]
        public void Run_ZeroCopyOutputs_PassThroughOfATemporary_KeepsTheTemporaryAlive()
        {
            using InferenceSession session = OpenSession("passthrough");
            using NDArray x64 = np.array(new double[] { 1, 2, 3 });
            IReadOnlyDictionary<string, NDArray> results = session.Run(new Dictionary<string, NDArray> { { "X", x64 } }, casting: "same_kind", zeroCopyOutputs: true);
            using NDArray view = results["X"];
            using NDArray y = results["Y"];
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the run disposed the handle that owned the temporary");
            view.typecode.Should().Be(NPTypeCode.Single, "the output has the model's dtype, not the fed array's");
            view.flags.writeable.Should().BeTrue("the temporary is the bridge's own copy");

            var churn = new List<NDArray>();
            for (int i = 0; i < 256; i++)
            {
                NDArray fresh = np.array(new float[] { 555, 555, 555 });
                Address(fresh).Should().NotBe(Address(view), "the temporary's buffer is still referenced by the view");
                churn.Add(fresh);
            }

            view.ToArray<float>().Should().Equal(new float[] { 1, 2, 3 });
            y.ToArray<float>().Should().Equal(new float[] { -1, -2, -3 });
            foreach (NDArray fresh in churn)
                fresh.Dispose();
        }

        /// <summary>A zero-copy pass-through output of a read-only input is read-only: the run must not hand out a writeable view of memory its owner protected.</summary>
        [TestMethod]
        public void Run_ZeroCopyOutputs_PassThroughOfAReadOnlyInput_IsReadOnly()
        {
            using InferenceSession session = OpenSession("passthrough");
            using NDArray x = np.array(new float[] { 1, 2, 3 });
            x.setflags(write: false);
            IReadOnlyDictionary<string, NDArray> results = session.Run(new Dictionary<string, NDArray> { { "X", x } }, zeroCopyOutputs: true);
            using NDArray view = results["X"];
            using NDArray y = results["Y"];
            Address(view).Should().Be(Address(x));
            view.flags.writeable.Should().BeFalse();
            view.ToArray<float>().Should().Equal(new float[] { 1, 2, 3 });
        }

        /// <summary>
        ///     An output that is an overridable initializer. Not fed: every form returns the default value, which ORT
        ///     copies (a write through a zero-copy view of it does not reach the session's weight). Fed: every form
        ///     returns the FED value; <c>RunInto</c> used to leave the default in the pre-allocated array while the run
        ///     computed with the fed one.
        /// </summary>
        [TestMethod]
        public void OverridableInitializerOutput_IsTheFedValueWhenFed_AndTheDefaultWhenNot()
        {
            using InferenceSession session = OpenSession("overridable_out");
            using NDArray x = np.array(new float[] { 1, 2, 3 });
            using NDArray w = np.array(new float[] { 100, 200, 300 });
            var withDefault = new Dictionary<string, NDArray> { { "x", x } };
            var withFed = new Dictionary<string, NDArray> { { "x", x }, { "w", w } };

            // ---- not fed: the default
            using (NDArray outW = Sentinels(3))
            using (NDArray outY = Sentinels(3))
            {
                session.RunInto(withDefault, new Dictionary<string, NDArray> { { "w", outW }, { "y", outY } });
                outW.ToArray<float>().Should().Equal(new float[] { 10, 20, 30 });
                outY.ToArray<float>().Should().Equal(new float[] { 11, 22, 33 });
            }

            IReadOnlyDictionary<string, NDArray> views = session.Run(withDefault, zeroCopyOutputs: true);
            using (NDArray viewW = views["w"])
            using (NDArray viewY = views["y"])
            {
                viewW.ToArray<float>().Should().Equal(new float[] { 10, 20, 30 });
                viewW.SetAtIndex<float>(1000f, 0);   // ORT's own copy of the weight, not the weight
            }

            IReadOnlyDictionary<string, NDArray> again = session.Run(withDefault);
            using (NDArray againW = again["w"])
            using (NDArray againY = again["y"])
            {
                againW.ToArray<float>().Should().Equal(new float[] { 10, 20, 30 }, "the session's weight is intact");
                againY.ToArray<float>().Should().Equal(new float[] { 11, 22, 33 });
            }

            // ---- fed: the fed value, in every form
            using (NDArray outW = Sentinels(3))
            using (NDArray outY = Sentinels(3))
            {
                session.RunInto(withFed, new Dictionary<string, NDArray> { { "w", outW }, { "y", outY } });
                outW.ToArray<float>().Should().Equal(new float[] { 100, 200, 300 }, "the output is the fed weight, not the default");
                outY.ToArray<float>().Should().Equal(new float[] { 101, 202, 303 });
            }

            IReadOnlyDictionary<string, NDArray> copies = session.Run(withFed);
            using (NDArray copyW = copies["w"])
            using (NDArray copyY = copies["y"])
            {
                copyW.ToArray<float>().Should().Equal(new float[] { 100, 200, 300 });
                Address(copyW).Should().NotBe(Address(w));
            }

            IReadOnlyDictionary<string, NDArray> fedViews = session.Run(withFed, zeroCopyOutputs: true);
            using (NDArray viewW = fedViews["w"])
            using (NDArray viewY = fedViews["y"])
            {
                Address(viewW).Should().Be(Address(w), "the fed weight passed through");
                w.Storage.InternalArray.IsUniquelyReferenced.Should().BeFalse("the view holds a reference on the fed array's buffer");
                viewY.ToArray<float>().Should().Equal(new float[] { 101, 202, 303 });
            }

            w.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("the reference is given back with the view");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     A TEXT input passed through to an output is not the fed-memory case: the string tensor the bridge fed is
        ///     ONNX Runtime's own object, with no NumSharp buffer behind it. Both forms of <c>Run</c> read it as text (a
        ///     copy, also with zero-copy outputs), and the numeric output next to it is unaffected.
        /// </summary>
        [TestMethod]
        public void Run_PassThroughOfATextInput_IsReadAsText()
        {
            using InferenceSession session = OpenSession("passthrough_text");
            using NDArray text = np.array(new[] { "a", "bb", "ccc" });
            using NDArray x = np.array(new float[] { 1, 2, 3 });
            var inputs = new Dictionary<string, NDArray> { { "text", text }, { "x", x } };

            foreach (bool zeroCopy in new[] { false, true })
            {
                IReadOnlyDictionary<string, NDArray> results = session.Run(inputs, zeroCopyOutputs: zeroCopy);
                using NDArray echo = results["text"];
                using NDArray y = results["y"];
                echo.typecode.Should().Be(NPTypeCode.String, $"zeroCopyOutputs: {zeroCopy}");
                echo.flags.owndata.Should().BeTrue("text is always a copy");
                NDArray.AsStringArray(echo).Should().Equal(new[] { "a", "bb", "ccc" }, $"zeroCopyOutputs: {zeroCopy}");
                y.ToArray<float>().Should().Equal(new float[] { -1, -2, -3 });
            }

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }
    }
}
