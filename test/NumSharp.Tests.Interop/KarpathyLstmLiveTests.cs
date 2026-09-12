using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist.Karpathy;

namespace NumSharp.Tests.Interop;

[TestClass]
public class KarpathyLstmLiveTests : InteropTestBase
{
    [DataTestMethod]
    [DataRow(false, false)]
    [DataRow(true, false)]
    [DataRow(true, true)]
    [TestCategory("KarpathyByteParity")]
    public void SourceSizedForwardAllIntermediatesAndFourGradients_ExactLiveNumpy(bool suppliedStates, bool reversedViews)
    {
        using var scope = NDScope.Open();
        var rng = np.random.RandomState(7);
        var w = BatchedLstm.Initialize(10, 4);
        var x = rng.randn(5, 3, 10); var c0 = rng.randn(3, 4); var h0 = rng.randn(3, 4);
        var dh = rng.randn(5, 3, 4); var dcn = rng.randn(3, 4); var dhn = rng.randn(3, 4);
        if (reversedViews) { x = x["::-1,:,::-1"]; w = w["::-1,:"]; dh = dh["::-1,:,:" ]; }
        using var forward = BatchedLstm.Forward(x, w, suppliedStates ? c0 : null, suppliedStates ? h0 : null);
        using var gradients = BatchedLstm.Backward(dh, forward, suppliedStates ? dcn : null, suppliedStates ? dhn : null);
        ExportTo("x", x); ExportTo("w", w); ExportTo("c0", c0); ExportTo("h0", h0);
        ExportTo("dh", dh); ExportTo("dcn", dcn); ExportTo("dhn", dhn);
        PyExec(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "KarpathyLstmReference.py")));
        PyExec(suppliedStates ? "f=forward(x,w,c0,h0)\ng=backward(dh,f,dcn,dhn)" : "f=forward(x,w)\ng=backward(dh,f)");
        var fields = new (string Expression, NDArray Value)[] {
            ("f['h']",forward.Hidden),("f['c']",forward.Cell),("f['ct']",forward.CellTanh),
            ("f['gates']",forward.GateInputs),("f['activated']",forward.Gates),("f['hin']",forward.Concatenated),
            ("g['dx']",gradients.Input),("g['dw']",gradients.Weights),
            ("g['dc0']",gradients.InitialCell),("g['dh0']",gradients.InitialHidden) };
        using (Gil()) foreach (var (expression, value) in fields)
        {
            using var expected = Scope.Eval(expression);
            GistParity.AssertUlps(value, expected, 0, "LSTM " + expression);
        }
    }

    [TestMethod]
    [TestCategory("KarpathyByteParity")]
    public void PackedInitialization_ExactSeededNumpy()
    {
        using var actual = BatchedLstm.Initialize(10, 4);
        PyExec("w=np.random.RandomState(7).randn(15,16)/np.sqrt(14)\nw[0,:]=0\nw[0,4:8]=3");
        using (Gil()) { using var expected = Scope.Eval("w"); GistParity.AssertExact(actual, expected, "LSTM Xavier/forget initialization"); }
    }

    [TestMethod, TestCategory("KarpathyShortRun"), DoNotParallelize]
    public void OriginalPinnedClassAndBothOriginalChecks_RunThroughPythonNet()
    {
        KarpathyOriginalSource.Load(Scope, "lstm");
        Assert.AreEqual("137a213cfe6ec6eb7c73dc73c18f356dfd676e6e36bb66d2be6667ca6be7a4c4", PyStr("original['__source_sha256__']"));
        Assert.AreEqual("('LSTM', 'checkSequentialMatchesBatch', 'checkBatchGradient')", PyStr("original['__loaded_names__']"));
        PyExec("""
            import io, contextlib
            saved_rng=np.random.get_state()
            try:
                np.random.seed(7)
                sequential_output=io.StringIO()
                with contextlib.redirect_stdout(sequential_output):
                    original['checkSequentialMatchesBatch']()
                np.random.seed(7)
                gradient_output=io.StringIO()
                with contextlib.redirect_stdout(gradient_output):
                    original['checkBatchGradient']()
                np.random.seed(7)
                original_w=original['LSTM'].init(10,4)
                original_x=np.random.randn(5,3,10)
                original_h0=np.random.randn(3,4)
                original_c0=np.random.randn(3,4)
                original_dh=np.random.randn(5,3,4)
                original_h,original_cn,original_hn,original_cache=original['LSTM'].forward(original_x,original_w,original_c0,original_h0)
                original_gradient=original['LSTM'].backward(original_dh,original_cache)
            finally:
                np.random.set_state(saved_rng)
            """);
        Assert.AreEqual(4L, PyLong("int(sum(line.strip()=='True' for line in sequential_output.getvalue().splitlines()))"));
        Assert.AreEqual(414L, PyLong("int(sum('checking param' in line for line in gradient_output.getvalue().splitlines()))"));
        Assert.IsTrue(PyBool("all(line.startswith('OK checking param') for line in gradient_output.getvalue().splitlines())"),
            "All 414 original finite-difference outputs must actually report OK on the fixed source fixture.");

        // This tier really executes the pinned author's class/check functions. Inputs produced by
        // that original class workflow cross through pythonnet into NumSharp, independently computed here.
        using var input = ImportOf("original_x"); using var weights = ImportOf("original_w");
        using var c0 = ImportOf("original_c0"); using var h0 = ImportOf("original_h0"); using var dh = ImportOf("original_dh");
        using var initialized = BatchedLstm.Initialize(10, 4, seed: 7);
        using var pass = BatchedLstm.Forward(input, weights, c0, h0);
        using var gradient = BatchedLstm.Backward(dh, pass);
        Exact(initialized, "original_w", "original LSTM.init");
        foreach (var (field, value) in new (string, NDArray)[] {
            ("Hout",pass.Hidden),("C",pass.Cell),("Ct",pass.CellTanh),("IFOG",pass.GateInputs),
            ("IFOGf",pass.Gates),("Hin",pass.Concatenated) })
            Exact(value, $"original_cache['{field}']", "pinned original cache " + field);
        var arrays = new[] { gradient.Input, gradient.Weights, gradient.InitialCell, gradient.InitialHidden };
        for (int i = 0; i < arrays.Length; i++) Exact(arrays[i], $"original_gradient[{i}]", "pinned original gradient " + i);
        Console.WriteLine("LSTM_ORIGINAL_SOURCE: both check functions executed; 4 True backward claims; 414/414 original gradient checks OK; original class and NumSharp caches/gradients byte-identical.");
    }

    [TestMethod, TestCategory("KarpathyByteParity")]
    public void FiveChunkShortRun_EveryCarryIntermediateAndGradient_ExactNumpy()
    {
        using var scope = NDScope.Open();
        var random = np.random.RandomState(17);
        var weights = BatchedLstm.Initialize(10, 4);
        var input = random.randn(5, 3, 10); var c0 = random.randn(3, 4); var h0 = random.randn(3, 4);
        var upstream = random.randn(5, 3, 4); var finalCell = random.randn(3, 4); var finalHidden = random.randn(3, 4);
        ExportTo("x", input); ExportTo("w", weights); ExportTo("c0", c0); ExportTo("h0", h0);
        ExportTo("dh", upstream); ExportTo("dcn", finalCell); ExportTo("dhn", finalHidden);
        LoadReference();
        PyExec("batch=forward(x,w,c0,h0)\nbatch_gradient=backward(dh,batch,dcn,dhn)\nchunks=[]\ncp=c0\nhp=h0");
        var passes = new BatchedLstm.ForwardPass[5];
        var gradients = new BatchedLstm.Gradients[5];
        try
        {
            NDArray cp = c0, hp = h0;
            var combinedHidden = np.empty((5, 3, 4), dtype: np.float64);
            for (int tick = 0; tick < 5; tick++)
            {
                passes[tick] = BatchedLstm.Forward(input[$"{tick}:{tick + 1},:,:"] , weights, cp, hp);
                cp = passes[tick].Cell["0,:,:"]; hp = passes[tick].Hidden["0,:,:"];
                combinedHidden[$"{tick},:,:"] = hp;
                PyExec($"chunk=forward(x[{tick}:{tick + 1}],w,cp,hp)\nchunks.append(chunk)\ncp=chunk['c'][-1]\nhp=chunk['h'][-1]");
                AssertForward(passes[tick], "chunk", $"chunk {tick} forward");
                Exact(cp, "cp", $"chunk {tick} final cell carry"); Exact(hp, "hp", $"chunk {tick} final hidden carry");
            }
            Exact(combinedHidden, "batch['h']", "five sequential ticks vs batched output");
            var combinedInput = np.zeros_like(input); var combinedWeights = np.zeros_like(weights);
            NDArray dcNext = finalCell, dhNext = finalHidden;
            PyExec("dcnext=dcn\ndhnext=dhn\ncombined_dx=np.zeros_like(x)\ncombined_dw=np.zeros_like(w)");
            for (int tick = 4; tick >= 0; tick--)
            {
                gradients[tick] = BatchedLstm.Backward(upstream[$"{tick}:{tick + 1},:,:"] , passes[tick], dcNext, dhNext);
                dcNext = gradients[tick].InitialCell; dhNext = gradients[tick].InitialHidden;
                combinedInput[$"{tick},:,:"] = gradients[tick].Input["0,:,:"];
                np.add(combinedWeights, gradients[tick].Weights, @out: combinedWeights);
                PyExec($"chunk_gradient=backward(dh[{tick}:{tick + 1}],chunks[{tick}],dcnext,dhnext)\ndcnext=chunk_gradient['dc0']\ndhnext=chunk_gradient['dh0']\ncombined_dx[{tick}]=chunk_gradient['dx'][0]\ncombined_dw+=chunk_gradient['dw']");
                AssertGradients(gradients[tick], "chunk_gradient", $"chunk {tick} backward");
                Exact(combinedInput, "combined_dx", $"chunk {tick} accumulated input derivatives");
                Exact(combinedWeights, "combined_dw", $"chunk {tick} accumulated weight derivatives");
            }
            Exact(combinedInput, "batch_gradient['dx']", "sequential vs batch all input gradients");
            Exact(combinedWeights, "batch_gradient['dw']", "sequential vs batch all weight gradients");
            Exact(dcNext, "batch_gradient['dc0']", "sequential vs batch initial cell gradient");
            Exact(dhNext, "batch_gradient['dh0']", "sequential vs batch initial hidden gradient");
        }
        finally
        {
            foreach (var gradient in gradients) gradient?.Dispose();
            foreach (var pass in passes) pass?.Dispose();
        }
    }

    [TestMethod, TestCategory("KarpathyByteParity")]
    public void SixUpdateRecurrentTrainingShortRun_AllStatesGradientsWeightsAndLoss_ExactNumpy()
    {
        using var scope = NDScope.Open();
        var random = np.random.RandomState(23);
        var input = random.randn(5, 3, 10); var weights = BatchedLstm.Initialize(10, 4);
        NDArray c = random.randn(3, 4) * .1, h = random.randn(3, 4) * .1;
        var target = np.tanh(random.randn(5, 3, 4)) * .25;
        ExportTo("x", input); ExportTo("w_initial", weights); ExportTo("c_initial", c); ExportTo("h_initial", h); ExportTo("target", target);
        LoadReference();
        PyExec("w=w_initial.copy()\ncp=c_initial.copy()\nhp=h_initial.copy()\ntraining_trace=[]");
        var lossTrace = np.empty(6, dtype: np.float64);
        for (int step = 0; step < 6; step++)
        {
            using var iteration = NDScope.Open();
            using var pass = BatchedLstm.Forward(input, weights, c, h);
            var error = pass.Hidden - target;
            var squared = np.square(error).reshape(-1);
            double loss = np.mean(squared, axis: 0).item<double>();
            var upstream = (2.0 / error.size) * error;
            using var gradient = BatchedLstm.Backward(upstream, pass);
            PyExec("f=forward(x,w,cp,hp)\nerror=f['h']-target\nloss=np.mean(np.square(error).ravel(),axis=0)\ng=backward((2.0/error.size)*error,f)\ntraining_trace.append(loss)");
            AssertForward(pass, "f", $"training step {step} cache");
            AssertGradients(gradient, "g", $"training step {step} gradients");
            Exact(np.array(loss), "np.asarray(loss)", $"training step {step} loss");
            lossTrace[step] = loss;
            np.subtract(weights, .05 * gradient.Weights, @out: weights);
            PyExec("w-=.05*g['dw']");
            Exact(weights, "w", $"training step {step} updated weights");

            // The carry changes the next segment's initial condition, so cross-segment losses
            // need not decrease. Check descent on THIS fixed-initial-state objective instead.
            using var updated = BatchedLstm.Forward(input, weights, c, h);
            double after = np.mean(np.square(updated.Hidden - target).reshape(-1), axis: 0).item<double>();
            PyExec("after=np.mean(np.square(forward(x,w,cp,hp)['h']-target).ravel(),axis=0)");
            Exact(np.array(after), "np.asarray(after)", $"training step {step} post-update objective");
            Assert.IsTrue(after < loss, $"SGD must decrease the fixed-context objective at step {step}: {loss:R} -> {after:R}");
            c = iteration.Returns(pass.Cell["-1,:,:"] .copy());
            h = iteration.Returns(pass.Hidden["-1,:,:"] .copy());
            PyExec("cp=f['c'][-1].copy()\nhp=f['h'][-1].copy()");
            Exact(c, "cp", $"training step {step} carried cell"); Exact(h, "hp", $"training step {step} carried hidden");
        }
        Exact(lossTrace, "np.asarray(training_trace)", "complete six-update loss trace");
        Console.WriteLine("LSTM_SHORT_TRAINING: six genuine SGD updates; every forward intermediate, all four gradients, weights, losses and recurrent carries byte-identical to independent NumPy. This optimizer workflow is new, not present in the original gist.");
    }

    private void LoadReference() => PyExec(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "KarpathyLstmReference.py")));
    private void Exact(NDArray actual, string expression, string label)
    {
        using (Gil()) { using var expected = Scope.Eval(expression); GistParity.AssertExact(actual, expected, label); }
    }
    private void AssertForward(BatchedLstm.ForwardPass pass, string root, string label)
    {
        foreach (var (field, value) in new (string, NDArray)[] {
            ("h",pass.Hidden),("c",pass.Cell),("ct",pass.CellTanh),
            ("gates",pass.GateInputs),("activated",pass.Gates),("hin",pass.Concatenated) })
            Exact(value, $"{root}['{field}']", label + " " + field);
    }
    private void AssertGradients(BatchedLstm.Gradients gradient, string root, string label)
    {
        foreach (var (field, value) in new (string, NDArray)[] {
            ("dx",gradient.Input),("dw",gradient.Weights),("dc0",gradient.InitialCell),("dh0",gradient.InitialHidden) })
            Exact(value, $"{root}['{field}']", label + " " + field);
    }
}
