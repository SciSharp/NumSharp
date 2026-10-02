using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist.Karpathy;

namespace NumSharp.Tests.Interop;

[TestClass]
public class KarpathyRnnLiveTests : InteropTestBase
{
    // Independent mathematical NumPy oracle; no Python-2 IO, source exec, or infinite loop.
    private const string Reference = """
        def evaluate_rnn(weights, inputs, targets, initial, clipping=True):
            wi,wr,wo,hidden_bias,output_bias=weights
            h=[initial.copy()]; onehots=[]; probabilities=[]; loss=0.0
            for token,target in zip(inputs,targets):
                one=np.zeros((wi.shape[1],1)); one[int(token),0]=1.0
                onehots.append(one)
                h.append(np.tanh(wi@one+wr@h[-1]+hidden_bias))
                logits=wo@h[-1]+output_bias
                p=np.exp(logits)/np.sum(np.exp(logits))
                probabilities.append(p)
                loss-=np.log(p[int(target),0])
            gradients=[np.zeros_like(x) for x in weights]
            future=np.zeros_like(initial)
            for t in range(len(inputs)-1,-1,-1):
                output_gradient=probabilities[t].copy()
                output_gradient[int(targets[t]),0]-=1.0
                gradients[2]+=output_gradient@h[t+1].T
                gradients[4]+=output_gradient
                hidden_gradient=wo.T@output_gradient+future
                raw=(1-h[t+1]*h[t+1])*hidden_gradient
                gradients[3]+=raw
                gradients[0]+=raw@onehots[t].T
                gradients[1]+=raw@h[t].T
                future=wr.T@raw
            if clipping:
                for gradient in gradients: np.clip(gradient,-5.0,5.0,out=gradient)
            return (np.asarray(loss),gradients,h[-1],np.concatenate(h[1:],axis=1).T,
                    np.concatenate(probabilities,axis=1).T)

        def sample_rnn(weights,initial,token,count,rng):
            wi,wr,wo,hidden_bias,output_bias=weights
            h=initial.copy(); sampled=[]
            for _ in range(count):
                one=np.zeros((wi.shape[1],1)); one[int(token),0]=1.0
                h=np.tanh(wi@one+wr@h+hidden_bias)
                logits=wo@h+output_bias
                p=np.exp(logits)/np.sum(np.exp(logits))
                token=int(rng.choice(range(wi.shape[1]),p=p.ravel()))
                sampled.append(token)
            return np.asarray(sampled,dtype=np.int64)
        """;

    private void ExportParameters(MinimalCharacterRnn.Model model)
    {
        string[] names = { "wxh", "whh", "why", "bh", "by" };
        for (int i = 0; i < names.Length; i++) ExportTo(names[i], model.Parameters.Arrays[i]);
        PyExec(Reference + "\nweights=[wxh,whh,why,bh,by]\n");
    }

    [DataTestMethod, TestCategory("KarpathyByteParity")]
    [DataRow(3, 4, 4, 7)]
    [DataRow(100, 5, 25, 42)]
    public void FullForwardLossAndBptt_IncludeOriginal100By25Shape_ExactLiveNumpy(int hiddenSize, int vocabularySize, int length, int seed)
    {
        using var scope = NDScope.Open();
        using var model = new MinimalCharacterRnn.Model(vocabularySize, hiddenSize, seed);
        var backing = np.zeros((hiddenSize, 2), dtype: np.float64);
        var hidden = backing[":,::2"];
        if (hiddenSize == 3) { hidden[0, 0] = .02; hidden[1, 0] = -.01; hidden[2, 0] = .03; }
        int[] input = Enumerable.Range(0, length).Select(i => i % vocabularySize).ToArray();
        int[] target = input.Select(i => (i + 1) % vocabularySize).ToArray();
        using var result = model.LossAndGradients(input, target, hidden);
        ExportParameters(model); ExportTo("initial", hidden);
        ExportTo("inputs", np.array(input)); ExportTo("targets", np.array(target));
        PyExec("expected=evaluate_rnn(weights,inputs,targets,initial)");
        using (Gil())
        {
            using var loss = Scope.Eval("expected[0]");
            using var lastHidden = Scope.Eval("expected[2]");
            using var hiddenTrace = Scope.Eval("expected[3]");
            using var probabilities = Scope.Eval("expected[4]");
            GistParity.AssertExact(np.array(result.Loss), loss, "RNN accumulated cross-entropy");
            GistParity.AssertExact(result.LastHidden, lastHidden, "RNN final hidden state");
            GistParity.AssertExact(result.HiddenStates, hiddenTrace, "RNN every forward hidden state");
            GistParity.AssertExact(result.Probabilities, probabilities, "RNN every probability");
            for (int i = 0; i < 5; i++)
            {
                using var expectedGradient = Scope.Eval($"expected[1][{i}]");
                GistParity.AssertExact(result.Gradients.Arrays[i], expectedGradient, $"RNN BPTT gradient group {i}");
            }
        }
    }

    [TestMethod, TestCategory("KarpathyByteParity")]
    public void ClippingAndTwoAdagradUpdates_AllParametersAndMemories_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        using var model = new MinimalCharacterRnn.Model(4, 3, seed: 7);
        using var gradients = model.Parameters.Copy();
        for (int group = 0; group < 5; group++)
        {
            var gradient = gradients.Arrays[group];
            gradient[":"] = .25;
            gradient[0, 0] = -9.0; gradient[-1, -1] = 9.0;
            ExportTo($"g{group}", gradient);
        }
        ExportParameters(model);
        PyExec("""
            updated=[x.copy() for x in weights]
            gradients=[x.copy() for x in (g0,g1,g2,g3,g4)]
            memories=[np.zeros_like(x) for x in weights]
            for gradient in gradients: np.clip(gradient,-5.0,5.0,out=gradient)
            for _ in range(2):
                for parameter,gradient,memory in zip(updated,gradients,memories):
                    memory+=gradient*gradient
                    parameter+=-.1*gradient/np.sqrt(memory+1e-8)
            """);
        MinimalCharacterRnn.ClipGradients(gradients);
        model.ApplyAdagrad(gradients); model.ApplyAdagrad(gradients);
        using (Gil())
        for (int i = 0; i < 5; i++)
        {
            using var expectedGradient = Scope.Eval($"gradients[{i}]");
            using var expectedParameter = Scope.Eval($"updated[{i}]");
            using var expectedMemory = Scope.Eval($"memories[{i}]");
            GistParity.AssertExact(gradients.Arrays[i], expectedGradient, $"clipped gradient {i}");
            GistParity.AssertExact(model.Parameters.Arrays[i], expectedParameter, $"Adagrad parameter {i}");
            GistParity.AssertExact(model.AdagradMemory.Arrays[i], expectedMemory, $"Adagrad memory {i}");
        }
    }

    [TestMethod, TestCategory("KarpathyByteParity")]
    public void Sampling_AllOriginal200Draws_ExactLiveNumpyFromSharedParameters()
    {
        using var scope = NDScope.Open();
        using var model = new MinimalCharacterRnn.Model(5, seed: 42);
        var hidden = np.zeros((100, 1), dtype: np.float64);
        var samples = model.Sample(hidden, 0, 200, np.random.RandomState(19));
        ExportParameters(model); ExportTo("initial", hidden);
        PyExec("samples=sample_rnn(weights,initial,0,200,np.random.RandomState(19))");
        using (Gil())
        {
            using var expected = Scope.Eval("samples");
            GistParity.AssertExact(np.array(samples.Select(i => (long)i).ToArray()), expected, "RNN all200 sampled tokens");
        }
    }

    [TestMethod, TestCategory("KarpathyByteParity")]
    public void BoundedTraining_AllFortyLossesParametersMemoriesAndSamples_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        string text = string.Concat(Enumerable.Repeat("hello world\n", 40));
        var vocabulary = new MinimalCharacterRnn.Vocabulary(text);
        var tokens = vocabulary.Encode(text);
        using var model = new MinimalCharacterRnn.Model(vocabulary.Count, 16, seed: 1);
        ExportParameters(model); ExportTo("corpus", np.array(tokens));
        // Copy the shared starting buffers before C# updates them. This is not an oracle over
        // already-updated C# weights; both optimizers independently evolve the same initial bits.
        PyExec("""
            weights=[x.copy() for x in weights]
            rng=np.random.RandomState(1)
            rng.randn(16,9); rng.randn(16,16); rng.randn(9,16)
            memory=[np.zeros_like(x) for x in weights]
            hidden=np.zeros((16,1)); position=0
            smooth=-np.log(1.0/9)*25
            trace=[]; sampled=[]
            for iteration in range(40):
                reset=iteration==0 or position+26>=len(corpus)
                if reset: hidden=np.zeros((16,1)); position=0
                inputs=corpus[position:position+25]; targets=corpus[position+1:position+26]
                if iteration%20==0: sampled.append(sample_rnn(weights,hidden,int(inputs[0]),40,rng))
                loss,gradient,hidden,_,_=evaluate_rnn(weights,inputs,targets,hidden)
                smooth=smooth*.999+loss*.001
                trace.append([iteration,position,int(reset),float(loss),float(smooth)])
                for parameter,g,mem in zip(weights,gradient,memory):
                    mem+=g*g
                    parameter+=-.1*g/np.sqrt(mem+1e-8)
                position+=25
            trace=np.asarray(trace,dtype=np.float64)
            sampled=np.stack(sampled)
            """);
        using var training = model.Train(tokens, 40, sampleEvery: 20, sampleLength: 40);
        var actualTrace = np.zeros((40, 5), dtype: np.float64);
        foreach (var step in training.Steps)
        {
            int i = step.Iteration;
            actualTrace[i, 0] = (double)i; actualTrace[i, 1] = (double)step.Position;
            actualTrace[i, 2] = step.ResetHidden ? 1.0 : 0.0;
            actualTrace[i, 3] = step.Loss; actualTrace[i, 4] = step.SmoothLoss;
        }
        var actualSamples = np.array(training.Samples.SelectMany(s => s.Tokens).Select(t => (long)t).ToArray()).reshape(2, 40);
        using (Gil())
        {
            using var expectedTrace = Scope.Eval("trace");
            using var expectedSamples = Scope.Eval("sampled");
            using var expectedHidden = Scope.Eval("hidden");
            GistParity.AssertExact(actualTrace, expectedTrace, "RNN all40 training steps");
            GistParity.AssertExact(actualSamples, expectedSamples, "RNN periodic training samples");
            GistParity.AssertExact(training.LastHidden, expectedHidden, "RNN training final hidden state");
            for (int i = 0; i < 5; i++)
            {
                using var expectedParameter = Scope.Eval($"weights[{i}]");
                using var expectedMemory = Scope.Eval($"memory[{i}]");
                GistParity.AssertExact(model.Parameters.Arrays[i], expectedParameter, $"RNN trained parameter {i}");
                GistParity.AssertExact(model.AdagradMemory.Arrays[i], expectedMemory, $"RNN trained memory {i}");
            }
        }
    }

    [TestMethod, TestCategory("KarpathyShortRun")]
    public void OriginalPinnedDefinitions_CompleteFourTrainingUpdatesAndSamples_ThroughPythonNet()
    {
        using var scope = NDScope.Open();
        using var model = new MinimalCharacterRnn.Model(2, hiddenSize: 100, seed: 17);
        int[] corpus = Enumerable.Range(0, 65).Select(i => i % 5 == 4 ? 1 : 0).ToArray();
        ExportParameters(model); ExportTo("short_corpus", np.array(corpus));
        KarpathyOriginalSource.Load(Scope, "rnn");
        PyExec("_rnn_saved_random_state=np.random.get_state()");
        try
        {
        PyExec("""
            # Execute only the two reviewed, hash-pinned ORIGINAL functions. The adapter
            # converted Python2 syntax; this finite driver replaces the source's while True.
            original_parameters=[x.copy() for x in weights]
            original.update(zip(('Wxh','Whh','Why','bh','by'),original_parameters))
            original['vocab_size']=2; original['hidden_size']=100
            original_memories=[np.zeros_like(x) for x in original_parameters]
            starting_parameters=[x.copy() for x in original_parameters]
            np.random.seed(17)
            np.random.randn(100,2); np.random.randn(100,100); np.random.randn(2,100)
            source_position=0; source_smooth=-np.log(.5)*25
            source_records=[]; source_samples=[]; source_clip_seen=False
            for source_iteration in range(4):
                source_reset=source_iteration==0 or source_position+26>=len(short_corpus)
                if source_reset: source_hidden=np.zeros((100,1)); source_position=0
                source_inputs=short_corpus[source_position:source_position+25].tolist()
                source_targets=short_corpus[source_position+1:source_position+26].tolist()
                source_samples.append(original['sample'](source_hidden,source_inputs[0],12))
                source_result=original['lossFun'](source_inputs,source_targets,source_hidden)
                source_loss=source_result[0]; source_gradients=source_result[1:6]; source_hidden=source_result[6]
                assert all(np.isfinite(g).all() and (np.abs(g)<=5).all() for g in source_gradients)
                source_clip_seen=source_clip_seen or any((np.abs(g)==5).any() for g in source_gradients)
                source_smooth=source_smooth*.999+source_loss*.001
                source_records.append([source_iteration,source_position,float(source_loss),float(source_smooth),int(source_reset)])
                for parameter,gradient,memory in zip(original_parameters,source_gradients,original_memories):
                    memory+=gradient*gradient
                    parameter+=-.1*gradient/np.sqrt(memory+1e-8)
                source_position+=25
            source_records=np.asarray(source_records,dtype=np.float64)
            source_samples=np.asarray(source_samples,dtype=np.int64)
            """);
        }
        finally { PyExec("np.random.set_state(_rnn_saved_random_state)"); }
        using var training = model.Train(corpus, 4, sampleEvery: 1, sampleLength: 12);
        Assert.AreEqual("lossFun,sample", PyStr("','.join(original['__loaded_names__'])"));
        Assert.AreEqual("98a0eaa61092f9541e883f3e8b6c02dae28b5658dbc5f692ffe0881be37a390a", PyStr("original['__source_sha256__']"));
        Assert.AreEqual(4L, PyLong("int(len(source_records))"));
        Assert.AreEqual(48L, PyLong("int(source_samples.size)"));
        Assert.IsTrue(PyBool("bool(np.isfinite(source_records).all() and np.isfinite(source_hidden).all())"));
        Assert.IsTrue(PyBool("bool(source_clip_seen)"));
        Assert.IsTrue(PyBool("bool(all(np.any(m>0) for m in original_memories))"));
        Assert.IsTrue(PyBool("bool(all(np.any(a!=b) for a,b in zip(original_parameters,starting_parameters)))"));
        Assert.AreEqual(17.318640025562782, PyFloat("float(source_records[0,2])"), 1e-12);
        Assert.AreEqual(4, training.Steps.Count);
        Assert.AreEqual(4, training.Samples.Count);
        Assert.IsTrue(training.Steps.All(step => double.IsFinite(step.Loss) && double.IsFinite(step.SmoothLoss)));
        Assert.IsTrue(training.Samples.All(sample => sample.Tokens.Length == 12 && sample.Tokens.All(token => token is 0 or 1)));
        CollectionAssert.AreEqual(new[] { 0, 2 }, training.Steps.Where(step => step.ResetHidden).Select(step => step.Iteration).ToArray());
        // This deliberately aggressive source-parameter fixture exercises clipping and resets;
        // four updates are not a claim of monotonic learning. The existing 40-step test measures learning.
        for (int iteration = 0; iteration < 4; iteration++)
            Assert.AreEqual(PyFloat($"float(source_records[{iteration},2])"), training.Steps[iteration].Loss,
                $"Functional original/port loss at iteration {iteration}; full byte-state gates are separate.");
    }

    [TestMethod, TestCategory("KarpathyByteParity")]
    public void FourSourceSizedUpdates_EveryRawClippedGradientHiddenAndOptimizerState_ByteExact()
    {
        using var scope = NDScope.Open();
        const int hiddenSize = 100, sequenceLength = 25;
        int[] corpus = Enumerable.Range(0, 65).Select(i => i % 5 == 4 ? 1 : 0).ToArray();
        using var model = new MinimalCharacterRnn.Model(2, hiddenSize, seed: 17);
        var random = np.random.RandomState(29);
        var hidden = np.zeros((hiddenSize, 1), dtype: np.float64);
        ExportParameters(model);
        PyExec("""
            # Snapshot the exported STARTING parameters once. The Python optimizer evolves its
            # own arrays; it never reads back the C# optimizer's updated parameter values.
            weights=[x.copy() for x in weights]
            memories=[np.zeros_like(x) for x in weights]
            hidden_ref=np.zeros((100,1))
            rng=np.random.RandomState(29)
            smooth_ref=-np.log(.5)*25
            position_ref=0
            """);
        int position = 0;
        double smooth = -np.log(np.array(.5)).item<double>() * sequenceLength;
        bool sawClipping = false;
        var positions = new int[4];
        for (int iteration = 0; iteration < 4; iteration++)
        {
            using var step = NDScope.Open();
            bool reset = iteration == 0 || position + sequenceLength + 1 >= corpus.Length;
            if (reset)
            {
                hidden.Dispose(); hidden = step.Returns(np.zeros((hiddenSize, 1), dtype: np.float64));
                position = 0;
            }
            positions[iteration] = position;
            int[] inputs = corpus.Skip(position).Take(sequenceLength).ToArray();
            int[] targets = corpus.Skip(position + 1).Take(sequenceLength).ToArray();
            ExportTo("inputs", np.array(inputs)); ExportTo("targets", np.array(targets));
            PyExec($"iteration={iteration}\nreset=iteration==0 or position_ref+26>=65\nif reset:\n    hidden_ref=np.zeros((100,1))\n    position_ref=0");
            PyExec("""
                expected_sample=sample_rnn(weights,hidden_ref,int(inputs[0]),12,rng)
                expected_raw=evaluate_rnn(weights,inputs,targets,hidden_ref,clipping=False)
                expected_clipped=[np.clip(g,-5.0,5.0) for g in expected_raw[1]]
                smooth_ref=smooth_ref*.999+float(expected_raw[0])*.001
                expected_stats=np.array([float(expected_raw[0]),smooth_ref,position_ref,int(reset)],dtype=np.float64)
                """);

            int[] sample = model.Sample(hidden, inputs[0], 12, random);
            using var result = model.LossAndGradients(inputs, targets, hidden, clipGradients: false);
            smooth = smooth * .999 + result.Loss * .001;
            ExactState(np.array(new[] { result.Loss, smooth, (double)position, reset ? 1.0 : 0.0 }),
                "expected_stats", $"step {iteration}: loss, smooth loss, position and reset");
            ExactState(np.array(sample.Select(token => (long)token).ToArray()), "expected_sample", $"step {iteration}: all sampled tokens");
            ExactState(result.HiddenStates, "expected_raw[3]", $"step {iteration}: all 25 hidden states");
            ExactState(result.Probabilities, "expected_raw[4]", $"step {iteration}: all next-token probabilities");
            ExactState(result.LastHidden, "expected_raw[2]", $"step {iteration}: carried hidden state");
            for (int group = 0; group < 5; group++)
            {
                ExactState(result.Gradients.Arrays[group], $"expected_raw[1][{group}]", $"step {iteration}: raw gradient group {group}");
                sawClipping |= np.any(np.abs(result.Gradients.Arrays[group]) > 5.0);
            }
            MinimalCharacterRnn.ClipGradients(result.Gradients);
            for (int group = 0; group < 5; group++)
                ExactState(result.Gradients.Arrays[group], $"expected_clipped[{group}]", $"step {iteration}: clipped gradient group {group}");
            model.ApplyAdagrad(result.Gradients);
            PyExec("""
                for parameter,gradient,memory in zip(weights,expected_clipped,memories):
                    memory+=gradient*gradient
                    parameter+=-.1*gradient/np.sqrt(memory+1e-8)
                hidden_ref=expected_raw[2].copy()
                position_ref+=25
                """);
            for (int group = 0; group < 5; group++)
            {
                ExactState(model.Parameters.Arrays[group], $"weights[{group}]", $"step {iteration}: updated parameter group {group}");
                ExactState(model.AdagradMemory.Arrays[group], $"memories[{group}]", $"step {iteration}: updated Adagrad memory group {group}");
            }
            hidden.Dispose(); hidden = step.Returns(result.LastHidden.copy());
            position += sequenceLength;
        }
        Assert.IsTrue(sawClipping, "The trace must exercise actual clipping, not only a no-op clipping pass.");
        CollectionAssert.AreEqual(new[] { 0, 25, 0, 25 }, positions);
    }

    private void ExactState(NDArray actual, string expression, string label)
    {
        using (Gil())
        {
            using var expected = Scope.Eval(expression);
            GistParity.AssertExact(actual, expected, label);
        }
    }
}
