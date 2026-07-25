#:project ../NeuralNetwork.NumSharp.csproj
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Nullable=disable
// P4 verification for NeuralNetwork.NumSharp — regularization & normalization.
// Run from THIS directory (file-based apps need a csproj-free CWD):
//   cd examples/NeuralNetwork.NumSharp/tests && dotnet run verify_p4.cs
//
// The VALUES and GRADIENTS of Dropout / BatchNorm / LayerNorm / Embedding /
// the regularizers are pinned against real Keras + jax.grad by
// verify_edge_cases.cs. This gate covers what an oracle over single layers
// cannot see:
//   * the Training flag actually being plumbed by the trainer and NeuralNet
//   * running statistics updating in training and NOT in inference
//   * Dropout seed-determinism through np.random
//   * Flatten / Reshape shape algebra both ways
//   * Embedding's null InputGrad surviving a real backward sweep
//   * regularizer penalties reaching the reported loss AND the optimizer
//   * the new layers round-tripping through .npz + architecture JSON
//   * independent finite-difference gradient checks (a second opinion on the
//     hand-derived backward passes, computed without Keras)
//   * the argument/shape guards
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetwork.NumSharp;
using NeuralNetwork.NumSharp.Callbacks;
using NeuralNetwork.NumSharp.Cost;
using NeuralNetwork.NumSharp.Layers;
using NeuralNetwork.NumSharp.MnistMlp;
using NeuralNetwork.NumSharp.Optimizers;
using NeuralNetwork.NumSharp.Regularizers;
using NeuralNetwork.NumSharp.Serialization;
using NumSharp;
using NumSharp.Backends;

int pass = 0, fail = 0;
void Check(string name, bool ok, string detail = "")
{
    if (ok) pass++;
    else { fail++; Console.WriteLine($"  FAIL {name} {detail}"); }
}
void CheckClose(string name, float actual, float expected, float tol = 1e-5f)
    => Check(name, Math.Abs(actual - expected) <= tol * Math.Max(1f, Math.Abs(expected)),
             $"actual={actual} expected={expected}");
void CheckEq(string name, long actual, long expected)
    => Check(name, actual == expected, $"actual={actual} expected={expected}");

string tmpRoot = Path.Combine(Path.GetTempPath(), "nn_p4_" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(tmpRoot);

try
{

// ======================================================================
Console.WriteLine("--- P4 Training flag (the contract change) ---");
// ======================================================================
{
    // Default is INFERENCE, so a bare Forward never drops.
    var d = new Dropout(0.5f);
    Check("Training defaults to false", !d.Training);
    var ones = np.ones(new Shape(50, 10), NPTypeCode.Single);
    d.Forward(ones);
    CheckClose("bare Forward is inference (no drop)", (float)np.sum(d.Output), 500f);

    // MlpTrainer must set it true in the train loop and false when scoring.
    var spy = new TrainingSpy();
    var xs = np.zeros(new Shape(8, 3), NPTypeCode.Single);
    var ys = np.array(new byte[] { 0, 1, 2, 0, 1, 2, 0, 1 });
    MlpTrainer.Train(new List<BaseLayer> { spy }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 1, batchSize: 4, numClasses: 3, shuffle: false, verbose: 0);
    CheckEq("trainer forwards with Training=true", spy.TrainingCalls, 2);
    CheckEq("trainer does not eval without a test/val set", spy.InferenceCalls, 0);

    var spy2 = new TrainingSpy();
    MlpTrainer.Train(new List<BaseLayer> { spy2 }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, xs, ys, epochs: 1, batchSize: 4, numClasses: 3, shuffle: false,
        validationSplit: 0.25f, verbose: 0);
    Check("trainer scores validation with Training=false", spy2.InferenceCalls > 0,
          $"inference={spy2.InferenceCalls}");
    Check("training and inference forwards both happened",
          spy2.TrainingCalls > 0 && spy2.InferenceCalls > 0);

    // MlpTrainer.Evaluate is an inference path.
    var spy3 = new TrainingSpy();
    MlpTrainer.Evaluate(new List<BaseLayer> { spy3 }, xs, ys, batchSize: 4);
    CheckEq("Evaluate is pure inference", spy3.TrainingCalls, 0);
    CheckEq("Evaluate ran", spy3.InferenceCalls, 2);

    // NeuralNet.Predict is inference; NeuralNet.Train is training.
    var spy4 = new TrainingSpy();
    var net = new NeuralNet(new SGD(lr: 0f), new SoftmaxCrossEntropy());
    net.Add(spy4);
    net.Predict(xs);
    CheckEq("NeuralNet.Predict is inference", spy4.TrainingCalls, 0);
    net.Train(xs, SoftmaxCrossEntropy.OneHot(ys, 3), 1, 4);
    Check("NeuralNet.Train is training", spy4.TrainingCalls > 0);
}

// ======================================================================
Console.WriteLine("--- P4 Dropout ---");
// ======================================================================
{
    // Seeded determinism through np.random (MT19937).
    float[] Draw(int seed)
    {
        np.random.seed(seed);
        var d = new Dropout(0.5f) { Training = true };
        d.Forward(np.ones(new Shape(20, 20), NPTypeCode.Single));
        var flat = np.reshape(d.Output, new Shape(400));
        return Enumerable.Range(0, 400).Select(i => flat.GetSingle(i)).ToArray();
    }
    var a = Draw(7); var b = Draw(7); var c = Draw(8);
    Check("dropout is seed-deterministic", a.SequenceEqual(b));
    Check("dropout differs across seeds", !a.SequenceEqual(c));

    // rate=0 is the identity even in training.
    var d0 = new Dropout(0f) { Training = true };
    d0.Forward(np.ones(new Shape(4, 4), NPTypeCode.Single));
    CheckClose("rate=0 passes through in training", (float)np.sum(d0.Output), 16f);
    d0.Backward(np.ones(new Shape(4, 4), NPTypeCode.Single));
    CheckClose("rate=0 gradient passes through", (float)np.sum(d0.InputGrad), 16f);

    // Expected value is preserved — the whole point of the inverted form.
    np.random.seed(99);
    var d5 = new Dropout(0.5f) { Training = true };
    d5.Forward(np.ones(new Shape(400, 400), NPTypeCode.Single));
    float mean = (float)np.sum(d5.Output) / 160000f;
    Check("inverted dropout preserves E[x]", Math.Abs(mean - 1f) < 0.02f, $"mean={mean}");

    // Guards.
    Check("rate>=1 rejected", ExpectThrow(() => new Dropout(1f)).Contains("[0, 1)"));
    Check("negative rate rejected", ExpectThrow(() => new Dropout(-0.1f)).Contains("[0, 1)"));
    Check("NaN rate rejected", ExpectThrow(() => new Dropout(float.NaN)).Contains("[0, 1)"));
}

// ======================================================================
Console.WriteLine("--- P4 BatchNormalization (state machine + FD gradients) ---");
// ======================================================================
{
    var x = np.array(new float[,] { { 1f, 2f }, { 3f, 5f }, { 7f, 11f } });

    // Training updates the running statistics; inference must NOT.
    var bn = new BatchNormalization(2);
    NDArray mm0 = bn.NonTrainable["moving_mean"].copy();
    bn.Training = false;
    bn.Forward(x);
    CheckClose("inference leaves moving_mean alone",
               (float)np.sum(np.abs(bn.NonTrainable["moving_mean"] - mm0)), 0f);

    bn.Training = true;
    bn.Forward(x);
    // moving = 0.99*0 + 0.01*mean; mean = [11/3, 6]
    CheckClose("training updates moving_mean [0]", bn.NonTrainable["moving_mean"].GetSingle(0), 0.01f * (11f / 3f));
    CheckClose("training updates moving_mean [1]", bn.NonTrainable["moving_mean"].GetSingle(1), 0.06f);
    // population var of col0 = 56/9; moving_var = 0.99*1 + 0.01*var
    CheckClose("training updates moving_variance [0]",
               bn.NonTrainable["moving_variance"].GetSingle(0), 0.99f + 0.01f * (56f / 9f), 1e-4f);

    // Running stats are NOT parameters — an optimizer must never see them.
    Check("running stats are not in Parameters",
          !bn.Parameters.ContainsKey("moving_mean") && !bn.Parameters.ContainsKey("moving_variance"));
    Check("gamma/beta are parameters", bn.Parameters.ContainsKey("gamma") && bn.Parameters.ContainsKey("beta"));
    // An optimizer step must not throw for want of a gradient.
    bn.Backward(np.ones(x.Shape, NPTypeCode.Single));
    var sgd = new SGD(lr: 0.1f);
    Check("optimizer steps a BatchNorm layer", ExpectThrow(() => sgd.Update(1, bn)) == "<no exception thrown>");

    // Normalized output has zero mean / unit variance in training (gamma=1,beta=0).
    var bn2 = new BatchNormalization(2, epsilon: 1e-8f) { Training = true };
    bn2.Forward(x);
    CheckClose("normalized mean ~ 0", (float)np.sum(np.mean(bn2.Output, axis: 0)), 0f, 1e-3f);
    CheckClose("normalized var ~ 1", (float)np.sum(np.var(bn2.Output, axis: 0)), 2f, 1e-3f);

    // Independent finite-difference check of the training backward (no Keras).
    FdCheck("bn train dx", () => { var l = new BatchNormalization(2) { Training = true }; Randomize(l, 2); return l; },
            np.array(new float[,] { { 1f, 2f }, { 3f, 5f }, { 7f, 11f }, { -2f, 0.5f } }),
            np.array(new float[,] { { 1f, -2f }, { 0.25f, 3f }, { -0.5f, 1f }, { 2f, 0.75f } }));

    // ... and of the INFERENCE backward, which takes a different branch.
    FdCheck("bn inference dx", () => { var l = new BatchNormalization(2) { Training = false }; Randomize(l, 2);
                                       l.NonTrainable["moving_mean"] = np.array(new float[] { 0.5f, 1.5f });
                                       l.NonTrainable["moving_variance"] = np.array(new float[] { 2f, 0.75f });
                                       return l; },
            np.array(new float[,] { { 1f, 2f }, { 3f, 5f }, { 7f, 11f } }),
            np.array(new float[,] { { 1f, -2f }, { 0.25f, 3f }, { -0.5f, 1f } }));

    // Guards.
    Check("3-D input rejected", ExpectThrow(() =>
        new BatchNormalization(2).Forward(np.zeros(new Shape(2, 2, 2), NPTypeCode.Single))).Contains("2-D"));
    Check("feature mismatch rejected", ExpectThrow(() =>
        new BatchNormalization(5).Forward(x)).Contains("5 features"));
}

// ======================================================================
Console.WriteLine("--- P4 LayerNormalization ---");
// ======================================================================
{
    var x = np.array(new float[,] { { 1f, 2f, -3f, 0.5f }, { 3f, 5f, 0.5f, -1f } });

    var ln = new LayerNormalization(4, epsilon: 1e-8f);
    ln.Forward(x);
    // Per-SAMPLE normalization: every ROW has zero mean and unit variance.
    var rowMean = np.mean(ln.Output, axis: 1);
    var rowVar = np.var(ln.Output, axis: 1);
    CheckClose("row means ~ 0", (float)np.sum(np.abs(rowMean)), 0f, 1e-3f);
    CheckClose("row vars ~ 1", (float)np.sum(rowVar), 2f, 1e-3f);
    Check("LayerNorm has no running state", ln.NonTrainable.Count == 0);

    // Batch-independence: scoring a row alone gives the same answer.
    var lnSolo = new LayerNormalization(4, epsilon: 1e-8f);
    lnSolo.Forward(x[$"0:1"]);
    for (int j = 0; j < 4; j++)
        CheckClose($"batch-independent [{j}]", lnSolo.Output.GetSingle(0, j), ln.Output.GetSingle(0, j), 1e-4f);

    // LayerNorm ignores the Training flag entirely.
    var lnT = new LayerNormalization(4) { Training = true };
    var lnE = new LayerNormalization(4) { Training = false };
    lnT.Forward(x); lnE.Forward(x);
    CheckClose("Training flag does not change LayerNorm",
               (float)np.sum(np.abs(lnT.Output - lnE.Output)), 0f);

    FdCheck("ln dx", () => { var l = new LayerNormalization(4); Randomize(l, 4); return l; },
            x, np.array(new float[,] { { 1f, -2f, 0.5f, 0.25f }, { 0.25f, 3f, -1f, 1f } }));

    Check("LN feature mismatch rejected", ExpectThrow(() =>
        new LayerNormalization(9).Forward(x)).Contains("9 features"));
}

// ======================================================================
Console.WriteLine("--- P4 Embedding ---");
// ======================================================================
{
    var w = np.array(new float[,] { { 0.1f, 0.2f }, { -0.3f, 0.4f }, { 0.5f, -0.6f } });
    var emb = new Embedding(3, 2);
    emb.Parameters["w"] = w;

    // Duplicate indices MUST accumulate — the whole reason backward is a
    // scatter-ADD and not an assignment.
    emb.Forward(np.array(new int[] { 1, 1, 1 }));
    CheckEq("gather shape rows", emb.Output.shape[0], 3);
    CheckClose("gathered row value", emb.Output.GetSingle(2, 0), -0.3f);
    emb.Backward(np.array(new float[,] { { 1f, 10f }, { 2f, 20f }, { 4f, 40f } }));
    CheckClose("duplicate grads accumulate (col 0)", emb.Grads["w"].GetSingle(1, 0), 7f);
    CheckClose("duplicate grads accumulate (col 1)", emb.Grads["w"].GetSingle(1, 1), 70f);
    CheckClose("untouched row gets zero", emb.Grads["w"].GetSingle(0, 0), 0f);
    CheckClose("untouched row 2 gets zero", emb.Grads["w"].GetSingle(2, 1), 0f);
    Check("InputGrad is null", emb.InputGrad is null);

    // Index dtypes this project's arrays actually carry.
    foreach (var (label, idx) in new (string, NDArray)[]
    {
        ("int32", np.array(new int[] { 0, 2 })),
        ("int64", np.array(new long[] { 0, 2 })),
        ("byte", np.array(new byte[] { 0, 2 })),
    })
    {
        var e2 = new Embedding(3, 2);
        e2.Parameters["w"] = w;
        e2.Forward(idx);
        CheckClose($"index dtype {label}", e2.Output.GetSingle(1, 0), 0.5f);
    }

    // 2-D (batch, timesteps) input.
    var e3 = new Embedding(3, 2);
    e3.Parameters["w"] = w;
    e3.Forward(np.array(new int[,] { { 0, 1 }, { 2, 1 } }));
    CheckEq("2-D output ndim", e3.Output.ndim, 3);
    CheckEq("2-D output batch", e3.Output.shape[0], 2);
    CheckEq("2-D output steps", e3.Output.shape[1], 2);
    CheckEq("2-D output dim", e3.Output.shape[2], 2);

    // An Embedding at the head of a real stack: the null InputGrad must not
    // break the trainer's backward sweep.
    np.random.seed(5);
    var model = new List<BaseLayer> { new Embedding(6, 4), new Flatten(), new FullyConnectedFused(4, 3, FusedActivation.None) };
    var ids = np.array(new int[] { 0, 1, 2, 3, 4, 5 });
    var labels = np.array(new byte[] { 0, 1, 2, 0, 1, 2 });
    var r = MlpTrainer.Train(model, new SoftmaxCrossEntropy(), new Adam(lr: 0.05f),
        ids, labels, null, null, epochs: 5, batchSize: 3, numClasses: 3, shuffle: false, verbose: 0);
    Check("embedding stack trains", r.EpochLoss[4] < r.EpochLoss[0],
          $"{r.EpochLoss[0]} -> {r.EpochLoss[4]}");
    Check("embedding weights moved", (float)np.sum(np.abs(model[0].Parameters["w"])) > 0f);

    // Guards.
    Check("out-of-range index rejected", ExpectThrow(() =>
        new Embedding(3, 2).Forward(np.array(new int[] { 5 }))).Contains("[0, 3)"));
    Check("negative index rejected", ExpectThrow(() =>
        new Embedding(3, 2).Forward(np.array(new int[] { -1 }))).Contains("[0, 3)"));
    Check("float indices rejected", ExpectThrow(() =>
        new Embedding(3, 2).Forward(np.array(new float[] { 1f }))).Contains("integer dtype"));
    Check("3-D indices rejected", ExpectThrow(() =>
        new Embedding(3, 2).Forward(np.zeros(new Shape(2, 2, 2), NPTypeCode.Int32))).Contains("3-D"));
}

// ======================================================================
Console.WriteLine("--- P4 Flatten / Reshape ---");
// ======================================================================
{
    var x = np.reshape(np.arange(24).astype(NPTypeCode.Single), new Shape(2, 3, 4));

    var f = new Flatten();
    f.Forward(x);
    CheckEq("flatten batch", f.Output.shape[0], 2);
    CheckEq("flatten features", f.Output.shape[1], 12);
    f.Backward(np.reshape(np.arange(24).astype(NPTypeCode.Single), new Shape(2, 12)));
    CheckEq("flatten backward restores ndim", f.InputGrad.ndim, 3);
    CheckEq("flatten backward restores dim1", f.InputGrad.shape[1], 3);
    CheckClose("flatten backward preserves values", (float)np.sum(f.InputGrad), 276f);

    var r = new Reshape(3, 4);
    r.Forward(np.reshape(np.arange(24).astype(NPTypeCode.Single), new Shape(2, 12)));
    CheckEq("reshape ndim", r.Output.ndim, 3);
    CheckEq("reshape dim1", r.Output.shape[1], 3);
    CheckEq("reshape dim2", r.Output.shape[2], 4);
    r.Backward(x);
    CheckEq("reshape backward ndim", r.InputGrad.ndim, 2);
    CheckEq("reshape backward cols", r.InputGrad.shape[1], 12);

    // One instance handles a partial final batch (batch read per call).
    r.Forward(np.zeros(new Shape(5, 12), NPTypeCode.Single));
    CheckEq("reshape adapts to a shorter batch", r.Output.shape[0], 5);

    Check("reshape size mismatch rejected", ExpectThrow(() =>
        new Reshape(3, 4).Forward(np.zeros(new Shape(2, 10), NPTypeCode.Single))).Contains("cannot map"));
    Check("reshape rejects -1", ExpectThrow(() => new Reshape(-1, 4)).Contains("positive"));
    Check("reshape rejects empty", ExpectThrow(() => new Reshape()).Contains("at least one"));
}

// ======================================================================
Console.WriteLine("--- P4 regularizers (wiring, not values) ---");
// ======================================================================
{
    var w = np.array(new float[,] { { 1f, -2f }, { 3f, -4f } });

    // Penalty/gradient consistency is oracle-pinned; here: the resolver.
    Check("Get('l2') resolves", BaseRegularizer.Get("l2") is L2);
    Check("Get('l1l2') resolves", BaseRegularizer.Get("l1l2") is L1L2);
    Check("Get('') is null", BaseRegularizer.Get("") is null);
    Check("Get(unknown) throws", ExpectThrow(() => BaseRegularizer.Get("nope")).Contains("Unknown regularizer"));

    // Attached to a layer: the penalty is reported and the gradient applied.
    var layer = new VecStub(new float[] { 3f, -4f });
    layer.Regularizers["w"] = new L2(0.5f);
    CheckClose("layer penalty = l2*sum(w^2)", layer.RegularizationPenalty(), 0.5f * 25f);

    layer.Grads["w"] = np.array(new float[] { 1f, 1f });
    layer.ApplyRegularizerGradients();
    CheckClose("grad gains 2*l2*w [0]", layer.Grads["w"].GetSingle(0), 1f + 2f * 0.5f * 3f);
    CheckClose("grad gains 2*l2*w [1]", layer.Grads["w"].GetSingle(1), 1f + 2f * 0.5f * -4f);

    // No regularizer attached: nothing changes, nothing is reported.
    var plain = new VecStub(new float[] { 3f, -4f });
    plain.Grads["w"] = np.array(new float[] { 1f, 1f });
    plain.ApplyRegularizerGradients();
    CheckClose("no regularizer leaves grads alone", plain.Grads["w"].GetSingle(0), 1f);
    CheckClose("no regularizer scores 0", plain.RegularizationPenalty(), 0f);

    // The trainer adds the penalty to the REPORTED loss and pushes the
    // gradient into the optimizer step.
    np.random.seed(3);
    var model = new List<BaseLayer> { new FullyConnectedFused(3, 3, FusedActivation.None) };
    var xs = np.zeros(new Shape(4, 3), NPTypeCode.Single);
    var ys = np.array(new byte[] { 0, 1, 2, 0 });

    var bare = MlpTrainer.Train(model, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 1, batchSize: 4, numClasses: 3, shuffle: false, verbose: 0);

    float expectedPenalty = 0.5f * (float)np.sum(model[0].Parameters["w"] * model[0].Parameters["w"]);
    model[0].Regularizers["w"] = new L2(0.5f);
    var penalized = MlpTrainer.Train(model, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 1, batchSize: 4, numClasses: 3, shuffle: false, verbose: 0);

    CheckClose("trainer adds the penalty to the reported loss",
               penalized.EpochLoss[0] - bare.EpochLoss[0], expectedPenalty, 1e-3f);

    // With lr>0 and a zero input the ONLY gradient on w is the regularizer's,
    // so the weights must shrink toward 0 (ridge decay), not stay put.
    np.random.seed(3);
    var decayModel = new List<BaseLayer> { new FullyConnectedFused(3, 3, FusedActivation.None) };
    decayModel[0].Regularizers["w"] = new L2(0.5f);
    float before = (float)np.sum(np.abs(decayModel[0].Parameters["w"]));
    MlpTrainer.Train(decayModel, new SoftmaxCrossEntropy(), new SGD(lr: 0.01f),
        xs, ys, null, null, epochs: 3, batchSize: 4, numClasses: 3, shuffle: false, verbose: 0);
    float after = (float)np.sum(np.abs(decayModel[0].Parameters["w"]));
    Check("regularizer gradient reaches the optimizer", after < before, $"{before} -> {after}");
}

// ======================================================================
Console.WriteLine("--- P4 serialization of the new layers ---");
// ======================================================================
{
    np.random.seed(17);
    var model = new List<BaseLayer>
    {
        new Embedding(10, 4),
        new Flatten(),
        new BatchNormalization(4),
        new Dropout(0.25f),
        new LayerNormalization(4),
        new Reshape(2, 2),
        new Flatten(),
        new FullyConnected(4, 3, "relu"),
    };

    // Move the BatchNorm running stats off their initial values so a lost
    // round-trip is visible.
    model[2].NonTrainable["moving_mean"] = np.array(new float[] { 0.5f, -1f, 2f, 0.25f });
    model[2].NonTrainable["moving_variance"] = np.array(new float[] { 2f, 3f, 0.5f, 1.5f });

    string ck = Path.Combine(tmpRoot, "p4.npz");
    ModelWeights.Save(model, ck);
    using (var npz = np.load_npz(ck))
    {
        var files = npz.Files.ToArray();
        Check("embedding weights archived", files.Contains("layer0/param/w"));
        Check("batchnorm gamma archived", files.Contains("layer2/param/gamma"));
        Check("batchnorm running stats archived",
              files.Contains("layer2/state/moving_mean") && files.Contains("layer2/state/moving_variance"));
        Check("layernorm beta archived", files.Contains("layer4/param/beta"));
        Check("stateless layers archive nothing",
              !files.Any(f => f.StartsWith("layer1/") || f.StartsWith("layer3/") || f.StartsWith("layer5/")));
    }

    // Architecture round-trip through JSON, then weights into it.
    string json = ModelArchitecture.ToJson(model);
    var rebuilt = ModelArchitecture.FromJson(json);
    CheckEq("json rebuilt layer count", rebuilt.Count, 8);
    Check("json rebuilt types", rebuilt[0] is Embedding && rebuilt[1] is Flatten &&
                                rebuilt[2] is BatchNormalization && rebuilt[3] is Dropout &&
                                rebuilt[4] is LayerNormalization && rebuilt[5] is Reshape);
    CheckClose("dropout rate survives json", ((Dropout)rebuilt[3]).Rate, 0.25f);
    CheckClose("batchnorm epsilon survives json", ((BatchNormalization)rebuilt[2]).Epsilon, 1e-3f);
    CheckClose("batchnorm momentum survives json", ((BatchNormalization)rebuilt[2]).Momentum, 0.99f);
    Check("reshape target survives json", ((Reshape)rebuilt[5]).TargetShape.SequenceEqual(new[] { 2, 2 }));
    CheckEq("embedding dims survive json", ((Embedding)rebuilt[0]).OutputDim, 4);

    ModelWeights.Load(rebuilt, ck);
    CheckClose("embedding weights restored",
               (float)MaxAbsDiff(model[0].Parameters["w"], rebuilt[0].Parameters["w"]), 0f);
    CheckClose("batchnorm running mean restored",
               (float)MaxAbsDiff(model[2].NonTrainable["moving_mean"], rebuilt[2].NonTrainable["moving_mean"]), 0f);
    CheckClose("batchnorm running variance restored",
               (float)MaxAbsDiff(model[2].NonTrainable["moving_variance"], rebuilt[2].NonTrainable["moving_variance"]), 0f);

    // A reloaded model must EVALUATE identically — the reason running stats
    // are in the archive at all.
    var probe = np.array(new float[,] { { 1f, 2f, 3f, 4f } });
    NDArray Run(List<BaseLayer> m)
    {
        NDArray act = probe;
        for (int i = 2; i <= 4; i++) { m[i].Training = false; m[i].Forward(act); act = m[i].Output; }
        return act;
    }
    CheckClose("reloaded model evaluates identically", (float)MaxAbsDiff(Run(model), Run(rebuilt)), 0f);
}

Console.WriteLine();
Console.WriteLine($"RESULT: {pass} passed, {fail} failed");
return fail == 0 ? 0 : 1;

}
finally
{
    try { Directory.Delete(tmpRoot, recursive: true); } catch { /* best effort */ }
}

// ======================================================================
// helpers
// ======================================================================

double MaxAbsDiff(NDArray a, NDArray b)
{
    var fa = np.reshape(a, new Shape((int)a.size)).astype(NPTypeCode.Single);
    var fb = np.reshape(b, new Shape((int)b.size)).astype(NPTypeCode.Single);
    double max = 0;
    for (int i = 0; i < fa.size; i++)
        max = Math.Max(max, Math.Abs(fa.GetSingle(i) - fb.GetSingle(i)));
    return max;
}

string ExpectThrow(Action a)
{
    try { a(); }
    catch (Exception ex) { return ex.Message; }
    return "<no exception thrown>";
}

// Gives gamma/beta non-trivial values so a backward that ignores them fails.
void Randomize(BaseLayer layer, int features)
{
    var gamma = new float[features];
    var beta = new float[features];
    for (int i = 0; i < features; i++) { gamma[i] = 1.5f - 0.5f * i; beta[i] = 0.25f * (i + 1); }
    layer.Parameters["gamma"] = np.array(gamma);
    layer.Parameters["beta"] = np.array(beta);
}

// Central-difference check of InputGrad against L = sum(Forward(x) * upstream).
// Independent of Keras: a second opinion on the hand-derived backward.
void FdCheck(string name, Func<BaseLayer> make, NDArray x, NDArray upstream)
{
    var layer = make();
    layer.Forward(x);
    layer.Backward(upstream);
    NDArray analytic = layer.InputGrad;

    const float h = 1e-2f;
    int rows = (int)x.shape[0], cols = (int)x.shape[1];
    float worst = 0f;

    float Loss(NDArray input)
    {
        var l = make();
        l.Forward(input);
        return (float)np.sum(l.Output * upstream);
    }

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            var xp = x.copy(); xp[$"{i}:{i + 1}"][$":, {j}:{j + 1}"] = (NDArray)(x.GetSingle(i, j) + h);
            var xm = x.copy(); xm[$"{i}:{i + 1}"][$":, {j}:{j + 1}"] = (NDArray)(x.GetSingle(i, j) - h);
            float fd = (Loss(xp) - Loss(xm)) / (2f * h);
            worst = Math.Max(worst, Math.Abs(fd - analytic.GetSingle(i, j)));
        }

    Check(name + " matches central differences", worst < 2e-2f, $"max|fd-analytic|={worst}");
}

// ---- test doubles -----------------------------------------------------

class VecStub : BaseLayer
{
    public VecStub(float[] w) : base("vec") { Parameters["w"] = np.array(w); }
}

// Echo layer that counts how many forwards ran in each mode.
class TrainingSpy : BaseLayer
{
    public int TrainingCalls, InferenceCalls;

    public TrainingSpy() : base("tspy") { }

    public override void Forward(NDArray x)
    {
        base.Forward(x);
        if (Training) TrainingCalls++; else InferenceCalls++;
        Output = x;
    }

    public override void Backward(NDArray grad) => InputGrad = grad;
}
